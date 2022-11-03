#!/usr/bin/env python
import sys
import os
import wandb
import socket
import setproctitle
import numpy as np
from pathlib import Path
import torch
from onpolicy.config import get_config
from pettingzoo.mpe import simple_spread_v2
from pettingzoo.mpe._mpe_utils.core import Agent, Landmark, World
from pettingzoo.mpe._mpe_utils.scenario import BaseScenario
from pettingzoo.mpe._mpe_utils.simple_env import SimpleEnv, make_env
from pettingzoo.utils.conversions import parallel_wrapper_fn
from onpolicy.scripts.train.env_wrappers import SubprocVecEnv, DummyVecEnv

"""Train script for MPEs."""

def make_train_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "MPE-simple.spread":
                envs = simple_spread_v2.parallel_env(N=3, local_ratio=0.5, max_cycles=int(all_args.num_env_steps), continuous_actions=False)
            envs.reset(seed=all_args.seed + rank * 1000)
            return envs
        return init_env
    if all_args.n_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])

def make_eval_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "MPE-simple.spread":
                envs = simple_spread_v2.parallel_env(N=3, local_ratio=0.5, max_cycles=int(all_args.num_env_steps), continuous_actions=False)
            envs.reset(seed=all_args.seed + rank * 1000)
            return envs
        return init_env
    if all_args.n_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)])

def parse_args(args, parser):
    parser.add_argument('--scenario_name', type=str,
                        default='simple_spread', help="Which scenario to run on")
    parser.add_argument("--num_landmarks", type=int, default=3)
    parser.add_argument('--num_agents', type=int,
                        default=2, help="number of players")

    all_args = parser.parse_known_args(args)[0]

    return all_args


def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)

    if all_args.algorithm_name == "rmappo":
        assert (all_args.use_recurrent_policy or all_args.use_naive_recurrent_policy), ("check recurrent policy!")
    elif all_args.algorithm_name == "mappo":
        assert (all_args.use_recurrent_policy == False and all_args.use_naive_recurrent_policy == False), ("check recurrent policy!")
    else:
        raise NotImplementedError

    assert (all_args.share_policy == True and all_args.scenario_name == 'simple_speaker_listener') == False, (
        "The simple_speaker_listener scenario can not use shared policy. Please check the config.py.")

    # cuda
    if all_args.cuda and torch.cuda.is_available():
        print("choose to use gpu...")
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("choose to use cpu...")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    # run dir
    run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[
                   0] + "/results") / all_args.env_name / all_args.scenario_name / all_args.algorithm_name / all_args.experiment_name
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    # wandb
    if all_args.use_wandb:
        run = wandb.init(config=all_args,
                         project=all_args.env_name,
                         entity=all_args.user_name,
                         notes=socket.gethostname(),
                         name=str(all_args.algorithm_name) + "_" +
                         str(all_args.experiment_name) +
                         "_seed" + str(all_args.seed),
                         group=all_args.scenario_name,
                         dir=str(run_dir),
                         job_type="training",
                         reinit=True)
    else:
        if not run_dir.exists():
            curr_run = 'run1'
        else:
            exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in run_dir.iterdir() if str(folder.name).startswith('run')]
            if len(exst_run_nums) == 0:
                curr_run = 'run1'
            else:
                curr_run = 'run%i' % (max(exst_run_nums) + 1)
        run_dir = run_dir / curr_run
        if not run_dir.exists():
            os.makedirs(str(run_dir))

    setproctitle.setproctitle(str(all_args.algorithm_name) + "-" + \
        str(all_args.env_name) + "-" + str(all_args.experiment_name) + "@" + str(all_args.user_name))

    # seed
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)
    
    num_env_steps = all_args.num_env_steps
    num_agents = all_args.num_agents

    # env init
    from pettingzoo.mpe import simple_spread_v2   
    envs = make_train_env(all_args)
    eval_envs = make_eval_env(all_args) if all_args.use_eval else None
    
    if all_args.n_rollout_threads == 1:
        envs = simple_spread_v2.parallel_env(N=3, local_ratio=0.5, max_cycles=int(num_env_steps), continuous_actions=False)
    else:    
        import supersuit as ss
        envs = simple_spread_v2.parallel_env(N=3, local_ratio=0.5, max_cycles=int(num_env_steps), continuous_actions=False)
        envs = ss.pettingzoo_env_to_vec_env_v1(envs)
        envs = ss.concat_vec_envs_v1(envs, all_args.n_rollout_threads, num_cpus=4, base_class='gym')
        
    # eval env init
    if all_args.n_rollout_threads == 1:
        eval_envs = simple_spread_v2.parallel_env(N=3, local_ratio=0.5, max_cycles=int(num_env_steps), continuous_actions=False)
    else:    
        eval_envs = simple_spread_v2.parallel_env(N=3, local_ratio=0.5, max_cycles=int(num_env_steps), continuous_actions=False)
        eval_envs = ss.pettingzoo_env_to_vec_env_v1(eval_envs)
        eval_envs = ss.concat_vec_envs_v1(eval_envs, all_args.n_rollout_threads, num_cpus=4, base_class='gym')    
                
    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": num_agents,
        "device": device,
        "run_dir": run_dir
    }

    # run experiments
    if all_args.share_policy:
        from onpolicy.runner.shared.mpe_runner import MPERunner as Runner
    else:
        from onpolicy.runner.separated.mpe_runner import MPERunner as Runner

    runner = Runner(config)
    runner.run()
    
    # post process
    envs.close()
    if all_args.use_eval and eval_envs is not envs:
        eval_envs.close()

    if all_args.use_wandb:
        run.finish()
    else:
        runner.writter.export_scalars_to_json(str(runner.log_dir + '/summary.json'))
        runner.writter.close()


if __name__ == "__main__":
    main(sys.argv[1:])
