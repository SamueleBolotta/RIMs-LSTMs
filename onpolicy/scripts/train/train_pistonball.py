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
from supersuit import color_reduction_v0, frame_stack_v1, resize_v1
from pettingzoo.butterfly import pistonball_v6
from pettingzoo.utils.conversions import parallel_wrapper_fn

"""Train script for MPEs."""

def parse_args(args, parser):
    parser.add_argument('--scenario_name', type=str,
                        default='pistonball_v6', help="Which scenario to run on")
    parser.add_argument('--num_agents', type=int,
                        default=20, help="number of players")
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
    from pettingzoo.butterfly import pistonball_v6    
    if all_args.n_rollout_threads == 1:
        stack_size = 4
        frame_size = (64, 64)
        envs = pistonball_v6.parallel_env()
        envs = color_reduction_v0(envs)
        envs = resize_v1(envs, frame_size[0], frame_size[1])
        envs = frame_stack_v1(envs, stack_size=stack_size)
        envs = ss.pettingzoo_env_to_vec_env_v1(envs)
        envs = ss.concat_vec_envs_v1(envs, 8, num_cpus=4, base_class=’gym’)
        
    # eval env init
    if all_args.n_rollout_threads == 1:
        stack_size = 4
        frame_size = (64, 64)
        eval_envs = pistonball_v6.parallel_env()
        eval_envs = color_reduction_v0(eval_envs)
        eval_envs = resize_v1(eval_envs, frame_size[0], frame_size[1])
        eval_envs = frame_stack_v1(eval_envs, stack_size=stack_size)
        eval_envs = ss.pettingzoo_env_to_vec_env_v1(eval_envs)
        eval_envs = ss.concat_vec_envs_v1(eval_envs, 8, num_cpus=4, base_class=’gym’)
        
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
