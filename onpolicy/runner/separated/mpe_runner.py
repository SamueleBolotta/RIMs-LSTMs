import time
import wandb
import os
import numpy as np
from itertools import chain
import torch

from onpolicy.utils.util import update_linear_schedule
from onpolicy.runner.separated.base_runner import Runner
import imageio

def generator_possible_agents(envs, num_agents):
    if envs == 'BUTTERFLY-pistonball': 
        basnm = 'piston'
    elif envs == 'BUTTERFLY-pong':
        basnm = "paddle"
    elif envs == 'MPE-simple.spread':
        basnm = "agent" 
    result = ["{}_{}".format(basnm, i) for i in range(0, num_agents)]
    return result

def unbatchify(x, num_ag):
    """Converts np array to PZ style arguments."""
    x = {a: x[i][0] for i, a in enumerate(num_ag)}
    return x

def batchify_obs(obs, device):
    """Converts PZ style observations to batch of torch arrays."""
    # convert to list of np arrays
        
    obs = np.stack([obs[a] for a in obs], axis=0)
    obs = obs[np.newaxis, :, :]
    # transpose to be (batch, channel, height, width)
    if len(obs.shape) == 4:
        obs_n = obs.transpose(0, -1, 1, 2)
    else:
        obs_n = obs
   
    # convert to torch
    obs = torch.tensor(obs_n).to(device)

    return obs, obs_n

def topetzoo(agent_id, envs, num_agents):
    if envs == 'BUTTERFLY-pistonball': 
        basnm = 'piston'
    elif envs == 'BUTTERFLY-pong':
        basnm = "paddle"
    elif envs == 'MPE-simple.spread':
        basnm = "agent" 
    result = ["{}_{}".format(basnm, i) for i in range(0, num_agents)]
    return result[agent_id]
    
        
def before_pz(actions, envs, num_agents):
    if envs == 'BUTTERFLY-pistonball': 
        basnm = 'piston'
    elif envs == 'BUTTERFLY-pong':
        basnm = "paddle"
    elif envs == 'MPE-simple.spread':
        basnm = "agent"
    actions_step = {"{}_{}".format(basnm, i):int(actions[0][i]) for i in range(0, num_agents)}
    return actions_step

def after_pz(obs, rewards, dones, infos):
       
    obs = np.array(list(obs.values()))
    rewards = np.array(list(rewards.values()))
    dones = np.array(list(dones.values()))
    infos = np.array(list(infos.values()))
    obs = obs[np.newaxis, :, :]
    rewards = rewards[np.newaxis, :, np.newaxis]
    dones = dones[np.newaxis, :]
    return obs, rewards, dones, infos

def _t2n(x):
    return x.detach().cpu().numpy()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                      
class MPERunner(Runner):
    def __init__(self, config):
        super(MPERunner, self).__init__(config)
       
    def run(self):

        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads
        possible_agents = generator_possible_agents(self.env_name, self.num_agents)
        
        for episode in range(episodes):
            if self.use_linear_lr_decay:
                for agent_id in range(self.num_agents):
                    self.trainer[agent_id].policy.lr_decay(episode, episodes)
                    
            next_obs = self.envs.reset()   
            start = time.time()

            for step in range(self.episode_length):
                #Convert PZ style observations to batch of np arrays, for each rollout thread
                obs_n_list = self.prep(next_obs, device)
                      
                #Warmup
                self.warmup(obs_n_list)
                
                #Sample actions
                values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env = self.collect(step)
                
                act_list = []
                for i in range(self.n_rollout_threads):
                    actions_pz = unbatchify(actions[i], possible_agents)
                    act_list.append(actions_pz)
                
                print("act list", act_list)
                print("actions_pz", actions_pz)
                next_obs, rewards, dones, infos = self.envs.step(act_list)
                   
                do, rew, ob_l, infs = self.after_step(next_obs, rewards, dones, infos)

                data = ob_l, rew, do, infs, values, actions, action_log_probs, rnn_states, rnn_states_critic 
               
                # insert data into buffer
                self.insert(data)

            # compute return and update network
            self.compute()
            train_infos = self.train()
            
            # post process
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads
            
            # save model
            if (episode % self.save_interval == 0 or episode == episodes - 1):
                self.save()

            # log information
            if episode % self.log_interval == 0:
                end = time.time()
                print("\n Scenario {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n"
                        .format(self.all_args.scenario_name,
                                self.algorithm_name,
                                self.experiment_name,
                                episode,
                                episodes,
                                total_num_steps,
                                self.num_env_steps,
                                int(total_num_steps / (end - start))))

                for agent_id in range(self.num_agents):
                    print("np.mean(self.buffer[agent_id].rewards)", np.mean(self.buffer[agent_id].rewards))
                    train_infos[agent_id].update({"average_episode_rewards": np.mean(self.buffer[agent_id].rewards) * self.episode_length})
                self.log_train(train_infos, total_num_steps)

            # eval
            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)
                
    def prep(self, next_obs, device):
             
        obs_list = []
        obs_n_list = []

        for i in range(self.n_rollout_threads):
            obs, obs_n = batchify_obs(next_obs[i], device)
            obs_list.append(obs)
            obs_n_list.append(obs_n)
            
        return obs_n_list
            
    def warmup(self, obs):
        
        obs = np.squeeze(obs, axis=1)
        
        share_obs = []
        for o in obs:
            share_obs.append(list(chain(*o)))
        share_obs = np.array(share_obs)

        for agent_id in range(self.num_agents):
            if not self.use_centralized_V:
                if obs.ndim == 5:
                    share_obs = np.array(list(obs[agent_id, :, :, :]))
                elif obs.ndim == 3:
                    share_obs = np.array(list(obs[agent_id, :]))

            self.buffer[agent_id].share_obs[0] = share_obs.copy()
            
            obs = obs[0]
            if obs.ndim == 5:
                self.buffer[agent_id].obs[0] = np.array(list(obs[agent_id, :, :, :])).copy()
            elif obs.ndim == 3:
                temp = np.array(list(obs[agent_id, :]))
                self.buffer[agent_id].obs[0] = np.array(list(obs[agent_id, :])).copy()

    def after_step(next_obs, rewards, dones, infos):
        
        ob_l = []
        rew = []
        do = []
        infs = []
                
        for i in range(self.n_rollout_threads):

            obs__, rewards__, dones__, infos__ = after_pz(next_obs[i], rewards[i], dones[i], infos[i])
            ob_l.append(obs__[0])
            rew.append(rewards__[0])
            do.append(dones__.tolist()[0])
            infs.append(infos__.tolist())
                    
            do = np.array(do)
            rew = np.array(rew)
            ob_l = np.array(ob_l)
            infs = tuple(infs)
            
            
        return do, rew, ob_l, infs
    
    @torch.no_grad()
    def collect(self, step):
        values = []
        actions = []
        temp_actions_env = []
        action_log_probs = []
        rnn_states = []
        rnn_states_critic = []

        for agent_id in range(self.num_agents):
            agent_id_pet = topetzoo(agent_id, self.env_name, self.num_agents)
            self.trainer[agent_id].prep_rollout()
            value, action, action_log_prob, rnn_state, rnn_state_critic \
                = self.trainer[agent_id].policy.get_actions(self.buffer[agent_id].share_obs[step],
                                                            self.buffer[agent_id].obs[step],
                                                            self.buffer[agent_id].rnn_states[step],
                                                            self.buffer[agent_id].rnn_states_critic[step],
                                                            self.buffer[agent_id].masks[step])
            # [agents, envs, dim]
            values.append(_t2n(value))
            action = _t2n(action)
            # rearrange action
            if self.envs.action_space(agent_id_pet).__class__.__name__ == 'MultiDiscrete':
                for i in range(self.envs.action_space(agent_id_pet).shape):
                    uc_action_env = np.eye(self.envs.action_space(agent_id_pet).high[i]+1)[action[:, i]]
                    if i == 0:
                        action_env = uc_action_env
                    else:
                        action_env = np.concatenate((action_env, uc_action_env), axis=1)
            elif self.envs.action_space(agent_id_pet).__class__.__name__ == 'Discrete':
                action_env = np.squeeze(np.eye(self.envs.action_space(agent_id_pet).n)[action], 1)
            else:
                raise NotImplementedError

            actions.append(action)
            temp_actions_env.append(action_env)
            action_log_probs.append(_t2n(action_log_prob))
            rnn_states.append(_t2n(rnn_state))
            rnn_states_critic.append( _t2n(rnn_state_critic))

        # [envs, agents, dim]
        actions_env = []
        for i in range(self.n_rollout_threads):
            one_hot_action_env = []
            for temp_action_env in temp_actions_env:
                one_hot_action_env.append(temp_action_env[i])
            actions_env.append(one_hot_action_env)

        values = np.array(values).transpose(1, 0, 2)
        actions = np.array(actions).transpose(1, 0, 2)
        action_log_probs = np.array(action_log_probs).transpose(1, 0, 2)
        rnn_states = np.array(rnn_states).transpose(1, 0, 2, 3)
        rnn_states_critic = np.array(rnn_states_critic).transpose(1, 0, 2, 3)

        return values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env

    def insert(self, data):
        obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic = data

        rnn_states[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
        rnn_states_critic[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

        share_obs = []
        for o in obs:
            share_obs.append(list(chain(*o)))
        share_obs = np.array(share_obs)

        for agent_id in range(self.num_agents):
            if not self.use_centralized_V:
                share_obs = np.array(list(obs[:, agent_id]))

            self.buffer[agent_id].insert(share_obs,
                                        np.array(list(obs[:, agent_id])),
                                        rnn_states[:, agent_id],
                                        rnn_states_critic[:, agent_id],
                                        actions[:, agent_id],
                                        action_log_probs[:, agent_id],
                                        values[:, agent_id],
                                        rewards[:, agent_id],
                                        masks[:, agent_id])

    @torch.no_grad()
    def eval(self, total_num_steps):
        eval_episode_rewards = []
        eval_obs = self.eval_envs.reset()

        eval_rnn_states = np.zeros((self.n_eval_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)

        for eval_step in range(self.episode_length):
            eval_temp_actions_env = []
            for agent_id in range(self.num_agents):
                agent_id_pet = topetzoo(agent_id, self.env_name, self.num_agents)
                self.trainer[agent_id].prep_rollout()
                eval_action, eval_rnn_state = self.trainer[agent_id].policy.act(np.array(list(eval_obs[:, agent_id])),
                                                                                eval_rnn_states[:, agent_id],
                                                                                eval_masks[:, agent_id],
                                                                                deterministic=True)

                eval_action = eval_action.detach().cpu().numpy()
                # rearrange action
                if self.eval_envs.action_spaces(agent_id_pet).__class__.__name__ == 'MultiDiscrete':
                    for i in range(self.eval_envs.action_spaces(agent_id_pet).shape):
                        eval_uc_action_env = np.eye(self.eval_envs.action_spaces(agent_id_pet).high[i]+1)[eval_action[:, i]]
                        if i == 0:
                            eval_action_env = eval_uc_action_env
                        else:
                            eval_action_env = np.concatenate((eval_action_env, eval_uc_action_env), axis=1)
                elif self.eval_envs.action_spaces(agent_id_pet).__class__.__name__ == 'Discrete':
                    eval_action_env = np.squeeze(np.eye(self.eval_envs.action_spaces(agent_id_pet).n)[eval_action], 1)
                else:
                    raise NotImplementedError

                eval_temp_actions_env.append(eval_action_env)
                eval_rnn_states[:, agent_id] = _t2n(eval_rnn_state)
                
            # [envs, agents, dim]
            eval_actions_env = []
            for i in range(self.n_eval_rollout_threads):
                eval_one_hot_action_env = []
                for eval_temp_action_env in eval_temp_actions_env:
                    eval_one_hot_action_env.append(eval_temp_action_env[i])
                eval_actions_env.append(eval_one_hot_action_env)

            # Obser reward and next obs
            eval_obs, eval_rewards, eval_terms, eval_infos = self.eval_envs.step(eval_actions_env)
            eval_episode_rewards.append(eval_rewards)

            eval_rnn_states[eval_terms == True] = np.zeros(((eval_terms == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
            eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
            eval_masks[eval_terms == True] = np.zeros(((eval_terms == True).sum(), 1), dtype=np.float32)

        eval_episode_rewards = np.array(eval_episode_rewards)
        
        for agent_id in range(self.num_agents):
            eval_average_episode_rewards = np.mean(np.sum(eval_episode_rewards[:, :, agent_id], axis=0))
            eval_train_infos.append({'eval_average_episode_rewards': eval_average_episode_rewards})
            print("eval average episode rewards of agent%i: " % agent_id + str(eval_average_episode_rewards))

        self.log_train(eval_train_infos, total_num_steps)  

    
