import numpy as np
import math
import torch

def generator_possible_agents(envs, num_agents):
    if envs == 'BUTTERFLY-pistonball': 
        basnm = 'piston'
    elif envs == 'BUTTERFLY-pong':
        basnm = "paddle"
    elif envs == 'MPE-simple.spread':
        basnm = "agent" 
    elif envs == 'SISL-multiwalker':
        basnm = "walker"
    result = ["{}_{}".format(basnm, i) for i in range(0, num_agents)]
    return result

def unbatchify(x, num_ag):
    """Converts np array to PZ style arguments."""
    x = {a: x[i][:] for i, a in enumerate(num_ag)}
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
    elif envs == 'SISL-multiwalker':
        basnm = "walker"   
    result = ["{}_{}".format(basnm, i) for i in range(0, num_agents)]
    return result[agent_id]
    
        
def before_pz(actions, envs, num_agents):
    if envs == 'BUTTERFLY-pistonball': 
        basnm = 'piston'
    elif envs == 'BUTTERFLY-pong':
        basnm = "paddle"
    elif envs == 'MPE-simple.spread':
        basnm = "agent"
    elif envs == 'SISL-multiwalker':
        basnm = "walker"
    actions_step = {"{}_{}".format(basnm, i):int(actions[0][i]) for i in range(0, num_agents)}
    return actions_step

def after_pz(obs, rewards, terms, truncs, infos):
       
    obs = np.array(list(obs.values()))
    rewards = np.array(list(rewards.values()))
    terms = np.array(list(terms.values()))
    truncs = np.array(list(truncs.values()))
    infos = np.array(list(infos.values()))
    print("shape of obs", np.shape(obs))
    obs = obs[np.newaxis, :, :]
    rewards = rewards[np.newaxis, :, np.newaxis]
    terms = terms[np.newaxis, :]
    truncs = truncs[np.newaxis, :]

    return obs, rewards, terms, truncs, infos

def _t2n(x):
    return x.detach().cpu().numpy()

def check(input):
    if type(input) == np.ndarray:
        return torch.from_numpy(input)
        
def get_gard_norm(it):
    sum_grad = 0
    for x in it:
        if x.grad is None:
            continue
        sum_grad += x.grad.norm() ** 2
    return math.sqrt(sum_grad)

def update_linear_schedule(optimizer, epoch, total_num_epochs, initial_lr):
    """Decreases the learning rate linearly"""
    lr = initial_lr - (initial_lr * (epoch / float(total_num_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def huber_loss(e, d):
    a = (abs(e) <= d).float()
    b = (e > d).float()
    return a*e**2/2 + b*d*(abs(e)-d/2)

def mse_loss(e):
    return e**2/2

def get_shape_from_obs_space(obs_space):
    if obs_space.__class__.__name__ == 'Box':
        obs_shape = obs_space.shape
    elif obs_space.__class__.__name__ == 'list':
        obs_shape = obs_space
    else:
        raise NotImplementedError
    return obs_shape

def get_shape_from_act_space(act_space):
    if act_space.__class__.__name__ == 'Discrete':
        act_shape = 1
    elif act_space.__class__.__name__ == "MultiDiscrete":
        act_shape = act_space.shape
    elif act_space.__class__.__name__ == "Box":
        act_shape = act_space.shape[0]
    elif act_space.__class__.__name__ == "MultiBinary":
        act_shape = act_space.shape[0]
    else:  # agar
        act_shape = act_space[0].shape[0] + 1  
    return act_shape


def tile_images(img_nhwc):
    """
    Tile N images into one big PxQ image
    (P,Q) are chosen to be as close as possible, and if N
    is square, then P=Q.
    input: img_nhwc, list or array of images, ndim=4 once turned into array
        n = batch index, h = height, w = width, c = channel
    returns:
        bigim_HWc, ndarray with ndim=3
    """
    img_nhwc = np.asarray(img_nhwc)
    N, h, w, c = img_nhwc.shape
    H = int(np.ceil(np.sqrt(N)))
    W = int(np.ceil(float(N)/H))
    img_nhwc = np.array(list(img_nhwc) + [img_nhwc[0]*0 for _ in range(N, H*W)])
    img_HWhwc = img_nhwc.reshape(H, W, h, w, c)
    img_HhWwc = img_HWhwc.transpose(0, 2, 1, 3, 4)
    img_Hh_Ww_c = img_HhWwc.reshape(H*h, W*w, c)
    return img_Hh_Ww_c
