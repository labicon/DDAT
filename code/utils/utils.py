# -*- coding: utf-8 -*-
"""
Created on Tue May 14 10:42:51 2024

@author: jeanb
"""

import copy
import torch
import random
import itertools
import numpy as np
import matplotlib.pyplot as plt


#%% Generic utils

def norm(x, dim=None, axis=None):
    """Calculates the norm of a vector x, either torch.Tensor or numpy.ndarray"""
    if type(x) == np.ndarray:
        if axis is not None:
            return np.linalg.norm(x, axis=axis)
        return np.linalg.norm(x)
    elif type(x) == torch.Tensor:
        if dim is not None:
            return torch.linalg.vector_norm(x, dim=dim)
        return torch.linalg.vector_norm(x).item()
    else:
        raise Exception(f"norm only works for torch.Tensor and numpy.ndarray, not {type(x)}")


def set_seed(seed: int):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


    
class Normalizer():
    def __init__(self, x: torch.Tensor):
        """Normalizing states for the Diffusion, same mean std for all timesteps
        But different mean for each state"""
        if len(x.shape) == 1: # saved mean
            self.mean, self.std = x, 0.
        elif len(x.shape) == 2: # single trajectory
            self.mean, self.std = x.mean(dim=0), x.std(dim=0)
        elif len(x.shape) == 3: # dataset of trajectories
            self.mean, self.std = x.mean(dim=(0,1)), x.std(dim=(0,1))
        print(f"Normalization vector has shape {self.mean.shape}")
        
    def normalize(self, x: torch.Tensor):
        """Normalize a trajectory starting"""
        return (x - self.mean) / self.std
    
    def unnormalize(self, x: torch.Tensor):
        """Unnormalize a whole trajectory"""
        return x * self.std + self.mean  
    



def vertices(LBs, UBs):
    """Calculates all the vertices corresponding to the hyperrectangle defined by LBs and UBs."""
    N = LBs.shape[0]
    m = LBs.shape[1]
    assert LBs.shape == UBs.shape, "Lower_Bounds and Upper_Bounds must have the same shape"

    L = list(itertools.product(range(2), repeat=m))

    if type(LBs) == torch.Tensor:
        LBs_UBs = torch.cat((LBs.reshape(N,1,m), UBs.reshape(N,1,m)), dim=1)
        v = torch.zeros(N, 2**m, m)
        cols = torch.arange(m)
        for i in range(2**m):
            rows = torch.tensor(L[i])
            v[:,i] = LBs_UBs[:, rows, cols]
            
    elif type(LBs) == np.ndarray:
        LBs_UBs = np.concatenate((LBs.reshape(N,1,m), UBs.reshape(N,1,m)), axis=1)
        v = np.zeros((N, 2**m, m))
        cols = np.arange(m)
        for i in range(2**m):
            v[:,i] = LBs_UBs[:, L[i], cols]
    
    else:
        raise Exception("Bounds are neither torch.Tensor nor numpy.ndarray")
        
    return v





  
#%% Admissibility utils

@torch.no_grad()
def open_loop(env, s0: np.array, actions: torch.Tensor, attr: torch.Tensor = None):
    """Compute the open-loop trajectory from s0 by applying the given actions

    Arguments:
        - s0 : initial state (state_size,)
        - actions : sequence of actions to apply open-loop (horizon, action_size)
        - attr : optional conditioning attribute (attr_dim,)

    Returns: 
        - reward : total reward collected over the trajectory
        - survival : ratio of time steps before trajectory fails
        - resulting trajectory of size (horizon+1, state_size)
    """
    if type(actions) == torch.Tensor:
        actions = actions.cpu().numpy()
    if type(s0) == torch.Tensor:
        s0 = s0.cpu().numpy()
    assert len(actions.shape) == 2, f"actions must be a 2D array and not {actions.shape}"
    assert actions.shape[1] == env.action_size, f"actions have the wrong size {actions.shape[1]} instead of {env.action_size}"
    assert s0.shape == (env.state_size,), f"initial state s0 has the wrong size {s0.shape} instead of ({env.state_size},)"
     
    H, _ = actions.shape
    total_reward = 0.
    env.reset_to(s0)
    traj = np.zeros((H+1, env.state_size))
    traj[0] = s0
    
    for t in range(H):
        next_obs, reward, done, _, _ = env.step(actions[t])
        traj[t+1] = next_obs
        total_reward += reward
        if done: break
        
    return total_reward, (t+1)/H, traj[:t+2]




@torch.no_grad()
def open_loop_stats(env, planner, ID, N_s0=100, s0=None, H=300, N_samples=4,
                    attr=None, N:int=None):
    """ Calculate the reward and survival for a given planner
    If s0 is not provided, 'N_s0' random initial states are drawn
    For each of these initial states 'N_samples' trajectories are generated and only the best one is kept
    Best meaning trajectory staying the longest time within bounds
    N: number of denoising steps, None: doesn't change what the ode does
    #is_planner_true: whether the planner produces exactly admissible trajectories, in that case no need for InverseDynamics
    """
    
    print("\nOpen-loop stats for " + planner.ode.filename + f" with {N_samples} samples per s0")
    if s0 is None: # generate random initial states
        s0 = np.zeros((N_s0, env.state_size))
        for i in range(N_s0):
            s0[i] = env.reset()
        s0 = torch.tensor(s0).float()
    
    # generate all trajectories at once
    if attr is not None:
        out = planner.best_traj(s0, attr, traj_len=H, n_samples_per_s0=N_samples, projector=planner.ode.projector, N=N)
    else:
        out = planner.best_traj(s0, traj_len=H, n_samples_per_s0=N_samples, projector=planner.ode.projector, N=N)
    if isinstance(out, tuple):
        if len(out) == 2:
            Trajs, Actions = out
        elif len(out) == 3:
            Trajs, Actions, Rewards = out
    else:
        Trajs = out
    
    reward_samples = np.zeros(N_s0)
    survival_samples = np.zeros(N_s0)
    
    for s_id in range(N_s0):
        ID_traj, _, reward_samples[s_id], _ = ID.closest_admissible_traj(Trajs[s_id])
        survival_samples[s_id] = ID_traj.shape[0]/H # InverseDynamics_trajectory stop when done
        print(f"reward {reward_samples[s_id]:.2f}   survival {survival_samples[s_id]:.2f}")
    
    mean_reward = reward_samples.mean()
    std_rewards = reward_samples.std()
    mean_survival = survival_samples.mean()
    std_survival = survival_samples.std()
    print(f"survival {mean_survival:.2f} +- {std_survival:.2f}")

    return mean_reward, std_rewards,  mean_survival, std_survival








def barplot_comparison(labels:list, list_of_survival:list, list_of_rewards:list,
                       list_of_survival_std:list = None,
                       list_of_rewards_std:list = None, title:str = ""):
    """ Bar plot of the rewards and survival of several models

    Arguments:
        - labels : list of the names of each models evaluated
        - list_of_survival : list of the corresponding survival of each model in [0, 1]
        - list_of_rewards : list of the corresponding reward of each model
        - list_of_survival_std : optional list of the survival standard deviation of each model
        - list_of_rewards_std : optional list of the reward standard deviation of each model
        - title : optional title of the plots
    """
    assert len(labels) == len(list_of_survival), "Number of labels not matching the number of survival stats"
    assert len(labels) == len(list_of_rewards), "Number of labels not matching the number of reward stats"
    
    nb_models = len(labels)
    
    fig, ax = plt.subplots()
    plt.rcParams.update({'font.size': 16})
    plt.rcParams['font.sans-serif'] = ['Palatino Linotype']
    ax.spines[['bottom', 'top', 'right', 'left']].set_color('w')
    ax.grid(axis='y')
    ax.set_axisbelow(True)
    ax.bar(labels, list_of_rewards, width=0.5)
    if list_of_rewards_std is not None:
        assert len(labels) == len(list_of_rewards_std), "Number of labels not matching the number of reward std"
        plt.errorbar(labels, list_of_rewards, yerr=list_of_rewards_std, fmt="o", color="tab:red", capsize=5.)
    ax.set_ylabel('reward')
    ax.set_title('Reward ' + title)
    ax.tick_params(bottom=False, axis='x', labelrotation=90) # no ticks at the bottom, only the labels
    plt.show()


    fig, ax = plt.subplots()
    plt.rcParams.update({'font.size': 16})
    plt.rcParams['font.sans-serif'] = ['Palatino Linotype']
    ax.spines[['bottom', 'top', 'right', 'left']].set_color('w')
    ax.grid(axis='y')
    ax.set_axisbelow(True)
    ax.bar(labels, list_of_survival, width=0.5)
    if list_of_survival_std is not None:
        assert len(labels) == len(list_of_survival_std), "Number of labels not matching the number of survival std"
        plt.errorbar(labels, list_of_survival, yerr=list_of_survival_std, fmt="o", color="tab:red", capsize=5.)
    ax.set_ylabel('survival')
    ax.set_title('Survival ' + title)
    ax.tick_params(bottom=False, axis='x', labelrotation=90)
    plt.show()


