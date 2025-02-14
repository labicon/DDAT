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
        if len(x.shape) == 2: # single trajectory
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


def open_loop(env, obs_0: np.array, actions: torch.Tensor):
    """Compute the open-loop trajectory obtained when starting from initial observation
    'obs_0' and applying the sequence 'actions' 
    Inputs: obs_0: (state_size)   actions: (horizon, action_size)
    Outputs: total_reward   ratio of time steps before trajectory fails
                resulting trajectory (horizon+1, action_size)"""
    
    if type(actions) == torch.Tensor:
        actions = actions.numpy()
    if type(obs_0) == torch.Tensor:
        obs_0 = obs_0.numpy()
        
    H, _ = actions.shape
    total_reward = 0.
    env.reset_to(obs_0)
    traj = np.zeros((H+1, env.state_size))
    traj[0] = obs_0
    
    for t in range(H):
        next_obs, reward, done, _, _ = env.step(actions[t])
        traj[t+1] = next_obs
        total_reward += reward
        if done: break
        
    return total_reward, (t+1)/H, traj[:t+2]





def open_loop_stats(env, planner, ID, N_s0=100, s0=None, H=300, N_samples=4,
                    attr=None, N:int=None):#, is_planner_true=False):
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




def closed_loop_stats(env, planner, N_samples, s0, test_H, H, mID):
    """planner: Planner of the ODE
    N_samples: int
    s0: initial state for the trajectory propagation
    test_H: int  horizon for the planning
    H: int full horizon
    mID: Inverse Dynamics

    reward_samples: 
    survival_samples:     """
    
    reward_samples = np.zeros(N_samples)
    survival_samples = np.zeros(N_samples)
    
    for sample_id in range(N_samples):
        s = copy.deepcopy(s0)
        sampled = np.zeros((H, env.state_size))
        admissible = np.zeros((H, env.state_size))
        
        for t in range(0, H, test_H):
            traj = planner.traj(s, traj_len=test_H, projector=planner.ode.projector)
            ID_traj, actions, _ = mID.closest_admissible_traj(traj[0])
            reward, survival, open_loop_traj = open_loop(env, s, actions)
            reward_samples[sample_id] += reward
            survival_samples[sample_id] += survival*test_H/H
            sampled[t:t+test_H] = traj[0]
            admissible[t:t+test_H] = open_loop_traj
            if survival < 1: # not survived horizon test_H
                t += round(survival*test_H +1)
                break
            s = open_loop_traj[-1] # replan from the last state
        # traj_comparison(env, sampled[:t], "sampled", admissible[:t], "admissible")
    
    return reward_samples, survival_samples




def barplot_comparison(labels, list_of_survival, list_of_rewards,
                       list_of_survival_std=None,
                       list_of_rewards_std=None, title="",
                       save_as_svg=None):
    """ Bar plot of the rewards and survival 
    save_as_svg="open_loop"
    """
    assert len(labels) == len(list_of_survival), "Number of labels not matching the number of survival stats"
    assert len(labels) == len(list_of_rewards), "Number of labels not matching the number of reward stats"
    
    
    # list_of_colors = ['silver', 'tab:blue', 'dimgray', 'gray', 'lightgray', 'darkgray', 'slategray']
    list_of_colors = ['silver', 'dimgray', 'gray', 'tab:blue', 'lightgray', 'darkgray', 'slategray']
    
    nb_models = len(labels)
    
    fig, ax = plt.subplots()
    plt.rcParams.update({'font.size': 16})
    plt.rcParams['font.sans-serif'] = ['Palatino Linotype']
    ax.spines[['bottom', 'top', 'right', 'left']].set_color('w')
    ax.grid(axis='y')
    ax.set_axisbelow(True)
    ax.bar(labels, list_of_rewards, width=0.5, color=list_of_colors[:nb_models])
    if list_of_rewards_std is not None:
        plt.errorbar(labels, list_of_rewards, yerr=list_of_rewards_std, fmt="o", color="tab:red", capsize=5.)
    ax.set_ylabel('reward')
    ax.set_title('Walker reward ' + title)
    ax.tick_params(bottom=False, axis='x', labelrotation=90) # no ticks at the bottom, only the labels
    if save_as_svg is not None:
        plt.savefig("Figures/"+save_as_svg+"_reward.svg", bbox_inches='tight', format="svg", dpi=1200)
    plt.show()


    fig, ax = plt.subplots()
    plt.rcParams.update({'font.size': 16})
    plt.rcParams['font.sans-serif'] = ['Palatino Linotype']
    ax.spines[['bottom', 'top', 'right', 'left']].set_color('w')
    ax.grid(axis='y')
    ax.set_axisbelow(True)
    ax.bar(labels, list_of_survival, width=0.5, color=list_of_colors[:nb_models])
    if list_of_survival_std is not None:
        plt.errorbar(labels, list_of_survival, yerr=list_of_survival_std, fmt="o", color="tab:red", capsize=5.)
    ax.set_ylabel('survival')
    ax.set_title('Walker survival ' + title)
    ax.tick_params(bottom=False, axis='x', labelrotation=90)
    if save_as_svg is not None:
        plt.savefig("Figures/"+save_as_svg+"_survival.svg", bbox_inches='tight', format="svg", dpi=1200)
    plt.show()


