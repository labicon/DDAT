# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 09:56:20 2025

@author: Jean-Baptiste Bouvier

Model-based inverse dynamics for the Quadcopter by matching the propeller velocity
"""



import torch
import numpy as np
from utils import norm



#%% Inverse Dynamics model


class ModelBasedInverseDynamics():
    """Model-based inverse dynamics for the Quadcopter by matching the propeller velocity"""
    
    def __init__(self, env, tol=1e-9):
        self.name = "InverseDynamics"
        self.env = env
        self.task = env.name
        self.action_min = env.action_min
        self.action_max = env.action_max
        self.state_size = env.state_size
        self.action_size = env.action_size
        
        self.propeller_states = [13, 14, 15, 16]
        self.dt = env.dt
        self.tol = tol # tolerance
            
        # model-based
        self.IRzz = env.IRzz
        
        
    def action(self, s0: np.ndarray, s1: np.ndarray):
        """Calculates the action transition s0 into s1 and the closest admissible s1"""
        assert s0.shape == (self.state_size,), "Only works for a single state of shape (state_size,)"
        assert s1.shape == (self.state_size,), "Only works for a single state of shape (state_size,)"
        
        w0 = s0[self.propeller_states]
        w1 = s1[self.propeller_states]
        a = self.IRzz * (w1 - w0)/self.dt
        
        self.env.reset_to(s0)
        pred_s1, reward, done, _, _ = self.env.step(a)
       
        return a, pred_s1, reward, done
    
   
   
    
    def closest_admissible_traj(self, traj):
        """Calculates the closest admissible trajectory, along with the array 
        of actions generating the given trajectory and the norm difference between
        the given and admissible trajectories"""
        
        N = traj.shape[0] # number of states in the trajectory
        assert traj.shape == (N, self.state_size), "Only works for a single trajectory of shape (N, state_size)"
        tensor = type(traj) == torch.Tensor
        if tensor: traj = traj.numpy()
        
        Actions = np.zeros((N-1, self.action_size))
        Admissible_traj = np.zeros((N, self.state_size))
        Admissible_traj[0] = traj[0]
        state_norm_dif = 0.
        
        reward = 0
        for i in range(N-1):            
            Actions[i], Admissible_traj[i+1], r, done = self.action(Admissible_traj[i], traj[i+1])
            state_norm_dif += norm(Admissible_traj[i+1] - traj[i+1])
            reward += r
            if done: 
                state_norm_dif = np.inf
                break
            
        if tensor: Admissible_traj = torch.tensor(Admissible_traj).float()
        return Admissible_traj[:i+2], Actions[:i+1], reward, state_norm_dif
    
        
    

#%% Testing
if __name__ == "__main__":
    from quadcopter import QuadcopterEnv, traj_comparison, plot_traj
    
    env = QuadcopterEnv()
    H = 200
    data = np.load("datasets/quad_4trajs.npz")
    loaded_trajs = data["trajs"]
    loaded_actions = data["actions"]
    
    #%% Reproducing trajectories
    for traj_id in range(loaded_trajs.shape[0]):
        s = env.reset_to(loaded_trajs[traj_id, 0])
        for t in range(199):
            s, _, done, _, _ = env.step(loaded_actions[traj_id, t])
            assert norm(s - loaded_trajs[traj_id, t+1]) < 1e-15, f"Trajectory {traj_id} is not reproducible at step {t}"
        print(f"Trajectory {traj_id} is reproducible")
    
    #%%
    
    ID = InverseDynamics(env, tol=1e-10)
    
    # for traj_id in range(loaded_trajs.shape[0]):
    #     ID_traj, ID_actions, _, distance = ID.closest_admissible_traj(loaded_trajs[traj_id], loaded_actions[traj_id])
    #     print(f"Traj {traj_id}: ID given inputs is {distance:.2f} from the truth")
        
    #     ID_traj, ID_actions, _, distance = ID.closest_admissible_traj(loaded_trajs[traj_id])
    #     print(f"Traj {traj_id}: ID without inputs is {distance:.2f} from the truth")
        
    #     end_id = ID_traj.shape[0]
    #     dif = np.linalg.norm(ID_traj - loaded_trajs[traj_id, :end_id], axis=1)
    #     traj_comparison(env, loaded_trajs[traj_id], "loaded", ID_traj, "ID", title=f"Traj {traj_id}")
        
    #     # plot_traj(env, loaded_trajs[traj_id], title=f"loaded traj {traj_id}")
    #     # plot_traj(env, ID_traj, title=f"ID traj {traj_id}")
    #     print(" ")
        
        
        
    #%% ID of non-admissible traj
    from utils import State_Normalizer, norm
    from DiT import ODE, Planner
    
    data = np.load("datasets/quad_1000trajs.npz")
    obs = torch.FloatTensor(data['trajs'])
    model_size = {"d_model": 128, "n_heads": 4, "depth": 3}

    normalizer = State_Normalizer(obs)
    nor_obs = normalizer.normalize(obs)
    state_sigma_data = nor_obs.std().item()
    ode = ODE(env, state_sigma_data, N=5, **model_size)
    ode.load(extra = "")
    planner = Planner(env, ode, normalizer)
    
    s0 = env.reset()
    N_samples = 4
    traj = planner.best_traj(s0, traj_len=H, n_samples_per_s0=N_samples, projector=None)
    traj = np.array(traj[0])
    ID_traj, ID_actions = ID.closest_admissible_traj(traj)[:2]
    print(f"Inverse dynamics traj diverges at step {ID_traj.shape[0]:}")

    traj_comparison(env, traj, "sampled", ID_traj, "ID",
                    title="S DiT trained without projections")

    repro_ID_traj = np.zeros_like(ID_traj)
    repro_ID_traj[0] = ID_traj[0]
    env.reset_to(ID_traj[0])
    for t in range(199):
        repro_ID_traj[t+1], _, done = env.step(ID_actions[t])[:3]
        if done: 
            print(t)
            break

    dif = repro_ID_traj[:t] - ID_traj[:t] 
    norm(dif[0])
    
        

