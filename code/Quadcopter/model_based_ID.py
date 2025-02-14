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
    
        

        

