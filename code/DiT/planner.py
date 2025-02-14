# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 17:13:39 2025

@author: Jean-Baptiste Bouvier

Planner generating trajectories by sampling a pre-trained ODE diffusion model
The normalization is handled in the ODE
"""


import torch
from ODE import ODE
from utils import open_loop

class Planner():
    """
    Planner to generate trajectories with a pre-trained diffusion transformer
    """
    def __init__(self, env, ode: ODE):
        """
        Arguments:
            env : Gym-like environment
            ode : pretrained DiT
        """
        self.env = env
        self.ode = ode
        self.modality = ode.modality 
        
        self.device = ode.device
        self.low_bound = torch.FloatTensor(env.low_bound) # lower bound on the state
        self.high_bound = torch.FloatTensor(env.high_bound) # upper bound on the state
        self.state_size = env.state_size
        self.action_size = env.action_size
        
        self.attr_dim = ode.attr_dim
        

    def _make_attr(self, s0, cmd):
        """Make sure the initial state and the command have the right shapes
        before concatenating them to create the conditioning attributes"""
        if type(s0) == np.ndarray:
            s0 = torch.tensor(s0, dtype=torch.float32, device=self.device)
        if len(s0.shape) == 1:
            s0 = s0.reshape((1, self.state_size))
        if type(cmd) == np.ndarray:
            cmd = torch.tensor(cmd, dtype=torch.float32, device=self.device)
        if len(cmd.shape) == 1:
            cmd = cmd.reshape((1, self.cmd_shape))
        
        assert cmd.shape[0] == s0.shape[0], "Need one cmd per s0" 
        
        if s0.device  != self.device:
            s0 = s0.to(self.device)
        if cmd.device != self.device:
            cmd = cmd.to(self.device)
        
        attr = torch.cat((nor_s0, cmd), dim=1)
        return attr

        
    @torch.no_grad()
    def traj_actions(self, s0, attr, traj_len, N:int = None):
        """Returns n_samples trajectories and action sequences of length traj_len
        ."""
        attr = self._make_attr(s0, cmd)
        
        pred = self.ode.sample(attr, traj_len, n_samples=attr.shape[0], N=N)
        traj_pred = self.normalizer.unnormalize(pred[:, :, :self.state_size])
        action_pred = pred[:, :, self.state_size:]
        
        return traj_pred, action_pred 
          
    
    @torch.no_grad()
    def best_traj(self, s0, cmd, traj_len, n_samples_per_s0=1, N:int=None, projector=None):
        """Returns 1 trajectory of length traj_len starting from each
        UNnormalized states s0.
        For each s0  n_samples_per_s0 are generated, the one with the longest survival is chosen"""
        
        assert projector == None
        if type(s0) == np.ndarray:
            s0 = torch.tensor(s0, dtype=torch.float32, device=self.device)
        if len(s0.shape) == 1:
            s0 = s0.reshape((1, self.state_size))
        N_s0 = s0.shape[0] # number of different initial states
        
        if type(cmd) == np.ndarray:
            cmd = torch.tensor(cmd, dtype=torch.float32, device=self.device)
        # if cmd.shape == (N_s0, self.command_size):
        #     cmd = cmd[:, 0].reshape((N_s0, 1))
        if cmd.shape == (self.command_size,):
            cmd = cmd[0].reshape((1, self.cmd_shape))
        # else:
        #     raise Exception(f"Invalid command size of {cmd.shape}, which should be either ({N_s0}, 1) or ({N_s0}, {self.command_size})")
            
        cmd = cmd.repeat_interleave(n_samples_per_s0, dim=0)
        S0 = s0.repeat_interleave(n_samples_per_s0, dim=0)
        n_samples = S0.shape[0] # total number of samples = N_s0 * n_samples_per_s0
        attr = self._make_attr(S0, cmd)
        
        Actions_pred = self.ode.sample(attr, traj_len, n_samples=n_samples, N=N)
        assert Actions_pred.shape == (n_samples, traj_len, self.action_size)
        Trajs_pred = np.zeros((n_samples, traj_len, self.state_size))
        Trajs = np.zeros((N_s0, traj_len, self.state_size))
        Actions = np.zeros((N_s0, traj_len, self.action_size))
        Survivals = np.zeros(N_s0)
        Rewards = np.zeros(N_s0)
        
        for s_id in range(N_s0): # index of the s0
            largest_survival = 0 # It would be unfair to other planners to keep the one with highest reward, they don't do that
            for sample_id in range(n_samples_per_s0):
                i = s_id*n_samples_per_s0 + sample_id
                reward, survival, traj = open_loop(self.env, s0[s_id], Actions_pred[i, :-1], cmd=cmd[i])
                Trajs_pred[i, :traj.shape[0]] = traj
                if survival > largest_survival:
                    largest_survival = survival
                    rwd_survivor = reward
                    best_sample_id = sample_id # keep the sample that stays in bounds the longest
                
            Trajs[s_id] = Trajs_pred[s_id*n_samples_per_s0 + best_sample_id].copy()
            Actions[s_id] = Actions_pred[s_id*n_samples_per_s0 + best_sample_id].cpu().numpy()
            Survivals[s_id] = largest_survival
            Rewards[s_id] = rwd_survivor
        
        return Trajs, Actions, Rewards, Survivals
       


  



