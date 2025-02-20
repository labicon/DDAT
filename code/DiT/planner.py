# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 17:13:39 2025

@author: Jean-Baptiste Bouvier

Planner generating trajectories by sampling a pre-trained ODE diffusion model
The normalization is handled in the ODE
"""


import torch
import numpy as np
from DiT.ODE import ODE
from utils.utils import open_loop

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
        assert self.modality in ["S", "SA", "A"]
        self.is_conditional = ode.is_conditional
        
        self.device = ode.device
        self.low_bound = torch.FloatTensor(env.low_bound).to(self.device) # lower bound on the state
        self.high_bound = torch.FloatTensor(env.high_bound).to(self.device) # upper bound on the state
        self.state_size = env.state_size
        self.action_size = env.action_size
        self.attr_dim = ode.attr_dim
        

    def _check_s0(self, s0):
        """Make sure the initial state has the right shape and device"""
        if type(s0) == np.ndarray:
            s0 = torch.tensor(s0, dtype=torch.float32, device=self.device)
        if len(s0.shape) == 1:
            s0 = s0.reshape((1, self.state_size))
        
        if s0.device  != self.device:
            s0 = s0.to(self.device)
        
        return s0
    
    
    def _check_attr(self, s0, attr):
        """Make sure the conditioning attribute has the right shape and device"""
        if type(attr) == np.ndarray:
            attr = torch.tensor(attr, dtype=torch.float32, device=self.device)
        if len(attr.shape) == 1:
            attr = attr.reshape((1, self.attr_dim))
        
        assert attr.shape[0] == s0.shape[0], "Need one attribute per s0" 
        
        if attr.device != self.device:
            attr = attr.to(self.device)
        
        return attr

    
    def _best_S_traj(self, s0:torch.Tensor, S_pred:torch.Tensor, 
                     N_s0:int, n_samples_per_s0:int, traj_len:int):
        """
        Prediction of states only, selection of trajectory staying in-bounds the longest
        """

        n_samples = N_s0 * n_samples_per_s0
        assert S_pred.shape == (n_samples, traj_len, self.state_size)
        Trajs = np.zeros((N_s0, traj_len, self.state_size))
        
        for s_id in range(N_s0): # index of the s0
            largest_survival = 0 # Keep trajectory surviving the longest
            for sample_id in range(n_samples_per_s0):
                i = s_id*n_samples_per_s0 + sample_id
                traj = S_pred[i].clone()
                for t in range(traj_len):
                    if (traj[t] < self.low_bound).any() or (traj[t] > self.high_bound).any():
                            break

                if t > largest_survival:
                    largest_survival = t
                    best_sample_id = sample_id # keep the sample that stays in bounds the longest
                
            Trajs[s_id] = S_pred[s_id*n_samples_per_s0 + best_sample_id].cpu().numpy()
        
        return Trajs

    
    def _best_SA_traj(self, s0:torch.Tensor, SA_pred:torch.Tensor, 
                     N_s0:int, n_samples_per_s0:int, traj_len:int):
        """
        Prediction of states and actions.
        Selection based on the best sequence of states
        """

        n_samples = N_s0 * n_samples_per_s0
        assert SA_pred.shape == (n_samples, traj_len, self.state_size + self.action_size)
        Trajs = np.zeros((N_s0, traj_len, self.state_size))
        Actions = np.zeros((N_s0, traj_len, self.action_size))
        
        for s_id in range(N_s0): # index of the s0
            largest_survival = 0 # Keep trajectory surviving the longest
            for sample_id in range(n_samples_per_s0):
                i = s_id*n_samples_per_s0 + sample_id
                traj = SA_pred[i, :, :self.state_size].clone()
                for t in range(traj_len):
                    if (traj[t] < self.low_bound).any() or (traj[t] > self.high_bound).any():
                            break
                
                if t > largest_survival:
                    largest_survival = t
                    best_sample_id = sample_id # keep the sample that stays in bounds the longest
                
            Trajs[s_id] = SA_pred[s_id*n_samples_per_s0 + best_sample_id, :, :self.state_size].cpu().numpy()
            Actions[s_id] = SA_pred[s_id*n_samples_per_s0 + best_sample_id, :, self.state_size:].cpu().numpy()
        
        return Trajs, Actions  


    def _best_A_traj(self, s0:torch.Tensor, A_pred:torch.Tensor, 
                     N_s0:int, n_samples_per_s0:int, traj_len:int, attr:torch.Tensor = None):
        """
        Prediction of actions only. Selection based on the open-loop trajectory staying in-bounds the longest
        """

        n_samples = N_s0 * n_samples_per_s0
        assert A_pred.shape == (n_samples, traj_len, self.action_size)
        Trajs_pred = np.zeros((n_samples, traj_len, self.state_size))
        Trajs = np.zeros((N_s0, traj_len, self.state_size))
        Actions = np.zeros((N_s0, traj_len, self.action_size))
        Survivals = np.zeros(N_s0)
        Rewards = np.zeros(N_s0)
        
        for s_id in range(N_s0): # index of the s0
            largest_survival = 0 # It would be unfair to other planners to keep the one with highest reward, they don't do that
            for sample_id in range(n_samples_per_s0):
                i = s_id*n_samples_per_s0 + sample_id
                reward, survival, traj = open_loop(self.env, s0[s_id], A_pred[i, :-1], attr=attr[i])
                Trajs_pred[i, :traj.shape[0]] = traj
                if survival > largest_survival:
                    largest_survival = survival
                    rwd_survivor = reward
                    best_sample_id = sample_id # keep the sample that stays in bounds the longest
                
            Trajs[s_id] = Trajs_pred[s_id*n_samples_per_s0 + best_sample_id].copy()
            Actions[s_id] = A_pred[s_id*n_samples_per_s0 + best_sample_id].cpu().numpy()
            Survivals[s_id] = largest_survival
            Rewards[s_id] = rwd_survivor
        
        return Trajs, Actions, Rewards, Survivals
          
    
    @torch.no_grad()
    def best_traj(self, s0:torch.Tensor, traj_len:int, attr:torch.Tensor = None,
                  n_samples_per_s0:int = 1, N:int = None, projector = None):
        """Returns 1 trajectory of length traj_len starting from each s0.
        For each s0 n_samples_per_s0 are generated, the one with the longest survival is chosen.
        
        Arguments:
            - s0 : Tensor of initial states (N_s0, state_size)
            - traj_len : horizon, i.e., length of the trajectories to generate including s0
            - attr: optional Tensor of conditioning attributes (N_s0, attr_dim)
            - n_samples_per_s0 : optional number of trajectories to generate for each initial condition before selecting the best
            - N : optional number of denoising steps of the diffusion model (default = 5 if None)
            - projector : optional whether to use a projector to generate trajectories

        Returns for modality = "S":
            - Trajs : sequence(s) of sampled states starting from s0 np.array(N_s0, traj_len, state_size)

        Returns for modality = "SA":
            - Trajs : sequence(s) of sampled states starting from s0 np.array(N_s0, traj_len, state_size)
            - Actions : sequence(s) of sampled actions np.array(N_s0, traj_len, action_size)
        
        Returns for modality = "A":
            - Trajs : sequence(s) of states starting from s0 generated open-loop from Actions np.array(N_s0, traj_len, state_size)
            - Actions : sequence(s) of sampled actions np.array(N_s0, traj_len, action_size)
            - Rewards : total rewards corresponding to each trajectories np.array(N_s0)
            - Survivals : survival of each trajectory np.array(N_s0)
        """
        
        
        s0 = self._check_s0(s0)
        N_s0 = s0.shape[0] # number of different initial states

        if self.is_conditional:
            assert attr is not None, f"Sampling {self.ode.filename} requires conditioning attributes"
            attr = self._check_attr(s0, attr)
            attr = attr.repeat_interleave(n_samples_per_s0, dim=0)
        
        S0 = s0.repeat_interleave(n_samples_per_s0, dim=0)
        n_samples = S0.shape[0] # total number of samples = N_s0 * n_samples_per_s0
        
        ### Prediction by the Diffusion model
        pred = self.ode.sample(s0=S0, attr=attr, traj_len=traj_len, n_samples=n_samples,
                                N=N, projector=projector)
        if self.modality == "S":
            return self._best_S_traj(s0, pred, N_s0, n_samples_per_s0, traj_len)
        elif self.modality == "SA":
            return self._best_SA_traj(s0, pred, N_s0, n_samples_per_s0, traj_len)
        else:
            return self._best_A_traj(s0, pred, N_s0, n_samples_per_s0, traj_len, attr)


       


  



