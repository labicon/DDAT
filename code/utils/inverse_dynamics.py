# -*- coding: utf-8 -*-
"""
Created on Thu May 23 10:57:41 2024

@author: Jean-Baptiste

Inverse Dynamics for the Hopper

manual model using the environment, testing extremal inputs, differential correction
with dichotomy of inputs until converging: performs very well!
"""

import copy
import torch
import numpy as np
import cvxpy as cp

from utils import vertices, norm



#%% Inverse Dynamics model


class InverseDynamics():
    """Inverse dyanmics using a manual approach, by testing a range of control
    inputs through the environment and iteratively refining estimate of inverse dynamics"""
    
    def __init__(self, env, tol=1e-9):
        self.name = "InverseDynamics"
        self.env = env
        self.task = env.name
        self.action_min = env.action_min
        self.action_max = env.action_max
        self.state_size = env.state_size
        self.action_size = env.action_size
        self.extremal_actions = vertices(self.action_min, self.action_max).squeeze()
        self.pos = env.position_states
        self.vel = env.velocity_states
        self.actu = env.actuated_states
        self.unac_vel = [s for s in self.vel if s in env.unactuated_states] # unactuated velocities
        self.dt = env.dt
        self.tol = tol # tolerance
     
        m = self.extremal_actions.shape[0]
        n = len(self.actu)
        
        ### Define convex optimization problem in cvxpy
        self.x = cp.Variable(m) # m coefficients, 1 for each vertex to describe point in convexhull
        constraints = [sum(self.x) == 1, self.x >= 0]
        self.Vertices = cp.Parameter((n, m))
        self.Point = cp.Parameter(n)
        objective = cp.Minimize(cp.pnorm(self.Vertices @ self.x - self.Point, p=2))
        self.problem = cp.Problem(objective, constraints)
        assert self.problem.is_dpp()
        
        
    def action(self, s0: np.ndarray, s1: np.ndarray, a0: np.ndarray):
        """Calculates the action transition s0 into s1 and the closest admissible s1"""
        assert s0.shape == (self.state_size,), "Only works for a single state of shape (state_size,)"
        assert s1.shape == (self.state_size,), "Only works for a single state of shape (state_size,)"
        a: np.ndarray
        
        a, pred_s1, reward, done = self._action(s0, s1, a0)
        if norm(pred_s1 - s1) > self.tol:
            a, pred_s1, reward, done = self._action(s0, s1, a, max_iter=12, shrink_rate=0.6)
            if norm(pred_s1 - s1) > self.tol:
                a, pred_s1, reward, done = self._action(s0, s1, a, max_iter=20, shrink_rate=0.7)
            
        return a, pred_s1, reward, done
    
        
    def _action(self, s0: np.ndarray, s1: np.ndarray, a: np.ndarray, max_iter=6, shrink_rate=0.5):
        
        self.env.reset_to(s0)
        self.env.env.data.qacc_warmstart = copy.deepcopy(self.warmstart)
        pred_s1, reward, done, _, _ = self.env.step(a)
        i = 0
        while i < max_iter and norm(pred_s1 - s1) > self.tol:
            actions = (a + self.extremal_actions*shrink_rate**i).clip(self.action_min, self.action_max)
            S1 = self._reachable_set(s0, actions)
            coefs = self._convex_coefficients(S1[:, self.actu], s1[self.actu])
            a = actions.T @ coefs
            a = a.clip(self.action_min, self.action_max).squeeze()
            self.env.reset_to(s0)
            pred_s1, reward, done, _, _ = self.env.step(a)
            i += 1
        
        self.warmstart = copy.deepcopy(self.env.env.data.qacc_warmstart)        
        return a, pred_s1, reward, done
    
   
    def _reachable_set(self, s0, actions):
        """Returns the states reachable from s0 with 'actions' """
        S1 = np.zeros((actions.shape[0], self.state_size))
        for i in range(actions.shape[0]):
            self.env.reset_to(s0)
            self.env.env.data.qacc_warmstart = copy.deepcopy(self.warmstart)
            S1[i] = self.env.step(actions[i])[0]
        return S1

   
    def _convex_coefficients(self, vertices, point):
        """Convex optimization returning the vector of coefficient 
        to describe 'point' as a convex combination of 'vertices'. """
        
        self.Vertices.value = vertices.T
        self.Point.value = point
        try:
            self.problem.solve(tol_feas=1e-10, tol_gap_abs=1e-10, tol_gap_rel=1e-10)
        except: # if 1e-10 accuracy is impossible
            print("Inverse Dynamics solver might be inacurate")
            if self.problem.status in ['optimal', 'optimal_inaccurate']: # solution is not very accurate, but still usable
                return self.x.value
            print(self.problem.status)
            self.problem.solve(verbose=True)
            
        return self.x.value
    
    
    def closest_admissible_traj(self, traj, pred_actions=None):
        """Calculates the closest admissible trajectory, along with the array 
        of actions generating the given trajectory and the norm difference between
        the given and admissible trajectories"""
        
        N = traj.shape[0] # number of states in the trajectory
        assert traj.shape == (N, self.state_size), "Only works for a single trajectory of shape (N, state_size)"
        tensor = type(traj) == torch.Tensor
        if tensor: traj = traj.numpy()
        
        if pred_actions is None:
            pred_actions = np.zeros((N, self.action_size))
        else:
            assert len(pred_actions.shape) == 2, f"Only works for a single action sequence of shape (horizon, action_size) and not {pred_actions.shape}"
            assert pred_actions.shape[1] == self.action_size, f"Only works for a single action sequence of shape (horizon, action_size) and not {pred_actions.shape}"
            if type(pred_actions) == torch.Tensor:
                pred_actions = pred_actions.numpy()
            pred_actions = pred_actions.clip(self.action_min, self.action_max)
        
        Actions = np.zeros((N-1, self.action_size))
        Admissible_traj = np.zeros((N, self.state_size))
        Admissible_traj[0] = traj[0]
        state_norm_dif = 0.
        self.warmstart = np.zeros(len(self.vel))
        
        reward = 0
        for i in range(N-1):            
            Actions[i], Admissible_traj[i+1], r, done = self.action(Admissible_traj[i], traj[i+1], pred_actions[i])
            state_norm_dif += norm(Admissible_traj[i+1] - traj[i+1])
            reward += r
            if done: 
                state_norm_dif = np.inf
                break
            
        if tensor: Admissible_traj = torch.tensor(Admissible_traj).float()
        return Admissible_traj[:i+2], Actions[:i+1], reward, state_norm_dif
        


#%% Testing
if __name__ == "__main__":
    from walker2d import WalkerEnv
    from plots import traj_comparison
    
    env = WalkerEnv()
    H = 300
    data = np.load("datasets/walker_10trajs_500steps.npz")
    loaded_trajs = data["Trajs"][:, :H]
    loaded_actions = data["Actions"][:, :H-1]
    
    
    #%%
    
    ID = InverseDynamics(env, tol=1e-9)
    
    for traj_id in range(loaded_trajs.shape[0]):
        ID_traj, _, _, distance = ID.closest_admissible_traj(loaded_trajs[traj_id])
        print(f"ID is {distance:.2f} from the truth")
        
        end_id = ID_traj.shape[0]
        dif = np.linalg.norm(ID_traj - loaded_trajs[traj_id, :end_id], axis=1)
        traj_comparison(env, loaded_trajs[traj_id], "loaded", ID_traj, "ID")
        
        
    #%%
    # traj = np.zeros((2, env.state_size))
    # traj[0] = env.reset()
    # a = np.zeros(6)
    # a[0] = 1.
    # traj[1] = env.step(a)[0]
    # out = ID.action(traj[0], traj[1])
    
        
        