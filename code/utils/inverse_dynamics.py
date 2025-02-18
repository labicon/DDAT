# -*- coding: utf-8 -*-
"""
Created on Thu May 23 10:57:41 2024

@author: Jean-Baptiste

Inverse Dynamics using the black-box environment model.

The approach tests extremal inputs, differential correction

"""

import copy
import torch
import numpy as np
import cvxpy as cp
from time import time
from tqdm import tqdm
from utils import vertices, norm



#%% Inverse Dynamics model


class InverseDynamics():
    """Inverse dyanmics using a manual approach, by testing a range of control
    inputs through the environment and iteratively refining estimate of inverse dynamics"""
    
    def __init__(self, env, tol:float = 1e-9, time_limit:float = 10):
        """
        Arguments:
            - env : a Gym-like environment to perform the black-box inverse dynamics through sampling
            - tol : optional desired tolerance on the single-step inverse dynamics
            - time_limit : optional time cutoff for the black-box optimization in seconds
        """
        self.name = "InverseDynamics"
        self.time_limit = time_limit # [seconds] to prevent infinite loop in black-box optimization
        self.env = env
        self.task = env.name
        self.action_min = env.action_min
        self.action_max = env.action_max
        self.state_size = env.state_size
        self.action_size = env.action_size
        self.extremal_actions = vertices(self.action_min, self.action_max).squeeze()
        # self.pos = env.position_states
        # self.vel = env.velocity_states
        # self.actu = env.actuated_states
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
        
        if env.name == "Quadcopter": # model-based
            self.propeller_states = [13, 14, 15, 16]
            self.IRzz = env.IRzz
        
        
    def action(self, s0: np.ndarray, s1: np.ndarray, a0: np.ndarray):
        """Calculates the action transition s0 into s1 and the closest admissible s1"""
        assert s0.shape == (self.state_size,), f"Only works for a single state of shape ({self.state_size},)"
        assert s1.shape == (self.state_size,), f"Only works for a single state of shape ({self.state_size},)"
        assert a0.shape == (self.action_size,), f"Only works for a single action of shape ({self.action_size},)"
        
        if self.env.name == "Quadcopter": # model-based inverse dynamics using rotor inertia and matching rotor velocities
            w0 = s0[self.propeller_states]
            w1 = s1[self.propeller_states]
            a = self.IRzz * (w1 - w0)/self.dt
            self.env.reset_to(s0)
            pred_s1, reward, done, _, _ = self.env.step(a)

        elif self.env.name in ["Hopper", "Walker"]:
            a, pred_s1, reward, done = self._action(s0, s1, a0)
            if norm(pred_s1 - s1) > self.tol:
                a, pred_s1, reward, done = self._action(s0, s1, a, max_iter=12, shrink_rate=0.6)
                if norm(pred_s1 - s1) > self.tol:
                    a, pred_s1, reward, done = self._action(s0, s1, a, max_iter=20, shrink_rate=0.7)
        
        else: # for HalfCheetah and GO1, GO2 the convex combinations are not great
            a, pred_s1, reward, done = self._action(s0, s1, a0, max_iter=6)
            if norm(pred_s1 - s1) > self.tol:
                a, cost = self._black_box_optimization(s0, s1, a, action_list="extremal")
                a, pred_s1, reward, done = self._action(s0, s1, a, max_iter=0)

        return a, pred_s1, reward, done
    
        
    def _action(self, s0: np.ndarray, s1: np.ndarray, a: np.ndarray, max_iter=6, shrink_rate=0.5):
        """
        Inverse dynamics finding the closest admissible next state to the desired one
        solving a convex combination of the reachable set vertices. Works exactly for control-affine dynamics
        Arguments:
            - s0 : current state
            - s1 : desired next state
            - a : action guess
        Returns:
            - a : better action guess
            - pred_s1 : admissible next state
            - reward : reward for the one step transition with pred_s1
            - done : whether the environment stopped the trajectory
        """
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
        the given and admissible trajectories.

        Arguments:
            - traj : a sequence of states of shape (H, state_size)
            - pred_actions : optional a sequence of actions approximately corresponding to traj,
                             used as first guesses in the optimization
        Returns:
            - adm_traj : an admissible trajectory starting from the same initial state as traj,
                        can be shorter than H if the environment returns 'done' before
            - actions : sequence of actions corresponding to adm_traj
            - reward : total reward for the adm_traj
            - state_norm_dif : cumulative norm of the difference between predicted states and
                                closest admissible ones.
        """
        
        H = traj.shape[0] # horizon, number of states in the trajectory
        assert traj.shape == (H, self.state_size), "Only works for a single trajectory of shape (N, state_size)"
        tensor = type(traj) == torch.Tensor
        if tensor: traj = traj.numpy()
        
        if pred_actions is None:
            pred_actions = np.zeros((H, self.action_size))
        else:
            assert len(pred_actions.shape) == 2, f"Only works for a single action sequence of shape ({H}, {self.action_size}) and not {pred_actions.shape}"
            assert pred_actions.shape[1] == self.action_size, f"Only works for a single action sequence of shape ({H}, {self.action_size}) and not {pred_actions.shape}"
            if type(pred_actions) == torch.Tensor:
                pred_actions = pred_actions.numpy()
            pred_actions = pred_actions.clip(self.action_min, self.action_max)
        
        Actions = np.zeros((H-1, self.action_size))
        Admissible_traj = np.zeros((H, self.state_size))
        Admissible_traj[0] = traj[0]
        state_norm_dif = 0.
        self.warmstart = np.zeros(len(self.vel))
        
        reward = 0
        for i in tqdm(range(H-1)):           
            Actions[i], Admissible_traj[i+1], r, done = self.action(Admissible_traj[i], traj[i+1], pred_actions[i])
            state_norm_dif += norm(Admissible_traj[i+1] - traj[i+1])
            reward += r
            if done: 
                state_norm_dif = np.inf
                break
            
        if tensor: Admissible_traj = torch.tensor(Admissible_traj).float()
        return Admissible_traj[:i+2], Actions[:i+1], reward, state_norm_dif
        
    
    # Black-box optimization functions
    def _black_box_cost(self, s0: np.ndarray, s1: np.ndarray, a: np.ndarray):
        """Evaluates the black-box cost of optimization variable a"""
        self.env.reset_to(s0)
        self.env.env.data.qacc_warmstart = self.warmstart.copy()
        pred_s1 = self.env.step(a)[0]
        return norm(pred_s1 - s1)
        
    
    def _black_box_optimization(self, s0: np.ndarray, s1: np.ndarray, a: np.ndarray,
                                action_list: str = "extremal"):
        """Find the action transition from s0 to s1 given first guess a.
        Try action directions until finding one reducing the cost, then dichotomy line search in that direction"""
        
        t0 = time()
        if action_list == "extremal":
            Actions = self.extremal_actions
        elif action_list == "randn":
            pass # random action list generated in each loop
        else:
            raise Exception("Undefined type of actions")
            
        cost = self._black_box_cost(s0, s1, a)
        radius = min(0.2, cost) # initialize the search radius at the same magnitude, upper bounded at 0.2
        while cost > self.tol and radius > self.tol/100 and time() - t0 < self.time_limit:
            shrink_radius = True
            if action_list == "randn": # generate new random actions
                Actions = np.random.randn((100, self.action_size))
                
            for i in range(Actions.shape[0]):
                a_i = (a + radius*Actions[i]).clip(self.action_min, self.action_max).squeeze()
                
                c = self._black_box_cost(s0, s1, a_i)
                if c < cost: # we have found a cost-reducing direction
                    shrink_radius = False
                    while c < cost: # push the action in that direction until the cost stops decreasing
                        cost = c
                        a = a_i.copy()
                        a_i = (a + radius*Actions[i]).clip(self.action_min, self.action_max).squeeze()
                        c = self._black_box_cost(s0, s1, a_i)
                    # Cost stopped decreasing in that direction, now dichotomy in [a_low, a_high]
                    a_low = (a - radius*Actions[i]).clip(self.action_min, self.action_max).squeeze()
                    c_low = self._black_box_cost(s0, s1, a_low)
                    a_high = a_i.copy() # previous action that went too far
                    c_high = c # cost higher than cost
                    
                    for _ in range(5): # dichotomy on [a_low, a]
                        a_m_low = (a_low + a)/2 # action between low and middle
                        c_m_low = self._black_box_cost(s0, s1, a_m_low)
                         
                        if c_m_low > cost:
                            if c_m_low > c_low + self.tol:
                                # print(f"non-convex with precision {c_m_low - c_low:.2e}")
                                break
                            a_low = a_m_low.copy() # move the low up to middle low
                            c_low = c_m_low
                        else: # c_m_low < cost
                            a_high = a.copy() # move the high to the middle
                            c_high = cost
                            a = a_m_low.copy()
                            cost = c_m_low
                    
                    for _ in range(5): # dichotomy on [a, a_high]
                         a_m_high = (a + a_high)/2 # action between middle and high
                         c_m_high = self._black_box_cost(s0, s1, a_m_high)
                          
                         if c_m_high > cost:
                             if c_m_high > c_high + self.tol:
                                 # print(f"non-convex with precision {c_m_high - c_high:.2e}")
                                 break
                             a_high = a_m_high.copy() # move the high down to middle high
                             c_high = c_m_high
                         else: # c_m_high < cost
                             a_low = a.copy() # move the high to the middle
                             c_low = cost
                             a = a_m_high.copy()
                             cost = c_m_high
                    
                    
            if shrink_radius: # i.e. didn't find a better direction
                radius *= 0.5 #           BS: radius always decreasing => prevents infinite loops where costs barely decreases for an infinity of iterations
                
        return a, cost


    
        
        