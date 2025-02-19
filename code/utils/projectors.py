# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 08:24:10 2024

@author: Jean-Baptiste Bouvier

Trajectory projectors to make state trajectories admissible
"""

import os
import torch
import cvxpy as cp
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils.utils import norm, vertices
from cvxpylayers.torch import CvxpyLayer




#%% Parent Projector

class Projector():
    """
    
    """
    def __init__(self, env, sigma_min:float = 0.0021, sigma_max:float = 0.2,
                 reference:bool = False, device:str = "cpu"):
        """Parent class for all projectors with projection curriculum.
        Arguments:
            - env : Gym-like environment
            - sigma_min : noise value under which all state transitions are projected
            - sigma_max : noise value over which none of the the state transitions are projected
            - reference : whether the projector uses a reference trajectory
            - device : "cpu" or "cuda"
        
        When sigma in [sigma_min, sigma_max]: projection probability proportional to sigma
        """
        
        self.env = env
        assert sigma_min >= 0., "Projections happen for sigma <= sigma_min which must be non-negative"
        self.sigma_min = sigma_min
        assert sigma_max >= sigma_min, "For the projection curriculum, sigma_min <= sigma_max"
        self.sigma_max = sigma_max
        self.reference = reference
        self.device = device
        self.state_size = self.env.state_size
        self.action_size = self.env.action_size
        self.a_min = torch.FloatTensor(self.env.action_min).to(self.device)
        self.a_max = torch.FloatTensor(self.env.action_max).to(self.device)
        if len(self.a_max.shape) == 1:
            self.a_max = self.a_max.reshape((1, self.action_size))
            self.a_min = self.a_min.reshape((1, self.action_size))
        self.extremal_actions = vertices(self.a_min, self.a_max).reshape((2**self.action_size, self.action_size)).to(self.device)
        self.nb_actions = self.extremal_actions.shape[0]
        
        self.pos = self.env.position_states
        self.vel = self.env.velocity_states
        self.dt = env.dt # time step of the environment
        
        # Convex optimization problem to project predicted next state on reachable set
        x = cp.Variable(self.nb_actions) # 1 coefficient for each vertex to describe point in convexhull
        
        # Projecting only the velocies
        P = cp.Parameter((len(self.vel), self.nb_actions)) # vertices
        z = cp.Parameter(len(self.vel))     # point to project into convexhull of vertices
        
        base_constraints = [sum(x) == 1, x >= 0] # ensure that we get a convex combination of vertices
        base_objective = cp.Minimize(cp.pnorm(P @ x - z, p=2)) # minimize 2-norm distance of convex combination to z
        base_problem = cp.Problem(base_objective, base_constraints)
        assert base_problem.is_dpp()
        self.base_cvxpylayer = CvxpyLayer(base_problem, parameters=[P, z], variables=[x])
    
        if self.reference: # use some velocity states as reference during the projection
            if env.name in ["GO1", "GO2"]:
                raise Exception("The Reference Projector is not implemented for the Unitree GO1 and GO2")
            elif env.name == "Quadcopter":
                self.ref_states_id = [7, 8, 9, 10, 11, 12] # velocities and angular velocities indices in state
                self.ref_vel_id = [0, 1, 2, 3, 4, 5] # velocities and angular velocities indices in qvel
                print("The Reference Projector matches the velocities and angular velocities of the Quadcopter")
            elif env.name in ["Hopper", "Walker", "HalfCheetah"]:
                self.ref_vel_id = [2] # velocities and angular velocities indices in qvel
                self.ref_states_id = [self.vel[2]] # velocities and angular velocities indices in state
                print(f"The Reference Projector matches the head angle velocity of the {self.env.name}")
            else:
                raise NotImplementedError(f"The Reference projector is not implemented for {env.name}")
            
            ref = cp.Parameter(len(self.ref_states_id)) # reference states to match
            
            penalized_objective = cp.Minimize(cp.pnorm(P @ x - z, p=2) + cp.pnorm(P[self.ref_vel_id] @ x - ref, p=2) )
            penalized_problem = cp.Problem(penalized_objective, base_constraints)
            assert penalized_problem.is_dpp()
            self.penalized_cvxpylayer = CvxpyLayer(penalized_problem, parameters=[P, z, ref], variables=[x])
            
        
        
    @torch.no_grad()
    def _reachable_vertices(self, States: torch.Tensor, actions: torch.Tensor):
        """Generates the reachable set from 'States'.
        Arguments:
            - States : current state of each trajectory (nb_traj, state_size)
            - actions : extremal actions to take for each trajectory (nb_trajs, nb_actions, action_size)
        Returns:
            - list of extremal vertices corresponding to the extremal actions (nb_traj, nb_actions, state_size)
        """
        N_trajs = States.shape[0]
        S0 = States.cpu().numpy()
        actions = actions.cpu().numpy()
        assert len(actions.shape) == 3 # whether the actions are different for each traj
        
        Reachable_vertices = np.zeros((N_trajs, self.nb_actions, self.state_size))
        
        for traj_id in range(N_trajs):
            for i in range(self.nb_actions):
                self.env.reset_to(S0[traj_id])
                Reachable_vertices[traj_id, i] = self.env.step(actions[traj_id, i])[0]
        
        return torch.FloatTensor(Reachable_vertices).cpu()


    def _projection_probabilities(self, sigma: torch.Tensor):
        """Calculates the probability of projection given sigma"""
        prob = torch.zeros_like(sigma)
        prob += sigma < self.sigma_min # probability = 1 for sigma < sigma_min
        idx = (sigma >= self.sigma_min) * (sigma < self.sigma_max)
        prob[idx] += (self.sigma_max - sigma[idx])/(self.sigma_max - self.sigma_min)
        return prob


    def _reshape_sigma(self, sigma, size):
        """Reshape the noise level sigma into a tensor of desired size"""
        if type(sigma) == float:
            sigma = torch.ones((size))*sigma
        elif type(sigma) == torch.Tensor:
            sigma = sigma.reshape((-1))
        elif type(sigma) == np.ndarray:
            sigma = torch.FloatTensor(sigma)
        else:
            raise Exception(f"sigma is neither a float, array, or a tensor but a {type(sigma)}")
        return sigma
    
    
    def project_traj(self, Trajs: torch.Tensor, Ref_Trajs: torch.Tensor = None,
                     sigma: float = 0., Actions: torch.Tensor = None):
        """Projects trajectories onto an admissible set at noise scale sigma, 
        trying to keep them close to their reference trajectories.
        Arguments:
            - Trajs :     Trajectories to project (N_trajs, horizon, state_size) 
            - Ref_Trajs : optional reference trajectories (N_trajs, horizon, state_size)
            - sigma : optional noise level of the trajectories
            - Actions :   optional predicted actions (N_trajs, horizon, action_size)
        Returns:
            - projected_trajectories :    (N_trajs, horizon, state_size) 
            - projected_actions :  (N_trajs, horizon, action_size) 
        """
        
        if self.reference:
            assert Ref_Trajs is not None
            assert Trajs.shape == Ref_Trajs.shape, f"Trajs {Trajs.shape} does not match Ref_Trajs {Ref_Trajs.shape}"
        
        N_trajs = Trajs.shape[0]
        N_steps = Trajs.shape[1]-1
        Proj_Trajs = Trajs.clone()
        if Actions is None:
            Proj_Actions = torch.zeros((N_trajs, N_steps+1, self.action_size), device=self.device)
        else:
            assert Actions.shape == (N_trajs, N_steps+1, self.action_size), f"Actions should be of shape ({N_trajs}, {N_steps+1}, {self.action_size})"
            if type(Actions) == torch.Tensor:
                Proj_Actions = Actions.clone() # return a tensor of actions
            else:
                Actions = torch.FloatTensor(Actions).to(self.device)
                Proj_Actions = Actions.clone()
                
        sigma = self._reshape_sigma(sigma, Trajs.shape[0])
        
        if min(sigma) > self.sigma_max: # no projections to be done
            return Proj_Trajs, Proj_Actions
        rand = torch.rand(N_steps, device=self.device) # random steps where projections are needed
        pp = self._projection_probabilities(sigma) # vector (N_trajs, 1) of probabilities
        idx = torch.arange(sigma.shape[0], device=self.device)
        
        for t_id in range(N_steps):
            # indices of the trajectories that need projections
            idx_proj = idx[rand[t_id] < pp]    
            if len(idx_proj) > 0: # needs exact projection
                S_t = Proj_Trajs[idx_proj, t_id, :] # projected current state
                S_t_dt = Trajs[idx_proj, t_id+1, :] # predicted next state to be made admissible
                
                if self.reference:
                    Ref_t_dt = Ref_Trajs[idx_proj, t_id+1, :] # reference next state
                else:
                    Ref_t_dt = None
                   
                if Actions is None:
                    A_t = None
                else:
                    A_t = Actions[idx_proj, t_id]
                    
                Proj_Trajs[idx_proj, t_id+1, :], Proj_Actions[idx_proj, t_id, :] = self.make_next_state_admissible(S_t, S_t_dt, Ref_t_dt=Ref_t_dt, sigma=sigma[idx_proj], A_t=A_t)
                
        return Proj_Trajs, Proj_Actions
    


    def make_next_state_admissible(self, S_t: torch.Tensor, S_t_dt: torch.Tensor,
                                   Ref_t_dt: torch.Tensor = None, sigma: float = 0.,
                                   A_t: torch.Tensor = None):
        """Projects next state S_t_dt in a reachable set approximation of S_t
        for the velocities. The positions are deduced from velocities and
        previous positions.
        Arguments:
            - S_t : current state of each trajectory (N_trajs, state_size)
            - S_t_dt : predicted next state of each trajectory (N_trajs, state_size)
            - Ref_t_dt : reference next state from which S_t_dt should not be too far away (N_trajs, state_size)
            - sigma : noise level of the trajectories
            - A_t: candidate actions for each trajectory (N_trajs, action_size)
        Returns:
            - adm_S_t_dt : admissible next state for each trajectory (N_trajs, state_size)
            - adm_A_t : corresponding admissible action for each trajectory (N_trajs, action_size)
        """
        
        N_trajs = S_t.shape[0]
        sigma = self._reshape_sigma(sigma, N_trajs)
        
        adm_S_t_dt = torch.zeros_like(S_t_dt, device=self.device) # admissible next state (to be computed)
        adm_A_t = torch.zeros((N_trajs, self.action_size), device=self.device)
        if A_t is None:
            Action_vertices = torch.broadcast_to(self.extremal_actions, (N_trajs, self.nb_actions, self.action_size))
        else:
            A_t = A_t.reshape((N_trajs, 1, self.action_size)).repeat_interleave(self.nb_actions, dim=1)
            Action_vertices = (A_t + 0.1*self.extremal_actions).clip(self.a_min, self.a_max) # action
        
        R = self._reachable_vertices(S_t, Action_vertices)
        
        for traj_id in range(N_trajs): # Can't be parallelized because of the projection
            
            # update the velocity states
            s_pred = S_t_dt[traj_id, self.vel].clone()
            Vertices = R[traj_id, :, self.vel].T.to(self.device) # vertices of the reachable set in velocity space
            
            ### Convex optimization: closest point to s_vel in the convex hull of the vertices of the reachable set
            if self.reference and sigma[traj_id] < self.sigma_min: # only use the reference at small noise level
                ref = Ref_t_dt[traj_id, self.ref_states_id].clone()
                solution, = self.penalized_cvxpylayer(Vertices, s_pred, ref)
            
            else: # either not reference or too much noise
                solution, = self.base_cvxpylayer(Vertices, s_pred)
           
            # solution = solution.to(self.device)
            adm_A_t[traj_id] = solution @ Action_vertices[traj_id]
            
            # Calculates the next state's velocities and positions
            adm_S_t_dt[traj_id, self.vel] = Vertices @ solution
            adm_S_t_dt[traj_id, self.pos] = self.env.pos_from_vel(S_t[traj_id], adm_S_t_dt[traj_id, self.vel])
        
        return adm_S_t_dt, adm_A_t
    




#%%



class Admissible_Projector(Projector):
    def __init__(self, env, sigma_min:float = 0.0021, sigma_max:float = None,
                 device:str = "cpu"):
        """Naive trajectory projector onto their closest admissible set using.
        convex optimization. Can be incorporated as a differentiable layer.
        Arguments:
            - env : Gym-like environment
            - sigma_min : noise value under which all state transitions are projected
            - sigma_max : optional noise value over which none of the the state transitions are projected
                        When sigma in [sigma_min, sigma_max]: projection probability proportional to sigma
            - device : "cpu" or "cuda"
         """
        
        if sigma_max is None:
            sigma_max = sigma_min # projects only for sigma < sigma_min
        super().__init__(env, sigma_min, sigma_max, reference=False, device=device)
        
        self.name = "Adm_proj_sigma_" + str(sigma_min)
    
        

class Reference_Projector(Projector):
    def __init__(self, env, sigma_min:float = 0.0021, sigma_max:float = 0.2,
                 device:str = "cpu"):
        """Reference trajectory projector onto their closest admissible set using 
        convex optimization. Can be incorporated as a differentiable layer.
        
        Arguments:
            - env : Gym-like environment
            - sigma_min : noise value under which all state transitions are projected
            - sigma_max : noise value over which none of the the state transitions are projected
            - device : "cpu" or "cuda"
        
        When sigma in [sigma_min, sigma_max]: projection probability proportional to sigma
        """
        
        super().__init__(env, sigma_min, sigma_max, reference=True, device=device)
        self.name = "Ref_proj_sigma_" + str(sigma_min) + "_" + str(sigma_max)
        

#%% State-Action Projector

class SA_Projector(Projector):
    def __init__(self, env, sigma_min:float = 0.0021, sigma_max:float = 0.2,
                 device:str = "cpu", width:int = 128):
        """State-Action projector without convex optimization, but using a 
        neural network to predict a feedback correction on the action leading to
        the admissible next state.
        
        Arguments:
            - env : Gym-like environment
            - sigma_min : noise value under which all state transitions are projected
            - sigma_max : noise value over which none of the the state transitions are projected
            - device : "cpu" or "cuda"
            - width : width of the feedback correction neural network
        
        When sigma in [sigma_min, sigma_max]: projection probability proportional to sigma
        """

        super().__init__(env, sigma_min, sigma_max, reference=False, device=device)
        self.name = "SA_proj_" + str(width) + "_sigma_"+ str(sigma_min) + "_" + str(sigma_max)

        # Network to predict the difference in action given the difference in velocity between two states obtained from a fixed initial state
        self.width = width
        self.net = nn.Sequential(nn.Linear(len(self.vel), self.width), nn.ReLU(),
                                 nn.Linear(self.width, self.width), nn.ReLU(),
                                 nn.Linear(self.width, self.action_size)).to(self.device)
        self.optim = torch.optim.AdamW(self.net.parameters(), lr=2e-4, weight_decay=1e-4)


    def train(self, Trajs: torch.Tensor, Actions: torch.Tensor, batch_size: int = 64,
              n_gradient_steps:int = 100_000, extra:str = ""):
        """Train the neural network of the projector to reconstitute the action
        given a desired change in final state.
        Arguments:
            - Trajs : dataset of admissible trajectories (N_trajs, H, state_size)
            - Actions : dataset of corresponding actions (N_trajs, H, action_size)
        """
        
        N_trajs, H, _ = Trajs.shape
        assert Trajs.shape[2] == self.state_size
        N = N_trajs*(H-1) # number of training points
    
        S_t   =  Trajs[:, :-1].reshape((N_trajs*(H-1), self.state_size)) # current state
        S_t_dt =  Trajs[:, 1:].reshape((N_trajs*(H-1), self.state_size)) # desired next state
        A_t = Actions[:, :H-1].reshape((N_trajs*(H-1), self.action_size)) # desired action
        S_t = S_t.cpu().numpy()
        
        print("Noising dataset")
        Noise = torch.randn_like(A_t, device=self.device) * (self.sigma_min + self.sigma_max/5) # sigma_min + 5*sigma < sigma_max
        Noised_A_t = (A_t + Noise).clip(self.a_min, self.a_max)
        Noised_S_t_dt = np.zeros((N, self.state_size))
        for i in range(N):
            self.env.reset_to(S_t[i])
            Noised_S_t_dt[i] = self.env.step(Noised_A_t[i].cpu().numpy())[0] # noised next state
        
        # Training inputs to the neural network
        Vel_dif = S_t_dt[:, self.vel] - torch.FloatTensor(Noised_S_t_dt[:, self.vel]).to(self.device) # difference in velocities
        # Target outputs to the neural network
        Action_dif = A_t - Noised_A_t
        
        print("Training the State-Action Projector")
        pbar = tqdm(range(n_gradient_steps))
        loss_avg = 0.
        self.training_loss = torch.zeros(n_gradient_steps)
        for step in range(n_gradient_steps):
            
            idx = np.random.randint(0, N, batch_size) # sample a random batch
            pred = self.net(Vel_dif[idx])
            loss = ((Action_dif[idx] - pred)**2).mean()
            self.optim.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(self.net.parameters(), 10.)
            self.optim.step()
            self.training_loss[step] = loss
            loss_avg += loss.item()
            if (step+1) % 100 == 0:
                pbar.set_description(f'step: {step+1} loss: {loss_avg / 100.:.4f} grad_norm: {grad_norm:.4f}')
                pbar.update(100)
                loss_avg = 0.
                self.save(extra)
                
        print('\nTraining completed!')
        self.save(extra)
        plt.plot(self.training_loss.detach().cpu().numpy())
        plt.title("Training loss for " + self.name)
        plt.show()
        self._freeze()


    def save(self, extra:str = ""):
         torch.save({'net': self.net.state_dict()}, self.env.name+"/trained_models/"+ self.name+extra+".pt")
         
         
    def load(self, extra:str = ""):
         name = self.env.name + "/trained_models/" + self.name + extra + ".pt"
         if os.path.isfile(name):
             print("Loading " + name)
             checkpoint = torch.load(name, map_location=self.device, weights_only=True)
             self.net.load_state_dict(checkpoint['net'])
             self._freeze()
             return True # loaded
         else:
             print("File " + name + " doesn't exist. Not loading anything.")
             return False # not loaded
    
    
    def make_next_state_admissible(self, S_t: torch.Tensor, S_t_dt: torch.Tensor,
                                   A_t: torch.Tensor, sigma: float = 0.,
                                   Ref_t_dt = None):
        """Makes S_t_dt admissible from S_t by applying action A_t plus a feedback
        correction action derived by the neural network based on the difference
        between S_t_dt and the state reached when applying A_t.
        
        Arguments:
            - S_t : current state of each trajectory (N_trajs, state_size)
            - S_t_dt : predicted next state of each trajectory (N_trajs, state_size)
            - sigma : noise level of the trajectories
            - A_t: candidate actions for each trajectory (N_trajs, action_size)
            - Ref_t_dt : compatibility argument not used here
        Returns:
            - adm_S_t_dt : admissible next state for each trajectory (N_trajs, state_size)
            - adm_A_t : corresponding admissible action for each trajectory (N_trajs, action_size)
        """
        assert Ref_t_dt is None, "SA-Projector cannot use a reference"
        
        N_trajs = S_t.shape[0]
        sigma = self._reshape_sigma(sigma, N_trajs)
        
        np_S_t_dt = S_t_dt.clone().detach().cpu().numpy() # next state computed with simulator (numpy)
        np_A_t = A_t.clone().detach().cpu().numpy()
        S_t = S_t.detach().cpu().numpy()
        
        for traj_id in range(N_trajs): # Can't be parallelized because of the environment calls
            self.env.reset_to(S_t[traj_id])
            s_ol = self.env.step(np_A_t[traj_id])[0] # open-loop next state
            # velocity difference between state prediction and applying the action prediction
            dif = S_t_dt[traj_id, self.vel] - torch.FloatTensor(s_ol[self.vel]).to(self.device)
            da_t = self.net(dif) # action correction
           
            A_t[traj_id] = (A_t[traj_id] + da_t).clip(self.a_min, self.a_max) # corrected action
            np_A_t[traj_id] += da_t.detach().cpu().numpy()
            self.env.reset_to(S_t[traj_id])
            np_S_t_dt[traj_id] = self.env.step(np_A_t[traj_id])[0] # corrected next state
        
        S_t_dt += torch.FloatTensor(np_S_t_dt).to(self.device) - S_t_dt.detach() # i.e. S_t_dt = adm_S_t_dt but without losing gradients
    
        return S_t_dt, A_t
    
    
    def _freeze(self):
        """Freeze the neural network after training or loading"""
        for param in self.net.parameters():
            param.requires_grad = False




#%% Action Projector


class Action_Projector(Projector):
    def __init__(self, env, sigma_min:float = 0.0021, sigma_max:float = 0.2,
                 device:str = "cpu"):
        """Action projector without convex optimization relying on an
            open-loop application of the predicted actions.
            
        Arguments:
            - env : Gym-like environment
            - sigma_min : noise value under which all state transitions are projected
            - sigma_max : noise value over which none of the the state transitions are projected
            - device : "cpu" or "cuda"
        
        When sigma in [sigma_min, sigma_max]: projection probability proportional to sigma
        """
        super().__init__(env, sigma_min, sigma_max, reference=False, device=device)
        self.name = "A_proj_sigma_"+ str(sigma_min) + "_" + str(sigma_max) 
        

    def make_next_state_admissible(self, S_t: torch.Tensor, S_t_dt: torch.Tensor,
                                   A_t: torch.Tensor, sigma: float = 0.,
                                   Ref_t_dt = None):
        """Modifies S_t_dt to make it admissible from S_t using A_t
        
        Arguments:
            - S_t : current state of each trajectory (N_trajs, state_size)
            - S_t_dt : predicted next state of each trajectory (N_trajs, state_size)
            - sigma : noise level of the trajectories
            - A_t: candidate actions for each trajectory (N_trajs, action_size)
            - Ref_t_dt : compatibility argument not used here
        Returns:
            - adm_S_t_dt : admissible next state for each trajectory (N_trajs, state_size)
            - adm_A_t : corresponding admissible action for each trajectory (N_trajs, action_size)
        """
        assert Ref_t_dt is None, "A-Projector cannot use a reference"
        
        N_trajs = S_t.shape[0]
        sigma = self._reshape_sigma(sigma, N_trajs)
        
        np_A_t = A_t.clone().detach().cpu().numpy()
        S_t = S_t.detach().cpu().numpy()
        
        for traj_id in range(N_trajs): # Can't be parallelized because of the environment calls
            self.env.reset_to(S_t[traj_id])
            s_ol = self.env.step(np_A_t[traj_id])[0] # open-loop next state
            dif = torch.FloatTensor(s_ol).to(self.device) - S_t_dt[traj_id]
            S_t_dt[traj_id] += dif.detach() # i.e. S_t_dt = s_ol but without losing gradients
        
        return S_t_dt, A_t
    
    
  