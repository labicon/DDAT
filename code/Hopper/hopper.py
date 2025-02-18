# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 13:45:36 2024

@author: Jean-Baptiste Bouvier

Wrapper for the Mujoco Hopper Environment with a new function: reset_to(state),
and a list of actuated and unactuated states.
Actions are clamped in [-1, 1]
Uses semi-implicit Euler integrator for step
12 states: 6 positions and 6 velocities,
the x-position is included so that each velocity has its corresponding position
"""

import torch
import numpy as np
import gymnasium as gym
from gymnasium.envs.registration import EnvSpec

from plots import plot_traj, traj_comparison

class HopperEnv():
    """
    Wrapper for the Hopper MuJoCo Environment
    
    | Num | Action                             | Min | Max |     Name    | Joint | Unit         |
    |-----|------------------------------------|-----|-----|-------------|-------|--------------|
    | 0   | Torque applied on the thigh rotor  | -1  | 1   | thigh_joint | hinge | torque (N m) |
    | 1   | Torque applied on the leg rotor    | -1  | 1   | leg_joint   | hinge | torque (N m) |
    | 2   | Torque applied on the foot rotor   | -1  | 1   | foot_joint  | hinge | torque (N m) |

    | Num | Observation                              | Min  | Max |    Name     | Joint | Unit                     |
    | --- | -----------------------------------------| ---- | --- | ----------- | ----- | ------------------------ |
    | 0   | x-coordinate of the top (location)       | -Inf | Inf | rootx       | slide | position (m)             |
    | 1   | z-coordinate of the top (height)         | -Inf | Inf | rootz       | slide | position (m)             |
    | 2   | angle of the top                         | -Inf | Inf | rooty       | hinge | angle (rad)              |
    | 3   | angle of the thigh joint                 | -Inf | Inf | thigh_joint | hinge | angle (rad)              |
    | 4   | angle of the leg joint                   | -Inf | Inf | leg_joint   | hinge | angle (rad)              |
    | 5   | angle of the foot joint                  | -Inf | Inf | foot_joint  | hinge | angle (rad)              |
    
    | 6   | velocity of the x-coordinate of the top  | -Inf | Inf | rootx       | slide | velocity (m/s)           |
    | 7   | velocity of the z-coordinate of the top  | -Inf | Inf | rootz       | slide | velocity (m/s)           |
    | 8   | angular velocity of the angle of the top | -Inf | Inf | rooty       | hinge | angular velocity (rad/s) |
    | 9   | angular velocity of the thigh hinge      | -Inf | Inf | thigh_joint | hinge | angular velocity (rad/s) |
    | 10  | angular velocity of the leg hinge        | -Inf | Inf | leg_joint   | hinge | angular velocity (rad/s) |
    | 11  | angular velocity of the foot hinge       | -Inf | Inf | foot_joint  | hinge | angular velocity (rad/s) |

    The reward consists of three parts:
    - *healthy_reward*: Every timestep that the hopper is healthy, it gets a reward of fixed value `healthy_reward`.
    - *forward_reward*: A reward of hopping forward which is measured
    as *`forward_reward_weight` * (x-coordinate before action - x-coordinate after action)/dt*.
    - *ctrl_cost*: A cost for penalising the hopper if it takes actions that are too large. 
    """

    def __init__(self, render_mode=None, seed=0):
        
        self.name = "Hopper"
        self.max_episode_steps = 2000
        self.reward_threshold = 1.3*self.max_episode_steps
        self.render_mode = render_mode
        
        # Specify the specs to remove all env wrappers to access the Mujoco env directly
        self.spec = EnvSpec(id = "Hopper-v4",           # The string used to create the environment with :meth:`gymnasium.make`
                       entry_point='gymnasium.envs.mujoco.hopper_v4:HopperEnv', # A string for the environment location
                       nondeterministic=False,      # If the observation of an environment cannot be repeated with the same initial state, random number generator state and actions.
                       max_episode_steps=None,      # The max number of steps that the environment can take before truncation
                       order_enforce=False,         # If to enforce the order of :meth:`gymnasium.Env.reset` before :meth:`gymnasium.Env.step` and :meth:`gymnasium.Env.render` functions
                       autoreset=False,             # If to automatically reset the environment on episode end
                       disable_env_checker=True,    # If to disable the environment checker wrapper in :meth:`gymnasium.make`, by default False (runs the environment checker)
                       kwargs={'render_mode': render_mode}, # Additional keyword arguments passed to the environment during initialisation
                       additional_wrappers=(),      #  A tuple of additional wrappers applied to the environment (WrapperSpec)
                       vector_entry_point=None)     # The location of the vectorized environment to create from
         
        # higher initial noise to generate more varied trajectories
        self.noise_scale = 5e-2 # base value is 5e-3
        self.env = gym.make(self.spec, reset_noise_scale=self.noise_scale,
                            exclude_current_positions_from_observation=False) # adds the x-position
        assert self.env.model.opt.integrator == 0, "Select 'Euler' for the integrator in the XML file at '<<NAME_OF_YOUR_CONDA_ENVIRONMENT>>/Lib/site-packages/gymnasium/envs/mujoco/assets' "
        assert self.env.frame_skip == 1, "Need frame_skip = 1, i.e., single time step between states. Change in hopper_v4.py line 209 and time step 0.008 in XML next to integrator"
        self.metadata = self.env.metadata
        
        self._seed = seed
        self.env.action_space.seed(self._seed)
        
        self.env.reset(seed = self._seed)
        self.state_size = 12
        self.action_size = 3
        
        self.action_max = np.array([[1., 1., 1.]])
        self.action_min = -self.action_max
        self.min_z = self.env._healthy_z_range[0] # 0.7, reset at 1.25
        self.angle_range = self.env._healthy_angle_range # (-0.2, 0.2), reset at 0
        self.state_range = self.env._healthy_state_range # (-100, 100), reset at 0
        self.dt = self.env.dt # 0.008s # time step
        
        self.actuated_states = [6, 7, 8, 9, 10, 11] # states that are directly actuated
        self.unactuated_states = [0, 1, 2, 3, 4, 5] # states that are not directly actuated
        self.velocity_states = [6, 7, 8, 9, 10, 11] 
        self.position_states = [0, 1, 2, 3, 4, 5] 
        self.state_labels = ["x", "z", "top", "thigh", "leg", "foot", "x", "z", "top", "thigh", "leg", "foot"]
        self.reset_state = np.array([0.0, 1.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        ### For random reset
        self.min_state = np.array([-10., self.min_z,  self.angle_range[0], self.angle_range[0], self.angle_range[0], self.angle_range[0], -1., -1., -1., -1., -1., -1.])
        self.max_state = np.array([10., 2*self.min_z, self.angle_range[1], self.angle_range[1], self.angle_range[1], self.angle_range[1],  1.,  1.,  1.,  1.,  1.,  1.])
        ### Bounds on admissible state range
        self.low_bound = torch.tensor([-np.inf, self.min_z, -0.2, -100., -100., -100., -100., -100., -100., -100., -100., -100.])
        self.high_bound = torch.tensor([np.inf, np.inf, 0.2, 100., 100., 100., 100., 100., 100., 100., 100., 100.])
        
        
    def reset(self):
        self.episode_step = 0
        self.state, _ = self.env.reset()
        self.env.data.qacc_warmstart[:] = 0 # reset for reproducibility
        return self.state.copy()
     
    
    def step(self, action):
        assert type(action) == np.ndarray, f"action must be a numpy array and not a {type(action)}"
        assert action.shape == (self.action_size,), f"action must be of size {self.action_size} and not {action.shape}"
        self.episode_step += 1
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.state = np.concatenate((self.env.data.qpos, self.env.data.qvel)) # because obs is truncated at +-10
        done = terminated or truncated or self.episode_step > self.max_episode_steps
        
        return self.state.copy(), reward, done, False, None
    
    
    def reset_to(self, state):
        """New function: reset the state to the one provided.
        Split the state into qpos and qvel to use  env.set_state(qpos, qvel)"""
        
        assert state.shape == (self.state_size,)
        qpos = state[:6] # desired positions (x, z, theta_top, theta_thigh, theta_leg, theta_foot)  
        qvel = state[6:] # desired velocities (d/dt positions)
        self.env.reset()
        self.env.set_state(qpos, qvel) # Mujoco method to set state
        self.env.data.qacc_warmstart[:] = 0
        self.state = np.concatenate((self.env.data.qpos, self.env.data.qvel))
        if np.linalg.norm(self.state - state) > 1e-10:
            print("states dont match") # observation is clipped at 10 (except maybe x)
        self.episode_step = 0
        return self.state.copy()
        
    def render(self):
        return self.env.render()
    
    def close(self):
        self.env.close()
 

    # Function called by the projectors
    def pos_from_vel(self, S_t, vel_t_dt):
        """
        Calculates the next state's position using implicit Euler integrator,
        does NOT need to know the dynamics.
        
        Arguments:
            - S_t : current state torch.tensor (12,)
            - vel_t_dt : next state's velocity torch.tensor (6,)
        Returns:
            - x_t_dt : next state's position torch.tensor (6,)
        """
        return S_t[:6] + self.dt * vel_t_dt


    # Plotting functions
    def plot_traj(self, Traj, title:str = ""):
        """Plots the top angle trajectory of the Hopper."""
        plot_traj(self, Traj, title)

    
    def traj_comparison(self, traj_1, label_1, traj_2, label_2, title:str = "",
                        traj_3=None, label_3=None, traj_4=None, label_4=None,
                        plot_height:bool = True, legend_loc='best'):
        """
        Compares up to 4 trajectories of the Hopper
        Arguments:
            - traj_1 : first trajectory of shape (H, 12)
            - label_1 : corresponding label to display
            - traj_2 : first trajectory of shape (H, 12)
            - label_2 : corresponding label to display
            - title: optional title of the plot
            - traj_3 : optional third trajectory of shape (H, 12)
            - label_3 : optional corresponding label to display
            - traj_4 : optional fourth trajectory of shape (H, 12)
            - label_4 : optional corresponding label to display
            - plot_height : optional whether to plot the Hopper height or just the top angle
            - legend_loc : optional location of the legend
        """
        traj_comparison(self, traj_1, label_1, traj_2, label_2, title,
                        traj_3, label_3, traj_4, label_4, legend_loc)

