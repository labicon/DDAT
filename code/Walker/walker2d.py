# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 13:45:36 2024

@author: Jean-Baptiste Bouvier

Wrapper for the Mujoco Walker Environment with a new function: reset_to(state),
and a list of actuated and unactuated states.
The 6 actions are clamped in [-1, 1]
Uses semi-implicit Euler integrator for step
18 states: 9 positions and 9 velocities,
the x-position is included so that each velocity has its corresponding position
"""

import numpy as np
import gymnasium as gym
from gymnasium.envs.registration import EnvSpec

from Walker.plots import plot_traj, traj_comparison

class WalkerEnv():
    """
    Wrapper for the Walker 2D MuJoCo Environment
    
    | Num | Action                                  | Control Min | Control Max | Name (in corresponding XML file) | Joint | Unit         |
    |-----|-----------------------------------------|-------------|-------------|----------------------------------|-------|--------------|
    | 0   | Torque applied on the right thigh rotor | -1          | 1           | thigh_joint                      | hinge | torque (N m) |
    | 1   | Torque applied on the right leg rotor   | -1          | 1           | leg_joint                        | hinge | torque (N m) |
    | 2   | Torque applied on the right foot rotor  | -1          | 1           | foot_joint                       | hinge | torque (N m) |
    | 3   | Torque applied on the left thigh rotor  | -1          | 1           | thigh_left_joint                 | hinge | torque (N m) |
    | 4   | Torque applied on the left leg rotor    | -1          | 1           | leg_left_joint                   | hinge | torque (N m) |
    | 5   | Torque applied on the left foot rotor   | -1          | 1           | foot_left_joint                  | hinge | torque (N m) |

    
    | Num | Observation                                        | Min  | Max | Name (in corresponding XML file) | Joint | Unit                     |
    | --- | -------------------------------------------------- | ---- | --- | -------------------------------- | ----- | ------------------------ |
    | 0   | x-coordinate of the torso                          | -Inf | Inf | rootx                            | slide | position (m)             |
    | 1   | z-coordinate of the torso (height of Walker2d)     | -Inf | Inf | rootz                            | slide | position (m)             |
    | 2   | angle of the torso                                 | -Inf | Inf | rooty                            | hinge | angle (rad)              |
    | 3   | angle of the right thigh joint                     | -Inf | Inf | thigh_joint                      | hinge | angle (rad)              |
    | 4   | angle of the right leg joint                       | -Inf | Inf | leg_joint                        | hinge | angle (rad)              |
    | 5   | angle of the right foot joint                      | -Inf | Inf | foot_joint                       | hinge | angle (rad)              |
    | 6   | angle of the left thigh joint                      | -Inf | Inf | thigh_left_joint                 | hinge | angle (rad)              |
    | 7   | angle of the left leg joint                        | -Inf | Inf | leg_left_joint                   | hinge | angle (rad)              |
    | 8   | angle of the left foot joint                       | -Inf | Inf | foot_left_joint                  | hinge | angle (rad)              |

    | 9   | velocity of the x-coordinate of the torso          | -Inf | Inf | rootx                            | slide | velocity (m/s)           |
    | 10  | velocity of the z-coordinate (height) of the torso | -Inf | Inf | rootz                            | slide | velocity (m/s)           |
    | 11  | angular velocity of the angle of the torso         | -Inf | Inf | rooty                            | hinge | angular velocity (rad/s) |
    | 12  | angular velocity of the right thigh hinge          | -Inf | Inf | thigh_joint                      | hinge | angular velocity (rad/s) |
    | 13  | angular velocity of the right leg hinge            | -Inf | Inf | leg_joint                        | hinge | angular velocity (rad/s) |
    | 14  | angular velocity of the right foot hinge           | -Inf | Inf | foot_joint                       | hinge | angular velocity (rad/s) |
    | 15  | angular velocity of the left thigh hinge           | -Inf | Inf | thigh_left_joint                 | hinge | angular velocity (rad/s) |
    | 16  | angular velocity of the left leg hinge             | -Inf | Inf | leg_left_joint                   | hinge | angular velocity (rad/s) |
    | 17  | angular velocity of the left foot hinge            | -Inf | Inf | foot_left_joint                  | hinge | angular velocity (rad/s) |
    
    ## Starting State
    All observations start in state
    (0.0, 1.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    with a uniform noise in the range of [-`reset_noise_scale`, `reset_noise_scale`] added to the values for stochasticity.
    """
    def __init__(self, render_mode=None, seed=0):
        
        self.name = "Walker"
        self.max_episode_steps = 1000
        self.render_mode = render_mode
        self.reward_threshold = 3000.
        
        # Specify the specs to remove all env wrappers to access the Mujoco env directly
        self.spec = EnvSpec(id = "Walker2d-v4",           # The string used to create the environment with :meth:`gymnasium.make`
                       entry_point='gymnasium.envs.mujoco.walker2d_v4:Walker2dEnv', # A string for the environment location
                       nondeterministic=False,      # If the observation of an environment cannot be repeated with the same initial state, random number generator state and actions.
                       max_episode_steps=None,      # The max number of steps that the environment can take before truncation
                       order_enforce=False,         # If to enforce the order of :meth:`gymnasium.Env.reset` before :meth:`gymnasium.Env.step` and :meth:`gymnasium.Env.render` functions
                       autoreset=False,             # If to automatically reset the environment on episode end
                       disable_env_checker=True,    # If to disable the environment checker wrapper in :meth:`gymnasium.make`, by default False (runs the environment checker)
                       kwargs={'render_mode': render_mode}, # Additional keyword arguments passed to the environment during initialisation
                       additional_wrappers=(),      #  A tuple of additional wrappers applied to the environment (WrapperSpec)
                       vector_entry_point=None)     # The location of the vectorized environment to create from
         
        self.env = gym.make(self.spec, exclude_current_positions_from_observation=False) # add x-position
        assert self.env.model.opt.integrator == 0, "Select 'Euler' for the integrator in the <option> line 13 of XML file at '<<NAME_OF_YOUR_CONDA_ENVIRONMENT>>/Lib/site-packages/gymnasium/envs/mujoco/assets' "
        assert self.env.frame_skip == 1, "Need frame_skip = 1, i.e., single time step between states. Change in walker2d_v4.py line 210 and time step 0.008 in XML next to integrator"
        self.metadata = self.env.metadata
        
        self._seed = seed
        self.env.action_space.seed(self._seed)
        
        self.env.reset(seed = self._seed)
        self.state_size = 18
        self.action_size = 6
        
        self.action_max = np.array([[1., 1., 1., 1., 1., 1.]])
        self.action_min = -self.action_max
        self.dt = self.env.dt # 0.008s # time step
        
        self.position_states = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        self.velocity_states = [9, 10, 11, 12, 13, 14, 15, 16, 17] 
        self.state_labels = ["x", "z", "front tip", "back thigh", "back shin", "back foot", "front thigh", "front shin", "front foot", "v_x", "v_z", "angle vel front tip", "angle vel back thigh", "angle vel back shin", "angle vel back foot", "angle vel front thigh", "angle vel front shin", "angle vel front foot"]
        self.action_labels = ["Torque back thigh", "Torque back shin", "Torque back foot", "Torque front thigh", "Torque front shin", "Torque front foot"]
         
        self.min_height, self.max_height = self.env.unwrapped._healthy_z_range # (0.8, 2)  
        self.min_angle, self.max_angle = self.env.unwrapped._healthy_angle_range # (-1, 1)
        self.low_bound = -np.inf * np.ones(self.state_size)
        self.low_bound[1] = self.min_height
        self.low_bound[2] = self.min_angle
        self.high_bound = np.inf * np.ones(self.state_size)
        self.high_bound[1] = self.max_height
        self.high_bound[2] = self.max_angle
        
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
        done = terminated or truncated or self.episode_step >= self.max_episode_steps
        
        return self.state.copy(), reward, done, False, None
    
    
    def reset_to(self, state):
        """New function: reset the state to the one provided.
        Split the state into qpos and qvel to use  env.set_state(qpos, qvel)"""
        
        assert state.shape == (self.state_size,)
        qpos = state[:9] # desired positions 
        qvel = state[9:] # desired velocities
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
            - S_t : current state torch.tensor (18,)
            - vel_t_dt : next state's velocity torch.tensor (9,)
        Returns:
            - x_t_dt : next state's position torch.tensor (9,)
        """
        return S_t[:9] + self.dt * vel_t_dt


    # Plotting functions
    def plot_traj(self, Traj, title:str = ""):
        """Plots the top angle trajectory of the Walker."""
        plot_traj(self, Traj, title)

    
    def traj_comparison(self, traj_1, label_1, traj_2, label_2, title:str = "",
                        traj_3=None, label_3=None, traj_4=None, label_4=None,
                        plot_height:bool = True, legend_loc='best'):
        """
        Compares up to 4 trajectories of the Walker
        Arguments:
            - traj_1 : first trajectory of shape (H, 18)
            - label_1 : corresponding label to display
            - traj_2 : first trajectory of shape (H, 18)
            - label_2 : corresponding label to display
            - title: optional title of the plot
            - traj_3 : optional third trajectory of shape (H, 18)
            - label_3 : optional corresponding label to display
            - traj_4 : optional fourth trajectory of shape (H, 18)
            - label_4 : optional corresponding label to display
            - plot_height : optional whether to plot the Walker height or just the top angle
            - legend_loc : optional location of the legend
        """
        traj_comparison(self, traj_1, label_1, traj_2, label_2, title,
                        traj_3, label_3, traj_4, label_4, legend_loc)


