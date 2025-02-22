# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 11:44:21 2025

@author: Kanghyun Ryu, modified by Jean-Baptiste Bouvier

Gymnasium Environment for the Unitree GO1
"""

import os
import torch
import numpy as np
from gymnasium import utils
from gymnasium.spaces import Box
from gymnasium.envs.mujoco import MujocoEnv

from GO1.plots import plot_traj, traj_comparison

DEFAULT_CAMERA_CONFIG = { "distance": 6.0}
menagerie_path = './GO1/Unitree_go1'

def rotate_vector_by_quaternion(vec, quat):
    """
    Rotate a vector by a unit quaternion.
    
    Args:
        vec (array-like): The vector to rotate (3-element array or list).
        quat (array-like): The quaternion (4-element array or list in the format [w, x, y, z]).
    
    Returns:
        np.ndarray: The rotated vector.
    """
    # Convert to numpy arrays
    vec = np.array(vec, dtype=np.float64)
    quat = np.array(quat, dtype=np.float64)

    assert quat.shape == (4,), "Quaternion must be a 4-element array."
    assert vec.shape == (3,), "Vector must be a 3-element array."
    
    # Extract quaternion components
    s, u = quat[0], quat[1:]
    r = 2 * (np.dot(u, vec) * u) + (s * s - np.dot(u, u)) * vec + 2 * s * np.cross(u, vec)
    return r



class Go1Env(MujocoEnv, utils.EzPickle):
    """
    This environment describes the Unitree GO 1 robot MuJoCo menagerie from
    https://github.com/google-deepmind/mujoco_menagerie/tree/main/unitree_go1

    ## Action Space
    The action space is a `Box(-1, 1, (12,), float32)`. An action represents the torques applied at the hinge joints.

    | Num | Action                                 | Min  | Max  | Name in xml | Joint | Unit         |
    | --- | ---------------------------------------| ---- | ---- | ----------- | ----- | ------------ |
    | 0   | Desired angle of the front right hip   | -0.7 | 0.52 | FR_hip      | hinge | angle (rad) |
    | 1   | Desired angle of the front right thigh | -1.0 | 2.1  | FR_thigh    | hinge | angle (rad) |
    | 2   | Desired angle of the front right calf  | -2.2 | -0.4 | FR_calf     | hinge | angle (rad) |
    | 3   | Desired angle of the front left hip    | -0.7 | 0.52 | FL_hip      | hinge | angle (rad) |
    | 4   | Desired angle of the front left thigh  | -1.0 | 2.1  | FL_thigh    | hinge | angle (rad) |
    | 5   | Desired angle of the front left calf   | -2.2 | -0.4 | FL_calf     | hinge | angle (rad) |
    | 6   | Desired angle of the rear right hip    | -0.7 | 0.52 | RR_hip      | hinge | angle (rad) |
    | 7   | Desired angle of the rear right thigh  | -1.0 | 2.1  | RR_thigh    | hinge | angle (rad) |
    | 8   | Desired angle of the rear right calf   | -2.2 | -0.4 | RR_calf     | hinge | angle (rad) |
    | 9   | Desired angle of the rear left hip     | -0.7 | 0.52 | RL_hip      | hinge | angle (rad) |
    | 10  | Desired angle of the rear left thigh   | -1.0 | 2.1  | RL_thigh    | hinge | angle (rad) |
    | 11  | Desired angle of the rear left calf    | -2.2 | -0.4 | RL_calf     | hinge | angle (rad) |
   
    
    ## Observation Space

    | Num | Observation                                | Min    | Max    | Name in xml | Joint | Unit                     |
    |-----|--------------------------------------------|--------|--------|-------------|-------|--------------------------|
    | 0   | x-coordinate of the trunk                  | -Inf   | Inf    | trunk       | free  | position (m)             |
    | 1   | y-coordinate of the trunk                  | -Inf   | Inf    | trunk       | free  | position (m)             |
    | 2   | z-coordinate of the trunk                  | -Inf   | Inf    | trunk       | free  | position (m)             |
    | 3   | w-orientation of the trunk (quaternion)    | -Inf   | Inf    | trunk       | free  | angle (rad)              |
    | 4   | x-orientation of the trunk (quaternion)    | -Inf   | Inf    | trunk       | free  | angle (rad)              |
    | 5   | y-orientation of the trunk (quaternion)    | -Inf   | Inf    | trunk       | free  | angle (rad)              |
    | 6   | z-orientation of the trunk (quaternion)    | -Inf   | Inf    | trunk       | free  | angle (rad)              |
    | 7   | angle of the front right hip               | -Inf   | Inf    | FR_hip      | hinge | angle (rad)              |
    | 8   | angle of the front right thigh             | -Inf   | Inf    | FR_thigh    | hinge | angle (rad)              |
    | 9   | angle of the front right calf              | -Inf   | Inf    | FR_calf     | hinge | angle (rad)              |
    | 10  | angle of the front left hip                | -Inf   | Inf    | FL_hip      | hinge | angle (rad)              |
    | 11  | angle of the front left thigh              | -Inf   | Inf    | FL_thigh    | hinge | angle (rad)              |
    | 12  | angle of the front left calf               | -Inf   | Inf    | FL_calf     | hinge | angle (rad)              |
    | 13  | angle of the rear right hip                | -Inf   | Inf    | RR_hip      | hinge | angle (rad)              |
    | 14  | angle of the rear right thigh              | -Inf   | Inf    | RR_thigh    | hinge | angle (rad)              |
    | 15  | angle of the rear right calf               | -Inf   | Inf    | RR_calf     | hinge | angle (rad)              |
    | 16  | angle of the rear left hip                 | -Inf   | Inf    | RL_hip      | hinge | angle (rad)              |
    | 17  | angle of the rear left thigh               | -Inf   | Inf    | RL_thigh    | hinge | angle (rad)              |
    | 18  | angle of the rear left calf                | -Inf   | Inf    | RL_calf     | hinge | angle (rad)              |
    
    | 19  | x-coordinate velocity of the trunk         | -Inf   | Inf    | trunk       | free  | velocity (m/s)           |
    | 20  | y-coordinate velocity of the trunk         | -Inf   | Inf    | trunk       | free  | velocity (m/s)           |
    | 21  | z-coordinate velocity of the trunk         | -Inf   | Inf    | trunk       | free  | velocity (m/s)           |
    | 22  | x-coordinate angular velocity of the trunk | -Inf   | Inf    | trunk       | free  | angular velocity (rad/s) |
    | 23  | y-coordinate angular velocity of the trunk | -Inf   | Inf    | trunk       | free  | angular velocity (rad/s) |
    | 24  | z-coordinate angular velocity of the trunk | -Inf   | Inf    | trunk       | free  | angular velocity (rad/s) |
    | 25  | angular velocity of the front right hip    | -Inf   | Inf    | FR_hip      | hinge | angular velocity (rad/s) |
    | 26  | angular velocity of the front right thigh  | -Inf   | Inf    | FR_thigh    | hinge | angular velocity (rad/s) |
    | 27  | angular velocity of the front right calf   | -Inf   | Inf    | FR_calf     | hinge | angular velocity (rad/s) |
    | 28  | angular velocity of the front left hip     | -Inf   | Inf    | FL_hip      | hinge | angular velocity (rad/s) |
    | 29  | angular velocity of the front left thigh   | -Inf   | Inf    | FL_thigh    | hinge | angular velocity (rad/s) |
    | 30  | angular velocity of the front left calf    | -Inf   | Inf    | FL_calf     | hinge | angular velocity (rad/s) |
    | 31  | angular velocity of the rear right hip     | -Inf   | Inf    | RR_hip      | hinge | angular velocity (rad/s) |
    | 32  | angular velocity of the rear right thigh   | -Inf   | Inf    | RR_thigh    | hinge | angular velocity (rad/s) |
    | 33  | angular velocity of the rear right calf    | -Inf   | Inf    | RR_calf     | hinge | angular velocity (rad/s) |
    | 34  | angular velocity of the rear left hip      | -Inf   | Inf    | RL_hip      | hinge | angular velocity (rad/s) |
    | 35  | angular velocity of the rear left thigh    | -Inf   | Inf    | RL_thigh    | hinge | angular velocity (rad/s) |
    | 36  | angular velocity of the rear left calf     | -Inf   | Inf    | RL_calf     | hinge | angular velocity (rad/s) |
    

    ## Rewards
    The reward is the sum of four parts:
    -reward_tracking_lin_vel: difference between xy velocity and the command
    -reward_tracking_ang_vel: difference between the yaw velocity and the command
    -reward_lin_vel_z: penalized z-axis velocity
    -reward_action_rate: penalized changes in action


    ## Starting State
    All observations start with 0 velocity from position
    [0, 0, 0.27, 1, 0, 0, 0, 0, 0.9, -1.8, 0, 0.9, -1.8, 0, 0.9, -1.8, 0, 0.9, -1.8] 
    with a uniform noise in the range of [-`reset_noise_scale`, `reset_noise_scale`]
    added to the position. 
    
    ## Episode End
    1. Any of the leg joint angles is **not** in [-0.7, 0.52]*[-1.0, 2.1]*[-2.2, -0.4]
    2. The z-coordinate of the trunk is **not** in [0.18, 1.0]
    3. The torso is upside down
    """
    
    metadata = {"render_modes": ["human", "rgb_array", "depth_array"], "render_fps": 50}
    
    def __init__(self, action_scale: float = 0.3, 
                 reset_noise_scale: float = 0.05, **kwargs):
        
        self.name = "GO1"
        self.action_size = 12
        self.state_size = 37
        self.command_size = 1 # only x-velocity
        
        self._xml_file = os.path.join(menagerie_path, 'scene_mjx_gym.xml')
        utils.EzPickle.__init__(self, self._xml_file, **kwargs)

        observation_space = Box(low=-100., high=100., shape=(self.state_size,), dtype=np.float64)
        self._action_scale = action_scale
        self._reset_noise_scale = reset_noise_scale

        self._xml_dt = 0.02 # timestep from the XML file
        MujocoEnv.__init__(self, self._xml_file, frame_skip=1,
                           observation_space=observation_space,
                           default_camera_config=DEFAULT_CAMERA_CONFIG, **kwargs)

        assert self.model.opt.integrator == 0, 'Use Euler integration for scene_mjx_gym.xml'
        assert self.frame_skip == 1, 'Use frame_skip=1 for Euler integration in scene_mjx_gym.xml'

        self._init_q = np.array([0, 0, 0.27, 1, 0, 0, 0, 0, 0.9, -1.8, 0, 0.9, -1.8, 0, 0.9, -1.8, 0, 0.9, -1.8])
        self._default_pose = np.array([0.0, 0.9, -1.8] * 4)
        self.lowers = np.array([-0.7, -1.0, -2.2] * 4)
        self.uppers = np.array([0.52, 2.1, -0.4] * 4)
        
        self.min_height = 0.18
        low_xyz_quat = np.array([-100., -100., self.min_height, -10., -10., -10., -10.])
        high_xyz_quat = np.array([100., 100., 1., 10., 10., 10., 10.])
        low_vel = -100 * np.ones(18)
        high_vel = 100 * np.ones(18)
        self.low_bound = np.concatenate((low_xyz_quat, self.lowers, low_vel))
        self.high_bound = np.concatenate((high_xyz_quat, self.uppers, high_vel))
        self.action_min = -1 * np.ones(self.action_size)
        self.action_max =  1 * np.ones(self.action_size)
        self.position_states = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
        self.velocity_states = list(range(19, 37))
        

    def sample_command(self, command:np.ndarray = None):
        lin_vel_x = [-0.6, 1.5]  # min max [m/s]
        # lin_vel_y = [-0.8, 0.8]  # min max [m/s]
        # ang_vel_yaw = [-0.7, 0.7]  # min max [rad/s]

        if command is None:
            lin_vel_x   = np.random.uniform(low=lin_vel_x[0],   high=lin_vel_x[1])
            # lin_vel_y   = np.random.uniform(low=lin_vel_y[0],   high=lin_vel_y[1])
            # ang_vel_yaw = np.random.uniform(low=ang_vel_yaw[0], high=ang_vel_yaw[1])
            new_cmd = np.array([lin_vel_x]) #, lin_vel_y, ang_vel_yaw])
        else:
            assert command.shape == (self.command_size,)
            new_cmd = command
        
        return new_cmd
    

    def reset(self, seed=None, options=None):
        qpos = self._init_q + np.random.uniform(low=-self._reset_noise_scale,
                                                high=self._reset_noise_scale,
                                                size=self.model.nq)
        qvel = np.zeros(self.model.nv)

        self.set_state(qpos, qvel)
        self.data.qacc_warmstart[:] = 0.0 # to enable reproducibility
        obs = self.get_full_state()

        self.info = {'command': self.sample_command(),
                     'last_act': np.zeros(self.action_size),
                     'step': 0}
        return obs


    def step(self, action):
        
        # Physics step
        self.data.qacc_warmstart[:] = 0.0 # to enable reproducibility
        motor_targets = self._default_pose + action * self._action_scale
        motor_targets = np.clip(motor_targets, self.lowers, self.uppers)
        self.do_simulation(motor_targets, self.frame_skip)
        joint_angles = self.data.qpos[7:]        
        obs = self.get_full_state()
        self.info['step'] += 1
        
        # done if joint limits are reached or robot is falling
        up = np.array([0.0, 0.0, 1.0])
        done = np.dot(rotate_vector_by_quaternion(up, self.data.qpos[3:7]), up) < 0
        done |= np.any(joint_angles < self.lowers)
        done |= np.any(joint_angles > self.uppers)
        done |= self.data.qpos[2] < self.min_height
        
        reward = self._reward_tracking_lin_vel() + self._reward_lin_vel_z() + self._reward_action_rate(action) # + self._reward_tracking_ang_vel()
        
        self.info['last_act'] = action
        
        if self.render_mode == "human":
            self.render()

        return obs, reward, done, False, self.info


    def reset_to(self, state, command=None):
        assert state.shape == (self.state_size,)
        self.reset()
        self.set_state(state[:19], state[19:])
        self.data.qacc_warmstart[:] = 0.0
        
        if command is None:
            command = self.sample_command()
        else:
            assert command.shape == (self.command_size,)

        self.info = {'command': self.sample_command(),
                     'last_act': np.zeros(self.action_size),
                     'step': 0}
        obs = self.get_full_state()
        return obs
    

    def get_full_state(self):
        qpos = self.data.qpos.ravel().copy()
        qvel = self.data.qvel.ravel().copy()
        return np.concatenate([qpos, qvel])
    
    
    def _reward_tracking_lin_vel(self):
        command = self.info['command']
        lin_vel = self.data.sensor('local_linvel').data
        # lin_vel_error = np.sum(np.square(command[:2] - lin_vel[:2]))
        lin_vel_error = np.sum(np.square(command[0] - lin_vel[0]))
        return 1.5*np.exp(-lin_vel_error)

    # def _reward_tracking_ang_vel(self):
    #     command = self.info['command']
    #     yaw_vel = self.data.qvel.ravel().copy()[5]
    #     ang_vel_error = np.sum(np.square(command[2] - yaw_vel))
    #     return 0.8*np.exp(-ang_vel_error)
    
    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        z_vel = self.data.qvel[2].copy()
        return -2*z_vel**2
    
    def _reward_action_rate(self, action):
        # Penalize changes in action
        return -0.01*np.sum(np.square(action - self.info['last_act']))


    # Function called by the projectors 
    def pos_from_vel(self, S_t, vel_t_dt):
        """
        Calculates next state's position using semi-implicit Euler integrator
        and quaternion formula, does not need to know the dynamics.

        Arguments:
            - S_t : current position torch.tensor (19,)
            - vel_t_dt : next state's velocity torch.tensor (18,)
        Returns:
            - pos_t_dt : next states's position torch.tensor (19,)
        """
        pos_t_dt = S_t[:19].copy() # copy the current position
        pos_t_dt[:3] += vel_t_dt[:3] * self.dt # linear position update
        pos_t_dt[-12:] += vel_t_dt[-12:] * self.dt # angles updates
        
        # Quaternion update
        q0 = S_t[3]
        q1 = S_t[4]
        q2 = S_t[5]
        q3 = S_t[6]
        
        p = vel_t_dt[3]
        q = vel_t_dt[4]
        r = vel_t_dt[5]
        
        pos_t_dt[3:7] += self.dt * torch.tensor([-0.5*p*q1 - 0.5*q*q2 - 0.5*q3*r,
                                                  0.5*p*q0 - 0.5*q*q3 + 0.5*q2*r,
                                                  0.5*p*q3 + 0.5*q*q0 - 0.5*q1*r,
                                                 -0.5*p*q2 + 0.5*q*q1 + 0.5*q0*r]).to(S_t.device)
        return pos_t_dt
    
    
    # Plotting functions
    def plot_traj(self, Traj, title:str = ""):
        """Plots the xy trajectory of the Unitree GO1."""
        plot_traj(self, Traj, title)

    
    def traj_comparison(self, traj_1, label_1, traj_2, label_2, title:str = "",
                        traj_3=None, label_3=None, traj_4=None, label_4=None,
                        plot_z:bool = True, legend_loc='best'):
        """
        Compares up to 4 trajectories of the Unitree GO1
        Arguments:
            - traj_1 : first trajectory of shape (H, 37)
            - label_1 : corresponding label to display
            - traj_2 : first trajectory of shape (H, 37)
            - label_2 : corresponding label to display
            - title: optional title of the plot
            - traj_3 : optional third trajectory of shape (H, 37)
            - label_3 : optional corresponding label to display
            - traj_4 : optional fourth trajectory of shape (H, 37)
            - label_4 : optional corresponding label to display
            - plot_z : optional whether to plot the body height
            - legend_loc : optional location of the legend
        """
        traj_comparison(self, traj_1, label_1, traj_2, label_2, title,
                        traj_3, label_3, traj_4, label_4,
                        plot_z, legend_loc)


