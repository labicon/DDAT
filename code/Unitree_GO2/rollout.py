# Make gymnasium environment
import gymnasium as gym
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium import utils
from gymnasium.spaces import Box

import math
from scipy.spatial.transform import Rotation as R
import numpy as np
import os
from typing import Optional
import imageio

DEFAULT_CAMERA_CONFIG = {
    "distance": 6.0,
}

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
    
    # Ensure the quaternion is normalized
    # quat = quat / np.linalg.norm(quat)

    assert quat.shape == (4,), "Quaternion must be a 4-element array."
    assert vec.shape == (3,), "Vector must be a 3-element array."
    
    # Extract quaternion components
    s, u = quat[0], quat[1:]
    r = 2 * (np.dot(u, vec) * u) + (s * s - np.dot(u, u)) * vec + 2 * s * np.cross(u, vec)
    
    return r

def quat_inv(quat):
    # Compute a inverse of quaternion
    # quat = quat / np.linalg.norm(quat)
    
    assert quat.shape == (4,), "Quaternion must be a 4-element array."
    return np.array([quat[0], -quat[1], -quat[2], -quat[3]])

menagerie_path = '../../assets/Unitree_GO2/'

class Go2JoystickGymEnv(MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 50,
    }
    def __init__(
        self,
        obs_noise: float = 0.05,
        action_scale: float = 0.3,
        frame_skip=10,
        **kwargs,
    ):
        
        self._xml_file = os.path.join(menagerie_path,'scene_mjx_gym.xml')
        utils.EzPickle.__init__(
            self,
            self._xml_file,
            **kwargs,
        )

        obs_shape = 31
        observation_space = Box(low=-100., high=100., shape=(obs_shape * 15,), dtype=np.float64)

        # self._obs_noise = obs_noise
        self._action_scale = action_scale

        self._xml_dt = 0.02 # timestep from the XML file
        MujocoEnv.__init__(
            self,
            self._xml_file,
            frame_skip,
            observation_space=observation_space,
            default_camera_config=DEFAULT_CAMERA_CONFIG,
            **kwargs,
        )

        self._init_q = np.array([0, 0, 0.27, 1, 0, 0, 0, 0, 0.9, -1.8, 0, 0.9, -1.8, 0, 0.9, -1.8, 0, 0.9, -1.8])
        self._default_pose = np.array([0.0, 0.9, -1.8] * 4)
        self.lowers = np.array([-0.7, -1.0, -2.2] * 4)
        self.uppers = np.array([0.52, 2.1, -0.4] * 4)

    def sample_command(self, command: Optional[np.ndarray] = None) -> np.ndarray:
        lin_vel_x = [-0.6, 1.5]  # min max [m/s]
        lin_vel_y = [-0.8, 0.8]  # min max [m/s]
        ang_vel_yaw = [-0.7, 0.7]  # min max [rad/s]

        if command is None:
            lin_vel_x = np.random.uniform(
                low=lin_vel_x[0], high=lin_vel_x[1]
            )
            lin_vel_y = np.random.uniform(
                low=lin_vel_y[0], high=lin_vel_y[1]
            )
            ang_vel_yaw = np.random.uniform(
                low=ang_vel_yaw[0], high=ang_vel_yaw[1]
            )
            new_cmd = np.array([lin_vel_x, lin_vel_y, ang_vel_yaw])
        else:
            new_cmd = command
        
        new_cmd = np.array([1.0, 0.0, 0.0])

        return new_cmd
    
    def reset(self, seed=None, options=None):
        qpos = self._init_q #+ np.random.uniform(low=-0.05, high=0.05, size=self.model.nq)
        qvel = np.zeros(self.model.nv)

        self.set_state(qpos, qvel)
        self.data.qacc_warmstart[:] = 0.0

        state_info = {
            'last_act': np.zeros(12),
            'last_vel': np.zeros(12),
            'command': self.sample_command(),
            'kick': np.array([0.0, 0.0]),
            'step': 0,
        }
        self.info = state_info

        obs_history = np.zeros(15 * 31)  # store 15 steps of history
        obs = self._get_obs(state_info, obs_history)
        self.obs = obs

        return obs, state_info

    def step(self, action):
        
        # Physics step
        motor_targets = self._default_pose + action * self._action_scale
        motor_targets = np.clip(motor_targets, self.lowers, self.uppers)
        self.do_simulation(motor_targets, self.frame_skip)

        # Observation data
        obs = self._get_obs(self.info, self.obs)
        joint_angles = self.data.qpos[7:]
        joint_vel = self.data.qvel[6:]
        self.obs = obs

        # done if joint limits are reached or robot is falling
        up = np.array([0.0, 0.0, 1.0])
        done = np.dot(rotate_vector_by_quaternion(up, self.data.qpos[3:7]), up) < 0
        done |= np.any(joint_angles < self.lowers)
        done |= np.any(joint_angles > self.uppers)
        done |= self.data.qpos[2] < 0.18

        # state management
        self.info['last_act'] = action

        # sample new command if more than 500 timesteps achieved
        if self.info['step'] > 500:
            self.info['command'] = self.sample_command()
        
        # reset the step counter when done
        if done or self.info['step'] > 500:
            self.info['step'] = 0

        # Reward
        reward = 0.0

        if self.render_mode == "human":
            self.render()

        return obs, reward, False, False, self.info


    def _get_obs(self, state_info, obs_history):
        torso_quat = self.data.qpos[3:7].copy()
        inv_torso_rot = quat_inv(torso_quat)
        torso_ang_vel = self.data.qvel[3:6].copy()
        local_rpyrate = rotate_vector_by_quaternion(torso_ang_vel, inv_torso_rot)

        obs = np.concatenate([
            np.array([local_rpyrate[2]]) * 0.25,                 # yaw rate
            rotate_vector_by_quaternion(np.array([0, 0, -1]), inv_torso_rot),    # projected gravity
            state_info['command'] * np.array([2.0, 2.0, 0.25]),  # command
            self.data.qpos[7:] - self._default_pose,           # motor angles
            state_info['last_act'],                              # last action
        ])

        # clip, no noise
        obs = np.clip(obs, -100.0, 100.0) # + self._obs_noise * np.random.uniform(-1, 1, obs.shape)

        # stack observations through time
        obs_history = np.roll(obs_history, obs.size)
        obs_history[:obs.size] = obs

        return obs_history

    def reset_to(self, qpos, qvel):
        self.reset()
        self.set_state(qpos, qvel)
        self.data.qacc_warmstart[:] = 0.0

        state_info = {
            'last_act': np.zeros(12),
            'last_vel': np.zeros(12),
            'command': self.sample_command(),
            'kick': np.array([0.0, 0.0]),
            'step': 0,
        }
        self.info = state_info

        obs_history = np.zeros(15 * 31)  # store 15 steps of history
        obs = self._get_obs(state_info, obs_history)
        self.obs = obs

        return obs
    
    def get_full_state(self):
        qpos = self.data.qpos.ravel().copy()
        qvel = self.data.qvel.ravel().copy()
        return np.concatenate([qpos, qvel])
    
# Supporting
import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.8"  # 0.9 causes too much lag.
import functools
from etils import epath
from typing import Any, Sequence, Optional, List
from ml_collections import config_dict

# Math
import jax
import jax.numpy as jp
import numpy as np
np.set_printoptions(precision=3, suppress=True, linewidth=100)  # More legible printing from numpy.
from jax import config  # Analytical gradients work much better with double precision.
# config.update("jax_debug_nans", True)
# config.update("jax_enable_x64", True)
# config.update('jax_default_matmul_precision', 'high')

from brax import base
from brax import envs
from brax import math
from brax.base import Motion, Transform
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf, model

import mujoco

# Algorithm
from brax.training.agents.ppo import train as ppo
from brax.training.agents.ppo import networks as ppo_networks


menagerie_path = '../../assets/Unitree_GO2/'
GO2_ROOT_PATH = epath.Path(menagerie_path)


def get_config():
  """Returns reward config for barkour quadruped environment."""

  def get_default_rewards_config():
    default_config = config_dict.ConfigDict(
        dict(
            # The coefficients for all reward terms used for training. All
            # physical quantities are in SI units, if no otherwise specified,
            # i.e. joint positions are in rad, positions are measured in meters,
            # torques in Nm, and time in seconds, and forces in Newtons.
            scales=config_dict.ConfigDict(
                dict(
                    # Tracking rewards are computed using exp(-delta^2/sigma)
                    # sigma can be a hyperparameters to tune.
                    # Track the base x-y velocity (no z-velocity tracking.)
                    tracking_lin_vel=1.5,
                    # Track the angular velocity along z-axis, i.e. yaw rate.
                    tracking_ang_vel=0.8,
                    # Below are regularization terms, we roughly divide the
                    # terms to base state regularizations, joint
                    # regularizations, and other behavior regularizations.
                    # Penalize the base velocity in z direction, L2 penalty.
                    lin_vel_z=-2.0,
                    # Penalize the base roll and pitch rate. L2 penalty.
                    ang_vel_xy=-0.05,
                    # Penalize non-zero roll and pitch angles. L2 penalty.
                    orientation=-5.0,
                    # L2 regularization of joint torques, |tau|^2.
                    torques=-0.0002,
                    # Penalize the change in the action and encourage smooth
                    # actions. L2 regularization |action - last_action|^2
                    action_rate=-0.01,
                    # Encourage long swing steps.  However, it does not
                    # encourage high clearances.
                    feet_air_time=0.2,
                    # Encourage no motion at zero command, L2 regularization
                    # |q - q_default|^2.
                    stand_still=-0.5,
                    # Early termination penalty.
                    termination=-1.0,
                    # Penalizing foot slipping on the ground.
                    foot_slip=-0.1,
                )
            ),
            # Tracking reward = exp(-error^2/sigma).
            tracking_sigma=0.25,
        )
    )
    return default_config

  default_config = config_dict.ConfigDict(
      dict(
          rewards=get_default_rewards_config(),
      )
  )

  return default_config


class Go2JoystickEnv(PipelineEnv):
  """Environment for training the go2 quadruped joystick policy in MJX."""

  def __init__(
      self,
      obs_noise: float = 0.05,
      action_scale: float = 0.3,
      kick_vel: float = 0.05,
      scene_file: str = 'scene_mjx.xml', # 'scene_mjx.xml'
      **kwargs,
  ):
    path = GO2_ROOT_PATH / scene_file
    sys = mjcf.load(path.as_posix())
    self._dt = 0.02  # this environment is 50 fps
    if scene_file == 'scene_mjx_euler.xml':
      sys = sys.tree_replace({'opt.timestep': 0.02})
    else:
      sys = sys.tree_replace({'opt.timestep': 0.004}) # 0.004 for original env or 0.02 for Euler

    # override menagerie params for smoother policy
    sys = sys.replace(
        dof_damping=sys.dof_damping.at[6:].set(0.5239),
        actuator_gainprm=sys.actuator_gainprm.at[:, 0].set(35.0),
        actuator_biasprm=sys.actuator_biasprm.at[:, 1].set(-35.0),
    )

    n_frames = kwargs.pop('n_frames', int(self._dt / sys.opt.timestep))
    super().__init__(sys, backend='mjx', n_frames=n_frames)

    self.reward_config = get_config()
    # set custom from kwargs
    for k, v in kwargs.items():
      if k.endswith('_scale'):
        self.reward_config.rewards.scales[k[:-6]] = v

    self._torso_idx = mujoco.mj_name2id(
        sys.mj_model, mujoco.mjtObj.mjOBJ_BODY.value, 'base'
    )
    self._action_scale = action_scale
    self._obs_noise = obs_noise
    self._kick_vel = kick_vel
    self._init_q = jp.array(sys.mj_model.keyframe('home').qpos)
    self._default_pose = sys.mj_model.keyframe('home').qpos[7:]
    self.lowers = jp.array([-0.7, -1.0, -2.2] * 4)
    self.uppers = jp.array([0.52, 2.1, -0.4] * 4)
    feet_site = [
        'FL_foot',
        'RL_foot',
        'FR_foot',
        'RR_foot',
    ]
    feet_site_id = [
        mujoco.mj_name2id(sys.mj_model, mujoco.mjtObj.mjOBJ_SITE.value, f)
        for f in feet_site
    ]
    assert not any(id_ == -1 for id_ in feet_site_id), 'Site not found.'
    self._feet_site_id = np.array(feet_site_id)
    lower_leg_body = [
        'FL_calf',
        'RL_calf',
        'FR_calf',
        'RR_calf',
    ]
    lower_leg_body_id = [
        mujoco.mj_name2id(sys.mj_model, mujoco.mjtObj.mjOBJ_BODY.value, l)
        for l in lower_leg_body
    ]
    assert not any(id_ == -1 for id_ in lower_leg_body_id), 'Body not found.'
    self._lower_leg_body_id = np.array(lower_leg_body_id)
    self._foot_radius = 0.0175
    self._nv = sys.nv

  def sample_command(self, rng: jax.Array) -> jax.Array:
    lin_vel_x = [-0.6, 1.5]  # min max [m/s]
    lin_vel_y = [-0.8, 0.8]  # min max [m/s]
    ang_vel_yaw = [-0.7, 0.7]  # min max [rad/s]

    _, key1, key2, key3 = jax.random.split(rng, 4)
    lin_vel_x = jax.random.uniform(
        key1, (1,), minval=lin_vel_x[0], maxval=lin_vel_x[1]
    )
    lin_vel_y = jax.random.uniform(
        key2, (1,), minval=lin_vel_y[0], maxval=lin_vel_y[1]
    )
    ang_vel_yaw = jax.random.uniform(
        key3, (1,), minval=ang_vel_yaw[0], maxval=ang_vel_yaw[1]
    )
    # new_cmd = jp.array([lin_vel_x[0], lin_vel_y[0], ang_vel_yaw[0]])
    new_cmd = jp.array([1.0, 0.0, 0.0])
    return new_cmd

  def reset(self, rng: jax.Array) -> State:  # pytype: disable=signature-mismatch
    rng, key = jax.random.split(rng)

    pipeline_state = self.pipeline_init(self._init_q, jp.zeros(self._nv))

    state_info = {
        'rng': rng,
        'last_act': jp.zeros(12),
        'last_vel': jp.zeros(12),
        'command': self.sample_command(key),
        'last_contact': jp.zeros(4, dtype=bool),
        'feet_air_time': jp.zeros(4),
        'rewards': {k: 0.0 for k in self.reward_config.rewards.scales.keys()},
        'kick': jp.array([0.0, 0.0]),
        'step': 0,
    }

    obs_history = jp.zeros(15 * 31)  # store 15 steps of history
    obs = self._get_obs(pipeline_state, state_info, obs_history)
    reward, done = jp.zeros(2)
    metrics = {'total_dist': 0.0}
    for k in state_info['rewards']:
      metrics[k] = state_info['rewards'][k]
    state = State(pipeline_state, obs, reward, done, metrics, state_info)  # pytype: disable=wrong-arg-types
    return state

  def step(self, state: State, action: jax.Array) -> State:  # pytype: disable=signature-mismatch
    rng, cmd_rng, kick_noise_2 = jax.random.split(state.info['rng'], 3)

    # kick
    push_interval = 10
    kick_theta = jax.random.uniform(kick_noise_2, maxval=2 * jp.pi)
    kick = jp.array([jp.cos(kick_theta), jp.sin(kick_theta)])
    kick *= jp.mod(state.info['step'], push_interval) == 0
    qvel = state.pipeline_state.qvel  # pytype: disable=attribute-error
    qvel = qvel.at[:2].set(kick * self._kick_vel + qvel[:2])
    state = state.tree_replace({'pipeline_state.qvel': qvel})

    # physics step
    motor_targets = self._default_pose + action * self._action_scale
    motor_targets = jp.clip(motor_targets, self.lowers, self.uppers)
    # jax.debug.print('motor_targets: {target}', target=motor_targets)
    pipeline_state = self.pipeline_step(state.pipeline_state, motor_targets)
    x, xd = pipeline_state.x, pipeline_state.xd

    # observation data
    obs = self._get_obs(pipeline_state, state.info, state.obs)
    joint_angles = pipeline_state.q[7:]
    joint_vel = pipeline_state.qd[6:]

    # foot contact data based on z-position
    foot_pos = pipeline_state.site_xpos[self._feet_site_id]  # pytype: disable=attribute-error
    foot_contact_z = foot_pos[:, 2] - self._foot_radius
    contact = foot_contact_z < 1e-3  # a mm or less off the floor
    contact_filt_mm = contact | state.info['last_contact']
    contact_filt_cm = (foot_contact_z < 3e-2) | state.info['last_contact']
    first_contact = (state.info['feet_air_time'] > 0) * contact_filt_mm
    state.info['feet_air_time'] += self.dt

    # done if joint limits are reached or robot is falling
    up = jp.array([0.0, 0.0, 1.0])
    done = jp.dot(math.rotate(up, x.rot[self._torso_idx - 1]), up) < 0
    done |= jp.any(joint_angles < self.lowers)
    done |= jp.any(joint_angles > self.uppers)
    done |= pipeline_state.x.pos[self._torso_idx - 1, 2] < 0.18

    # reward
    rewards = {
        'tracking_lin_vel': (
            self._reward_tracking_lin_vel(state.info['command'], x, xd)
        ),
        'tracking_ang_vel': (
            self._reward_tracking_ang_vel(state.info['command'], x, xd)
        ),
        'lin_vel_z': self._reward_lin_vel_z(xd),
        'ang_vel_xy': self._reward_ang_vel_xy(xd),
        'orientation': self._reward_orientation(x),
        'torques': self._reward_torques(pipeline_state.qfrc_actuator),  # pytype: disable=attribute-error
        'action_rate': self._reward_action_rate(action, state.info['last_act']),
        'stand_still': self._reward_stand_still(
            state.info['command'], joint_angles,
        ),
        'feet_air_time': self._reward_feet_air_time(
            state.info['feet_air_time'],
            first_contact,
            state.info['command'],
        ),
        'foot_slip': self._reward_foot_slip(pipeline_state, contact_filt_cm),
        'termination': self._reward_termination(done, state.info['step']),
    }
    rewards = {
        k: v * self.reward_config.rewards.scales[k] for k, v in rewards.items()
    }
    reward = jp.clip(sum(rewards.values()) * self.dt, 0.0, 10000.0)

    # state management
    state.info['kick'] = kick
    state.info['last_act'] = action
    state.info['last_vel'] = joint_vel
    state.info['feet_air_time'] *= ~contact_filt_mm
    state.info['last_contact'] = contact
    state.info['rewards'] = rewards
    state.info['step'] += 1
    state.info['rng'] = rng

    # sample new command if more than 500 timesteps achieved
    state.info['command'] = jp.where(
        state.info['step'] > 500,
        self.sample_command(cmd_rng),
        state.info['command'],
    )
    # reset the step counter when done
    state.info['step'] = jp.where(
        done | (state.info['step'] > 500), 0, state.info['step']
    )

    # log total displacement as a proxy metric
    state.metrics['total_dist'] = math.normalize(x.pos[self._torso_idx - 1])[1]
    state.metrics.update(state.info['rewards'])

    done = jp.float32(done)
    state = state.replace(
        pipeline_state=pipeline_state, obs=obs, reward=reward, done=done
    )
    return state

  def _get_obs(
      self,
      pipeline_state: base.State,
      state_info: dict[str, Any],
      obs_history: jax.Array,
  ) -> jax.Array:
    inv_torso_rot = math.quat_inv(pipeline_state.x.rot[0])
    local_rpyrate = math.rotate(pipeline_state.xd.ang[0], inv_torso_rot)

    obs = jp.concatenate([
        jp.array([local_rpyrate[2]]) * 0.25,                 # yaw rate
        math.rotate(jp.array([0, 0, -1]), inv_torso_rot),    # projected gravity
        state_info['command'] * jp.array([2.0, 2.0, 0.25]),  # command
        pipeline_state.q[7:] - self._default_pose,           # motor angles
        state_info['last_act'],                              # last action
    ])

    # clip, noise
    obs = jp.clip(obs, -100.0, 100.0) + self._obs_noise * jax.random.uniform(
        state_info['rng'], obs.shape, minval=-1, maxval=1
    )
    # stack observations through time
    obs = jp.roll(obs_history, obs.size).at[:obs.size].set(obs)

    return obs

  # ------------ reward functions----------------
  def _reward_lin_vel_z(self, xd: Motion) -> jax.Array:
    # Penalize z axis base linear velocity
    return jp.square(xd.vel[0, 2])

  def _reward_ang_vel_xy(self, xd: Motion) -> jax.Array:
    # Penalize xy axes base angular velocity
    return jp.sum(jp.square(xd.ang[0, :2]))

  def _reward_orientation(self, x: Transform) -> jax.Array:
    # Penalize non flat base orientation
    up = jp.array([0.0, 0.0, 1.0])
    rot_up = math.rotate(up, x.rot[0])
    return jp.sum(jp.square(rot_up[:2]))

  def _reward_torques(self, torques: jax.Array) -> jax.Array:
    # Penalize torques
    return jp.sqrt(jp.sum(jp.square(torques))) + jp.sum(jp.abs(torques))

  def _reward_action_rate(
      self, act: jax.Array, last_act: jax.Array
  ) -> jax.Array:
    # Penalize changes in actions
    return jp.sum(jp.square(act - last_act))

  def _reward_tracking_lin_vel(
      self, commands: jax.Array, x: Transform, xd: Motion
  ) -> jax.Array:
    # Tracking of linear velocity commands (xy axes)
    local_vel = math.rotate(xd.vel[0], math.quat_inv(x.rot[0]))
    lin_vel_error = jp.sum(jp.square(commands[:2] - local_vel[:2]))
    lin_vel_reward = jp.exp(
        -lin_vel_error / self.reward_config.rewards.tracking_sigma
    )
    return lin_vel_reward

  def _reward_tracking_ang_vel(
      self, commands: jax.Array, x: Transform, xd: Motion
  ) -> jax.Array:
    # Tracking of angular velocity commands (yaw)
    base_ang_vel = math.rotate(xd.ang[0], math.quat_inv(x.rot[0]))
    ang_vel_error = jp.square(commands[2] - base_ang_vel[2])
    return jp.exp(-ang_vel_error / self.reward_config.rewards.tracking_sigma)

  def _reward_feet_air_time(
      self, air_time: jax.Array, first_contact: jax.Array, commands: jax.Array
  ) -> jax.Array:
    # Reward air time.
    rew_air_time = jp.sum((air_time - 0.1) * first_contact)
    rew_air_time *= (
        math.normalize(commands[:2])[1] > 0.05
    )  # no reward for zero command
    return rew_air_time

  def _reward_stand_still(
      self,
      commands: jax.Array,
      joint_angles: jax.Array,
  ) -> jax.Array:
    # Penalize motion at zero commands
    return jp.sum(jp.abs(joint_angles - self._default_pose)) * (
        math.normalize(commands[:2])[1] < 0.1
    )

  def _reward_foot_slip(
      self, pipeline_state: base.State, contact_filt: jax.Array
  ) -> jax.Array:
    # get velocities at feet which are offset from lower legs
    # pytype: disable=attribute-error
    pos = pipeline_state.site_xpos[self._feet_site_id]  # feet position
    feet_offset = pos - pipeline_state.xpos[self._lower_leg_body_id]
    # pytype: enable=attribute-error
    offset = base.Transform.create(pos=feet_offset)
    foot_indices = self._lower_leg_body_id - 1  # we got rid of the world body
    foot_vel = offset.vmap().do(pipeline_state.xd.take(foot_indices)).vel

    # Penalize large feet velocity for feet that are in contact with the ground.
    return jp.sum(jp.square(foot_vel[:, :2]) * contact_filt.reshape((-1, 1)))

  def _reward_termination(self, done: jax.Array, step: jax.Array) -> jax.Array:
    return done & (step < 500)

  def render(
      self, trajectory: List[base.State], camera: str | None = None,
      width: int = 240, height: int = 320,
  ) -> Sequence[np.ndarray]:
    camera = camera or 'track'
    return super().render(trajectory, camera=camera, width=width, height=height)

## --------------------------- ##
# Test gym env
# Create the environment
# Register the Gym environment
gym.envs.register('joystick_go2', Go2JoystickGymEnv)
env = gym.make('joystick_go2', render_mode='human')
env = env.unwrapped
obs, _ = env.reset()

# Save obs and action history
obs_hist = []
act_hist = []

for t in range(500):
    act = np.zeros(12)
    obs_hist.append(env.get_full_state())
    act_hist.append(act)
    obs, _, terminated, _, info = env.step(act)

    if terminated:
        break

env.close()

## ------------
# Load Brax Policy
envs.register_environment('joystick_go2', Go2JoystickEnv)

env_name = 'joystick_go2'
# Get Brax policy
env = envs.get_environment(env_name)
eval_env = envs.get_environment(env_name)

make_networks_factory = functools.partial(
    ppo_networks.make_ppo_networks,
        policy_hidden_layer_sizes=(128, 128, 128, 128))
train_fn = functools.partial(
      ppo.train, num_timesteps=100_000_000, num_evals=10,
      reward_scaling=1, episode_length=1000, normalize_observations=True,
      action_repeat=1, unroll_length=20, num_minibatches=32,
      num_updates_per_batch=4, discounting=0.97, learning_rate=3.0e-4,
      entropy_cost=1e-2, num_envs=1, batch_size=256,
      network_factory=make_networks_factory,
      seed=0)

make_inference_fn, params, _= train_fn(environment=env,
                                       eval_env=eval_env,
                                       num_timesteps=0)

model_path = './logs/go2_policy'
params = model.load_params(model_path)
inference_fn = make_inference_fn(params)

import jax

jit_env_reset = jax.jit(eval_env.reset)
jit_env_step = jax.jit(eval_env.step)
jit_inference_fn = jax.jit(inference_fn)

rng = jax.random.PRNGKey(seed=1)

## --------------------------- ##
## Rollout trajectory from Gym environment

# Create the environment
env = gym.make('joystick_go2', render_mode='human')
env = env.unwrapped
obs, _ = env.reset()
env.info["command"] = np.array([0.0, 0.0, 0.0])

# Save obs and action history
obs_hist = []
act_hist = []

for t in range(1000):
    act_rng, rng = jax.random.split(rng)
    act, _ = jit_inference_fn(obs, act_rng)
    obs_hist.append(env.get_full_state())
    act_hist.append(act)
    obs, _, terminated, _, info = env.step(act)

    if t == 100:
        env.info["command"] = np.array([1.0, 0.0, 0.0])

    if terminated:
        break

# Save the history
trajectory = {
    'obs': np.array(obs_hist),
    'act': np.array(act_hist)
}

np.save('trajectory.npy', trajectory)
np.savetxt('obs_hist.csv', np.array(obs_hist), delimiter=',')
np.savetxt('act_hist.csv', np.array(act_hist), delimiter=',')

env.close()

## --------------------------- ##
## Load the trajectory and try to replay

env = gym.make('joystick_go2', render_mode='human')
env = env.unwrapped

trajectory = np.load('trajectory.npy', allow_pickle=True).item()
obs_hist = trajectory['obs']
act_hist = trajectory['act']

obs_init = obs_hist[0]
qpos_init = obs_init[:19]
qvel_init = obs_init[19:]

obs = env.reset_to(qpos_init, qvel_init)

for t in range(999):
    act = act_hist[t]
    obs, _, terminated, _, info = env.step(act)

    if terminated:
        break

env.close()
