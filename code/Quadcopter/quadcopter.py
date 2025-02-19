# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 14:09:01 2024

@author: Jean-Baptiste Bouvier

Quadcopter environment modified from John Viljoen's
https://github.com/johnviljoen/231A_project
"""

import torch
import numpy as np

from Quadcopter.plots import plot_traj, traj_comparison

class QuadcopterEnv():
    """
    This environment describes a fully nonlinear quadcopter

    ## Action Space
    | Num | Action                 | Min | Max | Name | Unit |
    | --- | ---------------------- | --- | --- | ---- | ---- |
    | 0   | Torque on first rotor  | -1  |  1  | w0d  |  N   |
    | 1   | Torque on second rotor | -1  |  1  | w1d  |  N   |
    | 2   | Torque on third rotor  | -1  |  1  | w2d  |  N   |
    | 3   | Torque on fourth rotor | -1  |  1  | w3d  |  N   |

    ## Observation Space
    | Num | Observation                            | Min  | Max | Name | Unit  |
    | --- | -------------------------------------- | ---- | --- | ---- | ----- |
    | 0   | x-coordinate of the center of mass     | -Inf | Inf |  x   |  m    |
    | 1   | y-coordinate of the center of mass     | -Inf | Inf |  y   |  m    |
    | 2   | z-coordinate of the center of mass     | -Inf | Inf |  z   |  m    |
    | 3   | w-orientaiont of the body (quaternion) | -Inf | Inf |  q0  |  rad  |
    | 4   | x-orientaiont of the body (quaternion) | -Inf | Inf |  q1  |  rad  |
    | 5   | y-orientaiont of the body (quaternion) | -Inf | Inf |  q2  |  rad  |
    | 6   | z-orientaiont of the body (quaternion) | -Inf | Inf |  q3  |  rad  |
    
    | 7   | x-velocity of the center of mass       | -Inf | Inf |  xd  |  m/s  |
    | 8   | y-velocity of the center of mass       | -Inf | Inf |  yd  |  m/s  |
    | 9   | z-velocity of the center of mass       | -Inf | Inf |  zd  |  m/s  |
    | 10  | x-angular velocity of the body         | -Inf | Inf |  p   | rad/s |
    | 11  | y-angular velocity of the body         | -Inf | Inf |  q   | rad/s |
    | 12  | z-angular velocity of the body         | -Inf | Inf |  r   | rad/s |
    | 13  | angular velocity of the first rotor    | -Inf | Inf |  w0  | rad/s |
    | 14  | angular velocity of the second rotor   | -Inf | Inf |  w1  | rad/s |
    | 15  | angular velocity of the third rotor    | -Inf | Inf |  w2  | rad/s |
    | 16  | angular velocity of the fourth rotor   | -Inf | Inf |  w3  | rad/s |

    
    ## Starting State
    All observations start from hover with a Gaussian noise of magnitude `reset_noise_scale'
    
    ## Episode End
    1. Any of the states goes out of bounds
    2. The Quadcopter collides with one of the cylinder obstacles

    NOTES:
    John integrated the proportional control of the rotors directly into the 
    equations of motion to more accurately reflect the closed loop system
    we will be controlling with a second outer loop. This inner loop is akin
    to the ESC which will be onboard many quadcopters which directly controls
    the rotor speeds to be what is commanded.
    """
    
    
    def __init__(self, reset_noise_scale:float = 1e-2, dt: float = 0.01,
                 cylinder_radii = [0.7, 0.7]):
        
        self.name = "Quadcopter"
        self.state_size = 17
        self.action_size = 4
        self.action_min = np.array([[-1., -1., -1., -1.]])
        self.action_max = np.array([[ 1.,  1.,  1.,  1.]])
        self.position_states = [0,1,2,3,4,5,6]
        self.velocity_states = [7,8,9,10,11,12,13,14,15,16]
        
        ### Obstacles: cylinders along the z-axis
        self.target_position = np.array([7., 0., 0.])
        self.N_cylinders = 2 # 2 cylinders
        self.cylinder_radii = cylinder_radii # radius of the cylinders
        self.cylinder_xc = [2.5, 5.2] # cylinders x center position
        self.cylinder_yc = [0.5, -0.5] # cylinders y center position
        
        
        ### Fundamental quad parameters
        self.g = 9.81 # gravity (m/s^2)
        self.mB = 1.2 # mass (kg)
        self.dxm = 0.16 # arm length (m)
        self.dym =  0.16 # arm length (m)
        self.dzm = 0.01  # arm height (m)
        self.IB = np.array([[0.0123, 0,      0     ],
                            [0,      0.0123, 0     ],
                            [0,      0,      0.0224]])  # Inertial tensor (kg*m^2)
        self.IRzz = 2.7e-5  # rotor moment of inertia (kg*m^2)
        self.Cd = 0.1  # drag coefficient (omnidirectional)
        self.kTh = 1.076e-5  # thrust coeff (N/(rad/s)^2)  (1.18e-7 N/RPM^2)
        self.kTo = 1.632e-7  # torque coeff (Nm/(rad/s)^2)  (1.79e-9 Nm/RPM^2)
        self.minThr = 0.1*4  # Minimum total thrust (N)
        self.maxThr = 9.18*4  # Maximum total thrust (N)
        self.minWmotor = 75  # Minimum motor rotation speed (rad/s)
        self.maxWmotor = 925  # Maximum motor rotation speed (rad/s)
        self.tau = 0.015  # Value for second order system for Motor dynamics
        self.kp = 1.0  # Value for second order system for Motor dynamics
        self.damp = 1.0  # Value for second order system for Motor dynamics
        self.usePrecession = True  # model precession or not
        self.w_hover = 522.9847140714692 # hardcoded hover rotor speed (rad/s)
        
        ### post init useful parameters for quad
        self.B0 = np.array([[self.kTh, self.kTh, self.kTh, self.kTh],
                            [self.dym * self.kTh, - self.dym * self.kTh, -self.dym * self.kTh,  self.dym * self.kTh],
                            [self.dxm * self.kTh,  self.dxm * self.kTh, -self.dxm * self.kTh, -self.dxm * self.kTh],
                            [-self.kTo, self.kTo, - self.kTo, self.kTo]]) # actuation matrix

        self.low_bound = np.array([-100, -100, -100, *[-np.inf]*4, *[-100]*3, *[-100]*3, *[self.minWmotor]*4])
                               # xyz       q0123         xdydzd    pdqdrd    w0123
        
        self.high_bound = np.array([100, 100, 100, *[np.inf]*4, *[100]*3, *[100]*3, *[self.maxWmotor]*4])
                                # xyz      q0123        xdydzd   pdqdrd   w0123
        
        self.dt = dt # time step
        self.reset_noise = reset_noise_scale
        self.hover_state = np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, *[522.9847140714692]*4])
        
        
    def reset(self):
        self.state = self.hover_state.copy() + self.reset_noise*np.random.randn(self.state_size)
        return self.state.copy()
        
    
    def reset_to(self, state):
        # assert all(state > self.x_lb) and all(state < self.x_ub), "state is out of bounds x_lb, x_ub"
        self.state = state.copy()
        return state


    def is_in_obstacle(self, x, y):
        """Checks whether (x,y) position is inside an obstacle"""
        for i in range(self.N_cylinders):
            inside = self.cylinder_radii[i]**2 >= (x - self.cylinder_xc[i])**2 + (y - self.cylinder_yc[i])**2 
            if inside:
                return True
        return False
    

    def step(self, u):
        
        q0 =    self.state[3]
        q1 =    self.state[4]
        q2 =    self.state[5]
        q3 =    self.state[6]
        xdot =  self.state[7]
        ydot =  self.state[8]
        zdot =  self.state[9]
        p =     self.state[10]
        q =     self.state[11]
        r =     self.state[12]
        wM1 =   self.state[13]
        wM2 =   self.state[14]
        wM3 =   self.state[15]
        wM4 =   self.state[16]
    
        # instantaneous thrusts and torques generated by the current w0...w3
        wMotor = np.stack([wM1, wM2, wM3, wM4])
        wMotor = np.clip(wMotor, self.minWmotor, self.maxWmotor) # this clip shouldn't occur within the dynamics
        th = self.kTh * wMotor ** 2 # thrust
        to = self.kTo * wMotor ** 2 # torque
    
        # state derivates (from sympy.mechanics derivation)
        xd = np.stack(
            [
                xdot,
                ydot,
                zdot,
                -0.5 * p * q1 - 0.5 * q * q2 - 0.5 * q3 * r,
                0.5 * p * q0 - 0.5 * q * q3 + 0.5 * q2 * r,
                0.5 * p * q3 + 0.5 * q * q0 - 0.5 * q1 * r,
                -0.5 * p * q2 + 0.5 * q * q1 + 0.5 * q0 * r,
                (self.Cd * np.sign(-xdot) * xdot**2
                    - 2 * (q0 * q2 + q1 * q3) * (th[0] + th[1] + th[2] + th[3])
                )
                /  self.mB, # xdd
                (
                     self.Cd * np.sign(-ydot) * ydot**2
                    + 2 * (q0 * q1 - q2 * q3) * (th[0] + th[1] + th[2] + th[3])
                )
                /  self.mB, # ydd
                (
                    - self.Cd * np.sign(zdot) * zdot**2
                    - (th[0] + th[1] + th[2] + th[3])
                    * (q0**2 - q1**2 - q2**2 + q3**2)
                    + self.g *  self.mB
                )
                /  self.mB, # zdd (the - in front turns increased height to be positive - SWU)
                (
                    ( self.IB[1,1] -  self.IB[2,2]) * q * r
                    -  self.usePrecession *  self.IRzz * (wM1 - wM2 + wM3 - wM4) * q
                    + (th[0] - th[1] - th[2] + th[3]) *  self.dym
                )
                /  self.IB[0,0], # pd
                (
                    ( self.IB[2,2] -  self.IB[0,0]) * p * r
                    +  self.usePrecession *  self.IRzz * (wM1 - wM2 + wM3 - wM4) * p
                    + (th[0] + th[1] - th[2] - th[3]) *  self.dxm
                )
                /  self.IB[1,1], #qd
                (( self.IB[0,0] -  self.IB[1,1]) * p * q - to[0] + to[1] - to[2] + to[3]) /  self.IB[2,2], # rd
                u[0]/self.IRzz, u[1]/self.IRzz, u[2]/self.IRzz, u[3]/self.IRzz # w0d ... w3d
            ]
        )
    
        self.state += xd * self.dt # one time step forward
        # Clip the rotor speeds within limits
        self.state[13:17] = np.clip(self.state[13:17], self.low_bound[13:17], self.high_bound[13:17])
        
        out_of_bound = any(self.state < self.low_bound) or any(self.state > self.high_bound) # out of bound state
        collided = self.is_in_obstacle(self.state[0], self.state[1])
        distance_to_target = np.linalg.norm(self.state[:3] - self.target_position)
        reward = 1 - out_of_bound - collided - distance_to_target
        terminated = out_of_bound or collided
    
        return self.state.copy(), reward, terminated, False, None


    # Function called by the projectors
    def pos_from_vel(self, S_t, vel_t_dt):
        """
        Calculates the next state's position using explicit Euler integrator
        and quaternion formula, does NOT need to know the dynamics.
        
        Arguments:
            - S_t : current state torch.tensor (17,)
            - vel_t_dt : (unused) next state's velocity torch.tensor (10,)
        Returns:
            - x_t_dt : next state's position torch.tensor (7,)
        """
        x_t_dt = S_t[:7].clone() # copy the current position
        q0 =    S_t[3]
        q1 =    S_t[4]
        q2 =    S_t[5]
        q3 =    S_t[6]
        xdot =  S_t[7]
        ydot =  S_t[8]
        zdot =  S_t[9]
        p =     S_t[10]
        q =     S_t[11]
        r =     S_t[12]
        
        x_t_dt += self.dt*torch.FloatTensor([xdot, ydot, zdot,
                                   -0.5*p*q1 - 0.5*q*q2 - 0.5*q3*r,
                                    0.5*p*q0 - 0.5*q*q3 + 0.5*q2*r,
                                    0.5*p*q3 + 0.5*q*q0 - 0.5*q1*r,
                                   -0.5*p*q2 + 0.5*q*q1 + 0.5*q0*r]).to(S_t.device)

        return x_t_dt

    # Plotting functions
    def plot_traj(self, Traj, title:str = ""):
        """Plots the xy trajectory of the Quadcopter."""
        plot_traj(self, Traj, title)

    
    def traj_comparison(self, traj_1, label_1, traj_2, label_2, title:str = "",
                        traj_3=None, label_3=None, traj_4=None, label_4=None,
                        legend_loc='best'):
        """
        Compares up to 4 xy trajectories of the Quadcopter
        Arguments:
            - traj_1 : first trajectory of shape (H, 17)
            - label_1 : corresponding label to display
            - traj_2 : first trajectory of shape (H, 17)
            - label_2 : corresponding label to display
            - title: optional title of the plot
            - traj_3 : optional third trajectory of shape (H, 17)
            - label_3 : optional corresponding label to display
            - traj_4 : optional fourth trajectory of shape (H, 17)
            - label_4 : optional corresponding label to display
            - legend_loc : optional location of the legend
        """
        traj_comparison(self, traj_1, label_1, traj_2, label_2, title,
                        traj_3, label_3, traj_4, label_4, legend_loc)

         











