# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 14:09:01 2024

@author: Jean-Baptiste

Quadcopter environment copied from John Viljoen
"""
import casadi as ca
import numpy as np


class QuadcopterEnv():
    """
    state:
        x = {x,y,z,q0,q1,q2,q3,xd,yd,zd,p ,q ,r ,w0,w1,w2,w3}
             0 1 2 3  4  5  6  7  8  9  10 11 12 13 14 15 16
    
    control:
        u = {w0d,w1d,w2d,w3d}
             0   1   2   3

    NOTES:
    
    John integrated the proportional control of the rotors directly into the 
    equations of motion to more accurately reflect the closed loop system
    we will be controlling with a second outer loop. This inner loop is akin
    to the ESC which will be onboard many quadcopters which directly controls
    the rotor speeds to be what is commanded.
    """
    
    
    def __init__(self, reset_noise=1e-2, dt=0.01, cylinder_radii = [0.9, 0.9]):
        """Reset normal noise on the whole state"""
        
        self.name = "quad"
        self.state_size = 17
        self.action_size = 4
        self.action_min = np.array([[-1., -1., -1., -1.]])
        self.action_max = np.array([[ 1.,  1.,  1.,  1.]])
        self.position_states = [0,1,2,3,4,5,6]
        self.velocity_states = [7,8,9,10,11,12,13,14,15,16]
        self.actuated_states = [7,8,9,10,11,12,13,14,15,16]
        
        ### Obstacles: cylinders along the z-axis
        self.target_position = np.array([7., 0., 0.])
        self.N_cylinders = 2 # 2 cylinders
        self.cylinder_radii = cylinder_radii # radius of the cylinders
        self.cylinder_xc = [2.5, 5.2] # cylinders x center position
        self.cylinder_yc = [0.5, -0.5] # cylinders y center position
        
        
        ### fundamental quad parameters
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
        self.reset_noise = reset_noise
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
        """Numpy"""
        
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


    def casadi_step(self, state: ca.MX, cmd: ca.MX):
         """Returns next state of the Casadi state"""

         # Import params to numpy for CasADI
         # ---------------------------
         IB = self.IB
         IBxx = IB[0, 0]
         IByy = IB[1, 1]
         IBzz = IB[2, 2]

         # Unpack state tensor for readability
         # ---------------------------
         q0 =    state[3]
         q1 =    state[4]
         q2 =    state[5]
         q3 =    state[6]
         xdot =  state[7]
         ydot =  state[8]
         zdot =  state[9]
         p =     state[10]
         q =     state[11]
         r =     state[12]
         wM1 =   state[13]
         wM2 =   state[14]
         wM3 =   state[15]
         wM4 =   state[16]

         # a tiny bit more readable
         ThrM1 = self.kTh * wM1 ** 2
         ThrM2 = self.kTh * wM2 ** 2
         ThrM3 = self.kTh * wM3 ** 2
         ThrM4 = self.kTh * wM4 ** 2
         TorM1 = self.kTo * wM1 ** 2
         TorM2 = self.kTo * wM2 ** 2
         TorM3 = self.kTo * wM3 ** 2
         TorM4 = self.kTo * wM4 ** 2

         # Wind Model (zero in expectation)
         # ---------------------------
         velW, qW1, qW2 = 0, 0, 0

         # State Derivatives (from PyDy) This is already the analytically solved vector of MM*x = RHS
         # ---------------------------
         DynamicsDot = ca.vertcat(
                 xdot,
                 ydot,
                 zdot,
                 -0.5 * p * q1 - 0.5 * q * q2 - 0.5 * q3 * r,
                 0.5 * p * q0 - 0.5 * q * q3 + 0.5 * q2 * r,
                 0.5 * p * q3 + 0.5 * q * q0 - 0.5 * q1 * r,
                 -0.5 * p * q2 + 0.5 * q * q1 + 0.5 * q0 * r,
                 (
                     self.Cd
                     * ca.sign(velW * ca.cos(qW1) * ca.cos(qW2) - xdot)
                     * (velW * ca.cos(qW1) * ca.cos(qW2) - xdot) ** 2
                     - 2 * (q0 * q2 + q1 * q3) * (ThrM1 + ThrM2 + ThrM3 + ThrM4)
                 )
                 / self.mB,
                 (
                     self.Cd
                     * ca.sign(velW * ca.sin(qW1) * ca.cos(qW2) - ydot)
                     * (velW * ca.sin(qW1) * ca.cos(qW2) - ydot) ** 2
                     + 2 * (q0 * q1 - q2 * q3) * (ThrM1 + ThrM2 + ThrM3 + ThrM4)
                 )
                 / self.mB,
                 (
                     -self.Cd * ca.sign(velW * ca.sin(qW2) + zdot) * (velW * ca.sin(qW2) + zdot) ** 2
                     - (ThrM1 + ThrM2 + ThrM3 + ThrM4)
                     * (q0 ** 2 - q1 ** 2 - q2 ** 2 + q3 ** 2)
                     + self.g * self.mB
                 )
                 / self.mB,
                 (
                     (IByy - IBzz) * q * r
                     - self.usePrecession * self.IRzz * (wM1 - wM2 + wM3 - wM4) * q
                     + (ThrM1 - ThrM2 - ThrM3 + ThrM4) * self.dym
                 )
                 / IBxx,  # uP activates or deactivates the use of gyroscopic precession.
                 (
                     (IBzz - IBxx) * p * r
                     + self.usePrecession * self.IRzz * (wM1 - wM2 + wM3 - wM4) * p
                     + (ThrM1 + ThrM2 - ThrM3 - ThrM4) * self.dxm
                 )
                 / IByy,  # Set uP to False if rotor inertia is not known (gyro precession has negigeable effect on drone dynamics)
                 ((IBxx - IByy) * p * q - TorM1 + TorM2 - TorM3 + TorM4) / IBzz,
                 cmd[0]/self.IRzz, cmd[1]/self.IRzz, cmd[2]/self.IRzz, cmd[3]/self.IRzz
         )

         if DynamicsDot.shape[1] == 17:
             print('fin')

         # State Derivative Vector
         next_state = state + DynamicsDot * self.dt
         return next_state


#%% Figures and animations

import matplotlib.pyplot as plt
import matplotlib.patches as pat
from scipy.spatial.transform import Rotation as R

def quaternion_to_rotation_matrix(q):
    """   Convert a quaternion into a rotation matrix.   """
    q0, q1, q2, q3 = q
    R = np.array([
        [1 - 2*(q2**2 + q3**2),     2*(q1*q2 - q0*q3),     2*(q1*q3 + q0*q2)],
        [    2*(q1*q2 + q0*q3), 1 - 2*(q1**2 + q3**2),     2*(q2*q3 - q0*q1)],
        [    2*(q1*q3 - q0*q2),     2*(q2*q3 + q0*q1), 1 - 2*(q1**2 + q2**2)]
    ])
    return R


def plot_traj(env, Traj, title=None):
    """Plot the important components of a given quadcopter trajectory"""
    
    assert len(Traj.shape) == 2, "Trajectory must be a 2D array"
    assert Traj.shape[1] == env.state_size, "Trajectory must contain the full state"
    T = Traj.shape[0]
    time = env.dt * np.arange(T)
    
    fig, ax = nice_plot()
    if title is not None:
        plt.title(title)
    plt.plot(time, Traj[:,0], label="x", linewidth=3)
    plt.plot(time, Traj[:,1], label="y", linewidth=3)
    plt.plot(time, Traj[:,2], label="z", linewidth=3)
    plt.xlabel("t (s)")
    plt.ylabel("orientation (deg)")
    plt.legend(frameon=False, labelspacing=0.3, handletextpad=0.2, handlelength=0.9)
    plt.show()
    
    fig, ax = nice_plot()
    if title is not None:
        plt.title(title)
    for rotor_id in range(1,5):
        plt.plot(time, Traj[:, -rotor_id], label=f"rotor {rotor_id}", linewidth=3)
    plt.xlabel("t (s)")
    plt.legend(frameon=False, labelspacing=0.3, handletextpad=0.2, handlelength=0.9)
    plt.show()
    
    rot = np.zeros((T, 3))
    quat_scalar_last = np.concatenate((Traj[:, 4:7],Traj[:, 3].reshape((T,1))), axis=1)
   
    for i in range(T):
        rot[i] = R.from_quat(quat_scalar_last[i]).as_euler('xyz', degrees=True)
        if i > 1:
            for angle in range(3):
                while rot[i, angle] > rot[i-1, angle] + 180:
                    rot[i, angle] -= 90*2
                while rot[i, angle] < rot[i-1, angle] - 180:
                    rot[i, angle] += 90*2
    
    fig, ax = nice_plot()
    if title is not None:
        plt.title(title)        
    plt.plot(time, rot[:, 0], label="roll x", linewidth=3)
    plt.plot(time, rot[:, 1], label="pitch y", linewidth=3)
    plt.plot(time, rot[:, 2], label="yaw z", linewidth=3)
    plt.xlabel("t (s)")
    plt.yticks([-180, -90, 0, 90, 180])
    plt.legend(frameon=False, labelspacing=0.3, handletextpad=0.2, handlelength=0.9)
    plt.show()


def plot_xy_path(env, Traj):

    radii = env.cylinder_radii # [0.9, 0.9] # radius
    xc = env.cylinder_xc # [2.5, 5.2] # cylinder x center position
    yc = env.cylinder_yc # [0.5, -0.5] # cylinder y center position

    fig, ax = nice_plot()
    plt.axis("equal")
    for i in range(len(radii)):
        cylinder = pat.Circle(xy=[xc[i], yc[i]], radius=radii[i], color="red")
        ax.add_patch(cylinder)
    plt.plot(Traj[:,0], Traj[:,1], linewidth=3)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()


def traj_comparison(env, traj_1, label_1, traj_2, label_2, title="",
                    traj_3=None, label_3=None, traj_4=None, label_4=None,
                    saveas: str = None, legend_loc='best'):
    
    """Compares given quadcopter trajectories.
    Optional argument 'saveas' takes the filename to save the plots if desired"""
    
    assert len(traj_1.shape) == 2, "Trajectory 1 must be a 2D array"
    assert len(traj_2.shape) == 2, "Trajectory 2 must be a 2D array"
    if traj_3 is not None:
        assert len(traj_3.shape) == 2, "Trajectory 3 must be a 2D array"
    if traj_4 is not None:
        assert len(traj_4.shape) == 2, "Trajectory 4 must be a 2D array"
    
    radii = env.cylinder_radii # [0.9, 0.9] # radius
    xc = env.cylinder_xc # [2.5, 5.2] # cylinder x center position
    yc = env.cylinder_yc # [0.5, -0.5] # cylinder y center position

    fig, ax = nice_plot()
    if title is not None:
        plt.title(title)
    plt.axis("equal")
    for i in range(len(radii)):
        cylinder = pat.Circle(xy=[xc[i], yc[i]], radius=radii[i], color="red")
        ax.add_patch(cylinder)
    plt.plot(traj_1[:,0], traj_1[:,1], label=label_1, linewidth=3)
    plt.plot(traj_2[:,0], traj_2[:,1], label=label_2, linewidth=3)
    if traj_3 is not None:
        plt.plot(traj_3[:,0], traj_3[:,1], label=label_3, linewidth=3)
    if traj_4 is not None:
        plt.plot(traj_4[:,0], traj_4[:,1], label=label_4, linewidth=3)
    plt.legend(frameon=False, labelspacing=0.3, handletextpad=0.2, handlelength=0.9, loc=legend_loc)
    plt.xlabel("x")
    plt.ylabel("y")
    if saveas is not None:
        plt.savefig(saveas + "_trajs.svg", bbox_inches='tight', format="svg", dpi=1200)
    plt.show()
    
    # TODO: make it compatible with QuadEnv
    # time_1 = env.dt * np.arange(traj_1.shape[0])
    # time_2 = env.dt * np.arange(traj_2.shape[0])
    # time_max = max(time_1[-1], time_2[-1])
    
    # fig, ax = nice_plot()
    # if title is not None:
    #     plt.title(title)
    # plt.plot(time_1, traj_1[:, 2]*180/np.pi, label=label_1, linewidth=3)
    # plt.plot(time_2, traj_2[:, 2]*180/np.pi, label=label_2, linewidth=3)
    # y_max = max(traj_1[:, 2].max(), traj_2[:, 2].max())
    # y_min = min(traj_1[:, 2].min(), traj_2[:, 2].min())
    # if traj_3 is not None:
    #     time_3 = env.dt * np.arange(traj_3.shape[0])
    #     time_max = max(time_max, time_3[-1])
    #     plt.plot(time_3, traj_3[:, 2]*180/np.pi, label=label_3, linewidth=3)
    #     y_max = max(traj_3[:, 2].max(), y_max)
    #     y_min = min(traj_3[:, 2].min(), y_min)
    # if traj_4 is not None:
    #     time_4 = env.dt * np.arange(traj_4.shape[0])
    #     time_max = max(time_max, time_4[-1])
    #     plt.plot(time_4, traj_4[:, 2]*180/np.pi, label=label_4, linewidth=3)
    #     y_max = max(traj_4[:, 2].max(), y_max)
    #     y_min = min(traj_4[:, 2].min(), y_min)
    
    # min_angle = env.min_angle*180/np.pi
    # max_angle = env.max_angle*180/np.pi
    # plt.plot([0., time_max], [min_angle, min_angle], color="red", linestyle="dashed", linewidth=1)
    # plt.plot([0., time_max], [max_angle, max_angle], color="red", linestyle="dashed", linewidth=1)
    # ax.set_ylim([max(min_angle-10, (y_min-0.1*abs(y_min))*180/np.pi), min(max_angle+10, y_max*180/np.pi*1.1)])
    
    # plt.ylabel("Front tip angle (deg)")
    # plt.xlabel("time (s)")
    # # ax.set_ylim([(y_min-0.1*abs(y_min))*180/np.pi, y_max*180/np.pi*1.1])
    # plt.legend(frameon=False, labelspacing=0.3, handletextpad=0.2, handlelength=0.9, loc=legend_loc)
    # if saveas is not None:
    #     plt.savefig(saveas + "_angle.svg", bbox_inches='tight', format="svg", dpi=1200)
    # plt.show()




def nice_plot():
    """Makes the plot nice"""
    fig = plt.gcf()
    ax = fig.gca()
    plt.rcParams.update({'font.size': 16})
    plt.rcParams['font.sans-serif'] = ['Palatino Linotype']
    ax.spines['bottom'].set_color('w')
    ax.spines['top'].set_color('w') 
    ax.spines['right'].set_color('w')
    ax.spines['left'].set_color('w')
    
    return fig, ax











from matplotlib import animation

class Animator:
    def __init__(self, env, x, r,
            max_frames = 500, # 500 works for my 16GB RAM machine
            dt = 0.1, # I think timestep of the frames of the gif
            save_path='figures/gifs/test.gif',
            title='test'
        ):

        t = np.arange(0, env.dt, x.shape[0]*env.dt)
        num_steps = len(t)
        max_frames = max_frames
        
        def compute_render_interval(num_steps, max_frames):
            render_interval = 1  # Start with rendering every frame.
            # While the number of frames using the current render interval exceeds max_frames, double the render interval.
            while num_steps / render_interval > max_frames:
                render_interval *= 2
            return render_interval
        
        render_interval = compute_render_interval(num_steps, max_frames)

        self.save_path = save_path
        self.dt = dt
        self.rp = None
        self.env = env
        self.x = x[::render_interval,:]
        self.t = t[::render_interval]
        self.r = r[::render_interval,:]

        # Instantiate the figure
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(projection='3d')

        # Draw the reference x y z (doesnt matter if static or dynamic)
        self.ax.plot(self.r[:,0], self.r[:,1], self.r[:,2], ':', lw=1.3, color='green')
        # these are the lines that draw the quadcopter
        self.line1, = self.ax.plot([], [], [], lw=2, color='red')
        self.line2, = self.ax.plot([], [], [], lw=2, color='blue')
        self.line3, = self.ax.plot([], [], [], '--', lw=1, color='blue')

        # Setting the limits correctly
        extra_each_side = 0.5

        x_min = min(np.min(self.x[:,0]), np.min(self.r[:,0]))
        y_min = min(np.min(self.x[:,1]), np.min(self.r[:,1]))
        z_min = min(np.min(self.x[:,2]), np.min(self.r[:,2]))
        x_max = max(np.max(self.x[:,0]), np.max(self.r[:,0]))
        y_max = max(np.max(self.x[:,1]), np.max(self.r[:,1]))
        z_max = max(np.max(self.x[:,2]), np.max(self.r[:,2]))

        max_range = 0.5*np.array([x_max-x_min, y_max-y_min, z_max-z_min]).max() + extra_each_side
        mid_x = 0.5*(x_max+x_min)
        mid_y = 0.5*(y_max+y_min)
        mid_z = 0.5*(z_max+z_min)
        
        self.ax.set_xlim3d([mid_x-max_range, mid_x+max_range])
        self.ax.set_xlabel('X')
        self.ax.set_ylim3d([mid_y-max_range, mid_y+max_range]) # NEU?

        self.ax.set_ylabel('Y')
        self.ax.set_zlim3d([mid_z-max_range, mid_z+max_range])
        self.ax.set_zlabel('Altitude')

        # add a dynamic time to the plot
        self.title_time = self.ax.text2D(0.05, 0.95, "", transform=self.ax.transAxes)

        # add the title itself
        title = self.ax.text2D(0.95, 0.95, title, transform=self.ax.transAxes, horizontalalignment='right')

    def update_lines(self, k):
        
        # current time
        tk = self.t[k]

        # history of x from 0 to current timestep k
        x_0k = self.x[0:k+1]
        xk = self.x[k]

        q_0k = np.array([
            self.x[0:k+1,3],
            self.x[0:k+1,4],
            self.x[0:k+1,5],
            self.x[0:k+1,6]
        ])
        qk = q_0k[:,-1]

        R = quaternion_to_rotation_matrix(qk)

        motor_points = R @ np.array([
            [self.env.dxm, -self.env.dym, self.env.dzm], 
            [0, 0, 0], 
            [self.env.dxm, self.env.dym, self.env.dzm], 
            [-self.env.dxm, self.env.dym, self.env.dzm], 
            [0, 0, 0], 
            [-self.env.dxm, -self.env.dym, self.env.dzm]
        ]).T
        motor_points[0:3,:] += xk[0:3][:,None]

        # plot the current point of the reference along the reference
        if self.rp is not None:
            self.rp.remove()
        self.rp = self.ax.scatter(self.r[k,0], self.r[k,1], self.r[k,2], color='magenta', alpha=1, marker = 'o', s = 25)

        self.line1.set_data(motor_points[0,0:3], motor_points[1,0:3])
        self.line1.set_3d_properties(motor_points[2,0:3])
        self.line2.set_data(motor_points[0,3:6], motor_points[1,3:6])
        self.line2.set_3d_properties(motor_points[2,3:6])
        self.line3.set_data(x_0k[:,0], x_0k[:,1])
        self.line3.set_3d_properties(x_0k[:,2])
        self.title_time.set_text(u"Time = {:.2f} s ".format(tk))

    def ini_plot(self):

        self.line1.set_data(np.empty([1]), np.empty([1]))
        self.line1.set_3d_properties(np.empty([1]))
        self.line2.set_data(np.empty([1]), np.empty([1]))
        self.line2.set_3d_properties(np.empty([1]))
        self.line3.set_data(np.empty([1]), np.empty([1]))
        self.line3.set_3d_properties(np.empty([1]))

        return self.line1, self.line2, self.line3   

    def animate(self):
        line_ani = animation.FuncAnimation(
            self.fig, 
            self.update_lines, 
            init_func=self.ini_plot, 
            frames=len(self.t)-1, 
            interval=(self.dt*10), 
            blit=False)

        line_ani.save(self.save_path, dpi=120, fps=25)

        return line_ani
    



