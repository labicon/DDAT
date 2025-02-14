# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 10:32:09 2024

@author: Jean-Baptiste Bouvier

Quadcopter specific plotting functions
"""

import numpy as np
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


def plot_traj(env, Traj, title=""):
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
    plt.ylabel("position (m)")
    plt.legend(frameon=False, labelspacing=0.3, handletextpad=0.2, handlelength=0.9)
    plt.show()
    
    fig, ax = nice_plot()
    if title is not None:
        plt.title(title)
    for rotor_id in range(1,5):
        plt.plot(time, Traj[:, -rotor_id], label=f"rotor {rotor_id}", linewidth=3)
    plt.xlabel("t (s)")
    plt.ylabel("angular velocity (rad/s)")
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
    plt.ylabel("orientation (deg)")
    plt.yticks([-180, -90, 0, 90, 180])
    plt.legend(frameon=False, labelspacing=0.3, handletextpad=0.2, handlelength=0.9)
    plt.show()

    radii = env.cylinder_radii # [0.9, 0.9] # radius
    xc = env.cylinder_xc # [2.5, 5.2] # cylinder x center position
    yc = env.cylinder_yc # [0.5, -0.5] # cylinder y center position

    fig, ax = nice_plot()
    plt.axis("equal")
    for i in range(len(radii)):
        cylinder = pat.Circle(xy=[xc[i], yc[i]], radius=radii[i], color="red")
        ax.add_patch(cylinder)
    plt.plot(Traj[:,0], Traj[:,1], linewidth=3)
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.show()


def traj_comparison(env, traj_1, label_1, traj_2, label_2, title="",
                    traj_3=None, label_3=None, traj_4=None, label_4=None,
                    legend_loc='best'):
    
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
    plt.show()
    
    


def nice_plot():
    """Makes the plot nice"""
    fig = plt.gcf()
    ax = fig.gca()
    plt.rcParams.update({'font.size': 16})
    ax.spines['bottom'].set_color('w')
    ax.spines['top'].set_color('w') 
    ax.spines['right'].set_color('w')
    ax.spines['left'].set_color('w')
    
    return fig, ax



