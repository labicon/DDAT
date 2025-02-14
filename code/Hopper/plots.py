# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 10:32:09 2024

@author: Jean-Baptiste Bouvier

Hopper specific plotting functions
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_traj(env, Traj, title=""):
    """Plots the important components of a hopper trajectory."""
        
    assert len(Traj.shape) == 2, "Trajectory must be a 2D array"
    assert Traj.shape[1] == env.state_size, "Trajectory must contain the full state"
    N = Traj.shape[0] # number of steps of the trajectory before terminating
    time = env.dt * np.arange(N)
    
    fig, ax = nice_plot()
    plt.title(title)
    plt.scatter(time, Traj[:, 1], s=10)
    plt.plot([0., time[-1]], [env.min_z, env.min_z], color="red")
    plt.ylabel("Hopper height (m)")
    plt.xlabel("time (s)")
    plt.show()
    
    fig, ax = nice_plot()
    plt.title(title)
    plt.scatter(time, Traj[:, 2]*180/np.pi, s=10)
    min_angle = env.angle_range[0]*180/np.pi
    max_angle = env.angle_range[1]*180/np.pi
    plt.plot([0., time[-1]], [min_angle, min_angle], color="red")
    plt.plot([0., time[-1]], [max_angle, max_angle], color="red")
    plt.ylabel("Top angle (deg)")
    plt.xlabel("time (s)")
    plt.show()
    
    
   
 
def traj_comparison(env, traj_1, label_1, traj_2, label_2, title="",
                    traj_3=None, label_3=None, traj_4=None, label_4=None,
                    plot_height=True, legend_loc='best'):
    """Compares given hopper trajectories.
    Optional argument 'saveas' takes the filename to save the plots if desired"""
    
    assert len(traj_1.shape) == 2, "Trajectory 1 must be a 2D array"
    assert len(traj_2.shape) == 2, "Trajectory 2 must be a 2D array"
    if traj_3 is not None:
        assert len(traj_3.shape) == 2, "Trajectory 3 must be a 2D array"
    if traj_4 is not None:
        assert len(traj_4.shape) == 2, "Trajectory 4 must be a 2D array"
    
    time_1 = env.dt * np.arange(traj_1.shape[0])
    time_2 = env.dt * np.arange(traj_2.shape[0])
    time_max = max(time_1[-1], time_2[-1])
    
    fig, ax = nice_plot()
    if title is not None:
        plt.title(title)
    plt.plot(time_1, traj_1[:, 2]*180/np.pi, label=label_1, linewidth=3)
    plt.plot(time_2, traj_2[:, 2]*180/np.pi, label=label_2, linewidth=3)
    y_max = max(traj_1[:, 2].max(), traj_2[:, 2].max())
    y_min = min(traj_1[:, 2].min(), traj_2[:, 2].min())
    if traj_3 is not None:
        time_3 = env.dt * np.arange(traj_3.shape[0])
        time_max = max(time_max, time_3[-1])
        plt.plot(time_3, traj_3[:, 2]*180/np.pi, label=label_3, linewidth=3)
        y_max = max(traj_3[:, 2].max(), y_max)
        y_min = min(traj_3[:, 2].min(), y_min)
    if traj_4 is not None:
        time_4 = env.dt * np.arange(traj_4.shape[0])
        time_max = max(time_max, time_4[-1])
        plt.plot(time_4, traj_4[:, 2]*180/np.pi, label=label_4, linewidth=3)
        y_max = max(traj_4[:, 2].max(), y_max)
        y_min = min(traj_4[:, 2].min(), y_min)
    min_angle = env.angle_range[0]*180/np.pi
    max_angle = env.angle_range[1]*180/np.pi
    plt.plot([0., time_max], [min_angle, min_angle], color="red", linestyle="dashed", linewidth=1)
    plt.plot([0., time_max], [max_angle, max_angle], color="red", linestyle="dashed", linewidth=1)
    plt.ylabel("Top angle (deg)")
    plt.xlabel("time (s)")
    ax.set_ylim([max(-30, (y_min-0.1*abs(y_min))*180/np.pi), min(30, y_max*180/np.pi*1.1)])
    plt.legend(frameon=False, labelspacing=0.3, handletextpad=0.2, handlelength=0.9, loc=legend_loc)
    plt.show()       

    if plot_height:
        fig, ax = nice_plot()
        if title is not None:
            plt.title(title)
        plt.plot(time_1, traj_1[:, 1], label=label_1, linewidth=3)
        plt.plot(time_2, traj_2[:, 1], label=label_2, linewidth=3)
        y_max = max(traj_1[:, 1].max(), traj_2[:, 1].max())
        y_min = min(traj_1[:, 1].min(), traj_2[:, 1].min())
        if traj_3 is not None:
            plt.plot(time_3, traj_3[:, 1], label=label_3, linewidth=3)
            y_max = max(traj_3[:, 1].max(), y_max)
            y_min = min(traj_3[:, 1].min(), y_min)
        if traj_4 is not None:
            plt.plot(time_4, traj_4[:, 1], label=label_4, linewidth=3)
            y_max = max(traj_4[:, 1].max(), y_max)
            y_min = min(traj_4[:, 1].min(), y_min)
        plt.plot([0., time_max], [env.min_z, env.min_z], color="red", linestyle="dashed", linewidth=1)
        plt.ylabel("Hopper height (m)")
        plt.xlabel("time (s)")
        ax.set_ylim([max(-5, y_min-0.1*abs(y_min)), min(5, y_max*1.1)])
        plt.legend(frameon=False, labelspacing=0.3, handletextpad=0.2, handlelength=0.9)
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
