# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 10:32:09 2024

@author: Jean-Baptiste Bouvier

Walker specific plotting functions
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_traj(env, Traj:np.ndarray, title:str = ""):
    """Plots the important components of a Walker trajectory."""
        
    assert len(Traj.shape) == 2, "Trajectory must be a 2D array"
    assert Traj.shape[1] == env.state_size, "Trajectory must contain the full state"
    N = Traj.shape[0] # number of steps of the trajectory before terminating
    time = np.arange(N)
    
    height_id = 1
    angle_id = 2
    
    fig, ax = nice_plot()
    plt.title(title)
    plt.plot(time, Traj[:, angle_id]*180/np.pi, linewidth=3)
    y_max = Traj[:, angle_id].max()
    y_min = Traj[:, angle_id].min()
    min_angle = env.min_angle*180/np.pi
    max_angle = env.max_angle*180/np.pi
    plt.plot([0., time[-1] ], [min_angle, min_angle], color="red", linestyle="dashed", linewidth=1)
    plt.plot([0., time[-1] ], [max_angle, max_angle], color="red", linestyle="dashed", linewidth=1)
    plt.ylabel("Torso angle (deg)")
    plt.xlabel("timesteps")
    ax.set_ylim([max(min_angle-10, (y_min-0.1*abs(y_min))*180/np.pi), min(max_angle+10, y_max*180/np.pi*1.1)])
    plt.show()       

    fig, ax = nice_plot()
    plt.title(title)
    plt.plot(time, Traj[:, height_id], linewidth=3)
    y_max = Traj[:, height_id].max()
    y_min = Traj[:, height_id].min()
    plt.plot([0., time[-1] ], [env.min_height, env.min_height], color="red", linestyle="dashed", linewidth=1)
    plt.plot([0., time[-1] ], [env.max_height, env.max_height], color="red", linestyle="dashed", linewidth=1)
    plt.ylabel("Torso height (m)")
    plt.xlabel("timesteps")
    ax.set_ylim([max(env.min_height-0.1, y_min-0.1*abs(y_min)), min(env.max_height+0.1, y_max*1.1)])
    plt.show()
        

    
   
 
def traj_comparison(env, traj_1, label_1, traj_2, label_2, title:str = "",
                    traj_3=None, label_3=None, traj_4=None, label_4=None,
                    plot_height=True, legend_loc='best'):
    """Compares given Walker trajectories."""
    
    assert len(traj_1.shape) == 2, "Trajectory 1 must be a 2D array"
    assert len(traj_2.shape) == 2, "Trajectory 2 must be a 2D array"
    if traj_3 is not None:
        assert len(traj_3.shape) == 2, "Trajectory 3 must be a 2D array"
    if traj_4 is not None:
        assert len(traj_4.shape) == 2, "Trajectory 4 must be a 2D array"
    
    time_1 = np.arange(traj_1.shape[0])
    time_2 = np.arange(traj_2.shape[0])
    time_max = max(time_1[-1], time_2[-1])
    
    height_id = 1
    angle_id = 2    
    
    fig, ax = nice_plot()
    if title is not None:
        plt.title(title)
    plt.plot(time_1, traj_1[:, angle_id]*180/np.pi, label=label_1, linewidth=3)
    plt.plot(time_2, traj_2[:, angle_id]*180/np.pi, label=label_2, linewidth=3)
    y_max = max(traj_1[:, angle_id].max(), traj_2[:, angle_id].max())
    y_min = min(traj_1[:, angle_id].min(), traj_2[:, angle_id].min())
    if traj_3 is not None:
        time_3 = np.arange(traj_3.shape[0])
        time_max = max(time_max, time_3[-1])
        plt.plot(time_3, traj_3[:, angle_id]*180/np.pi, label=label_3, linewidth=3)
        y_max = max(traj_3[:, angle_id].max(), y_max)
        y_min = min(traj_3[:, angle_id].min(), y_min)
    if traj_4 is not None:
        time_4 = np.arange(traj_4.shape[0])
        time_max = max(time_max, time_4[-1])
        plt.plot(time_4, traj_4[:, angle_id]*180/np.pi, label=label_4, linewidth=3)
        y_max = max(traj_4[:, angle_id].max(), y_max)
        y_min = min(traj_4[:, angle_id].min(), y_min)
    min_angle = env.min_angle*180/np.pi
    max_angle = env.max_angle*180/np.pi
    plt.plot([0., time_max], [min_angle, min_angle], color="red", linestyle="dashed", linewidth=1)
    plt.plot([0., time_max], [max_angle, max_angle], color="red", linestyle="dashed", linewidth=1)
    plt.ylabel("Torso angle (deg)")
    plt.xlabel("timesteps")
    ax.set_ylim([max(min_angle-10, (y_min-0.1*abs(y_min))*180/np.pi), min(max_angle+10, y_max*180/np.pi*1.1)])
    plt.legend(frameon=False, labelspacing=0.3, handletextpad=0.2, handlelength=0.9, loc=legend_loc)
    plt.show()       

    
    if plot_height:
        fig, ax = nice_plot()
        if title is not None:
            plt.title(title)
        plt.plot(time_1, traj_1[:, height_id], label=label_1, linewidth=3)
        plt.plot(time_2, traj_2[:, height_id], label=label_2, linewidth=3)
        y_max = max(traj_1[:, height_id].max(), traj_2[:, height_id].max())
        y_min = min(traj_1[:, height_id].min(), traj_2[:, height_id].min())
        if traj_3 is not None:
            plt.plot(time_3, traj_3[:, height_id], label=label_3, linewidth=3)
            y_max = max(traj_3[:, height_id].max(), y_max)
            y_min = min(traj_3[:, height_id].min(), y_min)
        if traj_4 is not None:
            plt.plot(time_4, traj_4[:, height_id], label=label_4, linewidth=3)
            y_max = max(traj_4[:, height_id].max(), y_max)
            y_min = min(traj_4[:, height_id].min(), y_min)
        plt.plot([0., time_max], [env.min_height, env.min_height], color="red", linestyle="dashed", linewidth=1)
        plt.plot([0., time_max], [env.max_height, env.max_height], color="red", linestyle="dashed", linewidth=1)
        plt.ylabel("Torso height (m)")
        plt.xlabel("timesteps")
        ax.set_ylim([max(env.min_height-0.1, y_min-0.1*abs(y_min)), min(env.max_height+0.1, y_max*1.1)])
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
