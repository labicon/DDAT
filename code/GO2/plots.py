# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 10:32:09 2024

@author: Jean-Baptiste Bouvier

Unitree GO2 specific plotting functions
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_traj(env, Traj, Actions=None, title="", plot_all=False):
    """Plots the trajectory of the Unitree GO2.
    state = [positions, velocities] """
        
    N = Traj.shape[0] # number of steps of the trajectory before terminating
    time = env.dt * np.arange(N)
    
    for state_id in range(env.state_size):
        fig, ax = nice_plot()
        plt.title(title)
        plt.scatter(time, Traj[:, state_id], s=10)
        plt.ylabel(env.state_labels[state_id])
        plt.xlabel("time (s)")
        plt.show()

    if Actions is not None:
        time = env.dt * np.arange(Actions.shape[0])
        for action_id in range(env.action_size):
            fig, ax = nice_plot()
            plt.title(title)
            plt.scatter(time, Actions[:, action_id], s=10)
            plt.ylabel(env.action_labels[action_id])
            plt.xlabel("time (s)")
            plt.show()
   
 
def traj_comparison(env, traj_1, label_1, traj_2, label_2, title="",
                    traj_3=None, label_3=None, traj_4=None, label_4=None,
                    saveas: str = None, plot_z = True, legend_loc='best'):
    """Compares given Unitree GO2 trajectories.
    Optional argument 'saveas' takes the filename to save the plots if desired"""
    
    
    assert len(traj_1.shape) == 2, f"Trajectory 1 must be a 2D array and not {len(traj_1.shape)}D"
    assert len(traj_2.shape) == 2, f"Trajectory 2 must be a 2D array and not {len(traj_2.shape)}D"
    assert traj_1.shape[1] == env.state_size, f"Trajectory 1 must have {env.state_size} components and not {traj_1.shape[1]}"
    assert traj_2.shape[1] == env.state_size, f"Trajectory 2 must have {env.state_size} components and not {traj_2.shape[1]}"
    if traj_3 is not None:
        assert len(traj_3.shape) == 2, f"Trajectory 3 must be a 2D array and not {len(traj_3.shape)}D"
        assert traj_3.shape[1] == env.state_size, f"Trajectory 3 must have {env.state_size} components and not {traj_3.shape[1]}"
    if traj_4 is not None:
        assert len(traj_4.shape) == 2, f"Trajectory 4 must be a 2D array and not {len(traj_4.shape)}D"
        assert traj_4.shape[1] == env.state_size, f"Trajectory 4 must have {env.state_size} components and not {traj_4.shape[1]}"
    
       
    fig, ax = nice_plot()
    if title is not None:
        plt.title(title)
    plt.plot(traj_1[:, 0], traj_1[:, 1], label=label_1, linewidth=3)
    plt.plot(traj_2[:, 0], traj_2[:, 1], label=label_2, linewidth=3)
    if traj_3 is not None:
        plt.plot(traj_3[:, 0], traj_3[:, 1], label=label_3, linewidth=3)
    if traj_4 is not None:
        plt.plot(traj_4[:, 0], traj_4[:, 1], label=label_4, linewidth=3)
    plt.ylabel("y (m)")
    plt.xlabel("x (m)")
    plt.legend(frameon=False, labelspacing=0.3, handletextpad=0.2, handlelength=0.9, loc=legend_loc)
    if saveas is not None:
        plt.savefig(saveas + "_xy_traj.pdf", bbox_inches='tight', format="pdf", dpi=1200)
    plt.show()       

    
    if plot_z:
        
        time_1 = env.dt * np.arange(traj_1.shape[0])
        time_2 = env.dt * np.arange(traj_2.shape[0])
        time_max = max(time_1[-1], time_2[-1])
        
        fig, ax = nice_plot()
        if title is not None:
            plt.title(title)
        plt.plot(time_1, traj_1[:, 2], label=label_1, linewidth=3)
        plt.plot(time_2, traj_2[:, 2], label=label_2, linewidth=3)
        z_max = max(traj_1[:, 2].max(), traj_2[:, 2].max())
        z_min = min(traj_1[:, 2].min(), traj_2[:, 2].min())
        if traj_3 is not None:
            time_3 = env.dt * np.arange(traj_3.shape[0])
            time_max = max(time_max, time_3[-1])
            plt.plot(time_3, traj_3[:, 2], label=label_3, linewidth=3)
            z_max = max(traj_3[:, 2].max(), z_max)
            z_min = min(traj_3[:, 2].min(), z_min)
        if traj_4 is not None:
            time_4 = env.dt * np.arange(traj_4.shape[0])
            time_max = max(time_max, time_4[-1])
            plt.plot(time_4, traj_4[:, 2], label=label_4, linewidth=3)
            z_max = max(traj_4[:, 2].max(), z_max)
            z_min = min(traj_4[:, 2].min(), z_min)
        plt.plot([0., time_max], [env.min_height, env.min_height], color="red", linestyle="dashed", linewidth=1)
        plt.ylabel("Torso height (m)")
        plt.xlabel("time (s)")
        ax.set_ylim([max(env.min_height-0.1, z_min-0.1*abs(z_min)), z_max*1.1])
        plt.legend(frameon=False, labelspacing=0.3, handletextpad=0.2, handlelength=0.9)
        if saveas is not None:
            plt.savefig(saveas + "_height.pdf", bbox_inches='tight', format="pdf", dpi=1200)
        plt.show()


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
