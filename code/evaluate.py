# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 15:46:22 2025

@author: Jean-Baptiste Bouvier

Main script to load and evaluate a pre-trained Diffusion Transfomer
"""

import torch
import numpy as np


from utils.loaders import make_env, load_proj
from utils.utils import set_seed, open_loop
from utils.inverse_dynamics import InverseDynamics
from DiT.ODE import ODE
from DiT.planner import Planner



#%% Hyperparameters

modality = "S" # whether diffusion predicts only states "S", states and actions "SA", or only actions "A"
env_name = "Hopper" # name of the environment in ["Hopper", "Walker", "HalfCheetah", "Quadcopter", "GO1", "GO2"]
proj_name = None # name of the projector in [None, "Ref", "Adm", "SA", "A"]
conditioning = None # attributes on which the diffusion model is conditioned in [None, "s0", "cmd", "s0_cmd"]

extra_name = "" # string to add to the diffusion model to differentiate it from others 
N_samples = 8 # number of sample trajectories to generate
time_limit = None # stops the training after this many seconds

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
set_seed(0) 


#%%

env, model_size, H, N_trajs = make_env(env_name, modality)
proj = load_proj(proj_name, env, device, modality)
print("Device:", device)


#%% Load a pretrained DiT

ode = ODE(env, modality=modality, device=device, **model_size, projector=proj)
assert ode.load(extra=extra_name), f"Model {ode.filename+extra_name} cannot be loaded"
planner = Planner(env, ode)

#%% Load inverse dynamics model

if "S" in modality: # no inverse dynamics for 'A' models since they directly generate admissible trajectories
    ID = InverseDynamics(env)

#%% Conditioning
s0 = env.reset()
if conditioning is None:
    attr = None
elif conditioning == "s0":
    attr = s0.copy()
elif "cmd" in conditioning:
    # cmd = env.sample_command()
    cmd = np.array([1., 0., 0.])
    if conditioning == "s0_cmd":
        attr = np.concatenate((s0, cmd))
    else:
        attr = cmd

#%% Evaluation

out = planner.best_traj(s0, traj_len=H, attr=attr, projector=ode.projector, n_samples_per_s0=N_samples)
if modality == "S":
    sampled_traj = out[0]
    ID_traj, ID_actions, reward, survival = ID.closest_admissible_traj(sampled_traj)
    env.traj_comparison(sampled_traj, "sampled", ID_traj, "ID", title=ode.filename)

elif modality == "SA":
    sampled_traj, actions = out
    reward, survival, open_loop_traj = open_loop(env, s0, actions[0], attr=attr)
    print(f"{env_name} gets reward of {reward:.2f} and survives {survival*100:.0f}%")
    env.plot_traj(open_loop_traj)

