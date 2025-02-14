# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 15:46:22 2025

@author: Jean-Baptiste Bouvier

Main script to load and evaluate a pre-trained Diffusion Transfomer
"""

import torch
import numpy as np


from utils.loaders import make_env, load_proj
from utils.utils import set_seed
from DiT.ODE import ODE
from DiT.planner import Planner



#%% Hyperparameters

modality = "S" # whether diffusion predicts only states "S", states and actions "SA", or only actions "A"
env_name = "Hopper" # name of the environment in ["Hopper", "Walker", "HalfCheetah", "Quadcopter", "GO1", "GO2"]
proj_name = None # name of the projector in [None, "Ref", "Adm", "SA", "A"]
conditioning = None # attributes on which the diffusion model is conditioned in [None, "s0", "cmd", "s0_cmd"]

extra_name = "" # string to add to the diffusion model to differentiate it from others 
n_gradient_steps = 11 # 10_000
batch_size = 64
time_limit = None # stops the training after this many seconds

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
set_seed(0) 


#%%

env, model_size, H, N_trajs = make_env(env_name)
proj = load_proj(env_name)
print("Device:", device)


#%% Load a pretrained DiT

ode = ODE(env, modality=modality, device=device, **model_size, projector=proj)
assert ode.load(extra=extra_name), f"Model {ode.filename+extra_name} cannot be loaded"
planner = Planner(env, ode)

#%% Evaluation

s0 = env.reset()





