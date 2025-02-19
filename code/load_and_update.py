# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 17:46:38 2025

@author: Jean-Baptiste Bouvier

Load an old model to update it by saving the normalizing constants with it
"""

import os
import torch

from utils.utils import set_seed
from DiT.ODE import ODE
from utils.loaders import make_env, load_datasets, load_proj


#%% Hyperparameters

modality = "S" # whether diffusion predicts only states "S", states and actions "SA", or only actions "A"
env_name = "Hopper" # name of the environment in ["Hopper", "Walker", "HalfCheetah", "Quadcopter", "GO1", "GO2"]
proj_name = None # name of the projector in [None, "Ref", "Adm", "SA", "A"]
conditioning = None # attributes on which the diffusion model is conditioned in [None, "s0", "cmd", "s0_cmd"]

extra_name = "" # string to add to the diffusion model to differentiate it from others 
n_gradient_steps = 10 # 10_000
batch_size = 64
time_limit = None # stops the training after this many seconds

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

set_seed(0) 
N_trajs = 1000 # number of trajectories in the dataset
H = 300 # horizon, length of each trajectory in the dataset



#%% Default environment parameters

print("Device:", device)
env, model_size, H, N_trajs = make_env(env_name)
x, attr, attr_dim = load_datasets(env_name, modality, conditioning, N_trajs, H, device)
proj = load_proj(proj_name, env, device, modality)



#%% Load ODE without normalizing constants

ode = ODE(env, modality=modality, attr_dim=attr_dim, device=device, **model_size,
          projector=proj)

name = ode.task + "/trained_models/" + ode.filename + extra_name + ".pt"
if os.path.isfile(name):
    print("Loading " + name)
    checkpoint = torch.load(name, map_location=ode.device, weights_only=True)
    ode.F.load_state_dict(checkpoint['model'])
    ode.F_ema.load_state_dict(checkpoint['model_ema'])
else:
    print("File " + name + " doesn't exist. Not loading anything.")
    
#%% Save ODE with normalizing constants
ode.train(x, attributes=attr, batch_size=batch_size, n_gradient_steps=0,
          extra=extra_name)

#%% Change normalization to single normalizer for all states (old normalization)

ode.normalizer.mean = x.mean()
ode.normalizer.std = x.std()
ode.save()

#%% Verification: loss should be small
ode.train(x, attributes=attr, batch_size=batch_size, n_gradient_steps=n_gradient_steps, 
          extra=extra_name, time_limit=time_limit)





