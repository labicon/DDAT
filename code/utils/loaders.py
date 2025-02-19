# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 17:48:31 2025

@author: Jean-Baptiste Bouvier

Utils to load the desired environments, projectors, datasets,...
"""
import torch
import numpy as np
from Walker.walker2d import WalkerEnv
from Hopper.hopper import HopperEnv
from HalfCheetah.half_cheetah import CheetahEnv
from Quadcopter.quadcopter import QuadcopterEnv
from GO1.GO1_env import Go1JoystickGymEnv
from GO2.GO2_env import Go2JoystickGymEnv

from utils.projectors import Reference_Projector, Admissible_Projector, SA_Projector, Action_Projector


#%% 

def make_env(env_name: str):
    """
    Creates the given environment

    Parameters
    ----------
    env_name : Name of the desired environment to create
        Should be one of "Hopper", "Walker", "HalfCheetah", "Quadcopter", "GO1", "GO2"

    Returns
    -------
    env : Gym-like environment
    model_size : dictionary of default parameters for DiT for the given environment
    H : horizon, default length of each trajectories in the environment
    N_trajs : default number of trajectories in the environment's datasets
    """
    assert env_name  in ["Hopper", "Walker", "HalfCheetah", "Quadcopter", "GO1", "GO2"], "Environment name not recognized"
    
    if env_name == "Hopper":
        env = HopperEnv()
        model_size = {"d_model": 64, "n_heads": 4, "depth": 3}
        H = 300 
        
    elif env_name == "Walker":
        env = WalkerEnv()
        model_size = {"d_model": 256, "n_heads": 4, "depth": 3}
        H = 300
        
    elif env_name == "HalfCheetah":
        env = CheetahEnv()
        model_size = {"d_model": 256, "n_heads": 4, "depth": 3}
        H = 200
        
    elif env_name == "Quadcopter":
        env = QuadcopterEnv()
        model_size = {"d_model": 128, "n_heads": 4, "depth": 3}
        H = 200
        
    elif env_name == "GO1":
        env = Go1JoystickGymEnv()
        model_size = {"d_model": 256, "n_heads": 4, "depth": 6}
        H = 500
        
    elif env_name == "GO2":
        env = Go2JoystickGymEnv()
        model_size = {"d_model": 256, "n_heads": 4, "depth": 6}
        H = 500

    N_trajs = 1000

    return env, model_size, H, N_trajs




#%% Loading training dataset and conditioning attributes

def load_datasets(env_name:str, modality:str, conditioning:str,
                  N_trajs:int, H:int, device):
    """
    Load a dataset of trajectories to train the diffusion models, 
    along with the dataset of conditioning attributes

    Parameters
    ----------
    env_name : name of the environment
    modality : whether DiT predicts states "S", states and actions "SA" or only actions "A" 
    conditioning: attributes types on which the model is conditioned None, "s0", "cmd", "s0_cmd"
    N_trajs : number of trajectories in the dataset
    H : horizon, length of the trajectories
    device : to place the loaded dataset on

    Returns
    -------
    x : torch.tensor of the dataset of size (N_trajs, H, modality_size)
    attr : torch.tensor of the conditioning attributes of size (N_trajs, attr_dim)
    attr_dim : int  Dimension of the conditioning attributes 
    """
    
    assert modality  in ["S", "SA", "A"], "Modality name not recognized"
    assert conditioning in [None, "s0", "cmd", "s0_cmd"], "Conditioning not recognized"
    
    # Training dataset
    dataset = np.load(f"datasets/{env_name}_{N_trajs}trajs_{H}steps.npz")
    if modality == "S":
        x = dataset['Trajs'][:, :H] # cut the length of trajectories to H
    elif modality == "SA":
        x = np.concatenate((dataset['Trajs'][:, :H], dataset['Actions'][:, :H]), dim=2)
    else: # modality == "A"
        x = dataset['Actions'][:, :H]
    x = torch.FloatTensor(x).to(device)
        
    # Conditioning attributes
    if conditioning == "s0":
        attr = dataset['Trajs'][:, 0] # initial states of trajectories
    elif conditioning == "s0_cmd":
        s0 = dataset['Trajs'][:, 0] # initial states of trajectories
        cmd = dataset['Cmd']
        attr = np.concatenate((s0, cmd), dim=1)
    elif conditioning == "cmd":
        attr = dataset['Cmd']
    
    if conditioning is None:
        attr, attr_dim = None, None
    else:
        attr = torch.FloatTensor(attr).to(device)
        attr_dim = attr.shape[1]

    return x, attr, attr_dim






def load_proj(proj_name:str, env, device:str, modality:str, dataset=None,
              extra_name:str = ""):
    """
    Default projector parameters

    Parameters
    ----------
    proj_name : name of the projector to load
    env : environment where the projector operates
    device : device on which projector operates
    modality : whether the diffusion predicts states "S", states and actions "SA", or only actions "A"
    dataset : optional dataset on which to train the SA-projector
    extra_name : name to for the SA-projector

    Returns
    -------
    loaded projector
    """
    assert proj_name in [None, "Ref", "Adm", "SA", "A"], "Projector name not recognized"
    assert modality in ["S", "SA", "A"], "Modality not recognized"
    
    if proj_name is None:
        proj = None
    else:
        assert modality != "A", "The action-only diffusion model does not support projections"
    
    if proj_name == "Ref":
        proj = Reference_Projector(env, sigma_min=0.0021, sigma_max=0.2, device=device)
        
    elif proj_name == "Adm":
        proj = Admissible_Projector(env, sigma_min=0.0021, device=device)
        
    elif proj_name == "SA":
        assert modality == "SA", "The state-action projector requires SA modality"
        proj = SA_Projector(env, sigma_min=0.0021, sigma_max=0.2, device=device)
        loaded = proj.load(extra=extra_name)
        if not loaded: # The SA-projector needs to be trained
            trajs = dataset[:, :, :env.state_size] # sequences of states in SA dataset
            actions = dataset[:, :, env.state_size:] # sequences of actions in SA dataset
            proj.train(Trajs=trajs, Actions=actions, extra=extra_name)
    
    elif proj_name == "A":
        assert modality == "SA", "The action projector requires SA modality"
        proj = Action_Projector(env, sigma_min=0.0021, sigma_max=0.2)
        
    return proj