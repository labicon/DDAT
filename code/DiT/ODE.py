# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 15:57:50 2025

@author: Jean-Baptiste Bouvier

ODE for the Diffusion Transformers (DiT)
Taken from https://github.com/ZibinDong/AlignDiff-ICLR2024/blob/main/utils/dit_utils.py
"""

import os
import time
import torch
import numpy as np
from tqdm import tqdm
from copy import deepcopy

from DiT.DiT import DiT1d, count_parameters
from utils.utils import Normalizer




class ODE():
    """
    Base diffusion 
    All the parameters come from the EDM paper:
        "Elucidating the Design Space of Diffusion-Based Generative Models"
    """
    def __init__(self, env, modality: str,
        attr_dim: int = None,
        sigma_min: float = 0.002, sigma_max: float = 80,
        rho: float = 7, p_mean: float = -1.2, p_std: float = 1.2, 
        d_model: int = 384, n_heads: int = 6, depth: int = 12,
        device: str = "cpu", N: int = 5, projector = None):
        """
        Arguments:
            env: a Gym-like environment
            modality: ["S", "SA", "A"] whether predicting states "S", states and actions "SA", or only actions "A"
            sigma_min: optional minimal noise level sigma from EDM
            sigma_max: optional maximal noise level sigma from EDM
            rho: optional power of the distribution of noise levels sigma between sigma_min and sigma_max following EDM
            p_mean: optional mean of the Gaussian sampling of noise level sigma from EDM
            p_std: optional std of the Gaussian sampling of noise level sigma from EDM
            d_model: width of a layer (must be divisible by n_heads) of the diffusion transformer
            n_heads: number of heads of the diffusion transformer
            depth: number of layers of the diffusion transformer
            device: ['cpu', 'cuda'] optional device to store the model
            N: optional number of denoising steps
            projector: optional projector to use during training of the model
        """
        
        self.projector = projector
        if projector is None:
            self.projector_name = ""
        else:
            self.projector_name = projector.name
        self.task = env.name
        self.specs = f"{d_model}_{n_heads}_{depth}"
        self.is_conditional = attr_dim is not None
        self.attr_dim = attr_dim
        assert modality in ["S", "SA", "A"], 'model must predict either states "S", states and actions "SA", or only actions "A" '
        self.modality = modality
        self.filename = f"{modality}_{self.is_conditional*'Cond_'}ODE_{self.task}_{self.projector_name}_specs_{self.specs}"
        
        self.state_size = env.state_size
        self.action_size = env.action_size
        # dimension of the predictions
        self.pred_dim = self.action_size*("A" in modality) + self.state_size*("S" in modality)
        
        if modality == "A":
            assert projector is None, "The action-only prediction model does not support projections"
            assert attr_dim >= self.state_size, "The action-only prediction model needs to be conditioned at least on the initial state"
        
        self.sigma_min, self.sigma_max = sigma_min, sigma_max
        self.rho, self.p_mean, self.p_std = rho, p_mean, p_std
        
        self.device = device
        self.F = DiT1d(self.pred_dim, attr_dim=attr_dim, d_model=d_model, n_heads=n_heads, depth=depth, dropout=0.1).to(device)
        self.F.train()
        # Exponential Moving Average (ema)
        self.F_ema = deepcopy(self.F).requires_grad_(False)
        self.F_ema.eval()
        self.optim = torch.optim.AdamW(self.F.parameters(), lr=2e-4, weight_decay=1e-4)
        self.set_N(N) # number of noise scales
        print(f'Initialized {self.filename} with {count_parameters(self.F)} parameters.')
        
    def ema_update(self, decay=0.999):
        for p, p_ema in zip(self.F.parameters(), self.F_ema.parameters()):
            p_ema.data = decay*p_ema.data + (1-decay)*p.data

    def set_N(self, N):
        self.N = N
        self.sigma_s = (self.sigma_max**(1/self.rho)+torch.arange(N, device=self.device)/(N-1)*\
            (self.sigma_min**(1/self.rho)-self.sigma_max**(1/self.rho)))**self.rho
        self.t_s = self.sigma_s
        self.scale_s = torch.ones_like(self.sigma_s) * 1.0
        self.dot_sigma_s = torch.ones_like(self.sigma_s) * 1.0
        self.dot_scale_s = torch.zeros_like(self.sigma_s)
        if self.t_s is not None:
            self.coeff1 = (self.dot_sigma_s/self.sigma_s + self.dot_scale_s/self.scale_s)
            self.coeff2 = self.dot_sigma_s/self.sigma_s*self.scale_s
            
    def c_skip(self, sigma): return self.sigma_data**2/(self.sigma_data**2+sigma**2)
    def c_out(self, sigma): return sigma*self.sigma_data/(self.sigma_data**2+sigma**2).sqrt()
    def c_in(self, sigma): return 1/(self.sigma_data**2+sigma**2).sqrt()
    def c_noise(self, sigma): return 0.25*(sigma).log()
    def loss_weighting(self, sigma): return (self.sigma_data**2+sigma**2)/((sigma*self.sigma_data)**2)
    def sample_noise_distribution(self, N):
        log_sigma = torch.randn((N,1,1),device=self.device)*self.p_std + self.p_mean
        return log_sigma.exp()
    
    def D(self, x, sigma, condition = None, mask = None, use_ema = False):
        c_skip, c_out, c_in, c_noise = self.c_skip(sigma), self.c_out(sigma), self.c_in(sigma), self.c_noise(sigma)
        F = self.F_ema if use_ema else self.F
        return c_skip*x + c_out*F(c_in*x, c_noise.squeeze(-1), condition, mask)
    
    def update(self, x, condition=None):
        """Updates the DiT module given a trajectory batch x: (batch, horizon, pred_dim)
        and their corresponding attributes condition: (batch, attr_dim) """
        sigma = self.sample_noise_distribution(x.shape[0])
        eps = torch.randn_like(x) * sigma
        eps[:, 0, :self.state_size] = 0. # preserve the first state observation since given
        loss_mask = torch.ones_like(x)
        loss_mask[:, 0, :self.state_size] = 0. # no loss on the first state since constant
        loss_mask[:, 0, self.state_size:] = 10. # higher coefficient for the first action
        
        if condition is None:
            mask = None
        else:
            mask = (torch.rand(*condition.shape, device=self.device) > 0.2).int()
        pred = self.D(x + eps, sigma, condition, mask)            
        
        if self.projector is not None:
            pred_s = self.normalizer.unnormalize(pred[:, :, :self.state_size]) # predicted states    
            ref_s = self.normalizer.unnormalize(x[:, :, :self.state_size]) # true states
            
            if "A" in self.modality:
                pred_a = pred[:, :, self.state_size:] # predicted actions
                ref_a = x[:, :, self.state_size:] # true actions
            else:
                pred_a, ref_a = None, None
            
            if self.projector.reference:
                s, a = self.projector.project_traj(Trajs=pred_s, Ref_Trajs=ref_s, sigma=sigma, Actions=ref_a) 
            else:
                s, a = self.projector.project_traj(Trajs=pred_s, sigma=sigma, Actions=pred_a)
            
            s = self.normalizer.normalize(s)
            if self.modality == "SA":
                pred = torch.cat((x, a), dim=2)
            else: # "S"
                pred = s
        
        loss = (loss_mask * self.loss_weighting(sigma) * (pred - x)**2).mean()
        self.optim.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.F.parameters(), 10.)
        self.optim.step()
        self.ema_update()
        return loss.item(), grad_norm.item()
    
    
    def train(self, x:torch.Tensor, attributes:torch.Tensor = None,
              n_gradient_steps:int = 10000,
              batch_size:int = 32, extra:str="", time_limit=None):
        """Trains the DiT module 
        
        Arguments:
            x: training dataset of size (nb trajs, horizon, modality_size)
            attributes: optional conditioning attributes matching x (nb_trajs, attr_dim)
            n_gradient_steps: number of training iterations
            batch_size: number of trajectories sampled per batch
            extra: optional string to add to the name of the model when saving it
            time_limit: training limit time in seconds
        """
        assert x.shape[2] == self.state_size*("S" in self.modality) + self.action_size*("A" in self.modality), "The third dimension of the dataset should match the modality size"
        
        if self.is_conditional:
            assert attributes is not None, "Conditional model requires attribute for training"
            self.attr_normalizer = Normalizer(attributes)
            nor_attr = self.attr_normalizer.normalize(attributes) # normalized conditioning
            
            
        print('Begins training of the Diffusion Transformer ' + self.filename + extra)
        if time_limit is not None:
            t0 = time.time()
            print(f"Training limited to {time_limit:.0f}s")
        
        if "S" in self.modality: # normalize only the states
            self.normalizer = Normalizer(x[:, :, :self.state_size])
            nor_s = self.normalizer.normalize(x[:, :, :self.state_size])
            x_normalized = torch.cat((nor_s, x[:, :, self.state_size:]), dim=2)
        else: # modality == "A"
            x_normalized = x.clone()
        self.sigma_data = x_normalized.std().item()
        N_trajs = x_normalized.shape[0]
        loss_avg = 0.
        
        pbar = tqdm(range(n_gradient_steps))
        for step in range(n_gradient_steps):
            # Training curriculum with projector
            if self.projector is not None:
                sigma_low = max(0.1*self.projector.sigma_max*(1 - 2*step/n_gradient_steps), self.sigma_min)
                self.p_mean = (np.log(self.sigma_max) + np.log(sigma_low))/2
                self.p_std = (np.log(self.sigma_max) - np.log(sigma_low))/10 # 5 sigmas on each sides of p_mean
            
            idx = np.random.randint(0, N_trajs, batch_size) # sample a random batch of trajectories
            x = x_normalized[idx].clone()
            if self.is_conditional:
                attr = nor_attr[idx].clone()
            else:
                attr = None
            loss, grad_norm = self.update(x, attr)
            loss_avg += loss
            if (step+1) % 10 == 0:
                pbar.set_description(f'step: {step+1} loss: {loss_avg / 10.:.4f} grad_norm: {grad_norm:.4f} ')
                pbar.update(10)
                self.save(extra)
                if time_limit is not None and time.time() - t0 > time_limit:
                    print(f"Time limit reached at {time.time() - t0:.0f}s")
                    break
                if loss_avg > 10: # prevents divergence, but might be too low at the start
                    print("Too high loss")
                    break
                loss_avg = 0.
                
        print('\nTraining completed!')
        
        
        
    
    @torch.no_grad()
    def sample(self, s0: torch.Tensor, traj_len:int, n_samples:int, attr = None,
               projector = None, w:float = 1.5, N:int = None):
        """Samples trajectories using Heun's 2nd order sampling from the EDM paper
        
        Arguments:
            s0: initial state of size (n_samples, state_size)
            traj_len: length of the trajectories to generate (including s0)
            n_samples: number of trajectories to generate
            attr: optional conditioning of the trajectories of size (n_samples, attr_dim)
            projector: optional projector to be used during sampling
            w: optional factor weighing the conditioned VS unconditioned samples
            N: optional number of denoising iterations
        """
        if N is not None and N != self.N: self.set_N(N)
        x = torch.randn((n_samples, traj_len, self.pred_dim), device=self.device) * self.sigma_s[0] * self.scale_s[0]
        if self.modality == "A":
            assert projector is None, "The action-only model does not support predictions"
        
        if "S" in self.modality:
            nor_s0 = self.normalizer.normalize(s0)
            x[:, 0, :self.state_size] = nor_s0 # set the initial state to s0
        
        if self.is_conditional:
            assert attr is not None, "Conditional model requires attribute for sampling"
            # Doubling x, attr since we sample eps(conditioned) - eps(unconditioned) see Section 4 of AlignDiff
            nor_attr = self.attr_normalizer.normalize(attr)
            attr_mask = torch.ones_like(nor_attr, device=self.device) 
            nor_attr = nor_attr.repeat(2, 1)
            attr_mask = attr_mask.repeat(2, 1)
            attr_mask[n_samples:] = 0
        
        for i in range(self.N):
            with torch.no_grad():
                if self.is_conditional:
                    D = self.D(x.repeat(2,1,1)/self.scale_s[i], torch.ones((2*n_samples,1,1),device=self.device)*self.sigma_s[i], nor_attr, attr_mask, use_ema=True)
                    D = w*D[:n_samples] + (1-w)*D[n_samples:]
                else:
                    D = self.D(x/self.scale_s[i], torch.ones((n_samples,1,1),device=self.device)*self.sigma_s[i], use_ema=True)
                
            d = self.coeff1[i] * x - self.coeff2[i] * D
            
            if i == self.N-1:
                dt = 0 - self.t_s[i]
            else:
                dt = self.t_s[i+1] - self.t_s[i]
                
            x = x + dt * d
            if "S" in self.modality:
                # at each denoising step, reset the initial state to s0
                x[:, 0, :self.state_size] = nor_s0 
           
            if projector is not None: # project the rest of the trajectory onto its admissible space
                sigma = self.sigma_s[i]*torch.ones((x.shape[0],1), device=self.device)
                s = self.normalizer.unnormalize(x[:, :, :self.state_size]) # sampled states
                
                if self.modality == "SA":
                    a = x[:, :, self.state_size:] # sampled actions
                else: # "S"
                    a = None

                if projector.reference:
                    # use the sampled traj as reference after training to minimize deviations from projections
                    s, a = projector.project_traj(Trajs=s, Ref_Trajs=s, sigma=sigma, Actions=a)
                else:
                    s, a = projector.project_traj(Trajs=s, sigma=sigma, Actions=a)
                
                x = self.normalizer.normalize(s)
                if self.modality == "SA":
                    x = torch.cat((x, a), dim=2)
        
        if "S" in self.modality:
            x[:, :, :self.state_size] = self.normalizer.unnormalize(x[:, :, :self.state_size])
        return x    
    
    
    def save(self, extra:str = ""):
        to_save = {'model': self.F.state_dict(),
                   'model_ema': self.F_ema.state_dict(),
                   'sigma_data': self.sigma_data}
        if "S" in self.modality:
             to_save['normalizer_mean'] = self.normalizer.mean
             to_save['normalizer_std'] = self.normalizer.std
        if self.is_conditional:
            to_save['attr_normalizer_mean'] = self.attr_normalizer.mean
            to_save['attr_normalizer_std'] = self.attr_normalizer.std
        torch.save(to_save, self.task+"/trained_models/"+ self.filename+extra+".pt")
        
    
    def load(self, extra:str = ""):    
        name = self.task+"/trained_models/" + self.filename + extra + ".pt"
        if os.path.isfile(name):
            print("Loading " + name)
            checkpoint = torch.load(name, map_location=self.device, weights_only=True)
            self.F.load_state_dict(checkpoint['model'])
            self.F_ema.load_state_dict(checkpoint['model_ema'])
            self.sigma_data = checkpoint['sigma_data']
            if "S" in self.modality:
                self.normalizer = Normalizer(checkpoint['normalizer_mean'])
                self.normalizer.std = checkpoint['normalizer_std']
            if self.is_conditional:
                self.attr_normalizer = Normalizer(checkpoint['attr_normalizer_mean'])
                self.attr_normalizer.std = checkpoint['attr_normalizer_std']   
            return True # loaded
        else:
            print("File " + name + " doesn't exist. Not loading anything.")
            return False # not loaded


    def update_projector(self, projector):
        """Upade the projector, changes the filename for saving/loading""" 
        assert "S" in self.modality, "The action-only model does not support predictions"
        self.projector = projector
        self.projector_name = projector.name
        self.filename = f"{self.modality}_{self.is_conditional*'Cond_'}ODE_{self.task}_{self.projector_name}_specs_{self.specs}"
        
