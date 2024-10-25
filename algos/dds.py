# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved

import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from tqdm import tqdm 

from models.classifier_guidance_model import ClassifierGuidanceModel
from .ddim import DDIM

"""
Batched Conjugate Gradient in PyTorch for 
    solve (I + gamma A* A) x = gamma * A* y + xhat0

Adapted from ODL: https://github.com/odlgroup/odl/blob/master/odl/solvers/iterative/iterative.py 
"""
def cg(op: callable, x, rhs, n_iter: int = 5, tol: float = 1e-10):
    # solve (I + gamma A* A) x = rhs
    # starting with x 

    # batch x 1 x h x w
    r = op(x)
    r = rhs - r
    p = r
    d = torch.zeros_like(x)

    # Only recalculate norm after update
    sqnorm_r_old = torch.linalg.norm(r.reshape(r.shape[0], -1), dim=1)**2 #r.norm() ** 2 

    for _ in range(n_iter):
        d = op(p)
        
        inner_p_d = (p * d).sum(dim=[1,2,3]) 

        alpha = sqnorm_r_old / inner_p_d
        x = x + alpha[:, None,None,None]*p # x = x + alpha*p
        r = r - alpha[:, None,None,None]*d # r = r - alpha*d

        sqnorm_r_new = torch.linalg.norm(r.reshape(r.shape[0], -1), dim=1)**2 
       
        beta = sqnorm_r_new / sqnorm_r_old
        sqnorm_r_old = sqnorm_r_new

        p = r + beta[:, None,None,None]*p # p = r + b * p

    return x 



class DDS(DDIM):
    def __init__(self, model: ClassifierGuidanceModel, cfg: DictConfig, H):
        self.model = model
        self.diffusion = model.diffusion
        self.H = H
        self.cfg = cfg
        self.awd = cfg.algo.awd
        self.eta = cfg.algo.eta
        self.gamma = cfg.algo.gamma
        self.max_iter = cfg.algo.max_iter

    @torch.no_grad()
    def sample(self, x, y, ts):
        y_0 = y
        n = x.size(0)
        H = self.H
    
        x = self.initialize(x, y, ts, y_0=y_0)
        ss = [-1] + list(ts[:-1])
        #xt_s = [x.cpu()]
        #x0_s = []

        def op(x):
            return x + self.gamma*self.H.H_adjoint(self.H.H(x))
        rhs = self.H.H_adjoint(y)

        xt = x
        for ti, si in tqdm(zip(reversed(ts), reversed(ss)), total=len(ts)):
            # print(ti, si)
            t = torch.ones(n).to(x.device).long() * ti
            # print("t after calc : ", t, ti)
            s = torch.ones(n).to(x.device).long() * si
            # print("t after s calc : ", t, ti)
            alpha_t = self.diffusion.alpha(t).view(-1, 1, 1, 1)
            alpha_s = self.diffusion.alpha(s).view(-1, 1, 1, 1)
            c1 = (
                (1 - alpha_t / alpha_s) * (1 - alpha_s) / (1 - alpha_t)
            ).sqrt() * self.eta
            c2 = ((1 - alpha_s) - c1**2).sqrt()

            # print("t in model : ", t)
            et, x0_pred = self.model(xt, y, t, scale=1.0)
            noisy_rhs = x0_pred + self.gamma * rhs
            
            xhat = cg(op=op,x=x0_pred, rhs=noisy_rhs, n_iter=self.max_iter)
            
            xs = alpha_s.sqrt() * xhat + c1 * torch.randn_like(xt) + c2 * et

            #import matplotlib.pyplot as plt 
            #fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(13,6)) 
            #im = ax1.imshow(xhat[0,0].detach().cpu().numpy(), cmap="gray")
            #fig.colorbar(im, ax=ax1)
            #ax1.set_title("x0hat tilde ")
            #im = ax2.imshow(x0_pred[0,0].detach().cpu().numpy(), cmap="gray")
            #ax2.set_title("x0hat ")
            #fig.colorbar(im, ax=ax2)

            #im = ax3.imshow(xs[0,0].detach().cpu().numpy(), cmap="gray")
            #ax3.set_title("next sample")
            #fig.colorbar(im, ax=ax3)
            #plt.show() 


            # xt_s.append(xs.cpu())
            # x0_s.append(x0_pred.cpu())
            xt = xs

        return x0_pred.detach().cpu(), xhat.detach().cpu()

    def initialize(self, x, y, ts, **kwargs):
        #y_0 = kwargs['y_0']
        #H = self.H
        #deg = self.cfg.algo.deg
        #n = x.size(0)
        #x_0 = H.H_pinv(y_0).view(*x.size()).detach()
        #ti = ts[-1]
        #t = torch.ones(n).to(x.device).long() * ti
        #alpha_t = self.diffusion.alpha(t).view(-1, 1, 1, 1)
        return torch.randn_like(x) #  alpha_t.sqrt() * x_0 + (1 - alpha_t).sqrt() * torch.randn_like(x_0)
