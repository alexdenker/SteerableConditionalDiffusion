# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved

"""

Implementation of DPS:

Hyungjin Chung, Jeongsol Kim, Michael T. Mccann, Marc L. Klasky, Jong Chul Ye
Diffusion Posterior Sampling for General Noisy Inverse Problems, 2022

Taken from https://github.com/NVlabs/RED-diff/blob/master/algos/dps.py
"""


import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from tqdm import tqdm 

from models.classifier_guidance_model import ClassifierGuidanceModel
from .ddim import DDIM


class DPS(DDIM):
    def __init__(self, model: ClassifierGuidanceModel, cfg: DictConfig, H):
        self.model = model
        self.diffusion = model.diffusion
        self.H = H
        self.cfg = cfg
        self.awd = cfg.algo.awd
        self.grad_term_weight = cfg.algo.grad_term_weight
        self.eta = cfg.algo.eta
        self.original = True #cfg.algo.original

    def sample(self, x, y, ts):
        y_0 = y
        n = x.size(0)
        H = self.H
    
        x = self.initialize(x, y, ts, y_0=y_0)
        ss = [-1] + list(ts[:-1])
        #xt_s = [x.cpu()]
        #x0_s = []

        xt = x
        for ti, si in tqdm(zip(reversed(ts), reversed(ss)), total=len(ss)):
            t = torch.ones(n).to(x.device).long() * ti
            s = torch.ones(n).to(x.device).long() * si
            alpha_t = self.diffusion.alpha(t).view(-1, 1, 1, 1)
            alpha_s = self.diffusion.alpha(s).view(-1, 1, 1, 1)
            c1 = ((1 - alpha_t / alpha_s) * (1 - alpha_s) / (1 - alpha_t)).sqrt() * self.eta
            c2 = ((1 - alpha_s) - c1 ** 2).sqrt()
            xt = xt.clone().to('cuda').requires_grad_(True)

            et, x0_pred = self.model(xt, y, t, scale=1.0)
            mat_norm = ((y_0 - H.H(x0_pred)).reshape(n, -1) ** 2).sum(dim=1).sqrt().detach()
            mat = ((y_0 - H.H(x0_pred)).reshape(n, -1) ** 2).sum()

            grad_term = torch.autograd.grad(mat, xt, retain_graph=True)[0]

            if self.original:
                coeff = self.grad_term_weight / mat_norm.reshape(-1, 1, 1, 1)
            else:
                coeff = alpha_s.sqrt() * alpha_t.sqrt() # - c2 * alpha_t.sqrt() / (1 - alpha_t).sqrt()

            grad_term = grad_term.detach()
            xs = alpha_s.sqrt() * x0_pred.detach() + c1 * torch.randn_like(xt) + c2 * et.detach() - grad_term * coeff
            #xt_s.append(xs.detach().cpu())
            #x0_s.append(x0_pred.detach().cpu())
            xt = xs

        return xs.detach().cpu(), x0_pred.detach().cpu()

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
