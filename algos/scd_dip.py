
import torch
from omegaconf import DictConfig
from tqdm import tqdm
import itertools 

from models.classifier_guidance_model import ClassifierGuidanceModel
from .ddim import DDIM

from adapt.adaptation import _score_model_adpt, _tune_lora_scale

import os 
import matplotlib.pyplot as plt 

def _has_lora(score):
    for _module in score.modules():
        if _module.__class__.__name__ in ['LoraInjectedLinear', 'LoraInjectedConv2d', 'LoraInjectedConv1d']:
            return True

def tv_loss(x):

    dh = torch.abs(x[..., :, 1:] - x[..., :, :-1])
    dw = torch.abs(x[..., 1:, :] - x[..., :-1, :])
    return torch.sum(dh[..., :-1, :] + dw[..., :, :-1])



class SCD_DIP(DDIM):
    def __init__(self, model: ClassifierGuidanceModel, cfg: DictConfig, H, save_dir=None):
        self.model = model
        self.diffusion = model.diffusion
        self.eta = cfg.algo.eta
    
        self.H = H 

        self.r = cfg.algo.r
        self.K = cfg.algo.K

        self.max_iter = cfg.algo.max_iter
        self.lr = cfg.algo.lr 
        self.alpha_tv = cfg.algo.alphatv
        self.skip = cfg.algo.skip

        self.save_dir = save_dir

    def sample(self, x, y, ts, **kwargs):
        
        _, trainable_params = _score_model_adpt(score=self.model.model,
                          method="lora",
                          r=self.r)
        
        #params_with_grad = [param for param in self.model.model.parameters() if param.requires_grad]
        #print(params_with_grad)
        xt = self.initialize(x, y, ts, **kwargs)
        n = xt.size(0)
            
        self.model.model.eval()
        optim = torch.optim.Adam(itertools.chain(*trainable_params), lr=self.lr, weight_decay=0.0)

        _tune_lora_scale(self.model.model, scale=1.0)

        t = torch.ones(n).to(x.device).long() * 800
        for _ in tqdm(range(self.K)):
            optim.zero_grad()
            
            _, x0_pred = self.model(xt, y, t)
            
            #print(self.alpha_tv, tv_loss(x0_pred), torch.mean((self.H.H(x0_pred) - y)**2))
            loss = torch.mean((self.H.H(x0_pred) - y)**2) + self.alpha_tv * tv_loss(x0_pred)

            loss.backward()

            optim.step()

        with torch.no_grad():
            _, x0_pred = self.model(xt, y, t)

        return x0_pred.cpu()

    def initialize(self, x, y, ts, **kwargs):
        return torch.randn_like(x)
