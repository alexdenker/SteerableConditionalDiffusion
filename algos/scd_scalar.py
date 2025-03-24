"""
This variation does not update the underlying model, but instead estimates the step size lambda to update the Tweedie estimate 

min_lambda || A (x0hat - lambda * gra ) - y || 


"""



import torch
from omegaconf import DictConfig
from tqdm import tqdm
import itertools 

from models.classifier_guidance_model import ClassifierGuidanceModel
from .ddim import DDIM

import os 
import matplotlib.pyplot as plt 

def tv_loss(x):

    dh = torch.abs(x[..., :, 1:] - x[..., :, :-1])
    dw = torch.abs(x[..., 1:, :] - x[..., :-1, :])
    return torch.sum(dh[..., :-1, :] + dw[..., :, :-1])

class SCDScalar(DDIM):
    def __init__(self, model: ClassifierGuidanceModel, cfg: DictConfig, H, save_dir=None):
        self.model = model
        self.diffusion = model.diffusion
        self.eta = cfg.algo.eta
    
        self.H = H 

        self.K = cfg.algo.K

        self.gamma = cfg.algo.gamma
        self.max_iter = cfg.algo.max_iter
        self.lr = cfg.algo.lr 
        self.skip = cfg.algo.skip

        self.save_dir = save_dir

    def sample(self, x, y, ts, **kwargs):
        
        lambd = torch.nn.parameter.Parameter(torch.tensor(0.0, device=x.device))
        
        xt = self.initialize(x, y, ts, **kwargs)
        n = xt.size(0)
        ss = [-1] + list(ts[:-1])
                
        self.model.model.eval()
        optim = torch.optim.AdamW([lambd], lr=self.lr, weight_decay=0.0)

        adapt_ts = list(ts[::self.skip])
        adapt_ts.extend(list(ts[0:4]))

        for ti, si in tqdm(zip(reversed(ts), reversed(ss)), total=len(ts)):
            
            t = torch.ones(n).to(x.device).long() * ti
            s = torch.ones(n).to(x.device).long() * si
            alpha_t = self.diffusion.alpha(t).view(-1, 1, 1, 1)
            alpha_s = self.diffusion.alpha(s).view(-1, 1, 1, 1)
            c1 = (
                (1 - alpha_t / alpha_s) * (1 - alpha_s) / (1 - alpha_t)
            ).sqrt() * self.eta
            c2 = ((1 - alpha_s) - c1**2).sqrt()

            with torch.no_grad():
                et_ddim, x0_pred = self.model(xt, y, t)
                grad = self.H.H_adjoint(self.H.H(x0_pred)) - self.H.H_adjoint(y)
            #weighting_gamma = self.gamma * 1./(ts[-1] - ti + 1.) 

            with torch.enable_grad():

                # skip adaptation steps to make sampling faster 
                # 1/2 || H(x) - y ||
                # H*(H(x) - y)
                
                if ti in adapt_ts:
                    for _ in range(self.K):
                        optim.zero_grad()
                        xhat = x0_pred - lambd * grad                   
                        loss = torch.mean((self.H.H(xhat) - y)**2) 

                        loss.backward()

                        optim.step()
            print("Lambda: ", lambd)
            with torch.no_grad():
                xhat = x0_pred - lambd * grad 

            xs = alpha_s.sqrt() * xhat + c1 * torch.randn_like(xt) + c2 * et_ddim
            
            if self.save_dir is not None:
                fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(13,6)) 
                im = ax1.imshow(xhat[0,0].detach().cpu().numpy(), cmap="gray")
                fig.colorbar(im, ax=ax1)
                ax1.set_title("x0hat tilde after adapt")
                im = ax2.imshow(x0_pred[0,0].detach().cpu().numpy(), cmap="gray")
                ax2.set_title("x0hat before adapt")
                fig.colorbar(im, ax=ax2)

                im = ax3.imshow(xs[0,0].detach().cpu().numpy(), cmap="gray")
                ax3.set_title("next sample")
                fig.colorbar(im, ax=ax3)

                plt.savefig(os.path.join(self.save_dir, f"img_at_{ti}.png"))
                plt.close()

            xt = xs

        return x0_pred.cpu()

    def initialize(self, x, y, ts, **kwargs):
        return torch.randn_like(x)
