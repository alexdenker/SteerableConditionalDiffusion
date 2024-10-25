
import torch
from omegaconf import DictConfig
from tqdm import tqdm

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


class SCD(DDIM):
    def __init__(self, model: ClassifierGuidanceModel, cfg: DictConfig, H, save_dir=None):
        self.model = model
        self.diffusion = model.diffusion
        self.eta = cfg.algo.eta
    
        self.H = H 

        self.r = cfg.algo.r
        self.K = cfg.algo.K

        self.gamma = cfg.algo.gamma
        self.max_iter = cfg.algo.max_iter
        self.lr = cfg.algo.lr 
        self.alpha_tv = cfg.algo.alphatv
        self.skip = cfg.algo.skip

        self.save_dir = save_dir

    def sample(self, x, y, ts, **kwargs):
        
        _score_model_adpt(score=self.model.model,
                          method="lora",
                          r=self.r)
        params_with_grad = [param for param in self.model.model.parameters() if param.requires_grad]
        
        xt = self.initialize(x, y, ts, **kwargs)
        n = xt.size(0)
        ss = [-1] + list(ts[:-1])
            
        def op(x):
            return x + self.gamma*self.H.H_adjoint(self.H.H(x))
        
        rhs = self.H.H_adjoint(y)
        self.model.model.eval()
        optim = torch.optim.Adam(params_with_grad, lr=self.lr, weight_decay=0.0)

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

            _tune_lora_scale(self.model.model, scale=0.0)
            with torch.no_grad():
                et_ddim, x0_pred_before = self.model(xt, y, t)

            #weighting_gamma = self.gamma * 1./(ts[-1] - ti + 1.) 

            with torch.enable_grad():
                _tune_lora_scale(self.model.model, scale=1.0)
                self.model.model.eval()
                
                # skip adaptation steps to make sampling faster 
                if ti in adapt_ts:
                    for _ in range(self.K):
                        optim.zero_grad()
                        
                        _, x0_pred = self.model(xt, y, t)
                        noisy_rhs = x0_pred + self.gamma * rhs
                        
                        #xhat = x0_pred - weighting_gamma * self.H.H_adjoint(self.H.H(x0_pred) - y)  #cg(op=op,x=x0_pred, rhs=noisy_rhs, n_iter=self.max_iter)
                        xhat = cg(op=op,x=x0_pred, rhs=noisy_rhs, n_iter=self.max_iter)


                        loss = torch.mean((self.H.H(xhat) - y)**2) + self.alpha_tv * tv_loss(xhat)
                        loss.backward()

                        optim.step()

            with torch.no_grad():
                new_eps, x0_pred = self.model(xt, y, t)
                noisy_rhs = x0_pred + self.gamma * rhs
                xhat = cg(op=op,x=x0_pred, rhs=noisy_rhs, n_iter=self.max_iter)
                #xhat = x0_pred - weighting_gamma * self.H.H_adjoint(self.H.H(x0_pred) - y)  #cg(op=op,x=x0_pred, rhs=noisy_rhs, n_iter=self.max_iter)
                #xhat = torch.clamp(xhat, 0, 1)
            xs = alpha_s.sqrt() * xhat + c1 * torch.randn_like(xt) + c2 * et_ddim
            
            if self.save_dir is not None:
                fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(13,6)) 
                im = ax1.imshow(xhat[0,0].detach().cpu().numpy(), cmap="gray")
                fig.colorbar(im, ax=ax1)
                ax1.set_title("x0hat tilde after adapt")
                im = ax2.imshow(x0_pred_before[0,0].detach().cpu().numpy(), cmap="gray")
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
