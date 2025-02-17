import argparse
import os

import numpy as np
import torch
import yaml
import time 
import matplotlib.pyplot as plt 

from models.utils import create_model
from models.diffusion import Diffusion
from models.classifier_guidance_model import ClassifierGuidanceModel
from algos.ddim import DDIM
from algos.reddiff import REDDIFF
from algos.scd import SCD
from algos.dps import DPS
from algos.dds import DDS

from dataset.aapm import AAPMDataset

from torch.utils.data import TensorDataset

from radon.tomography import Tomography

torch.manual_seed(5)
device = "cuda"

def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace

import argparse
parser = argparse.ArgumentParser(description='conditional sampling')

parser.add_argument('--method', default="reddiff")
parser.add_argument('--train_on', default="ellipses")
parser.add_argument('--test_on', default='aapm')
parser.add_argument('--grad_term_weight', default=1.0)
parser.add_argument('--eta', default=0.2) # has no influence on reddiff 
parser.add_argument('--part', default="val")

# params for inverse problems 
parser.add_argument('--num_angles', default=80)
parser.add_argument('--noise_std', default=0.01)

# joint params
parser.add_argument('--lr', default=0.0001)

# params for reddiff
parser.add_argument('--sigma_x0', default=0.0001)
parser.add_argument('--awd', default=True)
parser.add_argument('--cond_awd', default=False)
parser.add_argument('--obs_weight', default=1.0)
parser.add_argument('--denoise_term_weight', default="linear")

# params for DDS
parser.add_argument('--max_iter', default=3)
parser.add_argument('--gamma', default=10)

# params for SCD
parser.add_argument("--K", default=12)
parser.add_argument("--r", default=8)
parser.add_argument('--alphatv', default=1e-6)
parser.add_argument('--skip', default=20)


def coordinator(args):

    train_on = str(args.train_on) #"ellipses" # "aapm" "ellipses"
    test_on = str(args.test_on) #"aapm" # "aapm" "ellipses" "lodopab"
    method = str(args.method) #"reddiff"
    part = str(args.part) #"val" # "test"

    if train_on == "aapm":
        with open(os.path.join("configs", "aapm.yml"), "r") as f:
            model_config = yaml.safe_load(f)
    elif train_on == "ellipses":
        with open(os.path.join("configs", "diskellipses.yml"), "r") as f:
            model_config = yaml.safe_load(f)
    else:
        raise NotImplementedError
    model_config = dict2namespace(model_config)

    sde = Diffusion()

    model = create_model(**vars(model_config.model))
    model.convert_to_fp32()
    model.dtype = torch.float32
    model.load_state_dict(torch.load(model_config.data.model_path, weights_only=True))
    model.to("cuda")
    model.eval()

    if test_on == "aapm":
        dataset = AAPMDataset(part=part)
    elif test_on == "ellipses":
        dataset  = TensorDataset(torch.load(f"dataset/disk_ellipses_{part}_256.pt"))
    elif test_on == "walnut":
        dataset  = TensorDataset(torch.load(f"dataset/walnut.pt"))
    else:
        raise NotImplementedError

    print("Length of dataset: ", len(dataset))

    classifier_model = ClassifierGuidanceModel(model=model, classifier=None, diffusion=sde, cfg=None)

    if method == "reddiff":
        cfg_sampl = {
            "algo": 
                {"awd": args.awd,
                "cond_awd": args.cond_awd, 
                "grad_term_weight": float(args.grad_term_weight), 
                "eta": float(args.eta), 
                "sigma_x0": float(args.sigma_x0),
                "obs_weight": float(args.obs_weight),
                "denoise_term_weight": str(args.denoise_term_weight),
                "lr": float(args.lr)
                },
            "dataset": 
            {"image_size": 256, 
            "channels": 1},
            "exp": { 
                "save_evolution": False
            },
            "forward_op": {
                "num_angles": int(args.num_angles),
                "sigma_y":float(args.noise_std)
            }
        }
    elif method == "scd":
        cfg_sampl = {
            "algo": 
                {"awd": args.awd,
                "eta": float(args.eta), 
                "gamma": float(args.gamma),
                "K": int(args.K),
                "r": int(args.r),
                "lr": float(args.lr),
                "max_iter": int(args.max_iter),
                "alphatv": float(args.alphatv),
                "skip": int(args.skip)
                },
            "dataset": 
            {"image_size": 256, 
            "channels": 1},
            "exp": { 
                "save_evolution": False
            },
            "forward_op": {
                "num_angles": int(args.num_angles),
                "sigma_y":float(args.noise_std)
            }
        }
    elif method == "dps":
        cfg_sampl = {
            "algo": 
                {"awd": args.awd,
                "grad_term_weight": float(args.grad_term_weight), 
                "eta": float(args.eta), 
                },
            "dataset": 
            {"image_size": 256, 
            "channels": 1},
            "exp": { 
                "save_evolution": False
            },
            "forward_op": {
                "num_angles": int(args.num_angles),
                "sigma_y":float(args.noise_std)
            }
        }
    elif method == "dds":
        cfg_sampl = {
            "algo": 
                {"awd": args.awd,
                "eta": float(args.eta), 
                "gamma": float(args.gamma),
                "max_iter": int(args.max_iter)
                },
            "dataset": 
            {"image_size": 256, 
            "channels": 1},
            "exp": { 
                "save_5evolution": False
            },
            "forward_op": {
                "num_angles": int(args.num_angles),
                "sigma_y":float(args.noise_std)
            }
        }
    else:
        raise NotImplementedError
    sampl_config = dict2namespace(cfg_sampl)


    save_dir = os.path.join("results", train_on, test_on, method, f"angles={args.num_angles}", f'{time.strftime("%d-%m-%Y-%H-%M-%S")}')
    #save_dir = "tmp"
    print("save run to ", save_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_img_dir = os.path.join(save_dir, "imgs")
    if not os.path.exists(save_img_dir):
        os.makedirs(save_img_dir)

    with open(os.path.join(save_dir, "sampl_cfg.yaml"), "w") as f:
        yaml.dump(cfg_sampl, f)

    forward_op = Tomography(angles=cfg_sampl["forward_op"]["num_angles"], img_width=256, device="cuda")

    class ForwardModel():
        def __init__(self, likelihood):

            self.likelihood = likelihood

        def H(self, x):
            return self.likelihood.A(x)

        def H_pinv(self, y):
            return self.likelihood.A_dagger(y)
        
        def H_adjoint(self, y):
            return self.likelihood.A_adjoint(y)

    if method == "reddiff":
        sampler = REDDIFF(model=classifier_model, cfg=sampl_config, H=ForwardModel(forward_op))
    elif method == "scd":
        sampler = SCD(model=classifier_model, cfg=sampl_config, H=ForwardModel(forward_op))
    elif method == "dps":
        sampler = DPS(model=classifier_model, cfg=sampl_config, H=ForwardModel(forward_op))
    elif method == "dds":
        sampler = DDS(model=classifier_model, cfg=sampl_config, H=ForwardModel(forward_op))
    else:
        raise NotImplementedError

    for i in range(len(dataset)):

        if test_on == "aapm":
            x = dataset[i].unsqueeze(0)
            y = None 
        elif test_on == "ellipses" or test_on == "walnut":
            x = dataset[i][0].unsqueeze(0)
            y = None 
        else:
            raise NotImplementedError

        x = x.to("cuda")

        if y == None:
            y = forward_op.A(x)
            print("y: ", y.min(), y.max())
            y_noise = y + cfg_sampl["forward_op"]["sigma_y"] * torch.mean(torch.abs(y)) * torch.randn_like(y)
            y_noise[y_noise < 0] = 0
            x_fbp = forward_op.A_dagger(y_noise)
            
            print(y_noise.min(), y_noise.max())


        ts = torch.arange(0, sde.num_diffusion_timesteps).to(device)

        if method == "reddiff":
            x0_pred, mu = sampler.sample(x, y_noise, ts = ts, y_0=y_noise )
            x_mean = mu.detach().cpu()
            #print(x0_pred.shape, mu.shape)
        elif method == "scd":
            ts = torch.arange(0, sde.num_diffusion_timesteps).to(device)[::5]#[::20]

            x_mean = sampler.sample(x, y_noise, ts = ts)
            
            model = create_model(**vars(model_config.model))
            model.convert_to_fp32()
            model.dtype = torch.float32
            model.load_state_dict(torch.load(model_config.data.model_path, weights_only=True))
            model.to("cuda")
            model.eval()
            
            classifier_model = ClassifierGuidanceModel(model=model, classifier=None, diffusion=sde, cfg=None)

            # re-initialise SCD with clean model
            sampler = SCD(model=classifier_model, cfg=sampl_config, H=ForwardModel(forward_op))
        elif method == "dps":
            _, x_mean = sampler.sample(x, y_noise, ts = ts)
        elif method == "dds":
            ts = torch.arange(0, sde.num_diffusion_timesteps).to(device)[::20]
            print("Number of timesteps: ", len(ts))
            x_mean, _ = sampler.sample(x, y_noise, ts = ts)

        else:
            raise NotImplementedError

        fig, (ax1, ax2, ax3) = plt.subplots(1,3)

        ax1.imshow(x[0,0].cpu().numpy(), cmap="gray")
        ax2.imshow(x_fbp[0,0].cpu().numpy(), cmap="gray")
        ax3.imshow(x_mean[0,0].detach().cpu().numpy(), cmap="gray")

        plt.savefig(os.path.join(save_img_dir, f"reco_{i}.png"))
        plt.close()
        
        np.save(os.path.join(save_img_dir, "reco_{}.npy".format(i)), x_mean.cpu().numpy())
        np.save(os.path.join(save_img_dir, "gt_{}.npy".format(i)), x.cpu().numpy())

        #from PIL import Image
        #x_pil = (np.clip(x_mean[0,0].detach().cpu().numpy(), 0, 1) * 255).astype(np.uint8)
        #image = Image.fromarray(x_pil)
        #image.save(os.path.join("ellipses2walnut_scd.png")) 

        #x_pil = (np.clip(x[0,0].detach().cpu().numpy(), 0, 1) * 255).astype(np.uint8)
        #image = Image.fromarray(x_pil)
        #image.save(os.path.join("walnut.png")) 

        #x_pil = (np.clip(x_fbp[0,0].detach().cpu().numpy(), 0, 1) * 255).astype(np.uint8)
        #image = Image.fromarray(x_pil)
        #image.save(os.path.join("walnut_fbp.png")) 


if __name__ == '__main__':
	args = parser.parse_args()
	coordinator(args)