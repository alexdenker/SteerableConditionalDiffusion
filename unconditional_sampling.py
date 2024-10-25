import argparse
import os
from collections.abc import MutableMapping

import ml_collections.config_flags
import numpy as np
import torch
import torchvision
import yaml
from absl import app, flags
from omegaconf import OmegaConf
from PIL import Image
import time 
import matplotlib.pyplot as plt 

from models.utils import create_model
from models.diffusion import Diffusion
from models.classifier_guidance_model import ClassifierGuidanceModel
from algos.ddim import DDIM

def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def coordinator():
    
    with open(os.path.join("configs", "diskellipses.yml"), "r") as f:
        model_config = yaml.safe_load(f)
    model_config = dict2namespace(model_config)
    sde = Diffusion()

    pretrained_model = create_model(**vars(model_config.model))
    pretrained_model.convert_to_fp32()
    pretrained_model.dtype = torch.float32
    pretrained_model.load_state_dict(torch.load(model_config.data.model_path, weights_only=True))
    pretrained_model.to("cuda")
    pretrained_model.eval()

    classifier_model = ClassifierGuidanceModel(model=pretrained_model, classifier=None, diffusion=sde, cfg=None)

    cfg_sampl = {
        "algo": {"eta": 0.5, 
                "cond_awd": False,
                "sdedit": False}
    }
    sampl_config = dict2namespace(cfg_sampl)
    sampler = DDIM(model=classifier_model, cfg=sampl_config)
    x = torch.randn((1, 1, 256, 256), device="cuda")

    ts = torch.arange(0, sde.num_diffusion_timesteps).to("cuda")[::2]
    print(ts)
    with torch.no_grad():
        sample = sampler.sample(x, y=None, ts=ts)

    print(sample.shape)
    plt.figure()
    plt.imshow(sample[0,0,:,:].cpu().numpy(), cmap="gray")
    plt.colorbar()
    plt.show()

if __name__ == "__main__":
    coordinator()