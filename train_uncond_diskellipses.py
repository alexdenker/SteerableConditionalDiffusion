import argparse
import os

import torch
import yaml
import matplotlib.pyplot as plt 
from tqdm import tqdm 
from torch.utils.tensorboard import SummaryWriter
import torchvision
import time

from models.utils import create_model
from models.diffusion import Diffusion
from models.classifier_guidance_model import ClassifierGuidanceModel
from algos.ddim import DDIM
from ema_pytorch import EMA

from dataset.generate_ellipses import DiskDistributedEllipsesDataset

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
    device = "cuda"
    with open(os.path.join("configs", "diskellipses.yml"), "r") as f:
        model_config = yaml.safe_load(f)
    model_config = dict2namespace(model_config)
    sde = Diffusion()

    model = create_model(**vars(model_config.model))
    model.convert_to_fp32()
    model.dtype = torch.float32
    model.to(device)
    model.train()

    log_dir = os.path.join("saved_models", f'{time.strftime("%d-%m-%Y-%H-%M-%S")}')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
   
    dataset = DiskDistributedEllipsesDataset(
        fold = "train",
        shape=(256,256), 
        length=32000,
        diameter=0.4745,
        max_n_ellipse=140
    )
    
    num_epochs = 100
    batch_size = 3
    ema_decay = 0.999
    ema_warm_start_steps = 100 # only start updating ema after this amount of steps 
    log_freq = 50

    train_dl = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=1)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    ema = EMA(
        model,
        beta=ema_decay,  # exponential moving average factor
        update_after_step=ema_warm_start_steps,  # only after this number of .update() calls will it start updating
        update_every=10,  # how often to actually update, to save on compute (updates every 10th .update() call)
    )
    writer = SummaryWriter(log_dir=log_dir, comment='training-score-model')
    for epoch in range(num_epochs):

        model.train()
        avg_loss, num_items = 0, 0
        for idx, x in tqdm(enumerate(train_dl), total=len(train_dl)):
            x = x.to(device)
            optimizer.zero_grad()
            random_t = torch.randint(1, sde.num_diffusion_timesteps, (x.shape[0],), device=x.device)
            z = torch.randn_like(x)

            alpha_t = sde.alpha(random_t).view(-1, 1, 1, 1)
        
            perturbed_x = alpha_t.sqrt() * x + (1 - alpha_t).sqrt() * z

            zhat = model(perturbed_x, random_t)
            loss = torch.mean(torch.sum((z - zhat)**2, dim=(1,2,3)))
            loss.backward()
            optimizer.step() 
            avg_loss += loss.item() * x.shape[0]
            num_items += x.shape[0]
            if idx % log_freq == 0:
                writer.add_scalar('train/loss', loss.item(), epoch*len(train_dl) + idx) 
            
            ema.update()

        if epoch == num_epochs-1:
            torch.save(model.state_dict(), os.path.join(log_dir, 'model.pt'))
            torch.save(ema.ema_model.state_dict(), os.path.join(log_dir, 'ema_model.pt'))
        else:
            torch.save(model.state_dict(), os.path.join(log_dir, 'model_tmp.pt'))
            torch.save(ema.ema_model.state_dict(), os.path.join(log_dir, 'ema_model_tmp.pt'))
        
        writer.add_scalar('train/mean_loss_per_epoch', avg_loss / num_items, epoch + 1)
        model.eval()
        
        classifier_model = ClassifierGuidanceModel(model=ema.ema_model, classifier=None, diffusion=sde, cfg=None)

        cfg_sampl = {
            "algo": {"eta": 0.5, 
                    "cond_awd": False,
                    "sdedit": False}
        }
        sampl_config = dict2namespace(cfg_sampl)
        sampler = DDIM(model=classifier_model, cfg=sampl_config)
        x = torch.randn((8, 1, 256, 256), device="cuda")

        ts = torch.arange(0, sde.num_diffusion_timesteps).to("cuda")[::2]
        print("timesteps: ", ts)
        with torch.no_grad():
            sample = sampler.sample(x, y=None, ts=ts)

        sample_grid = torchvision.utils.make_grid(sample, normalize=True, scale_each=True,nrow=4)
        writer.add_image('unconditional samples (ema)', sample_grid, global_step=epoch)

if __name__ == "__main__":
    coordinator()