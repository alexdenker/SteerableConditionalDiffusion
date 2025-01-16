from typing import Optional
import torch 
import torch.nn as nn
from torch import Tensor

from adapt.lora import inject_trainable_lora_extended

import itertools

def tv_loss(x):

    dh = torch.abs(x[..., :, 1:] - x[..., :, :-1])
    dw = torch.abs(x[..., 1:, :] - x[..., :-1, :])
    return torch.mean(dh[..., :-1, :] + dw[..., :, :-1])

def _tune_lora_scale(score, scale: float = 1.0):
    for _module in score.modules():
        if _module.__class__.__name__ in [
            "LoraInjectedLinear",
            "LoraInjectedConv2d",
            "LoraInjectedConv1d",
        ]:
            _module.scale = scale


def _score_model_adpt(
    score: nn.Module, 
    im_size: Optional[int] = None, 
    method: str = 'lora',
    r: int = 4,
    ) -> None:
    
    for name, param in score.named_parameters():
        param.requires_grad = False

    if method == 'full':
        for name, param in score.named_parameters():
            if not "time_embed" in name:
                param.requires_grad = True

        new_num_params = sum([p.numel() for p in score.parameters()])
        trainable_params = sum([p.numel() for p in score.parameters() if p.requires_grad])
        print(f'% of trainable params: {trainable_params/new_num_params*100}')
    elif method == 'decoder': 
        for name, param in score.out.named_parameters():
            if not "emb_layers" in name and not "time_embed" in name:
                param.requires_grad = True
        for name, param in score.output_blocks.named_parameters():
            if not "emb_layers" in name and not "time_embed" in name:
                param.requires_grad = True
        
        new_num_params = sum([p.numel() for p in score.parameters()])
        trainable_params = sum([p.numel() for p in score.parameters() if p.requires_grad])
        print(f'% of trainable params: {trainable_params/new_num_params*100}')
    elif method == 'lora':
        """ 
        Implement LoRA: https://arxiv.org/pdf/2106.09685.pdf 

        Adding LoRA modules to nn.Conv1d, nn.Conv2d 
        
        """
        score.requires_grad_(False)

        require_grad_params, names = inject_trainable_lora_extended(score, r=r) 

        new_num_params = sum([p.numel() for p in score.parameters()])
        trainable_params = sum([p.numel() for p in score.parameters() if p.requires_grad])
        print(f'% of trainable params: {trainable_params/new_num_params*100}')

        return score, require_grad_params
        
    else: 
        raise NotImplementedError
    
    return score