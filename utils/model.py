import os
import json

import torch
import numpy as np

from model import CVAEJETSSynthesizer, MultiPeriodDiscriminator


def load_checkpoint(model, saved_state_dict):
    state_dict = model.state_dict()
    new_state_dict = {}
    for k, v in state_dict.items():
        try:
            new_state_dict[k] = saved_state_dict[k]
        except:
            print("%s is not in the checkpoint" % k)
            new_state_dict[k] = v
    model.load_state_dict(new_state_dict)

def get_model(args, configs, device, train=False):
    (preprocess_config, model_config, train_config) = configs

    model = CVAEJETSSynthesizer(preprocess_config, model_config, train_config).to(device)
    if args.restore_step:
        ckpt_path = os.path.join(
            train_config["path"]["ckpt_path"],
            "{}.pth.tar".format(args.restore_step),
        )
        ckpt = torch.load(ckpt_path, map_location=device)
        load_checkpoint(model, ckpt["model"])

    if train:
        discriminator = MultiPeriodDiscriminator().to(device)
        
        model_optimizer = torch.optim.AdamW(
            model.parameters(), 
            train_config["optimizer"]["learning_rate"], 
            betas=train_config["optimizer"]["betas"], 
            eps=train_config["optimizer"]["eps"])
        discriminator_optimizer = torch.optim.AdamW(
            discriminator.parameters(),
            train_config["optimizer"]["learning_rate"], 
            betas=train_config["optimizer"]["betas"], 
            eps=train_config["optimizer"]["eps"])

        if args.restore_step:
            discriminator.load_state_dict(ckpt["discriminator"])
            model_optimizer.load_state_dict(ckpt["model_optimizer"])
            discriminator_optimizer.load_state_dict(ckpt["discriminator_optimizer"])
            iteration = ckpt['iteration']
        else:
            iteration = 1

        scheduler_model = torch.optim.lr_scheduler.ExponentialLR(
            model_optimizer, gamma=train_config["optimizer"]["lr_decay"], last_epoch=iteration-2)
        scheduler_discriminator = torch.optim.lr_scheduler.ExponentialLR(
            discriminator_optimizer, gamma=train_config["optimizer"]["lr_decay"], last_epoch=iteration-2)
            
        model.train()
        discriminator.train()
        return model, discriminator, model_optimizer, discriminator_optimizer, scheduler_model, scheduler_discriminator, iteration

    model.eval()
    model.requires_grad_ = False
    return model


def get_param_num(model):
    num_param = sum(param.numel() for param in model.parameters())
    return num_param
