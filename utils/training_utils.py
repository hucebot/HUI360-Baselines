## Plotting function for debug...
import os
import sys
here = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join("..", here))

from utils.print_utils import prInfo, prSuccess, prWarning, prError, prDebug
# Import torch libraries
import torch
import torch.optim as optim
from timm.scheduler import CosineLRScheduler


def get_lr_scheduler(optimizer, config, iters_per_epoch = None):
    """Get the learning rate scheduler based on the configuration
    
    Note that lr_decay is used as a list with parameters for :
    - CosineAnnealingWarmRestarts (3 elements : initial_cycle_T, cycle_T_mult, eta_min)
    - CosineAnnealingWithWarmup (6 elements : initial_cycle_T, cycle_T_mult, eta_min, cycle_T_decay, initial_lr_before_warmup, warmup_steps)
    - iterupdate_CosineAnnealingWithWarmup (same 6 elements as CosineAnnealingWithWarmup, but parameters are
      specified in epochs and automatically converted to iterations using iters_per_epoch; stepped per iteration
      via scheduler.step_update() instead of per epoch)
    
    And lr_decay is used as a float for :
    - ExponentialDecay (gamma)
    - None (no scheduler)
    
    If initial_cycle_T == -1, then will use all epochs for CosineAnnealingWarmRestarts, CosineAnnealingWithWarmup
    and iterupdate_CosineAnnealingWithWarmup.

    Args:
        optimizer (torch.optim.Optimizer): The optimizer to schedule
        config (dict): The configuration dictionary (must contain lr_scheduler_type, lr_decay and epochs)
        iters_per_epoch (int, optional): Number of iterations per epoch (required for iterupdate_CosineAnnealingWithWarmup)

    Raises:
        ValueError: If the lr_scheduler_type is not supported

    Returns:
        torch.optim.lr_scheduler.LRScheduler: The learning rate scheduler
    """
    
    assert("epochs" in config and config["epochs"] is not None), f"epochs must be in config"
    assert("lr_scheduler_type" in config and config["lr_scheduler_type"] is not None), f"lr_scheduler_type must be in config"
    assert("lr_decay" in config and config["lr_decay"] is not None), f"lr_decay must be in config"

    if "lr_scheduler_type" in config and config["lr_scheduler_type"] is not None:
        
        if config["lr_scheduler_type"] == "ExponentialDecay":
            lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=config["lr_decay"])
            
        elif config["lr_scheduler_type"] == "CosineAnnealingWarmRestarts":
            assert(type(config["lr_decay"]) == list \
                and len(config["lr_decay"]) == 3 \
                and config["lr_decay"][0] > 2 \
                and config["lr_decay"][1] > 1 \
                and config["lr_decay"][2] >= 0), \
                f"lr_decay (used as a list with parameters for CosineAnnealingWarmRestarts) must be a list of 3 elements with T_0 > 2, T_mult > 1, and eta_min >= 0. Got {config['lr_decay']}"
            
            initial_cycle_T = config["lr_decay"][0]
            if initial_cycle_T == -1:
                prWarning("initial_cycle_T == -1, will use all epochs")
                initial_cycle_T = config["epochs"]
            cycle_T_mult = config["lr_decay"][1]
            eta_min = config["lr_decay"][2]
            
            lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=initial_cycle_T, T_mult=cycle_T_mult, eta_min=eta_min)
            
            prInfo(f"CosineAnnealingWarmRestarts scheduler created with initial_cycle_T: {initial_cycle_T}, cycle_T_mult: {cycle_T_mult}, eta_min: {eta_min}")
            
        elif config["lr_scheduler_type"] == "CosineAnnealingWithWarmup":
        
            assert(type(config["lr_decay"]) == list and len(config["lr_decay"]) == 6), f"lr_decay (used as a list with parameters for CosineAnnealingWithWarmup) must be a list of 6 elements. Got {config['lr_decay']}"
            initial_cycle_T = config["lr_decay"][0] # t_initial
            if initial_cycle_T == -1:
                prWarning("initial_cycle_T == -1, will use all epochs")
                initial_cycle_T = config["epochs"]
            cycle_T_mult = config["lr_decay"][1] # cycle_mul
            eta_min = config["lr_decay"][2] # lr_min
            cycle_T_decay = config["lr_decay"][3] # cycle_decay
            initial_lr_before_warmup = config["lr_decay"][4] # warmup_lr_init
            warmup_steps = config["lr_decay"][5] # warmup_t
        
            cycle_limit = config["epochs"] // initial_cycle_T
            if cycle_limit < 1:
                cycle_limit = 1
        
            lr_scheduler = CosineLRScheduler(
                optimizer,
                t_initial=initial_cycle_T,
                cycle_mul=cycle_T_mult,
                cycle_decay=cycle_T_decay,
                lr_min=eta_min,
                warmup_lr_init=initial_lr_before_warmup,
                warmup_t=warmup_steps,
                cycle_limit=cycle_limit,
                t_in_epochs=True,
                warmup_prefix=True if warmup_steps > 0 else False,
            )
            
            prInfo(f"CosineAnnealingWithWarmup scheduler created with initial_cycle_T: {initial_cycle_T}, cycle_T_mult: {cycle_T_mult}, eta_min: {eta_min}, cycle_T_decay: {cycle_T_decay}, initial_lr_before_warmup: {initial_lr_before_warmup}, warmup_steps: {warmup_steps}")
        
        
        elif config["lr_scheduler_type"] == "iterupdate_CosineAnnealingWithWarmup":
            assert(iters_per_epoch is not None), f"iters_per_epoch must be provided for iterupdate_CosineAnnealingWithWarmup"
            assert(type(config["lr_decay"]) == list and len(config["lr_decay"]) == 6), f"lr_decay (used as a list with parameters for iterupdate_CosineAnnealingWithWarmup) must be a list of 6 elements. Got {config['lr_decay']}"
            initial_cycle_T = config["lr_decay"][0] # in epochs
            if initial_cycle_T == -1:
                prWarning("initial_cycle_T == -1, will use all epochs")
                initial_cycle_T = config["epochs"]
            cycle_T_mult = config["lr_decay"][1]
            eta_min = config["lr_decay"][2]
            cycle_T_decay = config["lr_decay"][3]
            initial_lr_before_warmup = config["lr_decay"][4]
            warmup_steps = config["lr_decay"][5] # in epochs

            initial_cycle_T_iters = initial_cycle_T * iters_per_epoch
            warmup_steps_iters = warmup_steps * iters_per_epoch
            total_iters = config["epochs"] * iters_per_epoch

            cycle_limit = total_iters // initial_cycle_T_iters
            if cycle_limit < 1:
                cycle_limit = 1

            lr_scheduler = CosineLRScheduler(
                optimizer,
                t_initial=initial_cycle_T_iters,
                cycle_mul=cycle_T_mult,
                cycle_decay=cycle_T_decay,
                lr_min=eta_min,
                warmup_lr_init=initial_lr_before_warmup,
                warmup_t=warmup_steps_iters,
                cycle_limit=cycle_limit,
                t_in_epochs=False,
                warmup_prefix=True if warmup_steps > 0 else False,
            )

            prInfo(f"iterupdate_CosineAnnealingWithWarmup scheduler created with initial_cycle_T: {initial_cycle_T} epochs ({initial_cycle_T_iters} iters), cycle_T_mult: {cycle_T_mult}, eta_min: {eta_min}, cycle_T_decay: {cycle_T_decay}, initial_lr_before_warmup: {initial_lr_before_warmup}, warmup_steps: {warmup_steps} epochs ({warmup_steps_iters} iters), iters_per_epoch: {iters_per_epoch}")

        
        elif config["lr_scheduler_type"] == "none":
            lr_scheduler = None
            
        else:
            raise ValueError(f"LR scheduler type {config['lr_scheduler_type']} not supported")
        
    else:
        lr_scheduler = None

    return lr_scheduler
          