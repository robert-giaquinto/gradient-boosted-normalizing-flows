"""
Utility functions
"""
import torch
import torch.nn as nn
import os

import logging
logger = logging.getLogger(__name__)


def safe_log(z):
	return torch.log(z + 1e-7)


def load(model, optimizer, path, args, init_with_args=False):
    checkpoint = torch.load(path, map_location=args.device)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])

    # if scheduler is not None:
    #     if 'scheduler' in checkpoint and checkpoint['scheduler'] is not None:
    #         scheduler.load_state_dict(checkpoint['scheduler'])
        
    msg = f"Loaded pre-trained {os.path.split(path)[-1]}"

    if init_with_args and args.flow == "boosted":

        if args.loaded_init_component is None or args.loaded_all_trained is None:
            raise ValueError("Cannot initialize a boosted model loaded from file, intialization parameters needed.")

        model.component = args.loaded_init_component
        model.all_trained = args.loaded_all_trained
        if args.loaded_num_components is not None:
            model.num_components = args.loaded_num_components
        msg += f"  and initialized with passed argument component={model.component} (of {model.num_components}) and all_trained={str(model.all_trained)}"

    else:
        msg = f"Restoring {os.path.split(path)[-1]}"
        if 'component' in checkpoint:
            model.component = checkpoint['component']
            msg += f", and initialized with pre-saved component={model.component}  (of {model.num_components})"
        if 'all_trained' in checkpoint:
            model.all_trained = checkpoint['all_trained']
            msg += f" and all_trained={str(model.all_trained)}"

    logger.info(msg)
    model.to(args.device)
    
    
def save(model, optimizer, path, scheduler=None):
    if hasattr(model, 'component') and hasattr(model, 'all_trained'):
        torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict() if scheduler is not None else None,
            'all_trained': model.all_trained,
            'component': model.component
        }, path)
    else:
        torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict() if scheduler is not None else None,
        }, path)


class MyDataParallel(torch.nn.DataParallel):
    """
    Allow nn.DataParallel to call model's attributes.
    """
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)
