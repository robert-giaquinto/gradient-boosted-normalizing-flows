"""
Utility functions
"""
import torch
import torch.nn as nn
import logging
logger = logging.getLogger(__name__)


def safe_log(z):
	return torch.log(z + 1e-7)


def load(model, optimizer, path, args):
    checkpoint = torch.load(path, map_location=args.device)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])

    if 'all_trained' in checkpoint:
        model.all_trained = checkpoint['all_trained']
    else:
        model.all_trained = args.loaded_is_all_trained
        
    if 'component' in checkpoint:
        model.component = checkpoint['component']
    elif args.loaded_init_component is not None:
        logger.info(f"Initializing the loaded boosted model with component={args.loaded_init_component}")
        model.component = args.loaded_init_component        
        
    model.to(args.device)
    
    
def save(model, optimizer, path, boosted=False):
    if boosted:
        torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'all_trained': model.all_trained,
            'component': model.component
        }, path)
    else:
        torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict()
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
