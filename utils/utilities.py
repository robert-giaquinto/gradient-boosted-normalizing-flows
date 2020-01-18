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
    if type(checkpoint) is dict:
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
    else:
        model = checkpoint
    
    model.all_trained = args.loaded_is_all_trained
    if args.loaded_init_component is not None:
        logger.info(f"Initializing the loaded boosted model with component={args.loaded_init_component}")
        model.component = args.loaded_init_component
    elif args.epochs > 0:
        logger.info("Loaded model's component attribute is intialized to zero (by default), this will be the first component trained")
        
    model.to(args.device)
    
    
def save(model, optimizer, path):
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
