"""
Utility functions
"""
import torch
import torch.nn as nn


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
