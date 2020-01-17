"""
Utility functions
"""
import torch
import torch.nn as nn


def safe_log(z):
	return torch.log(z + 1e-7)


def load(model, optimizer, path, args):
    checkpoint = torch.load(path, map_location=args.device)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    model.all_trained = args.loaded_is_all_trained
    model = nn.DataParallel(model)
    model.to(args.device)
    
    
def save(model, optimizer, path):
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }, path)
