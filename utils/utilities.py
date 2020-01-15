"""
Utility functions
"""
import torch



def safe_log(z):
	return torch.log(z + 1e-7)


def load(model, optimizer, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    model.all_trained = True
    
    
def save(model, optimizer, path):
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }, path)
