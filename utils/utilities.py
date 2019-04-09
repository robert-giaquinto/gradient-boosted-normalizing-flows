"""
Utility functions
"""
import torch



def safe_log(z):
	return torch.log(z + 1e-7)
