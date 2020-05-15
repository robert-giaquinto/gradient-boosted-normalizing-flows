"""
Utility functions
"""
import torch
import torch.nn as nn
import os

import logging
logger = logging.getLogger(__name__)


def softmax(x):
    e_x = torch.exp(x - torch.max(x))
    return e_x / torch.sum(e_x)


def safe_log(z):
	return torch.log(z + 1e-7)


def init_log(args):
    log_format = '%(asctime)s : %(message)s'
    filename = os.path.join(args.snap_dir, "log.txt")
    print(f"Saving log output to file: {filename}")

    if args.print_log:
        #logging.basicConfig(filename=filename, format=log_format, datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)
        handlers = [
            logging.FileHandler(filename),
            logging.StreamHandler()
        ]
    else:
        #logging.basicConfig(format=log_format, level=logging.INFO)
        handlers = [logging.FileHandler(filename)]
        
    logging.basicConfig(level=logging.INFO,
                        format=log_format,
                        datefmt='%Y-%m-%d %H:%M:%S',
                        handlers=handlers)


def load(model, optimizer, path, args, init_with_args=False, scheduler=None, verbose=True):
    checkpoint = torch.load(path, map_location=args.device)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])

    if scheduler is not None:
        if 'scheduler' in checkpoint and checkpoint['scheduler'] is not None:
            scheduler.load_state_dict(checkpoint['scheduler'])
        
    msg = f"Loaded pre-trained {os.path.split(path)[-1]}"

    if init_with_args and args.boosted:

        if args.loaded_init_component is None or args.loaded_all_trained is None:
            raise ValueError("Cannot initialize a boosted model loaded from file, intialization parameters needed.")

        model.component = args.loaded_init_component
        model.all_trained = args.loaded_all_trained
        if args.loaded_num_components is not None:
            model.num_components = args.loaded_num_components
        msg += f"  and initialized with passed argument component={model.component} (of {list(range(model.num_components))}) and all_trained={str(model.all_trained)}"

    else:
        msg = f"Restoring {os.path.split(path)[-1]}"
        if 'component' in checkpoint:
            model.component = checkpoint['component']
            msg += f", and initialized with pre-saved component={model.component}  (of {list(range(model.num_components))})"
        if 'all_trained' in checkpoint:
            model.all_trained = checkpoint['all_trained']
            msg += f" and all_trained={str(model.all_trained)}"

    model.to(args.device)
    if verbose:
        logger.info(msg)

    
    
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


def squeeze2d(input, factor):
    if factor == 1:
        return input

    B, C, H, W = input.size()

    assert H % factor == 0 and W % factor == 0, "H or W modulo factor is not 0"

    x = input.view(B, C, H // factor, factor, W // factor, factor)
    x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
    x = x.view(B, C * factor * factor, H // factor, W // factor)

    return x


def unsqueeze2d(input, factor):
    if factor == 1:
        return input

    factor2 = factor ** 2

    B, C, H, W = input.size()

    assert C % (factor2) == 0, "C module factor squared is not 0"

    x = input.view(B, C // factor2, factor, factor, H, W)
    x = x.permute(0, 1, 4, 2, 5, 3).contiguous()
    x = x.view(B, C // (factor2), H * factor, W * factor)

    return x


def split_feature(tensor, type="split"):
    """
    type = ["split", "cross"]
    """
    image_input = len(tensor.size()) > 2

    C = tensor.size(1)
    if image_input:
        if type == "split":
            return tensor[:, :C // 2, ...], tensor[:, C // 2:, ...]
        elif type == "cross":
            return tensor[:, 0::2, ...], tensor[:, 1::2, ...]

    else:
        if type == "split":
            return tensor[:, :C // 2], tensor[:, C // 2:]
        elif type == "cross":
            return tensor[:, 0::2], tensor[:, 1::2]


def compute_same_pad(kernel_size, stride):
    if isinstance(kernel_size, int):
        kernel_size = [kernel_size]

    if isinstance(stride, int):
        stride = [stride]

    assert len(stride) == len(kernel_size),\
        "Pass kernel size and stride both as int, or both as equal length iterable"

    return [((k - 1) * s + 1) // 2 for k, s in zip(kernel_size, stride)]


def pixels(tensor):
    return int(tensor.size(2) * tensor.size(3))

