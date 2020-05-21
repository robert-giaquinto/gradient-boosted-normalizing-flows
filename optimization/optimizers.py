import torch
import torch.nn as nn
import torch.utils.data
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau

import logging

logger = logging.getLogger(__name__)


def init_optimizer(model, args, verbose=True):
    """
    group model parameters to more easily modify learning rates of components (flow parameters)
    """
    if verbose:
        logger.info('OPTIMIZER:')
        logger.info(f"Initializing {args.optimizer} optimizer with base learning rate={args.learning_rate}, weight decay={args.weight_decay}.")
    
    if args.flow == 'boosted':
        if verbose:
            logger.info("For boosted model, grouping parameters according to Component Id:")
            
        flow_params = {f"{c}": torch.nn.ParameterList() for c in range(args.num_components)}
        flow_labels = {f"{c}": [] for c in range(args.num_components)}
        vae_params = torch.nn.ParameterList()
        vae_labels = []
        for name, param in model.named_parameters():
            if name.startswith("flow"):
                pos1 = name.find(".") + 1
                pos2 = name[(pos1):].find(".") + pos1
                component_id = name[pos1:pos2]
                flow_params[component_id].append(param)
                flow_labels[component_id].append(name)
            else:
                vae_labels.append(name)
                vae_params.append(param)

        # collect all parameters into a single list
        # the first args.num_components elements in the parameters list correspond boosting parameters
        all_params = []
        for c in range(args.num_components):
            all_params.append(flow_params[f"{c}"])
            if verbose:
                logger.info(f"Grouping [{', '.join(flow_labels[str(c)])}] as Component {c}'s parameters.")

        # vae parameters are at the end of the list (may not exist if doing density estimation)
        if len(vae_params) > 0:
            all_params.append(vae_params)
            if verbose:
                logger.info(f"Grouping [{', '.join(vae_labels)}] as the VAE parameters.\n")

        if args.optimizer.lower() == "sgd":
            optimizer = optim.SGD([{'params': param_group} for param_group in all_params], lr=args.learning_rate, weight_decay=args.weight_decay)
        else:
            optimizer = optim.AdamW([{'params': param_group} for param_group in all_params], lr=args.learning_rate, weight_decay=args.weight_decay)
    else:
        if verbose:
            logger.info(f"Initializing {args.optimizer} optimizer for standard models with learning rate={args.learning_rate}.\n")

        if args.optimizer.lower() == "sgd":
            optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        else:
            optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    if args.no_lr_schedule:
        scheduler = None
        
    else:

        epochs = args.epochs_per_component if args.boosted else args.epochs
        if args.min_lr is None:
            raise ValueError("Must specify a min_lr for lr_schedules")
        
        if args.lr_schedule == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                                   factor=0.5,
                                                                   patience=args.patience * args.train_size,
                                                                   min_lr=args.min_lr,
                                                                   verbose=True,
                                                                   threshold_mode='abs')
            if verbose:
                logger.info(f"Using ReduceLROnPlateua as a learning-rate schedule, reducing LR by 0.5 after {args.patience * args.train_size} epochs until it reaches {args.min_lr}.")                
            
        elif args.lr_schedule == "cosine":
            msg = "Using a cosine annealing learning-rate schedule, "
            if args.lr_restarts > 1:
                steps_per_cycle = int(epochs / args.lr_restarts) * args.train_size
                scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                                                 T_0=steps_per_cycle,
                                                                                 eta_min=args.min_lr)
                msg += f"annealed over {steps_per_cycle} training steps ({int(epochs / args.lr_restarts)} epochs), restarting {args.lr_restarts} times within each learning cycle."

            else:
                total_steps = epochs * args.train_size
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                                       T_max=total_steps,
                                                                       eta_min=args.min_lr)
                msg += f"annealed over {total_steps} training steps ({epochs} epochs), restarting with each new component (if boosted)."

            if verbose:
                logger.info(msg)

        elif args.lr_schedule == "test":
            # LR range test, make sure optimizer is SGD
            args.warmup_epochs = 0  # no warmup allowed with LR range test
            steps = epochs * args.train_size
            scheduler = ExponentialLR(optimizer, init_lr=args.min_lr, num_steps=steps)
            if verbose:
                logger.info(f"Running LR range test: initial LR of {args.min_lr} increasing exponentially to {args.learning_rate} over {steps} steps.")

        elif args.lr_schedule == "cyclic":
            args.warmup_epochs = 0  # no warmup allowed with cyclic lr
            if args.lr_restarts > 1:
                # Cyclic Learning Rates
                steps_per_half_cycle = int(epochs / (2 * args.lr_restarts)) * args.train_size
                scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=args.min_lr, cycle_momentum=args.optimizer == "sgd",
                                                             max_lr=args.learning_rate, mode="triangular2", step_size_up=steps_per_half_cycle)
                if verbose:
                    logger.info(f"Using cyclic learning rate schedule between {args.min_lr} and {args.learning_rate}, max lr cut in half after each {2 * steps_per_half_cycle} steps ({epochs / args.lr_restarts} epochs)")

            else:
                # One Cycle Super Convergence
                scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.learning_rate,
                                                               steps_per_epoch=args.train_size, epochs=epochs)
                if verbose:
                    logger.info(f"Using one cyclic learning rate schedule with max LR of {args.learning_rate}")                 

    if args.warmup_epochs > 0:
        if verbose:
            logger.info(f"Gradually warming up learning rate from 0.0 to {args.learning_rate} over the first {args.warmup_epochs * args.train_size} steps.\n")

        warmup_steps = args.warmup_epochs * args.train_size
        warmup_scheduler = GradualWarmupScheduler(optimizer, total_epoch=warmup_steps, after_scheduler=scheduler)
        return optimizer, warmup_scheduler
    else:
        return optimizer, scheduler


class GradualWarmupScheduler(_LRScheduler):
    """
    Gradually warm-up(increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.

    From: https://github.com/ildoonet/pytorch-gradual-warmup-lr
    
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier: target learning rate = base lr * multiplier if multiplier > 1.0. if multiplier = 1.0, lr starts from 0 and ends up with the base_lr.
        total_epoch: target learning rate is reached at total_epoch, gradually
        after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
    """

    def __init__(self, optimizer, total_epoch, after_scheduler=None, multiplier=1.0):
        self.multiplier = multiplier
        if self.multiplier < 1.:
            raise ValueError('multiplier should be greater thant or equal to 1.')
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super(GradualWarmupScheduler, self).__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]

    def step_ReduceLROnPlateau(self, metrics, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch if epoch != 0 else 1  # ReduceLROnPlateau is called at the end of epoch, whereas others are called at beginning
        if self.last_epoch <= self.total_epoch:
            warmup_lr = [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]
            for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
                param_group['lr'] = lr
        else:
            if epoch is None:
                self.after_scheduler.step(metrics, None)
            else:
                self.after_scheduler.step(metrics, epoch - self.total_epoch)

    def step(self, epoch=None, metrics=None):
        if type(self.after_scheduler) != ReduceLROnPlateau:
            if self.finished and self.after_scheduler:
                if epoch is None:
                    self.after_scheduler.step(None)
                else:
                    self.after_scheduler.step(epoch - self.total_epoch)
                    #self.after_scheduler.step(epoch)
            else:
                return super(GradualWarmupScheduler, self).step(epoch)
        else:
            self.step_ReduceLROnPlateau(metrics, epoch)


class ExponentialLR(_LRScheduler):
    """Exponentially increases the learning rate between two boundaries over a number of
    iterations.
    
    Arguments:
        optimizer (torch.optim.Optimizer): wrapped optimizer.
        init_lr (float): the initial learning rate which is the lower boundary of the test.
        num_steps (int): the number of steps over which the test occurs.
        last_epoch (int): the index of last epoch. Default: -1.

    """

    def __init__(self, optimizer, init_lr, num_steps, last_epoch=-1):
        self.init_lr = init_lr
        self.num_steps = num_steps
        super(ExponentialLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        curr_step = self.last_epoch + 1
        r = curr_step / self.num_steps
        return [self.init_lr *  (base_lr / self.init_lr) ** r for base_lr in self.base_lrs]
