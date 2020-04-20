import argparse
import datetime
import torch
import numpy as np
import math
import random
import os
import logging
import time
from tensorboardX import SummaryWriter
from shutil import copyfile
import torch.backends.cudnn as cudnn
import torchvision.utils as tv_utils
import torch.optim as optim

from main_experiment import init_log
from utils.utilities import save, load
from utils.load_data import load_image_dataset
from utils.distributions import log_normal_diag
from utils.warmup_scheduler import GradualWarmupScheduler
from models.glow import Glow


logger = logging.getLogger(__name__)
G_MAX_LOSS = -10.0


parser = argparse.ArgumentParser(description='Gradient Boosted Flows for generative modeling of images')

parser.add_argument('--dataset', type=str, help='Dataset choice.', choices=['mnist', 'freyfaces', 'omniglot', 'caltech', 'cifar10', 'celeba'])
parser.add_argument('--experiment_name', type=str, default="density", help="A name to help identify the experiment being run when training this model.")
parser.add_argument('--manual_seed', type=int, default=123, help='manual seed, if not given resorts to random seed.')
parser.add_argument("--augment_images", action="store_true", help="Augment training images with random translations and horizontal flips")
parser.set_defaults(augment_images=False)

# gpu/cpu
#parser.add_argument('--gpu_id', type=int, default=0, metavar='GPU', help='choose GPU to run on.')
parser.add_argument('--gpu_ids', default=[0], type=eval, help='IDs of GPUs to use')
parser.add_argument('--num_workers', type=int, default=1, help='How many CPU cores to run on. Setting to 0 uses os.cpu_count() - 1.')
parser.add_argument('--no_cuda', action='store_true', default=False, help='disables CUDA training')
parser.add_argument('--no_benchmark', dest='benchmark', action='store_false', help='Turn off CUDNN benchmarking')
parser.set_defaults(benchmark=True)

parser.add_argument('--out_dir', type=str, default='./results/snapshots', help='Output directory for model snapshots etc.')
parser.add_argument('--data_dir', type=str, default='./data/raw/', help="Where raw data is saved.")
parser.add_argument('--exp_log', type=str, default='./results/image_experiment_log.txt', help='File to save high-level results from each run of an experiment.')
parser.add_argument('--print_log', dest="save_log", action="store_false", help='Add this flag to have progress printed to log (rather than saved to a file).')
parser.add_argument('--no_tensorboard', dest="tensorboard", action="store_false", help='Turns off saving results to tensorboard.')
parser.set_defaults(save_log=True)
parser.set_defaults(tensorboard=True)

# testing vs. just validation
fp = parser.add_mutually_exclusive_group(required=False)
fp.add_argument('--testing', action='store_true', dest='testing', help='evaluate on test set after training')
fp.add_argument('--validation', action='store_false', dest='testing', help='only evaluate on validation set')
parser.set_defaults(testing=True)

# optimization settings
parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train (default: 100)')
parser.add_argument('--early_stopping_epochs', type=int, default=100, help='number of early stopping epochs')
parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 64)')
parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay parameter in Adamax')
parser.add_argument("--warmup_epochs", type=int, default=5, help="Use this number of epochs to warmup learning rate linearly from zero to learning rate")
parser.add_argument("--num_init_batches", type=int,default=15, help="Number of batches to use for Act Norm initialisation")
parser.add_argument('--no_lr_schedule', action='store_true', default=False, help='Disables learning rate scheduler during training')
parser.add_argument('--lr_schedule', type=str, default=None, help="Type of LR schedule to use.", choices=['plateau', 'cosine', None])
parser.add_argument('--patience', type=int, default=5000, help='If using LR schedule, number of steps before reducing LR.')
parser.add_argument("--max_grad_clip", type=float, default=0, help="Max gradient value (clip above - for off)")
parser.add_argument("--max_grad_norm", type=float, default=50.0, help="Max norm of gradient (clip above - 0 for off)")

# model settings
parser.add_argument('--flow', type=str, default='glow', help="Type of flow to use", choices=['realnvp', 'glow', 'boosted'])
parser.add_argument("--num_flows", type=int, default=8, help="Number of flow layers per block")
parser.add_argument("--num_blocks", type=int, default=2, help="Number of blocks. Ignored for non glow models")
parser.add_argument('--h_size', type=int, default=256, help='Width of layers in base networks of realnvp and glow.')
parser.add_argument("--actnorm_scale", type=float, default=1.0, help="Act norm scale")
parser.add_argument("--flow_permutation", type=str, default="invconv", choices=["invconv", "shuffle", "reverse"], help="Type of flow permutation")
parser.add_argument("--flow_coupling", type=str, default="affine", choices=["additive", "affine"], help="Type of flow coupling")
parser.add_argument("--no_LU_decomposed", action="store_false", dest="LU_decomposed", help="Don't train with LU decomposed 1x1 convs")
parser.set_defaults(LU_decomposed=True)
parser.add_argument("--no_learn_top", action="store_false", dest="learn_top", help="Do not train top layer (prior)")
parser.set_defaults(learn_top=True)
parser.add_argument("--y_condition", action="store_true", help="Train using class condition")
parser.set_defaults(y_condition=False)
parser.add_argument("--y_multiclass", action="store_true", help="Y is a multiclass classification")
parser.set_defaults(y_multiclass=False)
parser.add_argument("--y_weight", type=float, default=0.01, help="Weight for class condition loss")

parser.add_argument('--sample_interval', type=int, default=5, help='How often (epochs) to save samples from the model')
parser.add_argument('--sample_size', type=int, default=16, help='Number of images to sample from model')
parser.add_argument("--temperature", type=float, default=1.0, help="Temperature of samples")

parser.add_argument('--num_dequant_blocks', default=0, type=int, help='Number of blocks in dequantization')
parser.add_argument('--dequant_dim', default=96, type=int, help='Number of channels in Flow++ dequantizer')
parser.add_argument('--drop_prob', type=float, default=0.2, help='Dropout probability in Flow++ dequantizer')
parser.add_argument('--use_attention', type=str2bool, default=True, help='Use attention in the coupling layers')

# Boosting parameters and optimization settings
parser.add_argument('--num_components', type=int, default=2, help='How many components are combined to form the GBF model')
parser.add_argument('--regularization_rate', type=float, default=1.0, help='Regularization penalty for boosting.')
parser.add_argument('--epochs_per_component', type=int, default=100, help='Number of epochs to train each component of a boosted model. Defaults to max(annealing_schedule, epochs_per_component). Ignored for non-boosted models.')
parser.add_argument('--rho_init', type=str, default='decreasing', choices=['decreasing', 'uniform'], help='Initialization scheme for boosted parameter rho') 
parser.add_argument('--component_type', type=str, default='glow', choices=['realnvp', 'glow'], help='Flow to boost.')



def parse_args():
    """
    Parse command line arguments and compute number of cores to use
    """
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.device = torch.device("cuda" if args.cuda else "cpu")
    if args.device == "cuda":
        cudnn.benchmark = args.benchmark
        
    args.shuffle = True
    args.batch_size *= max(1, len(args.gpu_ids))

    # Set a random seed if not given one
    if args.manual_seed is None:
        args.manual_seed = random.randint(1, 100000)

    random.seed(args.manual_seed)
    torch.manual_seed(args.manual_seed)
    torch.cuda.manual_seed_all(args.manual_seed)
    np.random.seed(args.manual_seed)

    # intialize snapshots directory for saving models and results
    args.model_signature = str(datetime.datetime.now())[0:19].replace(' ', '_').replace(':', '_').replace('-', '_')
    args.experiment_name = args.experiment_name + "_" if args.experiment_name is not None else ""
    args.snap_dir = os.path.join(args.out_dir, args.experiment_name + args.flow)
    args.snap_dir += f'_seed{args.manual_seed}_bs{args.batch_size}_K{str(args.num_flows)}_L{str(args.num_blocks)}_hdim{str(args.h_size)}'

    args.boosted = args.flow == "boosted"
    if args.boosted:
        if args.regularization_rate < 0.0:
            raise ValueError("For boosting the regularization rate should be greater than or equal to zero.")
        args.snap_dir += f'_{args.component_type}_C{str(args.num_components)}_reg{int(100*args.regularization_rate):d}'
    else:
        args.num_components = 1

    if args.flow == "realnvp" or args.component_type == "realnvp":
        args.snap_dir += f'_args.base_network_{str(args.num_base_layers)}'

    if args.flow == "glow" or args.component_type == "glow":
        args.snap_dir += f'_{args.flow_permutation}_{args.flow_coupling}'

    if args.lr_schedule is None or args.no_lr_schedule:
        args.no_lr_schedule = True
        args.lr_schedule = None
        lr_schedule = ''
    else:
        args.no_lr_schedule = False
        lr_schedule = f'LR{args.lr_schedule}'

    args.snap_dir += lr_schedule + '_on_' + args.dataset + "_" + args.model_signature + '/'
    if not os.path.exists(args.snap_dir):
        os.makedirs(args.snap_dir)

    init_log(args)
    
    # Set up multiple CPU/GPUs
    logger.info("COMPUTATION SETTINGS:")
    logger.info(f"Random Seed: {args.manual_seed}")
    if args.cuda:
        logger_msg = "Using CUDA GPU"
        #torch.cuda.set_device(args.gpu_ids)
    else:
        logger_msg = "Using CPU"
        if args.num_workers > 0:
            num_workers = args.num_workers
        else:
            num_workers = max(1, os.cpu_count() - 1)

        logger_msg += "\n\tCores available: {} (only requesting {})".format(os.cpu_count(), num_workers)
        torch.set_num_threads(num_workers)
        logger_msg += "\n\tConfirmed Number of CPU threads: {}".format(torch.get_num_threads())

    kwargs = {'num_workers': 0, 'pin_memory': True} if args.cuda else {}
    logger.info(logger_msg + "\n")
    return args, kwargs


def init_model(args):
    if args.flow == 'glow':
        model = Glow(args).to(args.device)
    # elif args.flow == 'boosted':
    #     model = GBF(args).to(args.device)
    # elif args.flow == 'realnvp':
    #     model = RealNVP(args).to(args.device)
    else:
        raise ValueError('Invalid flow choice')

    #if device == 'cuda':
    #    model = torch.nn.DataParallel(model, args.gpu_ids)

    return model



#
#  TODO UPDATE WITH MAINS INITOPTIMIZER
#
def init_optimizer(model, args):
    """
    group model parameters to more easily modify learning rates of components (flow parameters)
    """
    logger.info('OPTIMIZER:')
    warmup_mult = 1000.0
    base_lr = (args.learning_rate / warmup_mult) if args.warmup_epochs > 0 else args.learning_rate
    logger.info(f"Initializing Adamax optimizer with base learning rate={args.learning_rate}, weight decay={args.weight_decay}.")
    
    if args.flow == 'boosted':
        logger.info("For boosted model, grouping parameters according to Component Id:")

        flow_params = {f"{c}": torch.nn.ParameterList() for c in range(args.num_components)}
        flow_labels = {f"{c}": [] for c in range(args.num_components)}
        for name, param in model.named_parameters():
            pos = name.find(".")
            component_id = name[(pos + 1):(pos + 2)]
            flow_params[component_id].append(param)
            flow_labels[component_id].append(name)

        # collect all parameters into a single list
        # the first args.num_components elements in the parameters list correspond boosting parameters
        all_params = []
        for c in range(args.num_components):
            all_params.append(flow_params[f"{c}"])
            logger.info(f"Grouping [{', '.join(flow_labels[str(c)])}] as Component {c}'s parameters.")

        optimizer = optim.Adamax([{'params': param_group} for param_group in all_params], lr=base_lr, weight_decay=args.weight_decay)
    else:
        optimizer = optim.Adamax(model.parameters(), lr=base_lr, weight_decay=args.weight_decay)

    if args.no_lr_schedule:
        scheduler = None
    else:
        if args.lr_schedule == "plateau":
            logger.info(f"Using ReduceLROnPlateua as a learning-rate schedule, reducing LR by 0.5 after {args.patience} steps until it reaches 1e-5.")
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                                   factor=0.5,
                                                                   patience=args.patience,
                                                                   min_lr=1e-5,
                                                                   verbose=True,
                                                                   threshold_mode='abs')
        elif args.lr_schedule == "cosine":
            logger.info(f"Using CosineAnnealingLR as a learning-rate schedule, annealed over {args.epochs * args.train_size} training steps.")
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs * args.train_size)

    if args.warmup_epochs > 0:
        logger.info(f"Gradually warming up learning rate from {base_lr} to {args.learning_rate} over the first {args.warmup_epochs * args.train_size} steps.\n")
        warmup_scheduler = GradualWarmupScheduler(optimizer, multiplier=warmup_mult, total_epoch=args.warmup_epochs * args.train_size, after_scheduler=scheduler)
        return optimizer, warmup_scheduler
    else:
        return optimizer, scheduler


def compute_loss(z, z_mu, z_var, logdet, y, y_logits, dim_prod, args):
    reduction='mean'
    
    # Full objective - converted to bits per dimension
    nll = -1.0 * (log_normal_diag(z, z_mu, z_var, dim=[1,2,3]) + logdet)
    bpd = nll / (math.log(2.) * dim_prod)
    losses = {"bpd": torch.mean(bpd)}

    if args.y_condition:
        if args.multi_class:
            y_logits = torch.sigmoid(y_logits)
            loss_classes = F.binary_cross_entropy_with_logits(y_logits, y, reduction=reduction)
        else:
            loss_classes = F.cross_entropy(y_logits, torch.argmax(y, dim=1), reduction=reduction)

        losses["loss_classes"] = loss_classes
        losses["total_loss"] = losses["bpd"] + args.y_weight * loss_classes
        
    else:
        losses["total_loss"] = losses["bpd"]

    return losses


def compute_boosted_loss(z_mu, z_var, zg, g_ldj, zG, G_ldj, y, y_logits, dim_prod, args):
    reduction='mean'
    
    # Full objective - converted to bits per dimension
    g_lhood = log_normal_diag(zg, z_mu, z_var, dim=[1,2,3]) + g_ldj
    G_lhood = torch.max(log_normal_diag(zG, z_mu, z_var, dim=[1,2,3]) + G_ldj, torch.ones_like(G_ldj * G_MAX_LOSS))
    nll = -1.0 * g_lhood + G_lhood
    bpd = nll / (math.log(2.) * dim_prod)

    losses = {"g_nll": torch.mean(-1.0 * g_lhood)}
    losses = {"G_nll": torch.mean(-1.0 * G_lhood)}
    losses = {"nll": torch.mean(nll)}
    losses = {"bpd": torch.mean(bpd)}

    if args.y_condition:
        if args.multi_class:
            y_logits = torch.sigmoid(y_logits)
            loss_classes = F.binary_cross_entropy_with_logits(y_logits, y, reduction=reduction)
        else:
            loss_classes = F.cross_entropy(y_logits, torch.argmax(y, dim=1), reduction=reduction)

        losses["loss_classes"] = loss_classes
        losses["total_loss"] = losses["bpd"] + args.y_weight * loss_classes
        
    else:
        losses["total_loss"] = losses["bpd"]

    return losses
    


def sample(model, args, step=None):
    with torch.no_grad():
        if args.y_condition:
            y = torch.eye(args.y_classes)
            y = y.repeat(args.batch_size // args.y_classes + 1)
            y = y[:args.sample_size, :].to(args.device)
        else:
            y = None

        images = model(y_onehot=y, temperature=args.temperature, reverse=True)

        nrow = int(np.floor(np.sqrt(args.sample_size)))
        fname = f'samples_step{step}.png' if step is not None else 'samples.png'
        tv_utils.save_image(tv_utils.make_grid(images.cpu(), nrow=nrow), args.snap_dir + fname)


def evaluate(model, data_loader, args):
    model.eval()

    loss = 0.0
    with torch.no_grad():
        for batch_id, (x, y) in enumerate(data_loader):
            x = x.to(args.device)
            if args.y_condition:
                y = y.to(device)
            else:
                y = None

            z, z_mu, z_var, logdet, y_logits = model(x, y)
            losses = compute_loss(z, z_mu, z_var, logdet, y, y_logits, np.prod(x.shape[1:]), args)
            loss += losses['total_loss'].item()

    avg_loss = loss / len(data_loader)
    return avg_loss
    

def train(model, train_loader, val_loader, optimizer, scheduler, args):
    """
    TODO add timings for training
    """
    if args.tensorboard:
        writer = SummaryWriter(args.snap_dir)

    step = 0
        
    header_msg = f'| Epoch | {"TRAIN": <14}{"Loss": >4} {"Time": >12} | {"VALIDATION": <14}{"Loss": >4} | '
    header_msg += f'{"Component": >10} | {"All Trained": >12} | {"Rho": >32} | ' if args.boosted else ''
    header_msg += f'{"Improved": >10} |'
    logger.info('|' + "-"*(len(header_msg)-2) + '|')
    logger.info(header_msg)
    logger.info('|' + "-"*(len(header_msg)-2) + '|')
    
    best_loss = np.array([np.inf] * args.num_components)
    early_stop_count = 0
    converged_epoch = 0  # corrects the annealing schedule when a boosted component converges early
    if args.boosted:
        model.component = 0
        prev_lr = []
        for c in range(args.num_components):
            optimizer.param_groups[c]['lr'] = args.learning_rate if c == model.component else 0.0
            prev_lr.append(args.learning_rate)
        for n, param in model.named_parameters():
            param.requires_grad = True if n.startswith(f"flow_param.{model.component}") else False

    for epoch in range(1, args.epochs + 1):

        model.train()
        train_loss = []
        train_times = []
        
        for batch_id, (x, y) in enumerate(train_loader):

            t_start = time.time()
            optimizer.zero_grad()
            x = x.to(args.device)

            if args.y_condition:
                y = y.to(device)
            else:
                y = None

            # initialize ActNorm on first step
            if step < args.num_init_batches:
                with torch.no_grad():
                    model(x, y)
                    step += 1
                    continue

            if args.boosted:
                z_mu, z_var, zg, g_ldj, zG, G_ldj, y_logits = model(x, y)
                losses = compute_boosted_loss(z_mu, z_var, zg, g_ldj, zG, G_ldj, y, y_logits, np.prod(x.shape[1:]), args)
            else:
                z, z_mu, z_var, logdet, y_logits = model(x, y)
                losses = compute_loss(z, z_mu, z_var, logdet, y, y_logits, np.prod(x.shape[1:]), args)
            
            losses["total_loss"].backward()
            if args.max_grad_clip > 0:
                torch.nn.utils.clip_grad_value_(model.parameters(), args.max_grad_clip)
            if args.max_grad_norm > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                writer.add_scalar("grad_norm/grad_norm", grad_norm, step)

            if args.boosted:
                    prev_lr[model.component] = optimizer.param_groups[model.component]['lr']
            if args.tensorboard:
                for i in range(len(optimizer.param_groups)):
                    writer.add_scalar(f'lr/lr_{i}', optimizer.param_groups[i]['lr'], step)
            optimizer.step()
            if not args.no_lr_schedule:
                if args.lr_schedule == "plateau":
                    scheduler.step(metrics=losses['total_loss'])
                else:
                    scheduler.step()

            train_times.append(time.time() - t_start)
            train_loss.append(losses['total_loss'])
            if args.tensorboard:
                writer.add_scalar('step_loss/total_loss', losses['total_loss'].item(), step)
                writer.add_scalar('step_loss/bpd', losses['bpd'].item(), step)
                if args.y_condition:
                    writer.add_scalar('step_loss/loss_classes', losses['loss_classes'].item(), step)
                
            step += 1

        # Validation
        val_loss = evaluate(model, val_loader, args)

        # Sampling
        if epoch == 1 or epoch % args.sample_interval == 0:
            sample(model, args, step=step)

        # Reporting
        train_times = np.array(train_times)
        train_loss = torch.stack(train_loss).mean().item()
        epoch_msg = f'| {epoch: <5} | {train_loss:18.3f} {np.mean(train_times):12.1f} | {val_loss:18.3f} | '
        rho_str = '[' + ', '.join([f"{val:4.2f}" for val in model.rho.data]) + ']' if args.boosted else ''
        epoch_msg += f'{model.component: >10} | {str(model.all_trained)[0]: >12} | {rho_str: >32} | ' if args.boosted else ''
        if args.tensorboard:
            writer.add_scalar('epoch_loss/validation', val_loss, epoch)
            writer.add_scalar('epoch_loss/train', train_loss, epoch)

        # Assess convergence
        component = model.component if args.boosted else 0
        converged, model_improved, early_stop_count, best_loss = check_convergence(
            early_stop_count, val_loss, best_loss, epoch - converged_epoch, component, args)

        epoch_msg += f'{"T" if model_improved else "": >10}'
        if model_improved:
            fname = f'model_c{model.component}.pt' if args.boosted else 'model.pt'
            save(model, optimizer, args.snap_dir + fname, scheduler)

        if converged:
            logger.info(epoch_msg + ' |')

            if args.boosted:
                converged_epoch = epoch
                prev_lr[model.component] = optimizer.param_groups[model.component]['lr']  # save LR for LR scheduler in case we train this component again

                # revert back to the last best version of the model and update rho
                load(model, optimizer, args.snap_dir + f'model_c{model.component}.pt', args)
                model.update_rho(train_loader)
                if model.component > 0 or model.all_trained:
                    logger.info('Rho Updated: ' + ' '.join([f"{val:1.2f}" for val in model.rho.data]))

                train_components_once = args.epochs <= (args.epochs_per_component * args.num_components)
                if model.component == (args.num_components - 1) and (model.all_trained or train_components_once):
                    # stop the full model after all components have been trained
                    logger.info(f"Model converged, stopping training and saving final model to: {args.snap_dir + 'model.pt'}")
                    model.all_trained = True
                    save(model, optimizer, args.snap_dir + f'model.pt', scheduler)
                    break

                # else if not done training:
                # save model with updated rho
                save(model, optimizer, args.snap_dir + f'model_c{model.component}.pt', scheduler)
                
                # reset early_stop_count and train the next component
                model.increment_component()
                early_stop_count = 0
                # freeze all but the new component being trained
                for c in range(args.num_components):
                    optimizer.param_groups[c]['lr'] = prev_lr[c] if c == model.component else 0.0
                for n, param in model.named_parameters():
                    param.requires_grad = True if n.startswith(f"flow_param.{model.component}") or not n.startswith("flow_param") else False
            else:
                # if a standard model converges once, break
                logger.info(f"Model converged, stopping training.")
                break
                
        else:
            logger.info(epoch_msg + ' |')
            if epoch == args.epochs:
                if args.boosted:
                    # Save the best version of the model trained up to the current component with filename model.pt
                    # This is to protect against times when the model is trained/re-trained but doesn't run long enough
                    #   for all components to converge / train completely
                    copyfile(args.snap_dir + f'model_c{model.component}.pt', args.snap_dir + 'model.pt')
                    logger.info(f"Resaving last improved version of {f'model_c{model.component}.pt'} as 'model.pt' for future testing")
                else:
                    logger.info(f"Stopping training after {epoch} epochs of training.")

    logger.info('|' + "-"*(len(header_msg)-2) + '|\n\n')
    writer.close()
                    

def check_convergence(early_stop_count, v_loss, best_loss, epochs_since_prev_convergence, component, args):
    """
    Verify if a boosted component has converged
    """
    if args.flow == "boosted":
        # Consider the boosted model's component as converged if a pre-set number of epochs have elapsed
        time_to_update = epochs_since_prev_convergence % args.epochs_per_component == 0
    else:
        time_to_update = False

    model_improved = v_loss < best_loss[component]
    early_stop_flag = False
    if v_loss < best_loss[component]:
        early_stop_count = 0
        best_loss[component] = v_loss
    elif args.early_stopping_epochs > 0:
        # model didn't improve, do we consider it converged yet?
        early_stop_count += 1        
        early_stop_flag = early_stop_count > args.early_stopping_epochs

    converged = early_stop_flag or time_to_update
    return converged, model_improved, early_stop_count, best_loss



def main():
    # =========================================================================
    # PARSE EXPERIMENT SETTINGS, SETUP SNAPSHOTS DIRECTORY, LOGGING
    # =========================================================================
    args, kwargs = parse_args()

    # =========================================================================
    # LOAD DATA
    # =========================================================================
    logger.info('LOADING DATA:')
    train_loader, val_loader, test_loader, args = load_image_dataset(args, **kwargs)

    # =========================================================================
    # SAVE EXPERIMENT SETTINGS
    # =========================================================================
    logger.info(f'EXPERIMENT SETTINGS:\n{args}\n')
    torch.save(args, os.path.join(args.snap_dir, 'config.pt'))

    # =========================================================================
    # INITIALIZE MODEL AND OPTIMIZATION
    # =========================================================================
    model = init_model(args)
    optimizer, scheduler = init_optimizer(model, args)
    num_params = sum([param.nelement() for param in model.parameters()])
    logger.info(f"MODEL:\nNumber of model parameters={num_params}\n{model}\n")

    # =========================================================================
    # TRAINING
    # =========================================================================
    logger.info('TRAINING:')
    train(model, train_loader, val_loader, optimizer, scheduler, args)

    # =========================================================================
    # VALIDATION
    # =========================================================================
    logger.info('VALIDATION:')
    val_loss = evaluate(model, val_loader, args)

    # =========================================================================
    # TESTING
    # =========================================================================
    if args.testing:
        logger.info("TESTING:")



if __name__ == "__main__":
    main()









