import argparse
import datetime
import torch
import torch.nn as nn
import torch.utils.data
import torch.optim as optim
import numpy as np
import math
import random
import os
import logging
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter

from models.vae import VAE
from models.sylvester import OrthogonalSylvesterVAE, HouseholderSylvesterVAE, TriangularSylvesterVAE
from models.iaf import IAFVAE
from models.realnvp import RealNVPVAE
from models.boosted_vae import BoostedVAE
from models.planar import PlanarVAE
from models.radial import RadialVAE
from models.liniaf import LinIAFVAE
from models.affine import AffineVAE
from models.nlsq import NLSqVAE

from optimization.training import train
from optimization.evaluation import evaluate, evaluate_likelihood
from optimization.optimizers import init_optimizer

from utils.load_data import load_image_dataset
from utils.utilities import load, save, init_log


logger = logging.getLogger(__name__)


parser = argparse.ArgumentParser(description='PyTorch Ensemble Normalizing flows')
parser.add_argument('--experiment_name', type=str, default=None,
                    help="A name to help identify the experiment being run when training this model.")
parser.add_argument('--dataset', type=str, default='mnist', help='Dataset choice.',
                    choices=['mnist', 'freyfaces', 'omniglot', 'caltech', 'cifar10'])

parser.add_argument('--manual_seed', type=int, default=123, help='manual seed, if not given resorts to random seed.')
parser.add_argument('--freyseed', type=int, default=123, help="Seed for shuffling frey face dataset for test split.")

# gpu/cpu
parser.add_argument('--gpu_id', type=int, default=0, help='choose GPU to run on.')
parser.add_argument('--num_workers', type=int, default=1,
                    help='How many CPU cores to run on. Setting to 0 uses os.cpu_count() - 1.')
parser.add_argument('--no_cuda', action='store_true', default=False, help='disables CUDA training')
parser.add_argument('--no_benchmark', dest='benchmark', action='store_false', help='Turn off CUDNN benchmarking')
parser.set_defaults(benchmark=True)

# Reporting
parser.add_argument('--plot_interval', type=int, default=10, help='Number of epochs between reconstructions plots.')
parser.add_argument('--out_dir', type=str, default='./results/snapshots', help='Output directory for model snapshots etc.')
parser.add_argument('--data_dir', type=str, default='./data/raw/', help="Where raw data is saved.")
parser.add_argument('--exp_log', type=str, default='./results/experiment_log.txt', help='File to save high-level results from each run of an experiment.')
parser.add_argument('--save_intermediate_checkpoints', dest="save_intermediate_checkpoints",
                    action="store_true", help='Save versions of the boosted model after each component converges.')
parser.add_argument('--print_log', dest="print_log", action="store_true", help='Add this flag to have progress printed to log (rather just than saved to a file).')
parser.add_argument('--no_tensorboard', dest="tensorboard", action="store_false", help='Turns off saving results to tensorboard.')
parser.set_defaults(print_log=False)
parser.set_defaults(tensorboard=True)

parser.add_argument('--load', type=str, default=None, help='Path to load the model from')
at = parser.add_mutually_exclusive_group(required=False)
at.add_argument('--loaded_is_not_all_trained', action="store_false", dest='loaded_all_trained',
                help="Set this if you don't want the loaded boosted model to be consider all_trained (default=True)")
at.add_argument('--loaded_is_all_trained', action='store_true', dest='loaded_all_trained',
                help='Default setting, which assumes the loaded boosted model is all_trained.')
parser.set_defaults(loaded_all_trained=None)
parser.add_argument('--loaded_init_component', default=None, type=int, help='Boosted component to begin training on first from a loaded model.')
parser.add_argument('--loaded_num_components', default=None, type=int,
                    help='If loading a boosted model, this will limit the loaded model to only use the first "loaded_num_components" components in all tasks.')
parser.add_argument('--init_epoch', default=1, type=int, help='Epoch number to start at (helpful if loading a model')

sr = parser.add_mutually_exclusive_group(required=False)
sr.add_argument('--save_results', action='store_true', dest='save_results', help='Save results from experiments.')
sr.add_argument('--discard_results', action='store_false', dest='save_results', help='Do NOT save results from experiments.')
parser.set_defaults(save_results=True)

# testing vs. just validation
fp = parser.add_mutually_exclusive_group(required=False)
fp.add_argument('--testing', action='store_true', dest='testing', help='evaluate on test set after training')
parser.add_argument('--nll_samples', type=int, default=2000, help='Number of samples to use in evaluating NLL')
parser.add_argument('--nll_mb', type=int, default=500, help='Number of mini-batches to use in evaluating NLL')
fp.add_argument('--validation', action='store_false', dest='testing', help='only evaluate on validation set')
parser.set_defaults(testing=True)

# optimization settings
parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train (default: 100)')
parser.add_argument('--early_stopping_epochs', type=int, default=100, help='number of early stopping epochs')
parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 64)')
parser.add_argument('--eval_batch_size', type=int, default=64, help='batch size for evaluation')
parser.add_argument('--learning_rate', type=float, default=0.0005, help='learning rate')
parser.add_argument('--min_lr', type=float, default=1e-6, help='Minimum learning rate used in cyclic learning rates schedulers')
parser.add_argument('--annealing_schedule', type=int, default=100, help='Number of epochs to anneal the KL term. Set to 0 to turn beta annealing off. Applies this annealing schedule to each component of a boosted model.')
parser.add_argument('--max_beta', type=float, default=1.0, help='max beta for warm-up')
parser.add_argument('--min_beta', type=float, default=0.0, help='min beta for warm-up')
parser.add_argument('--no_annealing', action='store_true', default=False, help='disables annealing while training')
parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay parameter in Adamax')
parser.add_argument("--warmup_epochs", type=int, default=0, help="Use this number of epochs to warmup learning rate linearly from zero to learning rate")
parser.add_argument('--no_lr_schedule', action='store_true', default=False, help='Disables learning rate scheduler during training')
parser.add_argument('--lr_schedule', type=str, default=None, help="Type of LR schedule to use.", choices=['plateau', 'cosine', 'test', 'cyclic', None])
parser.add_argument('--patience', type=int, default=5, help='If using LR schedule, number of epochs before reducing LR.')
parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd'], help='Use AdamW or SDG as optimizer?')
parser.add_argument("--max_grad_clip", type=float, default=0, help="Max gradient value (clip above max_grad_clip, 0 for off)")
parser.add_argument("--max_grad_norm", type=float, default=100.0, help="Max norm of gradient (clip above max_grad_norm, 0 for off)")

# model parameters
parser.add_argument('--vae_layers', type=str, default='linear', choices=['linear', 'convolutional', 'simple'],
                    help="Type of layers in VAE's encoder and decoder.")
parser.add_argument('--z_size', type=int, default=64, help='how many stochastic hidden units')
parser.add_argument('--num_flows', type=int, default=2, help='Number of flow layers, ignored in absence of flows')
parser.add_argument('--flow', type=str, default='no_flow', help="Type of flows to use, no flows can also be selected",
                    choices=['planar', 'radial', 'iaf', 'liniaf', 'affine', 'nlsq', 'realnvp', 'householder', 'orthogonal', 'triangular', 'no_flow', 'boosted'])

# Sylvester parameters
parser.add_argument('--num_ortho_vecs', type=int, default=32, help=" For orthogonal flow: Number of orthogonal vectors per flow.")
parser.add_argument('--num_householder', type=int, default=8, help="For Householder Sylvester flow: Number of Householder matrices per flow.")

# RealNVP (and IAF) parameters
parser.add_argument('--h_size', type=int, default=256, help='Width of layers in base networks of iaf and realnvp. Ignored for all other flows.')
parser.add_argument('--coupling_network_depth', type=int, default=1, help='Number of extra hidden layers in the base network of iaf and realnvp. Ignored for all other flows.')
parser.add_argument('--coupling_network', type=str, default='tanh', choices=['relu', 'residual', 'tanh', 'random', 'mixed'],
                    help='Base network for RealNVP coupling layers. Random chooses between either Tanh or ReLU for every network, whereas mixed uses ReLU for the T network and TanH for the S network.')
parser.add_argument('--no_batch_norm', dest='batch_norm', action='store_false', help='Disables batch norm in realnvp layers, not recommended')
parser.set_defaults(batch_norm=True)

# Boosting parameters and optimization settings
parser.add_argument('--regularization_rate', type=float, default=1.0, help='Regularization penalty for boosting.')
parser.add_argument('--epochs_per_component', type=int, default=100, help='Number of epochs to train each component of a boosted model. Defaults to max(annealing_schedule, epochs_per_component). Ignored for non-boosted models.')
parser.add_argument('--lr_restarts', type=int, default=1, help='If using a cosine learning rate, how many times should the LR schedule restart?')
parser.add_argument('--rho_init', type=str, default='decreasing', choices=['decreasing', 'uniform'],
                    help='Initialization scheme for boosted parameter rho')
parser.add_argument('--rho_iters', type=int, default=100, help='Maximum number of SGD iterations for training boosting weights')
parser.add_argument('--rho_lr', type=float, default=0.005, help='Initial learning rate used for training boosting weights')
parser.add_argument('--num_components', type=int, default=2, help='How many components are combined to form the flow')
parser.add_argument('--component_type', type=str, default='affine',
                    choices=['realnvp', 'liniaf', 'affine', 'nlsq', 'random'],
                    help='When flow is boosted -- what type of flow should each component implement.')


def parse_args(main_args=None):
    """
    Parse command line arguments, initialize logging, and compute number of cores to use
    """
    args = parser.parse_args(main_args)
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.device = torch.device("cuda" if args.cuda else "cpu")
    if args.device == "cuda":
        cudnn.benchmark = args.benchmark    
    
    args.density_evaluation = False
    args.shuffle = True
    args.annealing_schedule = max(args.annealing_schedule, 1)
    args.epochs_per_component = max(args.epochs_per_component, args.annealing_schedule)
    args.init_epoch = min(max(1, args.init_epoch), args.epochs)

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
    if args.vae_layers == "linear":
        vae_type = "vae_"
    elif args.vae_layers == "simple":
        vae_type = "scvae_"
    elif args.vae_layers == "convolutional":
        vae_type = "cvae_"
    else:
        raise ValueError("vae_layers argument must be ['linear', 'convolutional', 'simple']")    
    
    args.snap_dir = os.path.join(args.out_dir, args.experiment_name + vae_type + args.flow)

    lr_schedule = f'_lr{str(args.learning_rate)[2:]}'
    if args.lr_schedule is None or args.no_lr_schedule:
        args.no_lr_schedule = True
        args.lr_schedule = None
    else:
        args.no_lr_schedule = False
        lr_schedule += f'{args.lr_schedule}'

    args.snap_dir += f'_seed{args.manual_seed}' + lr_schedule + '_' + args.dataset + f"_bs{args.batch_size}"

    args.boosted = args.flow == "boosted"
    if args.flow != 'no_flow':
        args.snap_dir += '_K' + str(args.num_flows)
        
    if args.flow == 'orthogonal':
        args.snap_dir += '_vectors' + str(args.num_ortho_vecs)
    if args.flow == 'householder':
        args.snap_dir += '_householder' + str(args.num_householder)
    if args.flow == 'iaf':
        args.snap_dir += '_hidden' + str(args.coupling_network_depth) + '_hsize' + str(args.h_size)
    if args.flow == 'boosted':
        if  (args.epochs_per_component % args.lr_restarts) != 0:
            raise ValueError(f"lr_restarts {args.lr_restarts} must evenly divide epochs_per_component {args.epochs_per_component}")
        if args.regularization_rate < 0.0:
            raise ValueError("For boosting the regularization_rate should be greater than or equal to zero.")
        args.snap_dir += '_' + args.component_type + '_C' + str(args.num_components) + '_reg' + f'{int(100*args.regularization_rate):d}'
    else:
        if (args.epochs % args.lr_restarts) != 0:
            raise ValueError(f"lr_restarts {args.lr_restarts} must evenly divide epochs {args.epochs}")

    if args.flow in ["realnvp"] or args.component_type in ["realnvp"]:
        args.snap_dir += '_' + args.coupling_network + str(args.coupling_network_depth) + '_hsize' + str(args.h_size)

    is_annealed = ""
    if not args.no_annealing and args.min_beta < 1.0:
        is_annealed += "_annealed"
    else:
        args.min_beta = 1.0

    args.snap_dir += is_annealed + f'_{args.model_signature}/'
    if not os.path.exists(args.snap_dir):
        os.makedirs(args.snap_dir)

    # intialize logger
    init_log(args)

    # Initalize computation settings
    # Set up multiple CPU/GPUs
    logger.info("COMPUTATION SETTINGS:")
    logger.info(f"Random Seed: {args.manual_seed}")
    if args.cuda:
        logger_msg = "Using CUDA GPU"
        torch.cuda.set_device(args.gpu_id)
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
    if args.flow == 'no_flow':
        model = VAE(args).to(args.device)
    elif args.flow == 'boosted':
        model = BoostedVAE(args).to(args.device)
    elif args.flow == 'planar':
        model = PlanarVAE(args).to(args.device)
    elif args.flow == 'radial':
        model = RadialVAE(args).to(args.device)
    elif args.flow == 'liniaf':
        model = LinIAFVAE(args).to(args.device)
    elif args.flow == 'affine':
        model = AffineVAE(args).to(args.device)
    elif args.flow == 'nlsq':
        model = NLSqVAE(args).to(args.device)
    elif args.flow == 'iaf':
        model = IAFVAE(args).to(args.device)
    elif args.flow == "realnvp":
        model = RealNVPVAE(args).to(args.device)
    elif args.flow == 'orthogonal':
        model = OrthogonalSylvesterVAE(args).to(args.device)
    elif args.flow == 'householder':
        model = HouseholderSylvesterVAE(args).to(args.device)
    elif args.flow == 'triangular':
        model = TriangularSylvesterVAE(args).to(args.device)
    else:
        raise ValueError('Invalid flow choice')

    return model


def main(main_args=None):
    """
    use main_args to run this script as function in another script
    """

    # =========================================================================
    # PARSE EXPERIMENT SETTINGS, SETUP SNAPSHOTS DIRECTORY, LOGGING
    # =========================================================================
    args, kwargs = parse_args(main_args)

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

    if args.load:
        logger.info(f'LOADING CHECKPOINT FROM PRE-TRAINED MODEL: {args.load}')
        init_with_args = args.flow == "boosted" and args.loaded_init_component is not None and args.loaded_all_trained is not None
        load(model, optimizer, args.load, args, init_with_args)

    # =========================================================================
    # TRAINING
    # =========================================================================
    training_required = args.epochs > 0 or args.load is None
    if training_required:
        logger.info('TRAINING:')
        if args.tensorboard:
            logger.info(f'Follow progress on tensorboard: tb {args.snap_dir}')

        train_loss, val_loss = train(train_loader, val_loader, model, optimizer, scheduler, args)

    # =========================================================================
    # VALIDATION
    # =========================================================================
    logger.info('VALIDATION:')
    if training_required:
        load(model, optimizer, args.snap_dir + 'model.pt', args)
    val_loss, val_rec, val_kl = evaluate(val_loader, model, args, results_type='Validation')

    # =========================================================================
    # TESTING
    # =========================================================================
    if args.testing:
        logger.info("TESTING:")
        test_loss, test_rec, test_kl = evaluate(test_loader, model, args, results_type='Test')
        test_nll = evaluate_likelihood(test_loader, model, args, S=args.nll_samples, MB=args.nll_mb, results_type='Test')


if __name__ == "__main__":
    main()


