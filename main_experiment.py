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

from models.vae import VAE
from models.sylvester_vae import OrthogonalSylvesterVAE, HouseholderSylvesterVAE, TriangularSylvesterVAE
from models.iaf_vae import IAFVAE
from models.realnvp_vae import RealNVPVAE
from models.boosted_vae import BoostedVAE
from models.planar_vae import PlanarVAE
from models.radial_vae import RadialVAE
from models.liniaf_vae import LinIAFVAE
from models.affine_vae import AffineVAE
from models.nlsq_vae import NLSqVAE
from optimization.training import train
from optimization.evaluation import evaluate, evaluate_likelihood
from utils.load_data import load_dataset
from utils.utilities import load, save


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
parser.add_argument('-nc', '--no_cuda', action='store_true', default=False,
                    help='disables CUDA training')

# Reporting
parser.add_argument('--plot_interval', type=int, default=10, help='Number of epochs between reconstructions plots.')
parser.add_argument('--out_dir', type=str, default='./results/snapshots', help='Output directory for model snapshots etc.')
parser.add_argument('--data_dir', type=str, default='./data/raw/', help="Where raw data is saved.")
parser.add_argument('--exp_log', type=str, default='./results/experiment_log.txt', help='File to save high-level results from each run of an experiment.')
parser.add_argument('--print_log', dest="save_log", action="store_false", help='Add this flag to have progress printed to log (rather than saved to a file).')
parser.set_defaults(save_log=True)

parser.add_argument('--load', type=str, default=None, help='Path to load the model from')
at = parser.add_mutually_exclusive_group(required=False)
at.add_argument('--loaded_is_not_all_trained', action="store_false", dest='loaded_is_all_trained',
                help="Set this if you don't want the loaded boosted model to be consider all_trained (default=True)")
at.add_argument('--loaded_is_all_trained', action='store_true', dest='loaded_is_all_trained',
                help='Default setting, which assumes the loaded boosted model is all_trained.')
parser.set_defaults(loaded_is_all_trained=True)

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
parser.add_argument('--early_stopping_epochs', type=int, default=0, help='number of early stopping epochs')
parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 64)')
parser.add_argument('--learning_rate', type=float, default=0.0005, help='learning rate')
parser.add_argument('--annealing_schedule', type=int, default=100, help='Number of epochs to anneal the KL term. Set to 0 to turn beta annealing off. Applies this annealing schedule to each component of a boosted model.')
parser.add_argument('--max_beta', type=float, default=1.0, help='max beta for warm-up')
parser.add_argument('--min_beta', type=float, default=0.0, help='min beta for warm-up')
parser.add_argument('--no_annealing', action='store_true', default=False, help='disables annealing while training')
parser.add_argument('--no_lr_schedule', action='store_true', default=False, help='Disables learning rate scheduler during training')

# model parameters
parser.add_argument('--vae_layers', type=str, default='linear', choices=['linear', 'convolutional'],
                    help="Type of layers in VAE's encoder and decoder.")
parser.add_argument('--z_size', type=int, default=64, help='how many stochastic hidden units')
parser.add_argument('--num_flows', type=int, default=2, help='Number of flow layers, ignored in absence of flows')
parser.add_argument('--flow', type=str, default='no_flow', help="Type of flows to use, no flows can also be selected",
                    choices=['planar', 'radial', 'iaf', 'liniaf', 'affine', 'nlsq', 'realnvp', 'householder', 'orthogonal', 'triangular', 'no_flow', 'boosted'])

# Sylvester parameters
parser.add_argument('--num_ortho_vecs', type=int, default=8, help=" For orthogonal flow: Number of orthogonal vectors per flow.")
parser.add_argument('--num_householder', type=int, default=8, help="For Householder Sylvester flow: Number of Householder matrices per flow.")

# RealNVP (and IAF) parameters
parser.add_argument('--h_size', type=int, default=16, help='Width of layers in base networks of iaf and realnvp. Ignored for all other flows.')
parser.add_argument('--num_base_layers', type=int, default=0, help='Number of extra hidden layers in the base network of iaf and realnvp. Ignored for all other flows.')
parser.add_argument('--base_network', type=str, default='relu', help='Base network for RealNVP coupling layers', choices=['relu', 'residual', 'tanh', 'random'])
parser.add_argument('--no_batch_norm', dest='batch_norm', action='store_false', help='Disables batch norm in realnvp layers')
parser.set_defaults(batch_norm=True)

# Boosting parameters and optimization settings
parser.add_argument('--component_threshold', type=float, default=0.0, help='Threshold for determining if a boosted component has converged.')
parser.add_argument('--regularization_rate', type=float, default=0.4, help='Regularization penalty for boosting.')
parser.add_argument('--epochs_per_component', type=int, default=100, help='Number of epochs to train each component of a boosted model. Defaults to max(annealing_schedule, epochs_per_component). Ignored for non-boosted models.')
parser.add_argument('--rho_init', type=str, default='decreasing', choices=['decreasing', 'uniform'],
                    help='Initialization scheme for boosted parameter rho') 
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
    args.density_evaluation = False
    args.shuffle = True
    args.annealing_schedule = max(args.annealing_schedule, 1)
    args.epochs_per_component = max(args.epochs_per_component, args.annealing_schedule)

    # Set a random seed if not given one
    if args.manual_seed is None:
        args.manual_seed = random.randint(1, 100000)

    random.seed(args.manual_seed)
    torch.manual_seed(args.manual_seed)
    np.random.seed(args.manual_seed)

    # intialize snapshots directory for saving models and results
    args.model_signature = str(datetime.datetime.now())[0:19].replace(' ', '_').replace(':', '_').replace('-', '_')
    args.experiment_name = args.experiment_name + "_" if args.experiment_name is not None else ""
    vae_type = "vae_" if args.vae_layers == "linear" else "cvae_"
    args.snap_dir = os.path.join(args.out_dir, args.experiment_name + vae_type + args.flow + '_')

    if args.flow != 'no_flow':
        args.snap_dir += 'K' + str(args.num_flows)
        
    if args.flow == 'orthogonal':
        args.snap_dir += '_vectors' + str(args.num_ortho_vecs)
    if args.flow == 'householder':
        args.snap_dir += '_householder' + str(args.num_householder)
    if args.flow == 'iaf':
        args.snap_dir += '_hidden' + str(args.num_base_layers) + '_hsize' + str(args.h_size)
    if args.flow == 'boosted':
        if args.component_threshold < 0.0:
            raise ValueError("For boosted the component_threshold must be greater than or equal to zero.")
        if args.regularization_rate < 0.0:
            raise ValueError("For boosting the regularization_rate should be greater than or equal to zero.")
        args.snap_dir += '_' + args.component_type + '_C' + str(args.num_components) + '_reg' + f'{int(100*args.regularization_rate):d}'

    if args.flow == "realnvp" or args.component_type == "realnvp":
        args.snap_dir += '_' + args.base_network + str(args.num_base_layers) + '_hsize' + str(args.h_size)

    is_annealed = ""
    if not args.no_annealing and args.min_beta < 1.0:
        is_annealed += "_annealed"
    else:
        args.min_beta = 1.0

    lr_schedule = ""
    if not args.no_lr_schedule:
        lr_schedule += "_lr_scheduling"

    args.snap_dir += lr_schedule + is_annealed + '_on_' + args.dataset + "_" + args.model_signature + '/'
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

    #args.num_data_workers = min(4, os.cpu_count() - 1)
    #logger_msg += f"\n\tUsing {args.num_data_workers} cores for loading and preprocessing data"
    #kwargs = {'num_workers': args.num_data_workers, 'pin_memory': True} if args.cuda else {'num_workers': args.num_data_workers}
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

    if torch.cuda.device_count() > 1:
        logger.info("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model).to(args.device)

    return model


def init_optimizer(model, args):
    """
    group model parameters to more easily modify learning rates of components (flow parameters)
    """
    logger.info('OPTIMIZER:')
    if args.flow == 'boosted':
        logger.info("Initializing optimizer for ensemble model, grouping parameters according to VAE and Component Id:")

        flow_params = {f"{c}": torch.nn.ParameterList() for c in range(args.num_components)}
        flow_labels = {f"{c}": [] for c in range(args.num_components)}
        vae_params = torch.nn.ParameterList()
        vae_labels = []
        for name, param in model.named_parameters():
            if name.startswith("flow"):
                pos = name.find(".")
                component_id = name[(pos + 1):(pos + 2)]
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
            logger.info(f"Grouping [{', '.join(flow_labels[str(c)])}] as Component {c}'s parameters.")

        # vae parameters are at the end of the list (may not exist if doing density estimation)
        if len(vae_params) > 0:
            all_params.append(vae_params)
            logger.info(f"Grouping [{', '.join(vae_labels)}] as the VAE parameters.\n")
            
        optimizer = optim.Adamax([{'params': param_group} for param_group in all_params], lr=args.learning_rate, eps=1.e-7)
    else:
        logger.info(f"Initializing optimizer for standard models with learning rate={args.learning_rate}.\n")
        optimizer = optim.Adamax(model.parameters(), lr=args.learning_rate, eps=1.e-7)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           factor=0.5,
                                                           patience=2000,
                                                           min_lr=5e-5,
                                                           verbose=False,
                                                           threshold_mode='abs')

    return optimizer, scheduler


def init_log(args):
    #log_format = '%(asctime)s %(name)-12s %(levelname)s : %(message)s'
    log_format = '%(asctime)s : %(message)s'
    if args.save_log:
        filename = os.path.join(args.snap_dir, "log.txt")
        print(f"Saving log output to file: {filename}")
        logging.basicConfig(filename=filename, format=log_format, datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)
    else:
        logging.basicConfig(format=log_format, level=logging.INFO)


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
    train_loader, val_loader, test_loader, args = load_dataset(args, **kwargs)

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
        load(model, optimizer, args.load, args)

    # =========================================================================
    # TRAINING
    # =========================================================================
    training_required = args.epochs > 0 or args.load is None
    if training_required:
        logger.info('TRAINING:')
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
        test_nll = evaluate_likelihood(test_loader, model, args, S=args.nll_samples, MB=args.nll_mb, results_type='Test')


if __name__ == "__main__":
    main()


