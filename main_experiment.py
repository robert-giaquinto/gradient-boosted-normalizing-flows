import argparse
import datetime
import torch
import torch.utils.data
import torch.optim as optim
import numpy as np
import math
import random
import os
import logging

from models.vae import VAE, IAFVAE, OrthogonalSylvesterVAE, HouseholderSylvesterVAE, TriangularSylvesterVAE
from models.boosted_vae import BoostedVAE
from models.bagged_vae import BaggedVAE
from models.planar_vae import PlanarVAE
from models.radial_vae import RadialVAE
from optimization.training import train
from optimization.evaluation import evaluate, evaluate_likelihood
from utils.load_data import load_dataset


logger = logging.getLogger(__name__)


parser = argparse.ArgumentParser(description='PyTorch Ensemble Normalizing flows')

parser.add_argument('--dataset', type=str, default='mnist', help='Dataset choice.',
                    choices=['mnist', 'freyfaces', 'omniglot', 'caltech', 'cifar10'])
parser.add_argument('--num_toy_samples', type=int, default=1000, help='Number of samples to use for toy datasets')

# seeds
parser.add_argument('--manual_seed', type=int, default=123,
                    help='manual seed, if not given resorts to random seed.')
parser.add_argument('--freyseed', type=int, default=123,
                    help="Seed for shuffling frey face dataset for test split. Ignored for other datasets.")

# gpu/cpu
parser.add_argument('--gpu_id', type=int, default=0, metavar='GPU', help='choose GPU to run on.')
parser.add_argument('--num_workers', type=int, default=1,
                    help='How many CPU cores to run on. Setting to 0 uses os.cpu_count() - 1.')
parser.add_argument('-nc', '--no_cuda', action='store_true', default=False,
                    help='disables CUDA training')

# Reporting
parser.add_argument('--log_interval', type=int, default=0, metavar='LOG_INTERVAL',
                    help='how many batches to wait before logging training status. Set to <0 to turn off.')
parser.add_argument('--plot_interval', type=int, default=10, metavar='PLOT_INTERVAL',
                    help='how many batches to wait before creating reconstruction plots. Set to <0 to turn off.')

parser.add_argument('--experiment_name', type=str, default=None,
                    help="A name to help identify the experiment being run when training this model.")

parser.add_argument('--out_dir', type=str, default='./results/snapshots', help='Output directory for model snapshots etc.')
parser.add_argument('--data_dir', type=str, default='./data/raw/', help="Where raw data is saved.")
parser.add_argument('--exp_log', type=str, default='./results/experiment_log.txt', help='File to save high-level results from each run of an experiment.')
parser.add_argument('--print_log', dest="save_log", action="store_false", help='Add this flag to have progress printed to log (rather than saved to a file).')
parser.set_defaults(save_log=True)

sr = parser.add_mutually_exclusive_group(required=False)
sr.add_argument('--save_results', action='store_true', dest='save_results', help='Save results from experiments.')
sr.add_argument('--discard_results', action='store_false', dest='save_results', help='Do NOT save results from experiments.')
parser.set_defaults(save_results=True)

# testing vs. just validation
fp = parser.add_mutually_exclusive_group(required=False)
fp.add_argument('--testing', action='store_true', dest='testing', help='evaluate on test set after training')
fp.add_argument('--validation', action='store_false', dest='testing', help='only evaluate on validation set')
parser.set_defaults(testing=True)

# optimization settings
parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train (default: 100)')
parser.add_argument('--early_stopping_epochs', type=int, default=20, help='number of early stopping epochs')

parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 64)')
parser.add_argument('--learning_rate', type=float, default=0.0005, help='learning rate')
parser.add_argument('--regularization_rate', type=float, default=0.01, help='Regularization penalty for boosting.')

parser.add_argument('--annealing_schedule', type=int, default=100, help='number of epochs for warm-up. Set to 0 to turn beta annealing off.')
parser.add_argument('--max_beta', type=float, default=1.0, help='max beta for warm-up')
parser.add_argument('--min_beta', type=float, default=0.0, help='min beta for warm-up')
parser.add_argument('--burnin', type=int, default=25, help='number of extra epochs to run the first component of a boosted model.')

# flow parameters
parser.add_argument('--flow', type=str, default='no_flow',
                    choices=['planar', 'radial', 'iaf', 'householder', 'orthogonal', 'triangular', 'no_flow', 'boosted', 'bagged'],
                    help="""Type of flows to use, no flows can also be selected""")
parser.add_argument('--num_flows', type=int, default=2, help='Number of flow layers, ignored in absence of flows')

# Sylvester parameters
parser.add_argument('--num_ortho_vecs', type=int, default=8,
                    help=" For orthogonal flow: How orthogonal vectors per flow do you need. Ignored for other flow types.")
parser.add_argument('--num_householder', type=int, default=8,
                    help="For Householder Sylvester flow: Number of Householder matrices per flow. Ignored for other flow types.")
parser.add_argument('--made_h_size', type=int, default=320,
                    help='Width of mades for iaf. Ignored for all other flows.')
parser.add_argument('--z_size', type=int, default=64, help='how many stochastic hidden units')

# Bagging/Boosting parameters
parser.add_argument('--num_components', type=int, default=8,
                    help='How many components are combined to form the flow')
parser.add_argument('--component_type', type=str, default='planar',
                    choices=['planar', 'radial', 'iaf', 'householder', 'orthogonal', 'triangular', 'random'],
                    help='When flow is bagged or boosted -- what type of flow should each component implement.')



def parse_args(main_args=None):
    """
    Parse command line arguments and compute number of cores to use
    """
    args = parser.parse_args(main_args)
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.device = torch.device("cuda" if args.cuda else "cpu")
    args.density_evaluation = False

    # Set a random seed if not given one
    if args.manual_seed is None:
        args.manual_seed = random.randint(1, 100000)

    random.seed(args.manual_seed)
    torch.manual_seed(args.manual_seed)
    np.random.seed(args.manual_seed)
    logger.info(f"Random Seed: {args.manual_seed}\n")

    args.shuffle = args.flow != "bagged"
    
    # Set up multiple CPU/GPUs
    logger.info("COMPUTATION SETTINGS:")
    if args.cuda:
        logger.info("\tUsing CUDA GPU")
        torch.cuda.set_device(args.gpu_id)
    else:
        logger.info("\tUsing CPU")
        if args.num_workers > 0:
            num_workers = args.num_workers
        else:
            num_workers = max(1, os.cpu_count() - 1)

        logger.info("\tCores available: {} (only requesting {})".format(os.cpu_count(), num_workers))
        torch.set_num_threads(num_workers)
        logger.info("\tConfirmed Number of CPU threads: {}".format(torch.get_num_threads()))

    # SETUP SNAPSHOTS DIRECTORY FOR SAVING MODELS
    args.model_signature = str(datetime.datetime.now())[0:19].replace(' ', '_').replace(':', '_').replace('-', '_')
    args.experiment_name = args.experiment_name + "_" if args.experiment_name is not None else ""
    args.snap_dir = os.path.join(args.out_dir, args.experiment_name + args.flow + '_')

    if args.flow != 'no_flow':
        args.snap_dir += 'flow_length_' + str(args.num_flows)
        
    if args.flow == 'orthogonal':
        args.snap_dir += '_num_vectors_' + str(args.num_ortho_vecs)
    elif args.flow == 'householder':
        args.snap_dir += '_num_householder_' + str(args.num_householder)
    elif args.flow == 'iaf':
        args.snap_dir += '_madehsize_' + str(args.made_h_size)
    elif args.flow in ['boosted', 'bagged']:
        args.snap_dir += '_' + args.component_type + '_num_components_' + str(args.num_components)

    args.snap_dir += '_on_' + args.dataset + "_" +args.model_signature + '/'

    kwargs = {'num_workers': 0, 'pin_memory': True} if args.cuda else {}
    return args, kwargs


def init_model(args):
    if args.flow == 'no_flow':
        model = VAE(args).to(args.device)
    elif args.flow == 'boosted':
        model = BoostedVAE(args).to(args.device)
    elif args.flow == 'bagged':
        model = BaggedVAE(args).to(args.device)
    elif args.flow == 'planar':
        model = PlanarVAE(args).to(args.device)
    elif args.flow == 'radial':
        model = RadialVAE(args).to(args.device)
    elif args.flow == 'iaf':
        model = IAFVAE(args).to(args.device)
    elif args.flow == 'orthogonal':
        model = OrthogonalSylvesterVAE(args).to(args.device)
    elif args.flow == 'householder':
        model = HouseholderSylvesterVAE(args).to(args.device)
    elif args.flow == 'triangular':
        model = TriangularSylvesterVAE(args).to(args.device)
    else:
        raise ValueError('Invalid flow choice')

    return model


def init_optimizer(model, args):
    """
    group model parameters to more easily modify learning rates of components (flow parameters)
    """
    logger.info('OPTIMIZER:')
    if args.flow in ['boosted', 'bagged']:
        logger.info("Initializing optimizer for ensemble model, grouping parameters according to VAE and Component Id:")

        flow_params = {f"{c}": torch.nn.ParameterList() for c in range(args.num_components)}
        flow_labels = {f"{c}": [] for c in range(args.num_components)}
        vae_params = torch.nn.ParameterList()
        vae_labels = []
        for name, param in model.named_parameters():            
            if name.startswith("amor_u") or name.startswith("amor_w") or name.startswith("amor_b") or args.density_evaluation:
                if args.density_evaluation:
                    component_id = name[(name.find(".") + 1):(name.find(".") + 2)]
                else:
                    component_id = name[(name.find(".") - 1):name.find(".")]

                flow_params[component_id].append(param)
                flow_labels[component_id].append(name)
            else:
                vae_labels.append(name)
                vae_params.append(param)

        # collect all parameters into a single list
        # the first args.num_components elements in the parameters list correspond bagged/boosting parameters
        all_params = []
        for c in range(args.num_components):
            all_params.append(flow_params[f"{c}"])
            logger.info(f"Grouping [{', '.join(flow_labels[str(c)])}] as Component {c}'s parameters.")

        # vae parameters are at the end of the list
        all_params.append(vae_params)
        logger.info(f"Grouping [{', '.join(vae_labels)}] as the VAE parameters.\n")
            
        optimizer = optim.Adamax([{'params': param_group} for param_group in all_params], lr=args.learning_rate, eps=1.e-7)
    else:
        logger.info(f"Initializing optimizer for standard models with learning rate={args.learning_rate}.\n")
        optimizer = optim.Adamax(model.parameters(), lr=args.learning_rate, eps=1.e-7)

    return optimizer


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
    if not os.path.exists(args.snap_dir):
        os.makedirs(args.snap_dir)
    init_log(args)

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
    optimizer = init_optimizer(model, args)
    num_params = sum([param.nelement() for param in model.parameters()])
    logger.info(f"MODEL:\nNumber of model parameters={num_params}\n{model}\n")

    # =========================================================================
    # TRAINING
    # =========================================================================
    logger.info('TRAINING:')
    train_loss, val_loss = train(train_loader, val_loader, model, optimizer, args)

    # =========================================================================
    # VALIDATION
    # =========================================================================
    logger.info('VALIDATION:')
    final_model = torch.load(args.snap_dir + 'model.pt')
    val_loss, val_rec, val_kl = evaluate(val_loader, final_model, args, results_type='Validation')

    # =========================================================================
    # TESTING
    # =========================================================================
    if args.testing:
        logger.info("TESTING:")
        test_nll = evaluate_likelihood(test_loader, final_model, args, results_type='Test')


if __name__ == "__main__":
    main()


