import argparse
import datetime
import torch
import numpy as np
import math
import random
import os
import logging
from tensorboardX import SummaryWriter
from shutil import copyfile
from utils.utilities import save, load

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
from main_experiment import init_model, init_optimizer, init_log
from toy_experiment import compute_kl_pq_loss

from utils.gas import GAS
from utils.bsds300 import BSDS300
from utils.hepmass import HEPMASS
from utils.miniboone import MINIBOONE
from utils.power import POWER


logger = logging.getLogger(__name__)
G_MAX_LOSS = -10.0


parser = argparse.ArgumentParser(description='PyTorch Ensemble Normalizing flows')

parser.add_argument('--dataset', type=str, help='Dataset choice.', choices=['power', 'gas', 'hepmass', 'miniboone', 'bsds300'])

# seeds
parser.add_argument('--manual_seed', type=int, default=123,
                    help='manual seed, if not given resorts to random seed.')

# testing vs. just validation
fp = parser.add_mutually_exclusive_group(required=False)
fp.add_argument('--testing', action='store_true', dest='testing', help='evaluate on test set after training')
parser.add_argument('--nll_samples', type=int, default=2000, help='Number of samples to use in evaluating NLL')
parser.add_argument('--nll_mb', type=int, default=500, help='Number of mini-batches to use in evaluating NLL')
fp.add_argument('--validation', action='store_false', dest='testing', help='only evaluate on validation set')
parser.set_defaults(testing=True)

# gpu/cpu
parser.add_argument('--gpu_id', type=int, default=0, metavar='GPU', help='choose GPU to run on.')
parser.add_argument('--num_workers', type=int, default=1,
                    help='How many CPU cores to run on. Setting to 0 uses os.cpu_count() - 1.')
parser.add_argument('--no_cuda', action='store_true', default=False, help='disables CUDA training')

parser.add_argument('--experiment_name', type=str, default="density",
                    help="A name to help identify the experiment being run when training this model.")
parser.add_argument('--out_dir', type=str, default='./results/snapshots', help='Output directory for model snapshots etc.')
parser.add_argument('--data_dir', type=str, default='./data/raw/', help="Where raw data is saved.")
parser.add_argument('--exp_log', type=str, default='./results/density_experiment_log.txt', help='File to save high-level results from each run of an experiment.')
parser.add_argument('--print_log', dest="save_log", action="store_false", help='Add this flag to have progress printed to log (rather than saved to a file).')
parser.set_defaults(save_log=True)
parser.add_argument('--no_tensorboard', dest="tensorboard", action="store_false", help='Turns off saving results to tensorboard.')
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

# optimization settings
parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train (default: 100)')
parser.add_argument('--early_stopping_epochs', type=int, default=100, help='number of early stopping epochs')
parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 64)')
parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
parser.add_argument('--no_lr_schedule', action='store_true', default=False, help='Disables learning rate scheduler during training')
parser.add_argument('--patience', type=int, default=5000, help='If using LR schedule, number of steps before reducing LR.')

# flow parameters
parser.add_argument('--flow', type=str, default='planar',
                    choices=['planar', 'radial', 'liniaf', 'affine', 'nlsq', 'boosted', 'iaf', 'realnvp'],
                    help="""Type of flows to use, no flows can also be selected""")
parser.add_argument('--num_flows', type=int, default=2, help='Number of flow layers, ignored in absence of flows')

parser.add_argument('--h_size', type=int, default=16, help='Width of layers in base networks of iaf and realnvp. Ignored for all other flows.')
parser.add_argument('--num_base_layers', type=int, default=1, help='Number of layers in the base network of iaf and realnvp. Ignored for all other flows.')
parser.add_argument('--base_network', type=str, default='relu', help='Base network for RealNVP coupling layers', choices=['relu', 'residual', 'tanh', 'random'])
parser.add_argument('--no_batch_norm', dest='batch_norm', action='store_false', help='Disables batch norm in realnvp layers')
parser.set_defaults(batch_norm=True)
parser.add_argument('--z_size', type=int, default=2, help='how many stochastic hidden units')

# Boosting parameters
parser.add_argument('--epochs_per_component', type=int, default=100,
                    help='Number of epochs to train each component of a boosted model. Defaults to max(annealing_schedule, epochs_per_component). Ignored for non-boosted models.')
parser.add_argument('--regularization_rate', type=float, default=0.5, help='Regularization penalty for boosting.')
parser.add_argument('--no_rho_update', action='store_true', default=False, help='Disable boosted rho update')
parser.add_argument('--rho_init', type=str, default='decreasing', choices=['decreasing', 'uniform'],
                    help='Initialization scheme for boosted parameter rho') 
parser.add_argument('--num_components', type=int, default=1,
                    help='How many components are combined to form the flow')
parser.add_argument('--component_type', type=str, default='affine', choices=['liniaf', 'affine', 'nlsq', 'realnvp'],
                    help='When flow is boosted -- what type of flow should each component implement.')



def parse_args(main_args=None):
    """
    Parse command line arguments and compute number of cores to use
    """
    args = parser.parse_args(main_args)
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.device = torch.device("cuda" if args.cuda else "cpu")

    args.dynamic_binarization = False
    #args.input_type = 'binary'
    args.density_evaluation = True
    args.shuffle = True

    # Set a random seed if not given one
    if args.manual_seed is None:
        args.manual_seed = random.randint(1, 100000)

    random.seed(args.manual_seed)
    torch.manual_seed(args.manual_seed)
    np.random.seed(args.manual_seed)

    # intialize snapshots directory for saving models and results
    args.model_signature = str(datetime.datetime.now())[0:19].replace(' ', '_').replace(':', '_').replace('-', '_')
    args.experiment_name = args.experiment_name + "_" if args.experiment_name is not None else ""
    args.snap_dir = os.path.join(args.out_dir, args.experiment_name + args.flow + '_')
    args.snap_dir += f"bs{args.batch_size}_"

    if args.flow != 'no_flow':
        args.snap_dir += 'K' + str(args.num_flows)

    args.boosted = args.flow == "boosted"
    if args.flow == 'boosted':
        if args.regularization_rate < 0.0:
            raise ValueError("For boosting the regularization rate should be greater than or equal to zero.")
        args.snap_dir += '_' + args.component_type + '_C' + str(args.num_components) + '_reg' + f'{int(100*args.regularization_rate):d}'
    else:
        args.num_components = 1

    if args.flow == 'iaf':
        args.snap_dir += '_hidden' + str(args.num_base_layers) + '_hsize' + str(args.h_size)

    if args.flow == "realnvp" or args.component_type == "realnvp":
        args.snap_dir += '_' + args.base_network + str(args.num_base_layers) + '_hsize' + str(args.h_size)
        
    lr_schedule = ""
    if not args.no_lr_schedule:
        lr_schedule += "_lr_scheduling"

    if args.dataset in ['u5', 'mog']:
        dataset = f"{args.dataset}_s{int(100 * args.mog_sigma)}_c{args.mog_clusters}"
    else:
        dataset = args.dataset
        
    args.snap_dir += lr_schedule + '_on_' + dataset + "_" + args.model_signature + '/'
    if not os.path.exists(args.snap_dir):
        os.makedirs(args.snap_dir)

    init_log(args)
    
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


def load_dataset(args):
    if args.dataset == 'gas':
        dataset = GAS('data/gas/ethylene_CO.pickle')
    elif args.dataset == 'bsds300':
        dataset = BSDS300('data/BSDS300/BSDS300.hdf5')
    elif args.dataset == 'hepmass':
        dataset = HEPMASS('data/hepmass')
    elif args.dataset == 'miniboone':
        dataset = MINIBOONE('data/miniboone/data.npy')
    elif args.dataset == 'power':
        dataset = POWER('data/power/data.npy')
    else:
        raise RuntimeError()

    # idle y's
    y_train = np.zeros((dataset.trn.x.shape[0], 1))
    y_val = np.zeros((dataset.val.x.shape[0], 1))
    y_test = np.zeros((dataset.tst.x.shape[0], 1))

    dataset_train = torch.utils.data.TensorDataset(
        torch.from_numpy(dataset.trn.x).float().to(args.device), torch.from_numpy(y_train))
    data_loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True)

    dataset_valid = torch.utils.data.TensorDataset(
        torch.from_numpy(dataset.val.x).float().to(args.device), torch.from_numpy(y_val))
    data_loader_valid = torch.utils.data.DataLoader(dataset_valid, batch_size=args.batch_size, shuffle=False)

    dataset_test = torch.utils.data.TensorDataset(
        torch.from_numpy(dataset.tst.x).float().to(args.device), torch.from_numpy(y_test))
    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False)
    
    args.z_size = dataset.n_dims

    logger.info(f"Dataset: {args.dataset} has {len(data_loader_train)}, {len(data_loader_valid)}, and {len(data_loader_test)} minibatches of size {args.batch_size} in train, validation, and test sets.")
    logger.info(f"Dimension={dataset.n_dims}. Total samples: {len(data_loader_train.sampler)}, {len(data_loader_valid.sampler)}, and {len(data_loader_test.sampler)} in train, validation, and test sets.\n")
    return args, data_loader_train, data_loader_valid, data_loader_test


def train(model, train_loader, val_loader, optimizer, scheduler, args):
    if args.tensorboard:
        writer = SummaryWriter(args.snap_dir)
        
    header_msg = f'| Epoch | {"TRAIN": <14}{"Loss": >4} | {"VALIDATION": <14}{"Loss": >4} | '
    header_msg += f'{"Component": >10} | {"All Trained": >12} | {"Rho": >32} | '
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

    for epoch in range(args.init_epoch, args.epochs + 1):

        model.train()
        train_loss = []
        for batch_id, (x, _) in enumerate(train_loader):
            x = x.to(args.device)
            optimizer.zero_grad()
            loss, loss_terms = compute_kl_pq_loss(model, x, beta=None, args=args)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
            optimizer.step()
            if not args.no_lr_schedule:
                scheduler.step(loss)
                if args.boosted:
                    prev_lr[model.component] = optimizer.param_groups[model.component]['lr']
                    
            train_loss.append(loss)
            if args.tensorboard:
                if args.boosted:
                    writer.add_scalar('loss/train_g', loss_terms[0], (epoch-1)*len(train_loader) + batch_id)
                    writer.add_scalar('loss/train_G', loss_terms[1], (epoch-1)*len(train_loader) + batch_id)
                else:
                    writer.add_scalar('loss/q_log_prob', loss_terms[0], (epoch-1)*len(train_loader) + batch_id)
                    writer.add_scalar('loss/log_det', loss_terms[1], (epoch-1)*len(train_loader) + batch_id)

        train_loss = torch.stack(train_loss).mean().item()

        # Validation
        model.eval()
        val_loss = torch.stack([compute_kl_pq_loss(model, x, beta=None, args=args)[0].detach() for (x,_) in val_loader], -1).mean().item()

        epoch_msg = f'| {epoch: <5} | {train_loss:18.3f} | {val_loss:18.3f} | '
        rho_str = '[' + ', '.join([f"{val:4.2f}" for val in model.rho.data]) + ']'
        epoch_msg += f'{model.component: >10} | {str(model.all_trained)[0]: >12} | {rho_str: >32} | ' if args.boosted else ''

        if args.tensorboard:
            writer.add_scalar('loss/validation', val_loss, epoch)
            writer.add_scalar('loss/train', train_loss, epoch)
            for i in range(len(optimizer.param_groups)):
                writer.add_scalar(f'lr_{i}', optimizer.param_groups[i]['lr'], epoch)

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
    args, train_loader, val_loader, test_loader = load_dataset(args)

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


    # =========================================================================
    # TESTING
    # =========================================================================
    if args.testing:
        logger.info("TESTING:")



if __name__ == "__main__":
    main()

