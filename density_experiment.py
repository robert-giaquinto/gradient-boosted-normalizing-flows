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
from collections import Counter

from optimization.optimizers import init_optimizer
from models.boosted_flow import BoostedFlow
from models.realnvp import RealNVPFlow
from models.glow import Glow
from utils.utilities import save, load, init_log, softmax
from utils.load_data import load_density_dataset
from utils.distributions import log_normal_diag, log_normal_standard, log_normal_normalized


logger = logging.getLogger(__name__)


parser = argparse.ArgumentParser(description='PyTorch Gradient Boosted Normalizing flows')

parser.add_argument('--dataset', type=str, help='Dataset choice.', choices=['power', 'gas', 'hepmass', 'miniboone', 'bsds300'])

# seeds
parser.add_argument('--manual_seed', type=int, default=1,
                    help='manual seed, if not given resorts to random seed.')

# testing vs. just validation
fp = parser.add_mutually_exclusive_group(required=False)
fp.add_argument('--testing', action='store_true', dest='testing', help='evaluate on test set after training')
fp.add_argument('--validation', action='store_false', dest='testing', help='only evaluate on validation set')
parser.set_defaults(testing=True)

# gpu/cpu
parser.add_argument('--gpu_id', type=int, default=0, metavar='GPU', help='choose GPU to run on.')
parser.add_argument('--num_workers', type=int, default=1, help='How many CPU cores to run on. Setting to 0 uses os.cpu_count() - 1.')
parser.add_argument('--no_cuda', action='store_true', default=False, help='disables CUDA training')
parser.add_argument('--no_benchmark', dest='benchmark', action='store_false', help='Turn off CUDNN benchmarking')
parser.set_defaults(benchmark=True)

parser.add_argument('--experiment_name', type=str, default="density",
                    help="A name to help identify the experiment being run when training this model.")
parser.add_argument('--out_dir', type=str, default='./results/snapshots', help='Output directory for model snapshots etc.')
parser.add_argument('--data_dir', type=str, default='./data/raw/', help="Where raw data is saved.")
parser.add_argument('--exp_log', type=str, default='./results/density_experiment_log.txt', help='File to save high-level results from each run of an experiment.')
parser.add_argument('--print_log', dest="print_log", action="store_true", help='Add this flag to have progress printed to log (rather than only saved to a file).')
parser.set_defaults(print_log=True)
parser.add_argument('--no_tensorboard', dest="tensorboard", action="store_false", help='Turns off saving results to tensorboard.')
parser.set_defaults(tensorboard=True)

parser.add_argument('--load', type=str, default=None, help='Path to load the model from')
parser.add_argument('--save_intermediate_checkpoints', dest="save_intermediate_checkpoints",
                    action="store_true", help='Save versions of the boosted model after each component converges.')
parser.set_defaults(save_intermediate_checkpoints=False)
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
parser.add_argument('--epochs', type=int, default=100, help='Maximum number of epochs to train')
parser.add_argument('--early_stopping_epochs', type=int, default=50, help='number of early stopping epochs')
parser.add_argument('--batch_size', type=int, default=512, help='input batch size for training')
parser.add_argument('--eval_batch_size', type=int, default=1024, help='batch size for evaluation')
parser.add_argument('--learning_rate', type=float, default=None, help='learning rate, if none use best values found during LR range test for that dataset')
parser.add_argument('--min_lr', type=float, default=None, help='Minimum learning rate used in cyclic learning rates schedulers')
parser.add_argument('--patience', type=int, default=5, help='If using LR schedule, number of epochs before reducing LR.')
parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay parameter in Adamax')
parser.add_argument("--num_init_batches", type=int,default=15, help="Number of batches to use for Act Norm initialisation")
parser.add_argument("--warmup_epochs", type=int, default=0, help="Use this number of epochs to warmup learning rate linearly from zero to learning rate")
parser.add_argument('--no_lr_schedule', action='store_true', default=False, help='Disables learning rate scheduler during training')
parser.add_argument('--lr_schedule', type=str, default=None, help="Type of LR schedule to use.", choices=['plateau', 'cosine', 'test', 'cyclic', None])
parser.add_argument('--lr_restarts', type=int, default=1, help='If using a cyclic/cosine learning rate, how many times should the LR schedule restart? Must evenly divide epochs')
parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd'], help='Use AdamW or SDG as optimizer?')
parser.add_argument("--max_grad_clip", type=float, default=0.0, help="Max gradient value (clip above max_grad_clip, 0 for off)")
parser.add_argument("--max_grad_norm", type=float, default=0.0, help="Max norm of gradient (clip above max_grad_norm, 0 for off)")

# flow parameters
parser.add_argument('--flow', type=str, default='glow', help="Type of flow to use", choices=['realnvp', 'glow', 'boosted'])
parser.add_argument("--num_flows", type=int, default=5, help="Number of flow layers per block")
parser.add_argument("--num_blocks", type=int, default=1, help="Number of blocks. Ignored for non glow models")
parser.add_argument("--actnorm_scale", type=float, default=1.0, help="Act norm scale")
parser.add_argument("--flow_permutation", type=str, default="shuffle", choices=["invconv", "shuffle", "reverse"], help="Type of flow permutation")
parser.add_argument("--flow_coupling", type=str, default="affine", choices=["additive", "affine"], help="Type of flow coupling")
parser.add_argument("--no_LU_decomposed", action="store_false", dest="LU_decomposed", help="Don't train with LU decomposed 1x1 convs")
parser.set_defaults(LU_decomposed=True)
parser.add_argument("--learn_top", action="store_true", dest="learn_top", help="Do not train top layer (prior)")
parser.set_defaults(learn_top=False)
parser.add_argument("--y_weight", type=float, default=0.01, help="Weight for class condition loss")
parser.add_argument("--y_condition", action="store_true", help="Train using class condition")
parser.set_defaults(y_condition=False)
parser.add_argument("--y_multiclass", action="store_true", help="Y is a multiclass classification")
parser.set_defaults(y_multiclass=False)
parser.add_argument('--use_attention', dest='use_attn', action='store_true', help='Use attention in the coupling layers')
parser.set_defaults(use_attn=False)

parser.add_argument('--h_size', type=int, help='Width of layers in base networks of iaf and realnvp. Ignored for all other flows.')
parser.add_argument('--h_size_factor', type=int, help='Sets width of hidden layers as h_size_factor * dimension of data.')
parser.add_argument('--no_batch_norm', dest='batch_norm', action='store_false', help='Disables batch norm in realnvp layers (not recommended)')
parser.set_defaults(batch_norm=True)
parser.add_argument('--coupling_network_depth', type=int, default=1, help='Number of layers in the coupling network of iaf and realnvp. Ignored for all other flows.')
parser.add_argument('--coupling_network', type=str, default='tanh', choices=['relu', 'residual', 'tanh', 'random', 'mixed'],
                    help='Base network for RealNVP coupling layers. Random chooses between either Tanh or ReLU for every network, whereas mixed uses ReLU for the T network and TanH for the S network.')

parser.add_argument('--sample_interval', type=int, default=5, help='How often (epochs) to save samples from the model')
parser.add_argument('--sample_size', type=int, default=16, help='Number of images to sample from model')
parser.add_argument("--temperature", type=float, default=1.0, help="Temperature of samples")

# Boosting parameters
parser.add_argument('--epochs_per_component', type=int, default=100,
                    help='Number of epochs to train each component of a boosted model. Defaults to max(annealing_schedule, epochs_per_component). Ignored for non-boosted models.')
parser.add_argument('--boosted_burnin_epochs', type=int, default=None,
                    help='(DEPRECATED) Number of epochs to warmup/burnin for EACH component before proceeding with a full training schedule')
parser.add_argument('--rho_init', type=str, default='decreasing', choices=['decreasing', 'uniform'],
                    help='Initialization scheme for boosted parameter rho')
parser.add_argument('--rho_iters', type=int, default=100, help='Maximum number of SGD iterations for training boosting weights')
parser.add_argument('--rho_lr', type=float, default=0.005, help='Initial learning rate used for training boosting weights')
parser.add_argument('--num_components', type=int, default=2,
                    help='How many components are combined to form the flow')
parser.add_argument('--component_type', type=str, default='affine', choices=['liniaf', 'affine', 'nlsq', 'realnvp', 'glow'],
                    help='When flow is boosted -- what type of flow should each component implement.')



def parse_args(main_args=None):
    """
    Parse command line arguments and compute number of cores to use
    """
    args = parser.parse_args(main_args)
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.device = torch.device("cuda" if args.cuda else "cpu")
    if args.device == "cuda":
        cudnn.benchmark = args.benchmark    

    args.boosted = args.flow == "boosted"
    args.dynamic_binarization = False
    args.density_evaluation = True
    args.shuffle = True
    args.init_epoch = min(max(1, args.init_epoch), args.epochs)

    if args.h_size is None and args.h_size_factor is None:
        raise ValueError("Must specify the hidden size h_size, or provide the size of hidden layer relative to the data with h_size_factor")
    elif args.h_size is not None and args.h_size_factor is not None:
        raise ValueError("You must only specify either h_size or h_size_factor, but not both.")
    elif args.h_size is None:
        h_size = f'{args.h_size_factor}xD'
    else:
        h_size = str(args.h_size)

    # default optimization settings for each optimizer and dataset
    if args.learning_rate is None:
        logger.info("No learning rate given, using default settings for this dataset")
        if args.optimizer == "adam":
            if args.dataset == "miniboone":
                args.learning_rate = 5e-4
                args.min_lr = 5e-6
                args.max_grad_norm = 20.0
                args.weight_decay = 1e-5
            elif args.dataset == "gas":
                args.learning_rate = 8e-4
                args.min_lr = 2e-6
                args.max_grad_norm = 20.0
                args.weight_decay = 1e-4
            elif args.dataset == "hepmass":
                args.learning_rate = 1e-3
                args.min_lr = 1e-5
                args.max_grad_norm = 10.0
                args.weight_decay = 1e-5
            elif args.dataset == "power":
                args.learning_rate = 1e-4
                args.min_lr = 1e-6
                args.max_grad_norm = 10.0
                args.weight_decay = 1e-3
            elif args.dataset == "bsds300":
                args.learning_rate = 1e-5
                args.min_lr = 1e-6
                args.max_grad_norm = 60.0
                args.weight_decay = 1e-5
        elif args.optimizer == "sgd":
            if args.dataset == "miniboone":
                args.learning_rate = 1e-3
                args.min_lr = 1e-4
                args.max_grad_norm = 20.0
                args.weight_decay = 1e-5
            elif args.dataset == "gas":
                args.learning_rate = 1e-4
                args.min_lr = 5e-6
                args.max_grad_norm = 10.0
                args.weight_decay = 1e-4
            elif args.dataset == "hepmass":
                args.learning_rate = 1e-3
                args.min_lr = 1e-5
                args.max_grad_norm = 10.0
                args.weight_decay = 1e-5
            elif args.dataset == "power":
                args.learning_rate = 1e-3
                args.min_lr = 1e-4
                args.max_grad_norm = 10.0
                args.weight_decay = 1e-5
            elif args.dataset == "bsds300":
                args.learning_rate = 6e-4
                args.min_lr =51e-5
                args.max_grad_norm = 60.0
                args.weight_decay = 1e-5
        else:
            raise ValueError(f"No default settings found for optimizer {args.optimizer}")
            

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
    args.snap_dir = os.path.join(args.out_dir,  f"{args.experiment_name}{args.dataset}_{args.flow}")

    if args.flow == 'boosted':
        args.snap_dir += f'_{args.component_type}_C{args.num_components}'
        if (args.epochs_per_component % args.lr_restarts) != 0:
            raise ValueError(f"lr_restarts {args.lr_restarts} must evenly divide epochs_per_component {args.epochs_per_component}")
    else:
        args.num_components = 1
        if (args.epochs % args.lr_restarts) != 0:
            raise ValueError(f"lr_restarts {args.lr_restarts} must evenly divide epochs {args.epochs}")

    args.snap_dir += f'_K{args.num_flows}'
    if args.flow == "glow" or args.component_type == "glow":
        args.num_dequant_blocks = 0
        args.snap_dir += f'_L{str(args.num_blocks)}_{args.flow_permutation}_{args.flow_coupling}'

    args.snap_dir += f'_{args.coupling_network}{args.coupling_network_depth}'

    lr_schedule = f'_lr{str(args.learning_rate)[2:]}'
    if args.lr_schedule is None or args.no_lr_schedule:
        args.no_lr_schedule = True
        args.lr_schedule = None
    else:
        args.no_lr_schedule = False
        lr_schedule += f'{args.lr_schedule}'
        epochs = args.epochs_per_component if args.boosted else args.epochs
        lr_schedule += f'{args.lr_restarts}x{int(epochs / args.lr_restarts)}' if args.lr_schedule in ['cosine', 'cyclic'] else ''
        
    args.snap_dir += f'_hsize{h_size}' + lr_schedule + f"_bs{args.batch_size}_seed{args.manual_seed}_{args.model_signature}/"
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


def init_model(args):
    if args.flow == 'glow':
        model = Glow(args).to(args.device)
    elif args.flow == 'boosted':
        model = BoostedFlow(args).to(args.device)
    elif args.flow == 'realnvp':
         model = RealNVPFlow(args).to(args.device)
    else:
        raise ValueError('Invalid flow choice')

    #if device == 'cuda':
    #    model = torch.nn.DataParallel(model, args.gpu_ids)

    return model


def train(model, data_loaders, optimizer, scheduler, args):
    writer = SummaryWriter(args.snap_dir) if args.tensorboard else None
        
    header_msg = f'| Epoch | {"TRAIN": <14}{"Loss": >4} | {"VALIDATION": <14}{"Loss": >4} | {"TIMING":<8}{"(sec)":>4} | {"Improved": >8} |'
    header_msg += f' {"Component": >9} | {"All Trained": >11} | {"Rho": >{min(8, args.num_components) * 6}} |' if args.boosted else ''
    logger.info('|' + "-"*(len(header_msg)-2) + '|')
    logger.info(header_msg)
    logger.info('|' + "-"*(len(header_msg)-2) + '|')
    
    best_loss = np.array([np.inf] * args.num_components)
    early_stop_count = 0
    converged_epoch = 0  # for boosting, helps keep track how long the current component has been training
    
    if args.boosted:
        #model.component = 0
        prev_lr = init_boosted_lr(model, optimizer, args)
    else:
        prev_lr = []

    grad_norm = None
    epoch_times = []
    epoch_train = []
    epoch_valid = []

    pval_loss = 0.0
    val_losses = {'g_nll': 9999999.9}
    step = 0
    for epoch in range(args.init_epoch, args.epochs + 1):

        model.train()
        train_loss = []
        t_start = time.time()
        
        for batch_id, (x, _) in enumerate(data_loaders['train']):

            # initialize data and optimizer
            x = x.to(args.device)
            optimizer.zero_grad()

            # initialize ActNorm on first steps
            if (args.flow =='glow' or args.component_type == 'glow') and step < args.num_init_batches:
                with torch.no_grad():
                    if args.boosted:
                        for i in range(args.num_components):
                            model(x=x, components=i)
                    else:
                        model(x=x)
                        
                    step += 1
                    continue

            # compute loss and gradients
            losses = compute_kl_pq_loss(model, x, args)
            train_loss.append(losses['nll'])
            losses['nll'].backward()

            # clip gradients if requested
            if args.max_grad_clip > 0:
                torch.nn.utils.clip_grad_value_(model.parameters(), args.max_grad_clip)
            if args.max_grad_norm > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            # Adjust learning rates for boosted model, keep fixed components frozen
            if args.boosted:
                update_learning_rates(prev_lr, model, optimizer, step, args)
                
            # batch level reporting
            batch_reporting(writer, optimizer, losses, grad_norm, step, args)
            
            # Perform gradient update, modify learning rate according to learning rate schedule
            optimizer.step()
            if not args.no_lr_schedule:
                prev_lr = update_scheduler(prev_lr, model, optimizer, scheduler, val_losses['g_nll'], step, args)

                if args.lr_schedule == "test":
                    if step % 50 == 0:
                        pval_loss = evaluate(model, data_loaders['val'], args)['nll']
                    
                    writer.add_scalar('step/val_nll', pval_loss, step)
                    
            step += 1

        # Validation, collect results
        val_losses = evaluate(model, data_loaders['val'], args)
        train_loss = torch.stack(train_loss).mean().item()
        epoch_times.append(time.time() - t_start)
        epoch_train.append(train_loss)
        epoch_valid.append(val_losses['nll'])

        # Assess convergence
        component = (model.component, model.all_trained) if args.boosted else 0
        converged, model_improved, early_stop_count, best_loss = check_convergence(
            early_stop_count, val_losses, best_loss, epoch - converged_epoch, component, args)
        if model_improved:
            fname = f'model_c{model.component}.pt' if args.boosted and args.save_intermediate_checkpoints else 'model.pt'
            save(model, optimizer, args.snap_dir + fname, scheduler)

        # epoch level reporting
        epoch_msg = epoch_reporting(writer, model, train_loss, val_losses, epoch_times, model_improved, epoch, args)
        
        if converged:
            logger.info(epoch_msg + ' |')
            logger.info("-"*(len(header_msg)))

            if args.boosted:
                converged_epoch = epoch

                # revert back to the last best version of the model and update rho
                fname = f'model_c{model.component}.pt' if args.save_intermediate_checkpoints else 'model.pt'
                load(model=model, optimizer=optimizer, path=args.snap_dir + fname, args=args, scheduler=scheduler, verbose=False)
                model.update_rho(data_loaders['train'])

                last_component = model.component == (args.num_components - 1)
                no_fine_tuning = args.epochs <= args.epochs_per_component * args.num_components
                fine_tuning_done = model.all_trained and last_component  # no early stopping if burnin employed
                if (fine_tuning_done or no_fine_tuning) and last_component:
                    # stop the full model after all components have been trained
                    logger.info(f"Model converged, training complete, saving: {args.snap_dir + 'model.pt'}")
                    model.all_trained = True
                    save(model, optimizer, args.snap_dir + f'model.pt', scheduler)
                    break

                # else if not done training: save model with updated rho
                save(model, optimizer, args.snap_dir + fname, scheduler)

                # tempory: look at results after each component
                test_loss = evaluate(model, data_loaders['test'], args)
                logger.info(f"Loss after training {model.component + 1} components: {test_loss['nll']:8.3f}")
                logger.info("-"*(len(header_msg)))
                
                # reset optimizer, scheduler, and early_stop_count and train the next component
                model.increment_component()                
                early_stop_count = 0
                optimizer, scheduler = init_optimizer(model, args, verbose=False)
                prev_lr = init_boosted_lr(model, optimizer, args)
            else:
                # if a standard model converges once, break
                logger.info(f"Model converged, stopping training.")
                break
                
        else:
            logger.info(epoch_msg + ' |')
            if epoch == args.epochs:
                if args.boosted and args.save_intermediate_checkpoints:
                    # Save the best version of the model trained up to the current component with filename model.pt
                    # This is to protect against times when the model is trained/re-trained but doesn't run long enough
                    # for all components to converge / train completely
                    copyfile(args.snap_dir + f'model_c{model.component}.pt', args.snap_dir + 'model.pt')
                    logger.info(f"Resaving last improved version of {f'model_c{model.component}.pt'} as 'model.pt' for future testing")
                else:
                    logger.info(f"Stopping training after {epoch} epochs of training.")

    logger.info('|' + "-"*(len(header_msg)-2) + '|\n')
    if args.tensorboard:
        writer.close()

    epoch_times, epoch_train, epoch_valid = np.array(epoch_times), np.array(epoch_train), np.array(epoch_valid)
    timing_msg = f"Stopped after {epoch_times.shape[0]} epochs. "
    timing_msg += f"Average train time per epoch: {np.mean(epoch_times):.2f} +/- {np.std(epoch_times, ddof=1):.2f}"
    logger.info(timing_msg + '\n')
    if args.save_results:
        np.savetxt(args.snap_dir + '/train_loss.csv', epoch_train, fmt='%f', delimiter=',')
        np.savetxt(args.snap_dir + '/valid_loss.csv', epoch_valid, fmt='%f', delimiter=',')
        np.savetxt(args.snap_dir + '/epoch_times.csv', epoch_times, fmt='%f', delimiter=',')
        with open(args.exp_log, 'a') as ff:
            timestamp = str(datetime.datetime.now())[0:19].replace(' ', '_')
            setup_msg = '\n'.join([timestamp, args.snap_dir]) + '\n' + repr(args)
            print('\n' + setup_msg + '\n' + timing_msg, file=ff)


def epoch_reporting(writer, model, train_loss, val_losses, epoch_times, model_improved, epoch, args):
    epoch_msg = f'| {epoch: <5} | {train_loss:18.3f} | {val_losses["nll"]:18.3f} | {epoch_times[-1]:13.1f} | {"T" if model_improved else "": >8}'
    if args.boosted:
        epoch_msg += f' | {model.component: >9} | {str(model.all_trained)[0]: >11}'
        rho_str = '[' + ', '.join([f"{wt:4.2f}" for wt in model.rho.data]) + ']'
        epoch_msg += f' | {rho_str: >{args.num_components * 6}}' if args.num_components <= 8 else ''
        epoch_msg += f' | {val_losses["ratio"]:12.3f}'
        epoch_msg += f' | {val_losses["g_nll"]:12.3f}'
    if args.tensorboard:
        writer.add_scalar('epoch/validation', val_losses['nll'], epoch)
        writer.add_scalar('epoch/train', train_loss, epoch)
        if args.boosted:
            writer.add_scalar('epoch/validation_g', val_losses['g_nll'], epoch)
            writer.add_scalar('epoch/validation_ratio', val_losses['ratio'], epoch)

    return epoch_msg


def batch_reporting(writer, optimizer, losses, grad_norm, step, args):
    if args.tensorboard:
        writer.add_scalar('step/train_nll', losses['nll'], step)
        
        if args.max_grad_norm > 0:
            writer.add_scalar("grad_norm/grad_norm", grad_norm, step)
            
        if args.boosted:
            writer.add_scalar('step/train_g', losses['g_nll'], step)
            writer.add_scalar('step/train_G', losses['G_nll'], step)
        else:
            writer.add_scalar('step/log_pz', losses['log_pz'], step)
            writer.add_scalar('step/log_det_jacobian', losses['log_det_jacobian'], step)
            
        for i in range(len(optimizer.param_groups)):
            writer.add_scalar(f'lr/lr_{i}', optimizer.param_groups[i]['lr'], step)


def update_learning_rates(prev_lr, model, optimizer, step, args):
    for c in range(args.num_components):
        optimizer.param_groups[c]['lr'] = prev_lr[c] if c == model.component else 0.0


def update_scheduler(prev_lr, model, optimizer, scheduler, loss, step, args):
    if args.lr_schedule == "plateau":
        scheduler.step(metrics=loss)
    else:
        scheduler.step()
        
    if args.boosted:
        prev_lr[model.component] = optimizer.param_groups[model.component]['lr']
    else:
        prev_lr = []

    return prev_lr


def init_boosted_lr(model, optimizer, args):
    learning_rates = []
    for c in range(args.num_components):
        if c != model.component:
            optimizer.param_groups[c]['lr'] = 0.0
            
        learning_rates.append(optimizer.param_groups[c]['lr'])

    for n, param in model.named_parameters():
        param.requires_grad = True if n.startswith(f"flows.{model.component}") else False

    return learning_rates

            
def evaluate(model, data_loader, args, results_type=None):
    model.eval()

    if args.boosted:
        G_nll, g_nll = [], []
        for (x, _) in data_loader:
            x = x.to(args.device)

            approximate_fixed_G = False
            
            if approximate_fixed_G:
                # randomly sample a component
                z_G, _, _, ldj_G, _ = model(x=x, components="1:c")
                G_nll_i = -1.0 * (log_normal_standard(z_G, reduce=True, dim=-1, device=args.device) + ldj_G)                
                G_nll.append(G_nll_i.detach())

            else:
                G_ll = torch.zeros(x.size(0))
                for c in range(model.component + 1):
                    z_G, _, _, ldj_G, _ = model(x=x, components=c)
                    if c == 0:
                        G_ll = log_normal_standard(z_G, reduce=True, dim=-1, device=args.device) + ldj_G                        
                    else:
                        rho_simplex = model.rho[0:(c+1)] / torch.sum(model.rho[0:(c+1)])
                        last_ll = torch.log(1 - rho_simplex[c]) + G_ll
                        next_ll = torch.log(rho_simplex[c]) + (log_normal_standard(z_G, reduce=True, dim=-1, device=args.device) + ldj_G)
                        uG_ll = torch.cat([last_ll.view(x.size(0), 1), next_ll.view(x.size(0), 1)], dim=1)
                        G_ll = torch.logsumexp(uG_ll, dim=1)
                        
                G_nll.append(-1.0 * G_ll.detach())

            # track new component progress just for insights
            if model.component > 0 or model.all_trained:
                z_g, mu_g, var_g, ldj_g, _ = model(x=x, components="c")
                g_nll_i = -1.0 * (log_normal_standard(z_g, reduce=True, dim=-1, device=args.device) + ldj_g)
                g_nll.append(g_nll_i.detach())

        G_nll = torch.cat(G_nll, dim=0)
        # remove nan for now
        not_inf = torch.isinf(G_nll) == False
        G_nll = G_nll[not_inf]
        not_nan = torch.isnan(G_nll) == False
        G_nll = G_nll[not_nan]

        mean_G_nll = G_nll.mean().item()
        losses = {'nll': mean_G_nll}

        if model.component > 0 or model.all_trained:
            g_nll = torch.cat(g_nll, dim=0)
            g_nll = g_nll[not_inf]
            g_nll = g_nll[not_nan]
            losses['g_nll'] = g_nll.mean().item()
            losses['ratio'] = torch.mean(g_nll - G_nll).item()
        else:
            losses['g_nll'] = mean_G_nll
            losses['ratio'] = 0.0
        
    else:
        nll = torch.stack([compute_kl_pq_loss(model, x.to(args.device), args)['nll'].detach() for (x,_) in data_loader], -1).mean().item()
        losses = {'nll': nll, 'g_nll': nll}
    
    if args.save_results and results_type is not None:
        results_msg = f'{results_type} set loss: {losses["nll"]:.6f}'
        logger.info(results_msg + '\n')
        with open(args.exp_log, 'a') as ff:
            print(results_msg, file=ff)

    return losses


def compute_kl_pq_loss(model, x, args):
    if args.flow == "boosted":

        if model.all_trained or model.component > 0:

            # 1. Compute likelihood/weight for each sample
            approximate_fixed_G = False
            
            if approximate_fixed_G:
                # randomly sample a component
                fixed = '-c' if model.all_trained else '1:c-1'
                z_G, _, _, ldj_G, _ = model(x=x, components=fixed)
                G_nll = -1.0 * (log_normal_standard(z_G, reduce=True, dim=-1, device=args.device) + ldj_G)
            else:
                # combine weighted likelihoods from each component
                G_ll = torch.zeros(x.size(0))
                for c in range(model.component):  # TODO: if model.all_trained then what?
                    z_G, _, _, ldj_G, _ = model(x=x, components=c)
                    if c == 0:
                        G_ll = log_normal_standard(z_G, reduce=True, dim=-1, device=args.device) + ldj_G
                    else:
                        rho_simplex = model.rho[0:(c+1)] / torch.sum(model.rho[0:(c+1)])
                        last_ll = torch.log(1 - rho_simplex[c]) + G_ll
                        next_ll = torch.log(rho_simplex[c]) + (log_normal_standard(z_G, reduce=True, dim=-1, device=args.device) + ldj_G)
                        uG_ll = torch.cat([last_ll.view(x.size(0), 1), next_ll.view(x.size(0), 1)], dim=1)
                        G_ll = torch.logsumexp(uG_ll, dim=1)
                
                G_nll = -1.0 * G_ll

            reweight_samples = True
            if reweight_samples:
                # 2. Sample x with replacement, weighted by G_nll
                #weights = torch.exp(G_nll - torch.logsumexp(G_nll, dim=0))  # normalize weights: large NLL => large weight
                weights = softmax(G_nll)
                orig_weights = weights
                if weights.min() < 0.0:
                    weights = weights - weights.min()
                if weights.max() > 0.1:
                    weights = torch.max(torch.min(weights, torch.tensor([0.1], device=args.device)), torch.tensor([0.01], device=args.device))
                if weights.sum() != 1.0:
                    weights = weights / torch.sum(weights)
                    
                reweighted_idx = torch.multinomial(weights, x.size(0), replacement=True)
                x_resampled = x[reweighted_idx]

                # 3. Compute g for resampled observations
                z_g, _, _, ldj_g, _ = model(x=x_resampled, components="c")
                g_nll = -1.0 * (log_normal_standard(z_g, reduce=True, dim=-1, device=args.device) + ldj_g)
                nll = torch.mean(g_nll)

            else:
                z_g, _, _, ldj_g, _ = model(x=x, components="c")
                g_nll = -1.0 * (log_normal_standard(z_g, reduce=True, dim=-1, device=args.device) + ldj_g)
                nll = torch.mean(g_nll - G_nll)

            losses = {"nll": nll}
            losses["G_nll"] = torch.mean(G_nll)
            losses["g_nll"] = torch.mean(g_nll)
            
        else:
            z_g, _, _, ldj_g, _ = model(x=x, components="c")
            g_nll = -1.0 * (log_normal_standard(z_g, reduce=True, dim=-1, device=args.device) + ldj_g)
            losses = {"nll": torch.mean(g_nll)}
            losses["g_nll"] = torch.mean(g_nll)
            losses["G_nll"] = torch.zeros_like(losses['g_nll'])
    else:
        z, _, _, log_det_j, _ = model(x=x)
        log_pz = log_normal_standard(z, reduce=True, dim=-1, device=args.device)
        nll = -1.0 * (log_pz + log_det_j)

        losses = {"nll": torch.mean(nll)}
        losses['log_det_jacobian'] = torch.mean(log_det_j)
        losses['log_pz'] = torch.mean(log_pz)

    if torch.isnan(losses['nll']).any():
        raise ValueError(f"Nan Encountered. nll={losses['nll']}, x={x}, losses={losses}")

    return losses


def check_convergence(early_stop_count, losses, best_loss, epoch, stage, args):
    """
    Verify if a boosted component has converged
    """
    if args.boosted:
        c, all_trained = stage            
        stage_complete = (epoch % args.epochs_per_component == 0)
        v_loss = losses['g_nll']
    else:
        c = stage
        stage_complete = False
        v_loss = losses['nll']

    model_improved = v_loss < best_loss[c]
    early_stop_flag = False
    if model_improved:
        early_stop_count = 0
        best_loss[c] = v_loss
    elif args.early_stopping_epochs > 0:
        # model didn't improve, do we consider it converged yet?
        early_stop_count += 1        
        early_stop_flag = early_stop_count > args.early_stopping_epochs

    converged = early_stop_flag or stage_complete
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
    data_loaders, args = load_density_dataset(args)

    
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
        load(model=model, optimizer=optimizer, path=args.load, args=args, init_with_args=init_with_args, scheduler=scheduler)
        logger.info(f'Warning: boosted models may only be loaded to train a new component (until pytorch bug is fixed), optimizer and scheduler will be reset. Non-boosted models may not be loaded at all (will fail).')
        optimizer, scheduler = init_optimizer(model, args, verbose=False)

    
    # =========================================================================
    # TRAINING
    # =========================================================================
    if args.epochs > 0:
        logger.info('TRAINING:')
        if args.tensorboard:
            logger.info(f'Follow progress on tensorboard: tb {args.snap_dir}')
            
        train(model, data_loaders, optimizer, scheduler, args)

    
    # =========================================================================
    # VALIDATION
    # =========================================================================
    logger.info('VALIDATION:')
    load(model=model, optimizer=optimizer, path=args.snap_dir + 'model.pt', args=args)
    val_loss = evaluate(model, data_loaders['val'], args, results_type='Validation')


    # =========================================================================
    # TESTING
    # =========================================================================
    if args.testing:
        logger.info("TESTING:")
        test_loss = evaluate(model, data_loaders['test'], args, results_type='Test')



if __name__ == "__main__":
    main()

