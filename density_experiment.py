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
from utils.utilities import save, load

from models.boosted_flow import BoostedFlow
from models.realnvp import RealNVPFlow
from models.glow import Glow

from utils.load_data import load_density_dataset
from utils.distributions import log_normal_diag, log_normal_standard, log_normal_normalized
from main_experiment import init_optimizer, init_log


logger = logging.getLogger(__name__)
G_MAX_LOSS = -10.0


parser = argparse.ArgumentParser(description='PyTorch Gradient Boosted Normalizing flows')

parser.add_argument('--dataset', type=str, help='Dataset choice.', choices=['power', 'gas', 'hepmass', 'miniboone', 'bsds300'])

# seeds
parser.add_argument('--manual_seed', type=int, default=1,
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
parser.add_argument('--no_benchmark', dest='benchmark', action='store_false', help='Turn off CUDNN benchmarking')
parser.set_defaults(benchmark=True)

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
parser.add_argument('--epochs', type=int, default=500, help='Maximum number of epochs to train (default: 1000)')
parser.add_argument('--early_stopping_epochs', type=int, default=25, help='number of early stopping epochs')
parser.add_argument('--batch_size', type=int, default=256, help='input batch size for training (default: 256)')
parser.add_argument('--eval_batch_size', type=int, default=1024, help='input batch size for training (default: 1024)')
parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
parser.add_argument('--patience', type=int, default=5000, help='If using LR schedule, number of steps before reducing LR.')
parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay parameter in Adamax')
parser.add_argument("--num_init_batches", type=int,default=15, help="Number of batches to use for Act Norm initialisation")
parser.add_argument("--warmup_epochs", type=int, default=2, help="Use this number of epochs to warmup learning rate linearly from zero to learning rate")
parser.add_argument('--no_lr_schedule', action='store_true', default=False, help='Disables learning rate scheduler during training')
parser.add_argument('--lr_schedule', type=str, default=None, help="Type of LR schedule to use.", choices=['plateau', 'cosine', None])
parser.add_argument("--max_grad_clip", type=float, default=0, help="Max gradient value (clip above max_grad_clip, 0 for off)")
parser.add_argument("--max_grad_norm", type=float, default=5.0, help="Max norm of gradient (clip above max_grad_norm, 0 for off)")

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
parser.add_argument('--no_batch_norm', dest='batch_norm', action='store_false', help='Disables batch norm in realnvp layers')
parser.set_defaults(batch_norm=True)
parser.add_argument('--coupling_network_depth', type=int, default=1, help='Number of layers in the coupling network of iaf and realnvp. Ignored for all other flows.')
parser.add_argument('--coupling_network', type=str, default='tanh', choices=['relu', 'residual', 'tanh', 'random', 'mixed'],
                    help='Base network for RealNVP coupling layers. Random chooses between either Tanh or ReLU for every network, whereas mixed uses ReLU for the T network and TanH for the S network.')

parser.add_argument('--sample_interval', type=int, default=5, help='How often (epochs) to save samples from the model')
parser.add_argument('--sample_size', type=int, default=16, help='Number of images to sample from model')
parser.add_argument("--temperature", type=float, default=1.0, help="Temperature of samples")

# Boosting parameters
parser.add_argument('--epochs_per_component', type=int, default=1000,
                    help='Number of epochs to train each component of a boosted model. Defaults to max(annealing_schedule, epochs_per_component). Ignored for non-boosted models.')
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

    args.dynamic_binarization = False
    #args.input_type = 'binary'
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

    lr_schedule = f'_lr{str(args.learning_rate)[2:]}'
    if args.lr_schedule is None or args.no_lr_schedule:
        args.no_lr_schedule = True
        args.lr_schedule = None
    else:
        args.no_lr_schedule = False
        lr_schedule += f'{args.lr_schedule}'

    args.snap_dir += f'_seed{args.manual_seed}' + lr_schedule + '_' + args.dataset + f"_bs{args.batch_size}"

    args.boosted = args.flow == "boosted"
    if args.flow == 'boosted':
        args.snap_dir += f'_{args.component_type}_C{args.num_components}'
    else:
        args.num_components = 1

    args.snap_dir += f'_K{args.num_flows}'
    if args.flow == "realnvp" or args.component_type == "realnvp":
        args.snap_dir += f'_{args.coupling_network}{args.coupling_network_depth}'
    if args.flow == "glow" or args.component_type == "glow":
        args.num_dequant_blocks = 0
        args.snap_dir += f'_L{str(args.num_blocks)}_{args.flow_permutation}_{args.flow_coupling}'

    args.snap_dir += f'_hsize{h_size}'
    args.snap_dir += f'_{args.model_signature}/'
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


def train(model, train_loader, val_loader, optimizer, scheduler, args):
    if args.tensorboard:
        writer = SummaryWriter(args.snap_dir)
        
    header_msg = f'| Epoch | {"TRAIN": <14}{"Loss": >4} | {"VALIDATION": <14}{"Loss": >4} | {"TIMING":<8}{"(sec)":>4} | {"Improved": >10} |'
    header_msg += f' {"Component": >10} | {"All Trained": >12} | {"Rho": >48} |' if args.boosted else ''
    logger.info('|' + "-"*(len(header_msg)-2) + '|')
    logger.info(header_msg)
    logger.info('|' + "-"*(len(header_msg)-2) + '|')
    
    best_loss = np.array([np.inf] * args.num_components)
    early_stop_count = 0
    converged_epoch = 0
    if args.boosted:
        model.component = 0
        prev_lr = []
        for c in range(args.num_components):
            if c != model.component:
                optimizer.param_groups[c]['lr'] = 0.0

            prev_lr.append(optimizer.param_groups[c]['lr'])

    epoch_times = []
    epoch_train = []
    epoch_valid = []
    step = 0
    for epoch in range(args.init_epoch, args.epochs + 1):

        model.train()
        train_loss = []
        t_start = time.time()
        for batch_id, (x, _) in enumerate(train_loader):

            x = x.to(args.device)
            optimizer.zero_grad()

            # initialize ActNorm on first step
            if (args.flow =='glow' or args.component_type == 'glow') and step < args.num_init_batches:
                with torch.no_grad():
                    if args.boosted:
                        for i in range(args.num_components):
                            model(x=x, components=i)
                    else:
                        model(x=x)
                        
                    step += 1
                    continue

            losses = compute_kl_pq_loss(model, x, args)
            train_loss.append(losses['nll'])
            
            losses['nll'].backward()

            if args.max_grad_clip > 0:
                torch.nn.utils.clip_grad_value_(model.parameters(), args.max_grad_clip)
            if args.max_grad_norm > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                if args.tensorboard:
                    writer.add_scalar("grad_norm/grad_norm", grad_norm, step)

            if args.boosted:
                # freeze all but the new component being trained
                if step > 0:
                    for c in range(args.num_components):
                        optimizer.param_groups[c]['lr'] = prev_lr[c] if c == model.component else 0.0
            if args.tensorboard:
                for i in range(len(optimizer.param_groups)):
                    writer.add_scalar(f'lr/lr_{i}', optimizer.param_groups[i]['lr'], step)
                    
            optimizer.step()
            if not args.no_lr_schedule:
                if args.lr_schedule == "plateau":
                    scheduler.step(metrics=losses['nll'])
                else:
                    scheduler.step(epoch=step)

                if args.boosted:
                    prev_lr[model.component] = optimizer.param_groups[model.component]['lr']

            if args.tensorboard:
                writer.add_scalar('step/train_nll', losses['nll'], step)
                if args.boosted:
                    writer.add_scalar('step/train_g', losses['g_nll'], step)
                    writer.add_scalar('step/train_G', losses['G_nll'], step)
                else:
                    writer.add_scalar('step/log_pz', losses['log_pz'], step)
                    writer.add_scalar('step/log_det_jacobian', losses['log_det_jacobian'], step)

            step += 1

        # Validation
        val_loss = evaluate(model, val_loader, args)
        train_loss = torch.stack(train_loss).mean().item()
        epoch_times.append(time.time() - t_start)
        epoch_train.append(train_loss)
        epoch_valid.append(val_loss)

        # Assess convergence
        component = model.component if args.boosted else 0
        converged, model_improved, early_stop_count, best_loss = check_convergence(
            early_stop_count, val_loss, best_loss, epoch - converged_epoch, component, args)
        if model_improved:
            fname = f'model_c{model.component}.pt' if args.boosted and args.save_intermediate_checkpoints else 'model.pt'
            save(model, optimizer, args.snap_dir + fname, scheduler)

        # epoch level reporting
        epoch_msg = f'| {epoch: <5} | {train_loss:18.3f} | {val_loss:18.3f} | {epoch_times[-1]:13.1f} | {"T" if model_improved else "": >10}'
        if args.boosted:
            rho_str = '[' + ', '.join([f"{val:4.2f}" for val in model.rho.data]) + ']'
            epoch_msg += f' | {model.component: >10} | {str(model.all_trained)[0]: >12} | {rho_str: >48}'
        if args.tensorboard:
            writer.add_scalar('epoch/validation', val_loss, epoch)
            writer.add_scalar('epoch/train', train_loss, epoch)

        if converged:
            logger.info(epoch_msg + ' |')

            if args.boosted:
                converged_epoch = epoch

                # revert back to the last best version of the model and update rho
                fname = f'model_c{model.component}.pt' if args.save_intermediate_checkpoints else 'model.pt'
                load(model=model, optimizer=optimizer, path=args.snap_dir + fname, args=args, scheduler=scheduler)
                model.update_rho(train_loader)
                if model.component > 0 or model.all_trained:
                    logger.info('Rho Updated: ' + ' '.join([f"{val:1.2f}" for val in model.rho.data]))

                last_component = model.component == (args.num_components - 1)
                no_fine_tuning = args.epochs <= args.epochs_per_component * args.num_components
                fine_tuning_done = model.all_trained and last_component
                if (fine_tuning_done or no_fine_tuning) and last_component:
                    # stop the full model after all components have been trained
                    logger.info(f"Model converged, training complete, saving: {args.snap_dir + 'model.pt'}")
                    model.all_trained = True
                    save(model, optimizer, args.snap_dir + f'model.pt', scheduler)
                    break

                # else if not done training: save model with updated rho
                save(model, optimizer, args.snap_dir + fname, scheduler)
                
                # reset early_stop_count and train the next component
                model.increment_component()
                early_stop_count = 0
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



def evaluate(model, data_loader, args, results_type=None):
    model.eval()
    loss = torch.stack([compute_kl_pq_loss(model, x.to(args.device), args)['nll'].detach() for (x,_) in data_loader], -1).mean().item()
    
    if args.save_results and results_type is not None:
        results_msg = f'{results_type} set loss: {loss:.6f}'
        logger.info(results_msg + '\n')
        with open(args.exp_log, 'a') as ff:
            print(results_msg, file=ff)

    return loss


def compute_kl_pq_loss(model, x, args):
    if args.flow == "boosted":
        z_g, mu_g, var_g, ldj_g, _ = model(x=x, components="c")
        g_nll = -1.0 * (log_normal_standard(z_g, reduce=False).view(z_g.shape[0], -1).sum(1, keepdim=True) + ldj_g)
        #g_nll = -1.0 * (log_normal_normalized(z_g, mu_g, var_g, reduce=False).view(z_g.shape[0], -1).sum(1, keepdim=True) + ldj_g)
        
        if model.all_trained or model.component > 0:
            fixed = '-c' if model.all_trained else '1:c-1'
            z_G, mu_G, var_G, ldj_G, _ = model(x=x, components=fixed)
            unconstrained_G_nll = log_normal_standard(z_G, reduce=False).view(z_G.shape[0], -1).sum(1, keepdim=True) + ldj_G
            #unconstrained_G_nll = log_normal_normalized(z_G, mu_G, var_G, reduce=False).view(z_G.shape[0], -1).sum(1, keepdim=True) + ldj_G
            G_nll = -1.0 * torch.max(unconstrained_G_nll,
                                     torch.ones_like(ldj_G) * G_MAX_LOSS)
            losses = {"nll": torch.mean(g_nll - G_nll)}
            losses["G_nll"] = torch.mean(G_nll)
            losses["g_nll"] = torch.mean(g_nll)
        else:
            losses = {"nll": torch.mean(g_nll)}
            losses["g_nll"] = torch.mean(g_nll)
            losses["G_nll"] = torch.zeros_like(losses['g_nll'])
    else:
        z, z_mu, z_var, log_det_j, _ = model(x=x)
        #log_pz = log_normal_normalized(z, z_mu, z_var, reduce=False).view(z.shape[0], -1).sum(1, keepdim=True)
        log_pz = log_normal_standard(z, reduce=False).view(z.shape[0], -1).sum(1, keepdim=True)
        nll = -1.0 * (log_pz + log_det_j)

        losses = {"nll": torch.mean(nll)}
        losses['log_det_jacobian'] = torch.mean(log_det_j)
        losses['log_pz'] = torch.mean(log_pz)

    if torch.isnan(losses['nll']).any():
        raise ValueError(f"Nan Encountered. nll={losses['nll']}, x={x}, losses={losses}")

    return losses


def check_convergence(early_stop_count, v_loss, best_loss, epochs_since_prev_convergence, component, args):
    """
    Verify if a boosted component has converged
    """
    if args.boosted:
        # Consider the boosted model's component as converged if a pre-set number of epochs have elapsed
        time_to_update = epochs_since_prev_convergence % args.epochs_per_component == 0
    else:
        time_to_update = False

    model_improved = v_loss < best_loss[component]
    early_stop_flag = False
    if model_improved:
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
    train_loader, val_loader, test_loader, args = load_density_dataset(args)

    
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

    
    # =========================================================================
    # TRAINING
    # =========================================================================
    if args.epochs > 0:
        logger.info('TRAINING:')
        train(model, train_loader, val_loader, optimizer, scheduler, args)

    
    # =========================================================================
    # VALIDATION
    # =========================================================================
    logger.info('VALIDATION:')
    load(model=model, optimizer=optimizer, path=args.snap_dir + 'model.pt', args=args)
    val_loss = evaluate(model, val_loader, args, results_type='Validation')


    # =========================================================================
    # TESTING
    # =========================================================================
    if args.testing:
        logger.info("TESTING:")
        test_loss = evaluate(model, test_loader, args, results_type='Test')



if __name__ == "__main__":
    main()

