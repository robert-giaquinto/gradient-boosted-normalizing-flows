import argparse
import datetime
import torch
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
from utils.load_data import load_dataset
from utils.density_plotting import plot
from utils.density_data import make_toy_density, make_toy_sampler
from main_experiment import init_model, init_optimizer, init_log


logger = logging.getLogger(__name__)


TOY_DATASETS = ["8gaussians", "2gaussians", "1gaussian",  "swissroll", "rings", "moons", "pinwheel", "cos", "2spirals", "checkerboard", "line", "circles", "joint_gaussian"]
ENERGY_FUNS = ['u0', 'u1', 'u2', 'u3', 'u4', 'u5']
G_MAX_LOSS = -10.0

parser = argparse.ArgumentParser(description='PyTorch Ensemble Normalizing flows')

parser.add_argument('--dataset', type=str, default='mnist', help='Dataset choice.', choices=TOY_DATASETS + ENERGY_FUNS)
parser.add_argument('--mog_sigma', type=float, default=1.5, help='Variance in location of mixture of gaussian data.',
                    choices=[i / 100.0 for i in range(50, 250)])
parser.add_argument('--mog_clusters', type=int, default=6, help='Number of clusters to use in the mixture of gaussian data.',
                    choices=range(1,13))

# seeds
parser.add_argument('--manual_seed', type=int, default=123,
                    help='manual seed, if not given resorts to random seed.')

# gpu/cpu
parser.add_argument('--gpu_id', type=int, default=0, metavar='GPU', help='choose GPU to run on.')
parser.add_argument('--num_workers', type=int, default=1,
                    help='How many CPU cores to run on. Setting to 0 uses os.cpu_count() - 1.')
parser.add_argument('--no_cuda', action='store_true', default=False, help='disables CUDA training')

# Reporting
parser.add_argument('--log_interval', type=int, default=1000,
                    help='how many batches to wait before logging training status. Set to <0 to turn off.')
parser.add_argument('--plot_interval', type=int, default=1000,
                    help='how many batches to wait before creating reconstruction plots. Set to <0 to turn off.')

parser.add_argument('--experiment_name', type=str, default="density_evaluation",
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
parser.add_argument('--plot_resolution', type=int, default=250, help='how many points to plot, higher gives better resolution')

# optimization settings
parser.add_argument('--num_steps', type=int, default=100000, help='number of training steps to take (default: 100000)')
parser.add_argument('--batch_size', type=int, default=256, help='input batch size for training (default: 64)')
parser.add_argument('--learning_rate', type=float, default=0.005, help='learning rate')
parser.add_argument('--regularization_rate', type=float, default=0.5, help='Regularization penalty for boosting.')

parser.add_argument('--iters_per_component', type=int, default=10000, help='how often to train each boosted component before changing')
parser.add_argument('--max_beta', type=float, default=1.0, help='max beta for warm-up')
parser.add_argument('--min_beta', type=float, default=0.0, help='min beta for warm-up')
parser.add_argument('--no_annealing', action='store_true', default=False, help='disables annealing while training')
parser.add_argument('--no_lr_schedule', action='store_true', default=False, help='Disables learning rate scheduler during training')

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
parser.add_argument('--no_rho_update', action='store_true', default=False, help='Disable boosted rho update')
parser.add_argument('--rho_init', type=str, default='decreasing', choices=['decreasing', 'uniform'],
                    help='Initialization scheme for boosted parameter rho') 
parser.add_argument('--num_components', type=int, default=4,
                    help='How many components are combined to form the flow')
parser.add_argument('--component_type', type=str, default='affine', choices=['liniaf', 'affine', 'nlsq', 'realnvp', 'realnvp2'],
                    help='When flow is boosted -- what type of flow should each component implement.')



def parse_args(main_args=None):
    """
    Parse command line arguments and compute number of cores to use
    """
    args = parser.parse_args(main_args)
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.device = torch.device("cuda" if args.cuda else "cpu")

    args.dynamic_binarization = False
    args.input_type = 'binary'
    args.input_size = [2]
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

    if args.flow in ['boosted', 'bagged']:
        if args.regularization_rate < 0.0:
            raise ValueError("For boosting the regularization rate should be greater than or equal to zero.")
        args.snap_dir += '_' + args.component_type + '_C' + str(args.num_components) + '_reg' + f'{int(100*args.regularization_rate):d}'

    if args.flow == 'iaf':
        args.snap_dir += '_hidden' + str(args.num_base_layers) + '_hsize' + str(args.h_size)

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

    if args.dataset in ['u5', 'mog']:
        dataset = f"{args.dataset}_s{int(100 * args.mog_sigma)}_c{args.mog_clusters}"
    else:
        dataset = args.dataset
        
    args.snap_dir += lr_schedule + is_annealed + '_on_' + dataset + "_" + args.model_signature + '/'
    if not os.path.exists(args.snap_dir):
        os.makedirs(args.snap_dir)

    init_log(args)
    
    # Set up multiple CPU/GPUs
    logger.info("COMPUTATION SETTINGS:")
    logger.info(f"Random Seed: {args.manual_seed}\n")
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

    kwargs = {'num_workers': 0, 'pin_memory': True} if args.cuda else {}
    return args, kwargs


def compute_kl_qp_loss(model, target_fn, beta, args):
    """
    Compute KL(q_inv || p) where q_inv is the inverse flow transform:
    
    (log_q_inv = log_q_base - logdet),

    and p is the target distribution (energy potential)
 
    Returns the minimization objective for density matching.

    ADAPTED FROM: https://arxiv.org/pdf/1904.04676.pdf (https://github.com/kamenbliznashki/normalizing_flows/blob/master/bnaf.py)
    """
    z0 = model.base_dist.sample((args.batch_size,))
    q_log_prob = model.base_dist.log_prob(z0).sum(1)
    
    if args.flow == "boosted":
        if model.component < model.num_components:
            density_from = '-c' if model.all_trained else '1:c-1'
            sample_from = 'c'
        else:
            density_from = '1:c'
            sample_from = '1:c'
            
        z_g, entropy_ldj, z_G, boosted_ldj = model.flow(z0, sample_from=sample_from, density_from=density_from)
        p_log_prob = -1.0 * target_fn(z_g[-1]) * beta  # p = exp(-potential) => log_p = - potential
        g_lhood = q_log_prob - entropy_ldj
        
        if model.component == 0 and model.all_trained == False:
            G_lhood = torch.zeros_like(g_lhood)
            loss = g_lhood - p_log_prob
        else:
            G_log_prob = model.base_dist.log_prob(z_G[0]).sum(1)
            G_lhood = torch.max(G_log_prob - boosted_ldj, torch.ones_like(boosted_ldj) * G_MAX_LOSS)
            loss =  G_lhood - p_log_prob + g_lhood * args.regularization_rate

        return loss.mean(0), (g_lhood.mean().item(), G_lhood.mean().item(), p_log_prob.mean().item())

    else:
        zk, logdet = model.flow(z0)
        p_log_prob = -1.0 * target_fn(zk) * beta  # p = exp(-potential) => log_p = - potential
        loss = q_log_prob - logdet - p_log_prob
        return loss.mean(0), (q_log_prob.mean().item(), logdet.mean().item(), p_log_prob.mean().item())


def compute_kl_pq_loss(model, data_sampler, beta, args):
    """
    Compute KL(p || q_fwd) where q_fwd is the forward flow transform (log_q_fwd = log_q_base + logdet),
    and p is the target distribution.

    Returns the minimization objective for density estimation (NLL under the flow since the
    entropy of the target dist is fixed wrt the optimization)

    ADAPTED FROM: https://arxiv.org/pdf/1904.04676.pdf (https://github.com/kamenbliznashki/normalizing_flows/blob/master/bnaf.py)
    """
    sample = data_sampler(args.batch_size).to(args.device)
    
    if args.flow == "boosted":
        density_from = '-c' if model.all_trained else '1:c-1'
        sample_from = 'c'
        
        Z_g, g_ldj, Z_G, G_ldj = model.flow(sample, sample_from=sample_from, density_from=density_from)
        g_log_prob = model.base_dist.log_prob(Z_g[-1]).sum(1)
        g_lhood = g_log_prob + g_ldj
        
        if model.all_trained or model.component > 0:
            G_log_prob = model.base_dist.log_prob(Z_G[-1]).sum(1)
            G_lhood = G_log_prob + G_ldj
            G_lhood = torch.max(G_lhood, torch.ones_like(G_ldj) * G_MAX_LOSS)
            loss =  -1.0 * g_lhood + G_lhood * args.regularization_rate
        else:
            G_lhood = torch.zeros_like(g_lhood)
            loss = -1.0 * g_lhood

        return loss.mean(0), (g_lhood.mean().item(), G_lhood.mean().item())
        
    else:
        z, logdet = model.flow(sample)
        q_log_prob = model.base_dist.log_prob(z).sum(1)
        loss = -1.0 * (q_log_prob + logdet)
        return loss.mean(0), (q_log_prob.mean().item(), logdet.mean().detach().item())


@torch.no_grad()
def rho_gradient(model, target_or_sample_fn, args):
    if args.density_matching:
        # density matching of a target function
        fixed_components = "-c" if model.all_trained else "1:c-1"
        z0 = model.base_dist.sample((args.num_components * args.batch_size * 25,))
        g_zk, g_ldj = [], []
        G_zk, G_ldj = [], []
        for z0_i in z0.split(args.batch_size, dim=0):
            gZ_i, _, _, g_ldj_i = model.flow(z0_i, sample_from="c", density_from="1:c")
            g_zk += [gZ_i[-1]]  # grab K-th element
            g_ldj += [g_ldj_i]
            GZ_i, _, _, G_ldj_i = model.flow(z0_i, sample_from=fixed_components, density_from="1:c")
            G_zk += [GZ_i[-1]]  # grab K-th element
            G_ldj += [G_ldj_i]
        
        g_zk, g_ldj = torch.cat(g_zk, 0), torch.cat(g_ldj, 0)
        G_zk, G_ldj = torch.cat(G_zk, 0), torch.cat(G_ldj, 0)
        
        q_log_prob = model.base_dist.log_prob(z0).sum(1)
        p_log_prob_g = -1.0 * target_or_sample_fn(g_zk)  # p = exp(-potential) => log_p = - potential
        loss_wrt_g = q_log_prob - g_ldj - p_log_prob_g
        p_log_prob_G = -1.0 * target_or_sample_fn(G_zk)  # p = exp(-potential) => log_p = - potential
        loss_wrt_G = q_log_prob - G_ldj - p_log_prob_G
        
    else:
        # estimate density from a sampler
        sample = target_or_sample_fn(args.num_components * args.batch_size * 25).to(args.device)
        g_zk, g_ldj = [], []
        G_zk, G_ldj = [], []
        for sample_i in sample.split(args.batch_size, dim=0):
            g_zk_i, _, _, g_ldj_i = model.flow(sample_i, sample_from="c", density_from="1:c")
            g_zk += [g_zk_i[-1]]
            g_ldj += [g_ldj_i]
            G_zk_i, _, _, G_ldj_i = model.flow(sample_i, sample_from="1:c-1", density_from="1:c")
            G_zk += [G_zk_i[-1]]
            G_ldj += [G_ldj_i]

        g_zk, g_ldj = torch.cat(g_zk, 0), torch.cat(g_ldj, 0)
        G_zk, G_ldj = torch.cat(G_zk, 0), torch.cat(G_ldj, 0)  

        loss_wrt_g = -1.0 * (model.base_dist.log_prob(g_zk).sum(1) + g_ldj)
        loss_wrt_G = -1.0 * (model.base_dist.log_prob(G_zk).sum(1) + G_ldj)

    return loss_wrt_g.mean(0).detach().item(), loss_wrt_G.mean(0).detach().item()


def update_rho(model, target_or_sample_fn, args):
    if model.component == 0 and model.all_trained == False:
        return
    
    model.eval()
    with torch.no_grad():

        rho_log = open(model.args.snap_dir + '/rho.log', 'a')
        print(f"\n\nUpdating weight for component {model.component} (all_trained={str(model.all_trained)})", file=rho_log)
        print('Initial Rho: ' + ' '.join([f'{val:1.2f}' for val in model.rho.data]), file=rho_log)
            
        step_size = 0.005
        tolerance = 0.00001
        min_iters = 25
        max_iters = 200 if model.all_trained else 100
        prev_rho = model.rho.data[model.component].item()
        for batch_id in range(max_iters):

            loss_wrt_g, loss_wrt_G = rho_gradient(model, target_or_sample_fn, args)
            if math.isnan(loss_wrt_g) or math.isnan(loss_wrt_G):
                print("NaN encountered, breaking", file=rho_log)
                model.rho[model.component] = 0.0
                break
            
            gradient = loss_wrt_g - loss_wrt_G                
            ss = step_size / (0.025 * batch_id + 1)
            clipped_gradient = max(min(ss * gradient, 0.01), -0.01)
            rho = min(max(prev_rho - clipped_gradient, 0.01), 1.0) # projected SGD with gradient clipping

            grad_msg = f'{batch_id: >3}. rho = {prev_rho:5.3f} -  {gradient:4.2f} * {ss:5.3f} = {rho:5.3f}'
            loss_msg = f"\tg vs G. Loss: ({loss_wrt_g:5.1f}, {loss_wrt_G:5.1f})."
            print(grad_msg + loss_msg, file=rho_log)
                    
            model.rho[model.component] = rho
            dif = abs(prev_rho - rho)
            prev_rho = rho

            if batch_id > min_iters and (batch_id > max_iters or dif < tolerance):
                break

        print('New Rho: ' + ' '.join([f'{val:1.2f}' for val in model.rho.data]), file=rho_log)
        rho_log.close()


def annealing_schedule(i, args):
    if args.min_beta == 1.0:
        return 1.0
    
    if args.flow == "boosted":
        if i >= args.iters_per_component * args.num_components or i == args.iters_per_component:
            rval = 1.0
        else:
            rval = 0.01 + ((i % args.iters_per_component) / args.iters_per_component)
    else:
        rval = 0.01 + i/10000.0

    rval = max(args.min_beta, min(args.max_beta, rval))
    return rval


def train(model, target_or_sample_fn, loss_fn, optimizer, scheduler, args):
    model.train()

    if args.flow == "boosted":
        model.component = 0
        prev_lr = []
        for c in range(args.num_components):
            optimizer.param_groups[c]['lr'] = args.learning_rate if c == model.component else 0.0
            prev_lr.append(args.learning_rate)
        for n, param in model.named_parameters():
            param.requires_grad = True if n.startswith(f"flow_param.{model.component}") else False
            
    for batch_id in range(args.num_steps+1):
        model.train()
        optimizer.zero_grad()
        beta = annealing_schedule(batch_id, args)

        # for "training-wheels" expriments: change the data distribution after a few iterations
        if args.dataset == "1gaussian" and batch_id == args.iters_per_component + 1:
            args.dataset = "2gaussians"
            target_or_sample_fn = make_toy_sampler(args)
        elif args.dataset == "u0" and batch_id == args.iters_per_component + 1:
            args.dataset = "u1"
            target_or_sample_fn = make_toy_density(args)

        loss, loss_terms = loss_fn(model, target_or_sample_fn, beta, args)
        loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
        optimizer.step()
        if not args.no_lr_schedule:
            scheduler.step(loss)
            if args.flow == "boosted":
                prev_lr[model.component] = optimizer.param_groups[model.component]['lr']

        boosted_component_converged = args.flow == "boosted" and batch_id % args.iters_per_component == 0 and batch_id > 0
        new_boosted_component = args.flow == "boosted" and batch_id % args.iters_per_component == 1
        if boosted_component_converged or new_boosted_component or batch_id % args.log_interval == 0:
            msg = f'{args.dataset}: step {batch_id:5d} / {args.num_steps}; loss {loss.item():8.3f} (beta={beta:5.4f})'
            if args.flow == "boosted":
                msg += f' | g vs G ({loss_terms[0]:8.3f}, {loss_terms[1]:8.3f})'
                msg += f' | p_log_prob {loss_terms[2]:8.3f}' if args.density_matching else ''
                msg += f' | c={model.component} (all={str(model.all_trained)[0]})'
                msg += f' | Rho=[' + ', '.join([f"{val:4.2f}" for val in model.rho.data]) + "]"
                msg += f' | ' + ' '.join([f"{optimizer.param_groups[c]['lr']:6.4f}" for c in range(model.num_components)])
            else:
                msg += f' | q_log_prob {loss_terms[0]:8.3f}'
                msg += f' | ldj {loss_terms[1]:8.3f}'
                msg += f' | p_log_prob {loss_terms[2]:7.3f}' if args.density_matching else ''
            logger.info(msg)

        if boosted_component_converged:
            if not args.no_rho_update:
                update_rho(model, target_or_sample_fn, args)
                
            model.increment_component()

            # update learning rates
            for c in range(model.num_components):
                optimizer.param_groups[c]['lr'] = prev_lr[c] if c == model.component else 0.0
            for n, param in model.named_parameters():
                param.requires_grad = True if n.startswith(f"flow_param.{model.component}") else False

        if (batch_id > 0 and batch_id % args.plot_interval == 0) or boosted_component_converged:
            with torch.no_grad():
                plot(batch_id, model, target_or_sample_fn, args)

            

 
def main(main_args=None):
    """
    use main_args to run this script as function in another script
    """

    # =========================================================================
    # PARSE EXPERIMENT SETTINGS, SETUP SNAPSHOTS DIRECTORY, LOGGING
    # =========================================================================
    args, kwargs = parse_args(main_args)

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
    args.density_matching = args.dataset.startswith('u')
    if args.density_matching:
        # target is energy potential to match
        target_or_sample_fn = make_toy_density(args)
        loss_fn = compute_kl_qp_loss
    else:
        # target is density to estimate to sample from
        target_or_sample_fn = make_toy_sampler(args)
        loss_fn = compute_kl_pq_loss


    train(model, target_or_sample_fn, loss_fn, optimizer, scheduler, args)
        

if __name__ == "__main__":
    main()

