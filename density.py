import argparse
import datetime
import torch
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
from utils.load_data import load_dataset
from utils.density_plotting import plot
from utils.density_data import make_toy_density, make_toy_sampler
from main_experiment import init_model, init_optimizer, init_log


logger = logging.getLogger(__name__)


TOY_DATASETS = ["8gaussians", "2gaussians", "1gaussian",  "swissroll", "rings", "moons", "pinwheel", "cos", "2spirals", "checkerboard", "line", "circles", "joint_gaussian"]
ENERGY_FUNS = ['u0', 'u1', 'u2', 'u3', 'u4', 'u5']

parser = argparse.ArgumentParser(description='PyTorch Ensemble Normalizing flows')

parser.add_argument('--dataset', type=str, default='mnist', help='Dataset choice.', choices=TOY_DATASETS + ENERGY_FUNS)

# seeds
parser.add_argument('--manual_seed', type=int, default=123,
                    help='manual seed, if not given resorts to random seed.')
parser.add_argument('--freyseed', type=int, default=123,
                    help="Seed for shuffling frey face dataset for test split. Ignored for other datasets.")

# gpu/cpu
parser.add_argument('--gpu_id', type=int, default=0, metavar='GPU', help='choose GPU to run on.')
parser.add_argument('--num_workers', type=int, default=1,
                    help='How many CPU cores to run on. Setting to 0 uses os.cpu_count() - 1.')
parser.add_argument('--no_cuda', action='store_true', default=False,help='disables CUDA training')

# Reporting
parser.add_argument('--log_interval', type=int, default=0,
                    help='how many batches to wait before logging training status. Set to <0 to turn off.')
parser.add_argument('--plot_interval', type=int, default=10,
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

# flow parameters
parser.add_argument('--flow', type=str, default='planar',
                    choices=['planar', 'radial', 'boosted', 'bagged'],
                    help="""Type of flows to use, no flows can also be selected""")
parser.add_argument('--num_flows', type=int, default=2, help='Number of flow layers, ignored in absence of flows')
parser.add_argument('--z_size', type=int, default=2, help='how many stochastic hidden units')

# Bagging/Boosting parameters
parser.add_argument('--num_components', type=int, default=4,
                    help='How many components are combined to form the flow')
parser.add_argument('--component_type', type=str, default='planar', choices=['planar', 'radial'],
                    help='When flow is bagged or boosted -- what type of flow should each component implement.')



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

    if args.flow in ['boosted', 'bagged']:
        if args.regularization_rate < 0.0:
            raise ValueError("For boosting the regularization rate should be greater than or equal to zero.")
        
        args.snap_dir += '_' + args.component_type + '_num_components_' + str(args.num_components) + '_regularization_' + f'{int(100*args.regularization_rate):d}'

    is_annealed = "_annealed" if args.min_beta < 1.0 else ""
    args.snap_dir += '_on_' + args.dataset + is_annealed + "_" +args.model_signature + '/'

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
    z = model.base_dist.sample((args.batch_size,))
    q_log_prob = model.base_dist.log_prob(z).sum(1)
    
    if args.flow == "boosted":
        if model.component < model.num_components:
            density_from = '-c' if model.all_trained else '1:c-1'
            sample_from = 'c'
        else:
            density_from = '1:c'
            sample_from = '1:c'
            
        zk, entropy_ldj, boosted_ldj = model.flow(z, sample_from=sample_from, density_from=density_from)
        p_log_prob = target_fn(zk) * beta
        
        if model.component == 0 and model.all_trained == False:
            logdet = entropy_ldj
            loss = q_log_prob - logdet + p_log_prob
        else:
            logdet = boosted_ldj + entropy_ldj * args.regularization_rate
            loss = (1 + args.regularization_rate) * q_log_prob - logdet + p_log_prob

        return loss.mean(0), (q_log_prob.mean().item(), entropy_ldj.mean().item(), boosted_ldj.mean().item(), p_log_prob.mean().item())

    else:
        zk, logdet = model.flow(z)
        
        p_log_prob = target_fn(zk) * beta
        loss = q_log_prob - logdet + p_log_prob
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
        if model.component < model.num_components:
            density_from = '-c' if model.all_trained else '1:c-1'
            sample_from = 'c'
        else:
            density_from = '1:c'
            sample_from = '1:c'
            
        z, entropy_ldj, boosted_ldj = model.flow(sample, sample_from=sample_from, density_from=density_from)
        q_log_prob = model.base_dist.log_prob(z).sum(1)
        
        if model.component == 0 and model.all_trained == False:
            loss = -1.0 * (q_log_prob + entropy_ldj)
        else:
            logdet = boosted_ldj + entropy_ldj * args.regularization_rate
            loss = -1.0 * ((1 + args.regularization_rate) * q_log_prob + logdet)

        return loss.mean(0), (q_log_prob.mean().item(), entropy_ldj.mean().detach().item(), boosted_ldj.mean().detach().item())
        
    else:
        z, logdet = model.flow(sample)
        q_log_prob = model.base_dist.log_prob(z).sum(1)
        loss = -1.0 * (q_log_prob + logdet)
        return loss.mean(0), (q_log_prob.mean().item(), logdet.mean().detach().item())


@torch.no_grad()
def rho_gradient(model, target_or_sample_fn, args):
    if args.density_matching:
        # density matching of a target function
        z = model.base_dist.sample((args.batch_size,))
        q_log_prob = model.base_dist.log_prob(z)

        g_zk, _, g_ldj = model.flow(z, sample_from="c", density_from="1:c")
        loss_wrt_g = q_log_prob - g_ldj + target_or_sample_fn(g_zk)

        G_zk, _, G_ldj = model.flow(z, sample_from="1:c-1", density_from="1:c")
        loss_wrt_G = q_log_prob - G_ldj + target_or_sample_fn(G_zk)

    else:
        # estimate density from a sampler
        sample = target_or_sample_fn(args.batch_size).to(args.device)

        g_zk, _, g_ldj = model.flow(sample, sample_from="c", density_from="1:c")
        loss_wrt_g = -1.0 * (model.base_dist.log_prob(g_zk) + g_ldj)
                
        G_zk, _, G_ldj = model.flow(sample, sample_from="1:c-1", density_from="1:c")
        loss_wrt_G = -1.0 * (model.base_dist.log_prob(G_zk) + G_ldj)

    #gradient = loss_wrt_g.mean(0).detach().item() - loss_wrt_G.mean(0).detach().item()
    return loss_wrt_g.mean(0).detach().item(), loss_wrt_G.mean(0).detach().item()


@torch.no_grad()
def update_rho(model, target_or_sample_fn, args):
    if model.component > 0:
        model.eval()
        with torch.no_grad():
                
            #grad_log = open(model.args.snap_dir + '/gradient.log', 'a')
            #print('\n\nInitial Rho: ' + ' '.join([f'{val:1.2f}' for val in model.rho.data]), file=grad_log)
            print('\n\nInitial Rho: ' + ' '.join([f'{val:1.2f}' for val in model.rho.data]))
            
            step_size = 0.005
            tolerance = 0.0001
            min_iters = 25
            max_iters = 250
            prev_rho = 1.0 / model.num_components
            
            for batch_id in range(max_iters):

                loss_wrt_g, loss_wrt_G = rho_gradient(model, target_or_sample_fn, args)
                gradient = loss_wrt_g - loss_wrt_G
                ss = step_size / (0.01 * batch_id + 1)
                rho = min(max(prev_rho - ss * gradient, 0.025), 1.0)

                grad_msg = f'{batch_id: >3}. rho = {prev_rho:5.3f} -  {gradient:4.2f} * {ss:5.3f} = {rho:5.3f}'
                loss_msg = f"\tg vs G. Loss: ({loss_wrt_g:5.1f}, {loss_wrt_G:5.1f})."

                #print(grad_msg + loss_msg, file=grad_log)
                print(grad_msg + loss_msg)
                    
                model.rho[model.component] = rho
                dif = abs(prev_rho - rho)
                prev_rho = rho

                if batch_id > min_iters and (batch_id > max_iters or dif < tolerance):
                    break

            #print('New Rho: ' + ' '.join([f'{val:1.2f}' for val in model.rho.data]), file=grad_log)
            #grad_log.close()
            print('New Rho: ' + ' '.join([f'{val:1.2f}' for val in model.rho.data]))


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

    if args.flow == "bagged":
        component_loss = np.zeros((2, args.num_components))
    elif args.flow == "boosted":
        model.component = 0
        for c in range(args.num_components):
            optimizer.param_groups[c]['lr'] = args.learning_rate if c == model.component else 0.0

            
    for batch_id in range(args.num_steps):
        model.train()
        optimizer.zero_grad()
        beta = annealing_schedule(batch_id, args)

        if args.dataset == "1gaussian" and batch_id == args.iters_per_component + 1:
            args.dataset = "2gaussians"
            target_or_sample_fn = make_toy_sampler(args)
        elif args.dataset == "u0" and batch_id == args.iters_per_component + 1:
            args.dataset = "u1"
            target_or_sample_fn = make_toy_density(args)

        loss, loss_terms = loss_fn(model, target_or_sample_fn, beta, args)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
        optimizer.step()
        #scheduler.step(loss)
        
        new_boosted_component = batch_id % args.iters_per_component == 1 and args.flow == "boosted" and batch_id != 1
        if new_boosted_component or batch_id % args.log_interval == 0:
            msg = f'{args.dataset}: step {batch_id:5d} / {args.num_steps}; loss {loss.item():6.3f} (beta={beta:5.4f})'
            msg += f' | q_log_prob {loss_terms[0]:6.3f}'

            if args.flow == "boosted":
                msg += f' | ldjs ({loss_terms[1]:6.3f}, {loss_terms[2]:6.3f})'
                msg += f' | p_log_prob {loss_terms[3]:6.3f}' if args.density_matching else ''
                msg += f' | c={model.component} (all={str(model.all_trained)[0]})'
                msg += f' | Rho=[' + ', '.join([f"{val:4.2f}" for val in model.rho.data]) + "]"
                msg += f' | LR=[' + ', '.join([f"{optimizer.param_groups[c]['lr']:6.5f}" for c in range(args.num_components)]) + "]"
            else:
                msg += f' | ldj {loss_terms[1]:6.3f}'
                msg += f' | p_log_prob {loss_terms[2]:6.3f}' if args.density_matching else ''

            logger.info(msg)

        if new_boosted_component or batch_id % args.plot_interval == 0:
            plot(batch_id, model, target_or_sample_fn, args)

        if args.flow == "bagged":
            if batch_id % args.iters_per_component == 0 and batch_id > 0:
                #model.update_rho(component_loss)
                component_loss = np.zeros((2, args.num_components))
            else:
                component_loss[0, model.component] += loss.item()
                component_loss[1, model.component] += args.batch_size
        elif args.flow == "boosted":
            if batch_id % args.iters_per_component == 0 and batch_id > 0:
                #update_rho(model, target_or_sample_fn, args)
                if model.component == model.num_components - 1:
                    # loop through and retrain each component
                    model.component = 0
                    model.all_trained = True
                else:
                    model.component = min(model.component + 1, model.num_components - 1)

                for c in range(args.num_components):
                    optimizer.param_groups[c]['lr'] = args.learning_rate if c == model.component else 0.0
            

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
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           factor=0.5,
                                                           patience=2000,
                                                           min_lr=5e-4,
                                                           verbose=True,
                                                           threshold_mode='abs')
    
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

