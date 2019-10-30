import torch
import os

import logging
logger = logging.getLogger(__name__)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


@torch.no_grad()
def plot(batch_id, model, potential_or_sampling_fn, args):
    # TODO for plotting boosted show both the flow density according to boosted_ldj but seperately for entropy_ldj
    # TODO plot each component seperately
    model.eval()
    
    #n_pts = 1000
    n_pts = 250
    range_lim = 5

    # construct test points
    test_grid = setup_grid(range_lim * 2, n_pts, args)

    # plot
    if args.density_matching:
        fig, axs = plt.subplots(2, 2, figsize=(12,12), subplot_kw={'aspect': 'equal'})
        #fig.subplots_adjust(top=0.9)
        plot_potential(potential_or_sampling_fn, axs[0, 0], test_grid, n_pts)
        plot_flow_samples(model, axs[0, 1], n_pts, args.batch_size, args, "all")

        plot_inv_flow_density(model, axs[1, 0], test_grid, n_pts, args.batch_size, args, "current")
        plot_flow_samples(model, axs[1, 1], n_pts, args.batch_size, args, "current")

        #plot_q0_density(model, axs[1, 0], test_grid, n_pts, args.batch_size, args)
        #plot_inv_flow_density(model, axs[1, 2], test_grid, n_pts, args.batch_size, args, "all")

    else:
        fig, axs = plt.subplots(1, 3, figsize=(12,4), subplot_kw={'aspect': 'equal'})
        plot_samples(potential_or_sampling_fn, axs[0], range_lim, n_pts)
        plot_fwd_flow_density(model, axs[1], test_grid, n_pts, args.batch_size, args, "current")
        plot_fwd_flow_density(model, axs[2], test_grid, n_pts, args.batch_size, args, "all")

    # format
    for ax in plt.gcf().axes: format_ax(ax, range_lim)
    plt.tight_layout(rect=[0, 0, 1.0, 0.95])

    title = f'{args.flow.title()} Flow, K={args.num_flows}'
    title += f', C={args.num_components}, Reg={args.regularization_rate:.2f}' if args.flow == "boosted" else ''
    fig.suptitle(title, y=0.98, fontsize=20)

    # save
    plt.savefig(os.path.join(args.snap_dir, f'vis_step_{batch_id}.png'))
    plt.close()
    

def setup_grid(range_lim, n_pts, args):
    x = torch.linspace(-range_lim, range_lim, n_pts)
    xx, yy = torch.meshgrid((x, x))
    zz = torch.stack((xx.flatten(), yy.flatten()), dim=1)
    return xx, yy, zz.to(args.device)


def format_ax(ax, range_lim):
    ax.set_xlim(-range_lim, range_lim)
    ax.set_ylim(-range_lim, range_lim)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.invert_yaxis()

    
def plot_potential(potential_fn, ax, test_grid, n_pts):
    xx, yy, zz = test_grid
    ax.pcolormesh(xx, yy, torch.exp(-potential_fn(zz)).view(n_pts, n_pts).cpu().data, cmap=plt.cm.viridis)
    ax.set_title('Target Density', fontdict={'fontsize': 20})

    
def plot_samples(samples_fn, ax, range_lim, n_pts):
    samples = samples_fn(n_pts**2).numpy()
    ax.hist2d(samples[:,0], samples[:,1], range=[[-range_lim, range_lim], [-range_lim, range_lim]], bins=n_pts, cmap=plt.cm.viridis)
    ax.set_title('Target Samples', fontdict={'fontsize': 20})

    
def plot_flow_samples(model, ax, n_pts, batch_size, args, sample_from):
    z = model.base_dist.sample((n_pts**2,))
    zk = torch.cat([flow(model, z_, args, sample_from)[0] for z_ in z.split(batch_size, dim=0)], 0)
    zk = zk.cpu().numpy()
    
    # plot
    ax.hist2d(zk[:,0], zk[:,1], bins=n_pts, cmap=plt.cm.viridis)
    ax.set_facecolor(plt.cm.viridis(0.))
    caption = f"Training c={model.component}" if sample_from == "current" else f"From All Components"
    ax.set_title('Flow Samples ' + caption, fontdict={'fontsize': 20})

    
def plot_fwd_flow_density(model, ax, test_grid, n_pts, batch_size, args, sample_from):
    """
    plots square grid and flow density; where density under the flow is exp(log_flow_base_dist + logdet)
    """
    xx, yy, zz = test_grid
    
    # compute posterior approx density
    zzk, logdet = [], []
    for zz_i in zz.split(batch_size, dim=0):        
        zzk_i, logdet_i = flow(model, zz_i, args, sample_from)
        zzk += [zzk_i]
        logdet += [logdet_i]
        
    zzk, logdet = torch.cat(zzk, 0), torch.cat(logdet, 0)
    log_prob = (1 + args.regularization_rate) * model.base_dist.log_prob(zzk).sum(1) + logdet
    prob = log_prob.exp().cpu()
    
    # plot
    ax.pcolormesh(xx, yy, prob.view(n_pts,n_pts).data, cmap=plt.cm.viridis)
    ax.set_facecolor(plt.cm.jet(0.))
    caption = f"While Training c={model.component}" if sample_from == "current" else f"Sampling From All Components"
    ax.set_title('Flow Density ' + caption, fontdict={'fontsize': 20})

    
def plot_inv_flow_density(model, ax, test_grid, n_pts, batch_size, args, sample_from):
    """
    plots transformed grid and density; where density is exp(loq_flow_base_dist - logdet)
    """        
    xx, yy, zz = test_grid
    
    # compute posterior approx density
    zzk, logdet = [], []
    for zz_i in zz.split(batch_size, dim=0):
        zzk_i, logdet_i = flow(model, zz_i, args, sample_from)
        zzk += [zzk_i]
        logdet += [logdet_i]
    
    zzk, logdet = torch.cat(zzk, 0), torch.cat(logdet, 0)
    log_q0 = model.base_dist.log_prob(zz).sum(1)
    #log_qk = log_q0 - logdet
    log_qk = (1 + args.regularization_rate) * log_q0 - logdet
    qk = log_qk.exp().cpu()
    zzk = zzk.cpu()

    # plot
    ax.pcolormesh(zzk[:,0].view(n_pts,n_pts).data, zzk[:,1].view(n_pts,n_pts).data, qk.view(n_pts,n_pts).data, cmap=plt.cm.viridis)
    ax.set_facecolor(plt.cm.viridis(0.0))
    caption = f"While Training c={model.component}" if sample_from == "current" else f"Sampling From All Components"
    ax.set_title('Flow Density ' + caption, fontdict={'fontsize': 20})


def plot_q0_density(model, ax, test_grid, n_pts, batch_size, args):
    """
    plots transformed grid and density; where density is exp(loq_flow_base_dist - logdet)
    """        
    xx, yy, zz = test_grid
    log_q0 = model.base_dist.log_prob(zz).sum(1)    
    q0 = log_q0.exp().cpu()
    
    # plot
    ax.pcolormesh(zz[:,0].view(n_pts,n_pts).data, zz[:,1].view(n_pts,n_pts).data, q0.view(n_pts,n_pts).data, cmap=plt.cm.viridis)
    ax.set_facecolor(plt.cm.viridis(0.))
    ax.set_title('Base q_0 Density', fontdict={'fontsize': 20})
    

def flow(model, z, args, sample_from="current"):
    if args.flow == "boosted":
        if sample_from == "all":
            zk, _, logdet = model.flow(z, sample_from="1:c", density_from="1:c")
        elif sample_from == "current":
            density_from = '-c' if model.all_trained else '1:c-1'
            zk, entropy_ldj, boosted_ldj = model.flow(z, sample_from="c", density_from=density_from)
            logdet = args.regularization_rate * entropy_ldj + boosted_ldj
        else:
            raise ValueError("sample_from can only be current or all")
                               
    else:
        zk, logdet = model.flow(z)

    return zk, logdet
