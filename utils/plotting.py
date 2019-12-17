import numpy as np
import matplotlib
import os
import torch

# noninteractive background
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def plot_training_curve(train_loss, validation_loss, fname='training_curve.png', labels=None):
    """
    Plots train_loss and validation loss as a function of optimization iteration
    :param train_loss: np.array of train_loss (1D or 2D)
    :param validation_loss: np.array of validation loss (1D or 2D)
    :param fname: output file name
    :param labels: if train_loss and validation loss are 2D, then labels indicate which variable is varied
    accross training curves.
    :return: None
    """
    plt.close()

    matplotlib.rcParams.update({'font.size': 14})
    matplotlib.rcParams['mathtext.fontset'] = 'stix'
    matplotlib.rcParams['font.family'] = 'STIXGeneral'

    if len(train_loss.shape) == 1:
        # Single training curve
        fig, ax = plt.subplots(nrows=1, ncols=1)
        figsize = (6, 4)

        if train_loss.shape[0] == validation_loss.shape[0]:
            # validation score evaluated every iteration
            x = np.arange(train_loss.shape[0])
            ax.plot(x, train_loss, '-', lw=2., color='black', label='train')
            ax.plot(x, validation_loss, '-', lw=2., color='blue', label='val')

        elif train_loss.shape[0] % validation_loss.shape[0] == 0:
            # validation score evaluated every epoch
            x = np.arange(train_loss.shape[0])
            ax.plot(x, train_loss, '-', lw=2., color='black', label='train')

            x = np.arange(validation_loss.shape[0])
            x = (x + 1) * train_loss.shape[0] / validation_loss.shape[0]
            ax.plot(x, validation_loss, '-', lw=2., color='blue', label='val')
        else:
            raise ValueError('Length of train_loss and validation_loss must be equal or divisible')

        miny = np.minimum(validation_loss.min(), train_loss.min()) - 20.
        maxy = np.maximum(validation_loss.max(), train_loss.max()) + 30.
        ax.set_ylim([miny, maxy])

    elif len(train_loss.shape) == 2:
        # Multiple training curves

        cmap = plt.cm.brg

        cNorm = matplotlib.colors.Normalize(vmin=0, vmax=train_loss.shape[0])
        scalarMap = matplotlib.cm.ScalarMappable(norm=cNorm, cmap=cmap)

        fig, ax = plt.subplots(nrows=1, ncols=1)
        figsize = (6, 4)

        if labels is None:
            labels = ['%d' % i for i in range(train_loss.shape[0])]

        if train_loss.shape[1] == validation_loss.shape[1]:
            for i in range(train_loss.shape[0]):
                color_val = scalarMap.to_rgba(i)

                # validation score evaluated every iteration
                x = np.arange(train_loss.shape[0])
                ax.plot(x, train_loss[i], '-', lw=2., color=color_val, label=labels[i])
                ax.plot(x, validation_loss[i], '--', lw=2., color=color_val)

        elif train_loss.shape[1] % validation_loss.shape[1] == 0:
            for i in range(train_loss.shape[0]):
                color_val = scalarMap.to_rgba(i)

                # validation score evaluated every epoch
                x = np.arange(train_loss.shape[1])
                ax.plot(x, train_loss[i], '-', lw=2., color=color_val, label=labels[i])

                x = np.arange(validation_loss.shape[1])
                x = (x+1) * train_loss.shape[1] / validation_loss.shape[1]
                ax.plot(x, validation_loss[i], '-', lw=2., color=color_val)

        miny = np.minimum(validation_loss.min(), train_loss.min()) - 20.
        maxy = np.maximum(validation_loss.max(), train_loss.max()) + 30.
        ax.set_ylim([miny, maxy])

    else:
        raise ValueError('train_loss and validation_loss must be 1D or 2D arrays')

    ax.set_xlabel('iteration')
    ax.set_ylabel('loss')
    plt.title('Training and validation loss')

    fig.set_size_inches(figsize)
    fig.subplots_adjust(hspace=0.1)
    plt.savefig(fname, bbox_inches='tight')

    plt.close()


def plot_reconstructions(data, recon_mean, loss, args, epoch, size_x=4, size_y=4):
    if not os.path.exists(args.snap_dir + 'reconstruction/'):
        os.makedirs(args.snap_dir + 'reconstruction/')

    if epoch == 1:
        # VISUALIZATION: plot real images
        plot_images(args, data.data.cpu().numpy()[0:size_x*size_y], args.snap_dir + 'reconstruction/', 'real',
                    size_x=size_x, size_y=size_y)

    if args.input_type == 'multinomial':
        # data is already between 0 and 1
        num_classes = 256

        # Find largest class logit
        tmp = recon_mean.view(-1, num_classes, *args.input_size).max(dim=1)[1]
        recon_mean = tmp.float() / (num_classes - 1.)

    # VISUALIZATION: plot reconstructions
    epoch_label = 'final' if epoch is None else f"{epoch:03d}"
    if args.input_type == 'multinomial':
        fname = epoch_label + '_bpd_%5.3f' % loss.item()
    elif args.input_type == 'binary':
        fname = epoch_label + '_elbo_%6.4f' % loss.item()

    plot_images(args, recon_mean.data.cpu().numpy()[0:size_x*size_y],
        args.snap_dir + 'reconstruction/', fname, size_x=size_x, size_y=size_y)


def plot_images(args, x_sample, dir, file_name, size_x=3, size_y=3):

    fig = plt.figure(figsize=(size_x, size_y))
    # fig = plt.figure(1)
    gs = gridspec.GridSpec(size_x, size_y)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(x_sample):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        sample = sample.reshape((args.input_size[0], args.input_size[1], args.input_size[2]))
        sample = sample.swapaxes(0, 2)
        sample = sample.swapaxes(0, 1)
        if (args.input_type == 'binary') or (args.input_type in ['multinomial'] and args.input_size[0] == 1):
            sample = sample[:, :, 0]
            plt.imshow(sample, cmap='gray', vmin=0, vmax=1)
        else:
            plt.imshow(sample)

    plt.savefig(dir + file_name + '.png', bbox_inches='tight')
    plt.close(fig)


def plot_decoded_random_sample(args, model, size_x=3, size_y=3):
    """
    Use model to decode a random sample of the latent space
    Save in plot
    """
    # decode random sample
    z_rs = torch.normal(torch.from_numpy(np.zeros((25, model.z_size))).float(), 1.).to(args.device)
    x_rs_recon = model.decode(z_rs)
    samples = x_rs_recon.data.cpu().numpy()[:size_x*size_y]

    fig = plt.figure(figsize=(size_x, size_y))
    gs = gridspec.GridSpec(size_x, size_y)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(args.input_size[1], args.input_size[2]), cmap='Greys_r')

    plt.savefig(args.snap_dir + 'random_sample_decoded.png', bbox_inches='tight')
    plt.close(fig)


def plot_decoded_manifold(model, args, x_lim=5, y_lim=5, nx = 25):
    """
    Use model to decode a range of values within the latent space
    and plot.
    """
    #visualize 2D latent space
    ny = nx
    x_values = np.linspace(-x_lim, x_lim, nx)
    y_values = np.linspace(-y_lim, y_lim, ny)
    canvas = np.empty((args.input_size[1]*ny, args.input_size[2]*nx))

    for i, yi in enumerate(x_values):
        for j, xi in enumerate(y_values):
            zz = np.array( [[xi], [yi]], dtype='float32' ).T
            z_rs = torch.from_numpy(zz).to(args.device)
            x_rs_recon = model.decode(z_rs)
            x = x_rs_recon.data.cpu().numpy().flatten().reshape(args.input_size[1], args.input_size[2])

            xi0 = (nx-i-1) * args.input_size[1]
            xi1 = (nx-i) * args.input_size[1]
            yi0 = j * args.input_size[2]
            yi1 = (j+1) * args.input_size[2]
            canvas[xi0:xi1, yi0:yi1] = x

    fig = plt.figure(figsize=(8, 10))
    plt.imshow(canvas, origin="upper", cmap='Greys_r')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(args.snap_dir, 'latentSpace2D.png'), bbox_inches='tight')
    plt.close(fig)


def plot_data_manifold(model, data_loader, args, limit=None):
    """
    Pass data through encoder (and flows if possible) and then plot
    latent dimension.

    ASSUMES args.z_size == 2
    """
    if model.z_size > 2:
        raise ValueError("Can only plot latent dimension of data when z_size==2")

    X = torch.cat([x for x, y in list(data_loader)], 0).to(args.device)
    # MNIST should have just integer labels in a vector
    Y = torch.cat([y for x, y in list(data_loader)], 0).data.cpu().numpy().astype(int)[:, 0]

    if args.flow == "no_flow":
        z_mean, z_var = model.encode(X)
        Z = model.reparameterize(z_mean, z_var).data.cpu().numpy()
    else:
        # not efficient, but much easier to just pass model through entire network
        _, _, _, _, _, Z = model(X)
        Z = Z.data.cpu().numpy()

    if len( np.shape(Y) ) > 2:
        # For one-hot vectors
        Y_label = np.argmax(Y, 1)
    else:
        Y_label = Y

    fig = plt.figure()
    plt.scatter(Z[:, 0], Z[:, 1], c=Y_label,
                alpha=0.5, edgecolors='k', cmap='gist_ncar')
    plt.colorbar()

    if limit != None:
        # set axes range
        limit = np.abs(limit)
        plt.xlim(-limit, limit)
        plt.ylim(-limit, limit)

    plt.savefig(args.snap_dir + 'latent_manifold_of_data.png', bbox_inches='tight')
    plt.close(fig)



def format_ax(ax, range_lim):
    ax.set_xlim(-range_lim, range_lim)
    ax.set_ylim(-range_lim, range_lim)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.invert_yaxis()

@torch.no_grad()
def plot_flow_samples(epoch, model, data_loader, args):
    fig, axs = plt.subplots(1, 2, figsize=(12,12), subplot_kw={'aspect': 'equal'})
    alpha = 0.3
    range_lim = 10

    X = torch.cat([x for x, y in list(data_loader)], 0).to(args.device)
    if args.flow == "boosted":
        h, z_mu, z_var = model.encode(X)
    else:
        z_mu, z_var = model.encode(X)
    z0 = model.reparameterize(z_mu, z_var).data
    mu = z_mu.data.mean(0)

    mask_1 = (z0[:, 0] >= mu[0]) & (z0[:, 1] >= mu[1])
    mask_2 = (z0[:, 0] >= mu[0]) & (z0[:, 1] < mu[1])
    mask_3 = (z0[:, 0] < mu[0]) & (z0[:, 1] >= mu[1])
    mask_4 = (z0[:, 0] < mu[0]) & (z0[:, 1] < mu[1])

    if args.flow == 'boosted':
        zk = torch.cat([model.flow(z_, sample_from="1:c", h=h_)[0][-1].data \
                        for z_, h_ in zip(z0.split(args.batch_size, dim=0), h.split(args.batch_size, dim=0))], 0)
    else:
        zk, _ = model.flow(z0)

    z0 = z0.cpu().detach()
    zk = zk.cpu().detach()

    axs[0].set_title('Base $z_0$', fontdict={'fontsize': 20})
    axs[0].scatter(z0[:, 0][mask_1], z0[:, 1][mask_1], color='C0', alpha=alpha)
    axs[0].scatter(z0[:, 0][mask_2], z0[:, 1][mask_2], color='C1', alpha=alpha)
    axs[0].scatter(z0[:, 0][mask_3], z0[:, 1][mask_3], color='C3', alpha=alpha)
    axs[0].scatter(z0[:, 0][mask_4], z0[:, 1][mask_4], color='C4', alpha=alpha)

    axs[1].set_title('Transformed $z_K$', fontdict={'fontsize': 20})
    axs[1].scatter(zk[:, 0][mask_1], zk[:, 1][mask_1], color='C0', alpha=alpha)
    axs[1].scatter(zk[:, 0][mask_2], zk[:, 1][mask_2], color='C1', alpha=alpha)
    axs[1].scatter(zk[:, 0][mask_3], zk[:, 1][mask_3], color='C3', alpha=alpha)
    axs[1].scatter(zk[:, 0][mask_4], zk[:, 1][mask_4], color='C4', alpha=alpha)

    for ax in plt.gcf().axes: format_ax(ax, range_lim)
    plt.tight_layout(rect=[0, 0, 1.0, 0.95])

    title = f'{args.flow.title()} Flow, K={args.num_flows}'
    title += f', Annealed' if args.min_beta < 1.0 else ', No Annealing'
    title += f', C={args.num_components}, Reg={args.regularization_rate:.2f}, Training $c_{model.component}$' if args.flow == "boosted" else ''
    fig.suptitle(title, y=0.98, fontsize=20)
    if epoch is None:
        plt.savefig(os.path.join(args.snap_dir, f'flow_samples_final.png'))
    else:
        plt.savefig(os.path.join(args.snap_dir, f'flow_samples_{epoch:0>4d}.png'))
    plt.close()

    #axs[0, 0].set_xlim(-range_lim, range_lim)
    #axs[0, 0].set_ylim(-range_lim, range_lim)
    #axs[0, 0].get_xaxis().set_visible(False)
    #axs[0, 0].get_yaxis().set_visible(False)
    #ax.invert_yaxis()

    # for s, title in zip([z0, zk], ['Base Distribution $z_0$', f'Boosted $z_K$ c=1:{model.num_components if model.all_trained else model.component}']):
    #     plt.figure(figsize=(8, 8))
    #     plt.title(title)
    #     #plt.xlim(-range_lim, range_lim)
    #     #plt.ylim(-range_lim, range_lim)
    #     #plt.get_xaxis().set_visible(False)
    #     #plt.get_yaxis().set_visible(False)
    #     #plt.invert_yaxis()
    #     plt.scatter(s[:, 0][mask_1], s[:, 1][mask_1], color='C0', alpha=alpha, label='C0')
    #     plt.scatter(s[:, 0][mask_2], s[:, 1][mask_2], color='C1', alpha=alpha, label='C1')
    #     plt.scatter(s[:, 0][mask_3], s[:, 1][mask_3], color='C3', alpha=alpha, label='C3')
    #     plt.scatter(s[:, 0][mask_4], s[:, 1][mask_4], color='C4', alpha=alpha, label='C4')
    #     plt.legend()

        


