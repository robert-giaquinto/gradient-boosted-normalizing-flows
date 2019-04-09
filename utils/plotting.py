import numpy as np
import matplotlib
import os

# noninteractive background
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def plot_training_curve(train_loss, validation_loss, fname='training_curve.pdf', labels=None):
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
    epoch_label = 'final' if epoch is None else str(epoch)
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

