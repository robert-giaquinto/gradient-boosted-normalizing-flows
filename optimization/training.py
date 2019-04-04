from __future__ import print_function
import torch
from functools import reduce
import numpy as np
import random

from optimization.loss import calculate_loss, calculate_loss_array
from utils.visual_evaluation import plot_reconstructions
from utils.log_likelihood import calculate_likelihood
from scipy.misc import logsumexp
# from torchsummary import summary

import numpy as np


def train(epoch, train_loader, model, opt, args):
    model.train()

    num_trained = 0
    total_data = len(train_loader.sampler)
    total_batches = len(train_loader)

    train_loss = np.zeros(total_batches)
    train_bpd = np.zeros(total_batches)
    train_rec = np.zeros(total_batches)
    train_kl = np.zeros(total_batches)

    # set warmup coefficient
    beta = min([(epoch * 1.) / max([args.warmup, 1.]), args.max_beta])

    for batch_id, (data, _) in enumerate(train_loader):
        data = data.to(args.device)

        if args.dynamic_binarization:
            data = torch.bernoulli(data)

        opt.zero_grad()
        x_mean, z_mu, z_var, ldj, z0, zk = model(data)

        # adjust learning rates if performing boosting
        if args.flow == "boosted":
            # set the learning rate of all but one weak learner to zero
            # TODO explore exploit strategy for selecting weak learner
            # for now, just to uniform random
            learner_chosen = random.randint(0, args.num_learners - 1)

            # first num_learners groups correspond to the flow parameters for each weak learner
            for c in range(args.num_learners):
                if c == learner_chosen:
                    opt.param_groups[c]['lr'] = args.learning_rate
                else:
                    opt.param_groups[c]['lr'] = 0.0

            # only one row of log det jacobian is retained
            ldj = ldj[learner_chosen, :]

        loss, rec, kl = calculate_loss(x_mean, data, z_mu, z_var, z0, zk, ldj, args, beta=beta)
        loss.backward()
        opt.step()

        train_loss[batch_id] = loss.item()
        train_rec[batch_id] = rec.item()
        train_kl[batch_id] = kl.item()

        num_trained += len(data)
        pct_complete = 100. * batch_id / total_batches
        if args.log_interval > 0 and batch_id % args.log_interval == 0:
            msg = 'Epoch: {:3d} [{:5d}/{:5d} ({:2.0f}%)] \tLoss: {:11.6f}\trec: {:11.3f}\tkl: {:11.6f}'

            if args.input_type == 'binary':
                print(msg.format(
                    epoch, num_trained, total_data, pct_complete, loss.item(), rec.item(), kl.item()))
            else:
                msg += '\tbpd: {:8.6f}'
                bpd = loss.item() / (np.prod(args.input_size) * np.log(2.))
                print(msg.format(
                    epoch, num_trained, total_data, pct_complete,loss.item(), rec.item(), kl.item(), bpd))

    if args.input_type == 'binary':
        print('====> Epoch: {:3d} Average train loss: {:.4f}'.format(
            epoch, train_loss.sum() / total_batches))
    else:
        train_bpd = train_loss / (np.prod(args.input_size) * np.log(2.))
        print('====> Epoch: {:3d} Average train loss: {:.4f}, average bpd: {:.4f}'.format(
            epoch, train_loss.sum() / total_batches, train_bpd.sum() / total_batches))

    return train_loss, train_rec, train_kl


def evaluate(data_loader, model, args, save_plots=True, epoch=None):
    model.eval()

    loss = 0.0
    rec = 0.0
    kl = 0.0

    for batch_id, (data, _) in enumerate(data_loader):
        data = data.to(args.device)

        x_mean, z_mu, z_var, ldj, z0, zk = model(data)

        batch_loss, batch_rec, batch_kl = calculate_loss(x_mean, data, z_mu, z_var, z0, zk, ldj, args)
        loss += batch_loss.item()
        rec += batch_rec.item()
        kl += batch_kl.item()

        # PRINT RECONSTRUCTIONS
        save_this_epoch = epoch is None or epoch==1 or \
            (args.plot_interval > 0 and epoch % args.plot_interval == 0)
        if batch_id == 0 and save_plots and save_this_epoch:
            plot_reconstructions(data=data, recon_mean=x_mean, loss=batch_loss, args=args, epoch=epoch)

    avg_loss = loss / len(data_loader)
    avg_rec = rec / len(data_loader)
    avg_kl = kl / len(data_loader)
    return avg_loss, avg_rec, avg_kl




def evaluate_likelihood(data_loader, model, args, S=5000, MB=1000):
    """
    Calculate negative log likelihood using importance sampling
    """

    model.eval()

    X = torch.cat([x for x, y in list(data_loader)], 0).to(args.device)

    # set auxiliary variables for number of training and test sets
    N_test = X.size(0)

    likelihood_test = []

    if S <= MB:
        R = 1
    else:
        R = S // MB
        S = MB

    for j in range(N_test):
        if j % 100 == 0:
            print('Progress: {:.2f}%'.format(j / (1. * N_test) * 100))

        x_single = X[j].unsqueeze(0)

        a = []
        for r in range(0, R):
            # Repeat it for all training points
            x = x_single.expand(S, *x_single.size()[1:]).contiguous()
            x_mean, z_mu, z_var, ldj, z0, zk = model(x)
            a_tmp = calculate_loss_array(x_mean, x, z_mu, z_var, z0, zk, ldj, args)
            a.append(-a_tmp.cpu().data.numpy())

        # calculate max
        a = np.asarray(a)
        a = np.reshape(a, (a.shape[0] * a.shape[1], 1))
        likelihood_x = logsumexp(a)
        likelihood_test.append(likelihood_x - np.log(len(a)))

    likelihood_test = np.array(likelihood_test)
    nll = -np.mean(likelihood_test)
    return nll

