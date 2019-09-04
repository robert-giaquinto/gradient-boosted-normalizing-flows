import torch
from functools import reduce
import numpy as np
import random
import datetime
import time

from optimization.loss import calculate_loss, calculate_loss_array
from utils.plotting import plot_reconstructions
from scipy.misc import logsumexp
from optimization.evaluation import evaluate, evaluate_likelihood


def train(train_loader, val_loader, model, optimizer, args):
    header_msg = f'| Epoch |  TRAIN{"Loss": >12}{"Reconstruction": >18}{"KL": >12}{"| ": >4}'
    header_msg += f'{"VALIDATION": >11}{"Loss": >12}{"Reconstruction": >18}{"KL": >12}{"|": >3}'
    print('|' + "-"*(len(header_msg)-2) + '|')
    print(header_msg)
    print('|' + "-"*(len(header_msg)-2) + '|')

    if args.flow == "boosted":
        t_loss, t_rec, t_kl, v_loss, v_rec, v_kl, t_times  = train_boosted(
            train_loader, val_loader, model, optimizer, args)
    else:
        t_loss, t_rec, t_kl, v_loss, v_rec, v_kl, t_times  = train_vae(
            train_loader, val_loader, model, optimizer, args)

    # save training and validation results
    print('|' + "-"*(len(header_msg)-2) + '|')
    timing_msg = f"\nStopped after {t_times.shape[0]} epochs\n"
    timing_msg += f"Average train time per epoch: {np.mean(t_times):.2f} +/- {np.std(t_times, ddof=1):.2f}"
    print(timing_msg)

    if args.save_results:
        np.savetxt(args.snap_dir + '/train_loss.csv', t_loss, fmt='%f', delimiter=',')
        np.savetxt(args.snap_dir + '/train_rec.csv', t_rec, fmt='%f', delimiter=',')
        np.savetxt(args.snap_dir + '/train_kl.csv', t_kl, fmt='%f', delimiter=',')
        np.savetxt(args.snap_dir + '/val_loss.csv', v_loss, fmt='%f', delimiter=',')
        np.savetxt(args.snap_dir + '/val_rec.csv', v_rec, fmt='%f', delimiter=',')
        np.savetxt(args.snap_dir + '/val_kl.csv', v_kl, fmt='%f', delimiter=',')

        with open(args.exp_log, 'a') as ff:
            timestamp = str(datetime.datetime.now())[0:19].replace(' ', '_')
            setup_msg = ' '.join([timestamp, args.flow, args.dataset]) + "\n" + repr(args)
            print("\n" + setup_msg + "\n" + timing_msg, file=ff)

    return t_loss, v_loss


def train_boosted(train_loader, val_loader, model, optimizer, args):
    train_loss = []
    train_rec = []
    train_kl = []
    val_loss = []
    val_rec = []
    val_kl = []

    # for early stopping
    best_loss = np.inf
    best_bpd = np.inf
    e = 0
    epoch = 0
    train_times = []

    model.learner = 0
    for c in range(args.num_learners):
        optimizer.param_groups[c]['lr'] = args.learning_rate if c == model.learner else 0.0

    for epoch in range(1, args.epochs + 1):
        t_start = time.time()
        tr_loss, tr_rec, tr_kl = train_epoch(epoch, train_loader, model, optimizer, args)
        train_times.append(time.time() - t_start)
        train_loss.append(tr_loss)
        train_rec.append(tr_rec)
        train_kl.append(tr_kl)

        v_loss, v_rec, v_kl = evaluate(val_loader, model, args, save_plots=True, epoch=epoch)
        val_loss.append(v_loss)
        val_rec.append(v_rec)
        val_kl.append(v_kl)

        epoch_msg = f'| {epoch: <6}|{tr_loss.mean():19.3f}{tr_rec.mean():18.3f}{tr_kl.mean():12.3f}'
        epoch_msg += f'{"| ": >4}{v_loss:23.3f}{v_rec:18.3f}{v_kl:12.3f}'
        print(epoch_msg + f'{"| ": >4}')

        # compute value of grad elbo
        

        # new learner performance converged?
        if epoch % 5 == 0:
            model.update_rho()
            model.learner += 1

            # set the learning rate of all but one weak learner to zero
            for c in range(args.num_learners):
                optimizer.param_groups[c]['lr'] = args.learning_rate if c == model.learner else 0.0

        # early-stopping: does adding a new learner help?
        if v_loss < best_loss:
            e = 0
            best_loss = v_loss
            torch.save(model, args.snap_dir + 'model.pt')
        elif (args.early_stopping_epochs > 0) and (epoch >= args.warmup):
            e += 1
            if e > args.early_stopping_epochs:
                break

    train_loss = np.hstack(train_loss)
    train_rec = np.hstack(train_rec)
    train_kl = np.hstack(train_kl)
    val_loss = np.array(val_loss)
    val_rec = np.array(val_rec)
    val_kl = np.array(val_kl)
    train_times = np.array(train_times)
    return train_loss, train_rec, train_kl, val_loss, val_rec, val_kl, train_times


def train_vae(train_loader, val_loader, model, optimizer, args):
    train_loss = []
    train_rec = []
    train_kl = []
    val_loss = []
    val_rec = []
    val_kl = []

    # for early stopping
    best_loss = np.inf
    best_bpd = np.inf
    e = 0
    epoch = 0
    train_times = []

    for epoch in range(1, args.epochs + 1):
        t_start = time.time()
        tr_loss, tr_rec, tr_kl = train_epoch(epoch, train_loader, model, optimizer, args)
        train_times.append(time.time() - t_start)
        train_loss.append(tr_loss)
        train_rec.append(tr_rec)
        train_kl.append(tr_kl)

        v_loss, v_rec, v_kl = evaluate(val_loader, model, args, save_plots=True, epoch=epoch)
        val_loss.append(v_loss)
        val_rec.append(v_rec)
        val_kl.append(v_kl)

        epoch_msg = f'| {epoch: <6}|{tr_loss.mean():19.3f}{tr_rec.mean():18.3f}{tr_kl.mean():12.3f}'
        epoch_msg += f'{"| ": >4}{v_loss:22.3f}{v_rec:19.3f}{v_kl:12.3f}'
        print(epoch_msg + f'{"| ": >4}')

        # early-stopping: does adding a new learner help?
        if v_loss < best_loss:
            e = 0
            best_loss = v_loss
            torch.save(model, args.snap_dir + 'model.pt')
        elif (args.early_stopping_epochs > 0) and (epoch >= args.warmup):
            e += 1
            if e > args.early_stopping_epochs:
                break

    train_loss = np.hstack(train_loss)
    train_rec = np.hstack(train_rec)
    train_kl = np.hstack(train_kl)
    val_loss = np.array(val_loss)
    val_rec = np.array(val_rec)
    val_kl = np.array(val_kl)
    train_times = np.array(train_times)
    return train_loss, train_rec, train_kl, val_loss, val_rec, val_kl, train_times


def train_epoch(epoch, train_loader, model, opt, args):
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

        loss, rec, kl = calculate_loss(x_mean, data, z_mu, z_var, z0, zk, ldj, args, beta=beta)
        loss.backward()
        opt.step()

        train_loss[batch_id] = loss.item()
        train_rec[batch_id] = rec.item()
        train_kl[batch_id] = kl.item()

        num_trained += len(data)
        pct_complete = 100. * batch_id / total_batches
        if args.log_interval > 0 and batch_id % args.log_interval == 0:
            msg = 'Epoch: {:3d} [{:5d}/{:5d} ({:2.0f}%)]   \tLoss: {:11.6f}\trec: {:11.3f}\tkl: {:11.6f}'

            if args.input_type == 'binary':
                print(msg.format(
                    epoch, num_trained, total_data, pct_complete, loss.item(), rec.item(), kl.item()))
            else:
                msg += '\tbpd: {:8.6f}'
                bpd = loss.item() / (np.prod(args.input_size) * np.log(2.))
                print(msg.format(
                    epoch, num_trained, total_data, pct_complete,loss.item(), rec.item(), kl.item(), bpd))

    return train_loss, train_rec, train_kl



