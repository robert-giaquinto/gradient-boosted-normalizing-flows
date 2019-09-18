import torch
from functools import reduce
import numpy as np
import random
import datetime
import time

from optimization.loss import calculate_loss, calculate_loss_array, variational_loss
from utils.plotting import plot_training_curve
from scipy.misc import logsumexp
from optimization.evaluation import evaluate


def train(train_loader, val_loader, model, optimizer, args):
    header_msg = f'| Epoch |  TRAIN{"Loss": >12}{"Reconstruction": >18}{"KL": >12}'
    header_msg += f'{"Log g": >12}{"| ": >4}' if args.flow == "boosted" else f'{"| ": >4}'
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

        plot_training_curve(t_loss, v_loss, fname=args.snap_dir + 'training_curve.png')

        with open(args.exp_log, 'a') as ff:
            timestamp = str(datetime.datetime.now())[0:19].replace(' ', '_')
            setup_msg = ' '.join([timestamp, args.flow, args.dataset]) + "\n" + repr(args)
            print("\n" + setup_msg + "\n" + timing_msg, file=ff)

    return t_loss, v_loss


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
        tr_loss, tr_rec, tr_kl = train_epoch_vae(epoch, train_loader, model, optimizer, args)
        train_times.append(time.time() - t_start)
        train_loss.append(tr_loss)
        train_rec.append(tr_rec)
        train_kl.append(tr_kl)

        v_loss, v_rec, v_kl = evaluate(val_loader, model, args, epoch=epoch)
        val_loss.append(v_loss)
        val_rec.append(v_rec)
        val_kl.append(v_kl)

        epoch_msg = f'| {epoch: <6}|{tr_loss.mean():19.3f}{tr_rec.mean():18.3f}{tr_kl.mean():12.3f}'
        epoch_msg += f'{"| ": >4}{v_loss:22.3f}{v_rec:19.3f}{v_kl:12.3f}'
        print(epoch_msg + f'{"| ": >4}')

        # early-stopping: does adding a new component help?
        if v_loss < best_loss:
            e = 0
            best_loss = v_loss
            torch.save(model, args.snap_dir + 'model.pt')
        elif (args.early_stopping_epochs > 0) and (epoch >= args.annealing_schedule):
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


def train_epoch_vae(epoch, train_loader, model, opt, args):
    model.train()

    num_trained = 0
    total_data = len(train_loader.sampler)
    total_batches = len(train_loader)

    train_loss = np.zeros(total_batches)
    train_bpd = np.zeros(total_batches)
    train_rec = np.zeros(total_batches)
    train_kl = np.zeros(total_batches)

    # set beta annealing coefficient
    beta = min([(epoch * 1.) / max([args.annealing_schedule, 1.]), args.max_beta])

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
    prev_tr_reg = np.inf
    component_threshold = 0.01 if args.experiment_name != "debug" else 0.5
    e = 0
    epoch = 0
    converged_epoch = 0
    converged = False
    train_times = []

    model.component = 0
    for c in range(args.num_components):
        optimizer.param_groups[c]['lr'] = args.learning_rate if c == model.component else 0.0

    for epoch in range(1, args.epochs + 1):
        beta = boosted_annealing_schedule(epoch, converged_epoch, model.component, model.num_components, args)
        
        t_start = time.time()
        tr_loss, tr_rec, tr_kl, tr_reg = train_epoch_boosted(epoch, train_loader, model, optimizer, beta, args)
        train_times.append(time.time() - t_start)
        train_loss.append(tr_loss)
        train_rec.append(tr_rec)
        train_kl.append(tr_kl)

        v_loss, v_rec, v_kl = evaluate(val_loader, model, args, epoch=epoch)
        val_loss.append(v_loss)
        val_rec.append(v_rec)
        val_kl.append(v_kl)

        # new component performance converged?
        avg_tr_reg = tr_reg.mean()
        converged = abs(avg_tr_reg - prev_tr_reg) < component_threshold and prev_tr_reg > avg_tr_reg and model.component < model.num_components
        if converged:
            # save this epoch to correct the annealing schedule when a component converges early
            converged_epoch = epoch

        epoch_msg = f'| {epoch: <6}|{tr_loss.mean():19.3f}{tr_rec.mean():18.3f}{tr_kl.mean():12.3f}{tr_reg.mean():12.3f}'
        epoch_msg += f'{"| ": >4}{v_loss:23.3f}{v_rec:18.3f}{v_kl:12.3f} {beta:8.3f} {converged}'
        
        # update rho weights if appropriate
        annealing_schedule = args.annealing_schedule
        if model.component == 0:
            annealing_schedule += args.burnin
            offset = 0
        else:
            offset = args.burnin if converged_epoch == 0 else converged_epoch
        
        time_to_update = converged or (epoch - offset) % annealing_schedule == 0
        currently_in_training = model.component < model.num_components and epoch > args.burnin
        if currently_in_training and time_to_update:
            model.update_rho(train_loader)
            epoch_msg += f'  | Rho: ' + ' '.join([f"{val:1.2f}" for val in model.rho.data]) 
            model.component += 1

            # set the learning rate of all but one component to zero
            for c in range(args.num_components):
                optimizer.param_groups[c]['lr'] = args.learning_rate if c == model.component or model.component == model.num_components else 0.0

        print(epoch_msg + f'{"| ": >4}')
        # early-stopping: does adding a new component help?
        if v_loss < best_loss:
            e = 0
            best_loss = v_loss
            torch.save(model, args.snap_dir + 'model.pt')
        elif (args.early_stopping_epochs > 0) and (model.component == model.num_components and beta == 1.0):
            e += 1
            if e > args.early_stopping_epochs:
                break

        # save log g(z|x) to check for convergence of new component
        prev_tr_reg = avg_tr_reg

    train_loss = np.hstack(train_loss)
    train_rec = np.hstack(train_rec)
    train_kl = np.hstack(train_kl)
    val_loss = np.array(val_loss)
    val_rec = np.array(val_rec)
    val_kl = np.array(val_kl)
    train_times = np.array(train_times)
    return train_loss, train_rec, train_kl, val_loss, val_rec, val_kl, train_times


def train_epoch_boosted(epoch, train_loader, model, opt, beta, args):
    model.train()

    num_trained = 0
    total_data = len(train_loader.sampler)
    total_batches = len(train_loader)

    train_loss = np.zeros(total_batches)
    train_bpd = np.zeros(total_batches)
    train_rec = np.zeros(total_batches)
    train_kl = np.zeros(total_batches)
    train_reg = np.zeros(total_batches)

    for batch_id, (data, _) in enumerate(train_loader):
        data = data.to(args.device)

        if args.dynamic_binarization:
            data = torch.bernoulli(data)

        opt.zero_grad()
        x_recon, z_mu, z_var, ldj, z0, zk = model(data, sample_from="fixed")
        loss, rec, kl = calculate_loss(x_recon, data, z_mu, z_var, z0, zk, ldj, args, beta=beta)

        if args.regularization_rate > 0:
            # compute regularization terms
            _, g_z_mu, g_z_var, g_ldj, g_z0, _ = model(data, sample_from="new")
            regularizer = variational_loss(g_z_mu, g_z_var, g_z0, g_ldj)
            loss = loss + args.regularization_rate * regularizer
            train_reg[batch_id] = regularizer.item()
        
        loss.backward()
        opt.step()

        train_loss[batch_id] = loss.item()
        train_rec[batch_id] = rec.item()
        train_kl[batch_id] = kl.item()

        num_trained += len(data)
        pct_complete = 100. * batch_id / total_batches
        if args.log_interval > 0 and batch_id % args.log_interval == 0:
            msg = 'Epoch: {:3d} [{:5d}/{:5d} ({:2.0f}%)]   \tLoss: {:11.6f}\trec: {:11.3f}\tkl: {:11.6f}\tregularizer: {:11.6f}'

            if args.input_type == 'binary':
                print(msg.format(
                    epoch, num_trained, total_data, pct_complete, loss.item(), rec.item(), kl.item(), regularizer.item() * args.regularization_rate))
            else:
                msg += '\tbpd: {:8.6f}'
                bpd = loss.item() / (np.prod(args.input_size) * np.log(2.))
                print(msg.format(
                    epoch, num_trained, total_data, pct_complete,loss.item(), rec.item(), kl.item(), bpd))

    return train_loss, train_rec, train_kl, train_reg


def boosted_annealing_schedule(epoch, converged_epoch, current_component, total_components, args):
    """
    Annealing schedule for boosting will:
    1. Reset with each new component trained.
    2. Have a first component with a seperate, longer cycle because of the added "burnin" epochs
    3. One the last component, finish the annealing schedule even if the new component converged and then keep beta=1.0
    """
    if current_component == 0:
        # first component runs longer than the rest
        epochs_per_component = max(args.annealing_schedule + args.burnin, 1)
        epoch_offset = 1
    else:
        epochs_per_component = max(args.annealing_schedule, 1)
        epoch_offset = (args.burnin if converged_epoch == 0 else converged_epoch) + 1

    non_zero_offset = 1.0 / epochs_per_component
    beta = min((((epoch - epoch_offset) % epochs_per_component) / epochs_per_component) + non_zero_offset, args.max_beta)
    
    if current_component == total_components and (epoch - epoch_offset) >= args.annealing_schedule:
        # all components trained and rho updated for all components, make sure annealing rate doesn't continue to cycle
        beta = 1.0


    return beta
