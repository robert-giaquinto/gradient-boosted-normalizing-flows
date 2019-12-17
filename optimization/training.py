import torch
from functools import reduce
import numpy as np
import random
import datetime
import time
import logging

from optimization.loss import calculate_loss, calculate_loss_array, calculate_boosted_loss
from utils.plotting import plot_training_curve
from scipy.special import logsumexp
from optimization.evaluation import evaluate

logger = logging.getLogger(__name__)


def train(train_loader, val_loader, model, optimizer, scheduler, args):
    header_msg = f'| Epoch |  TRAIN{"Loss": >12}{"Reconstruction": >18}{"KL": >12}'
    header_msg += f'{"Fixed": >12}{"Converged": >12}{"| ": >4}' if args.flow == "boosted" else f'{"| ": >4}'
    #header_msg += f'{"Entropy": >12}{"P(sample all)": >16}{"| ": >4}' if args.flow == "boosted" else f'{"| ": >4}'
    header_msg += f'{"VALIDATION": >11}{"Loss": >12}{"Reconstruction": >18}{"KL": >12}{"| ": >4}{"Annealing": >12}{"|": >3}'
    logger.info('|' + "-"*(len(header_msg)-2) + '|')
    logger.info(header_msg)
    logger.info('|' + "-"*(len(header_msg)-2) + '|')

    if args.flow == "boosted":
        t_loss, t_rec, t_kl, v_loss, v_rec, v_kl, t_times  = train_boosted(
            train_loader, val_loader, model, optimizer, scheduler, args)
    else:
        t_loss, t_rec, t_kl, v_loss, v_rec, v_kl, t_times  = train_vae(
            train_loader, val_loader, model, optimizer, scheduler, args)

    # save training and validation results
    logger.info('|' + "-"*(len(header_msg)-2) + '|')
    timing_msg = f"\nStopped after {t_times.shape[0]} epochs"
    timing_msg += f"\nAverage train time per epoch: {np.mean(t_times):.2f} +/- {np.std(t_times, ddof=1):.2f}\n"
    logger.info(timing_msg)

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


def train_vae(train_loader, val_loader, model, optimizer, scheduler, args):
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
        tr_loss, tr_rec, tr_kl = train_epoch_vae(epoch, train_loader, model, optimizer, scheduler, args)    
        train_times.append(time.time() - t_start)
        train_loss.append(tr_loss)
        train_rec.append(tr_rec)
        train_kl.append(tr_kl)

        v_loss, v_rec, v_kl = evaluate(val_loader, model, args, epoch=epoch)
        val_loss.append(v_loss)
        val_rec.append(v_rec)
        val_kl.append(v_kl)

        beta = max(min([(epoch * 1.) / max([args.annealing_schedule, 1.]), args.max_beta]), args.min_beta)
        epoch_msg = f'| {epoch: <6}|{tr_loss.mean():19.3f}{tr_rec.mean():18.3f}{tr_kl.mean():12.3f}'
        epoch_msg += f'{"| ": >4}{v_loss:22.3f}{v_rec:19.3f}{v_kl:12.3f}{"| ": >4}{beta:12.3f}'
        logger.info(epoch_msg + f'{"| ": >4}')

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


def train_epoch_vae(epoch, train_loader, model, optimizer, scheduler, args):
    model.train()
    num_trained = 0
    total_samples = len(train_loader.sampler)
    total_batches = len(train_loader)
    train_loss = np.zeros(total_batches)
    train_bpd = np.zeros(total_batches)
    train_rec = np.zeros(total_batches)
    train_kl = np.zeros(total_batches)

    # set beta annealing coefficient
    beta = max(min([(epoch * 1.) / max([args.annealing_schedule, 1.]), args.max_beta]), args.min_beta)

    for batch_id, (x, _) in enumerate(train_loader):
        x = x.to(args.device)

        if args.dynamic_binarization:
            x = torch.bernoulli(x)

        optimizer.zero_grad()
        x_mean, z_mu, z_var, ldj, z0, zk = model(x)
        loss, rec, kl = calculate_loss(x_mean, x, z_mu, z_var, z0, zk, ldj, args, beta=beta)
        loss.backward()
        optimizer.step()
        if not args.no_lr_schedule:
            scheduler.step(loss)

        train_loss[batch_id] = loss.item()
        train_rec[batch_id] = rec.item()
        train_kl[batch_id] = kl.item()

        num_trained += len(x)
        pct_complete = 100. * batch_id / total_batches
        if args.log_interval > 0 and batch_id % args.log_interval == 0:
            msg = 'Epoch: {:3d} [{:5d}/{:5d} ({:2.0f}%)]   \tLoss: {:11.6f}\trec: {:11.3f}\tkl: {:11.6f}'

            if args.input_type == 'binary':
                logger.info(msg.format(
                    epoch, num_trained, total_samples, pct_complete, loss.item(), rec.item(), kl.item()))
            else:
                msg += '\tbpd: {:8.6f}'
                bpd = loss.item() / (np.prod(args.input_size) * np.log(2.))
                logger.info(msg.format(
                    epoch, num_trained, total_samples, pct_complete,loss.item(), rec.item(), kl.item(), bpd))

    return train_loss, train_rec, train_kl


def train_boosted(train_loader, val_loader, model, optimizer, scheduler, args):
    train_loss = []
    train_rec = []
    train_kl = []
    val_loss = []
    val_rec = []
    val_kl = []

    # for early stopping
    best_loss = np.inf
    best_bpd = np.inf
    prev_tr_fixed = np.inf
    component_threshold = 0.0 #0.0001
    e = 0
    epoch = 0
    converged_epoch = 0  # corrects the annealing schedule when a component converges early
    converged = False
    train_times = []

    model.component = 0
    for c in range(args.num_components):
        optimizer.param_groups[c]['lr'] = args.learning_rate if c == model.component else 0.0

    for epoch in range(1, args.epochs + 1):

        # compute annealing rate for KL loss term
        beta = kl_annealing_rate(epoch, model.component, args)

        # occasionally sample from all components to keep decoder from focusing solely on new component
        #prob_all = sample_from_all_prob(epoch, converged_epoch, model.component, args)
            
        t_start = time.time()
        tr_loss, tr_rec, tr_kl, tr_fixed = train_epoch_boosted(epoch, train_loader, model, optimizer, scheduler, beta, args)
        tr_fixed = tr_fixed.mean()
        train_times.append(time.time() - t_start)
        train_loss.append(tr_loss)
        train_rec.append(tr_rec)
        train_kl.append(tr_kl)

        # TODO FIX EVAL
        v_loss, v_rec, v_kl = evaluate(val_loader, model, args, epoch=epoch)
        val_loss.append(v_loss)
        val_rec.append(v_rec)
        val_kl.append(v_kl)

        # new component performance converged?
        converged = check_convergence(tr_fixed, prev_tr_fixed, component_threshold, model.component, model.num_components, epoch, args.burnin)
        if converged:
            converged_epoch = epoch

        epoch_msg = f'| {epoch: <6}|{tr_loss.mean():19.3f}{tr_rec.mean():18.3f}{tr_kl.mean():12.3f}{tr_fixed:12.3f}{str(converged): >12}'
        #epoch_msg = f'| {epoch: <6}|{tr_loss.mean():19.3f}{tr_rec.mean():18.3f}{tr_kl.mean():12.3f}{tr_fixed:12.3f}{prob_all:16.2f}'
        epoch_msg += f'{"| ": >4}{v_loss:23.3f}{v_rec:18.3f}{v_kl:12.3f}{"| ": >4}{beta:12.3f}'

        # is it time to update rho?
        time_to_update = check_time_to_update(epoch, args.annealing_schedule, args.burnin, model.component, converged_epoch)
        if time_to_update or converged:
            converged_epoch = epoch  # default convergence due to number of epochs elapsed
            
            model.update_rho(train_loader)
            epoch_msg += '  | Rho: ' + ' '.join([f"{val:1.2f}" for val in model.rho.data])
            model.increment_component()

            # set the learning rate of all but one component to zero
            for c in range(args.num_components):
                optimizer.param_groups[c]['lr'] = args.learning_rate if c == model.component else 0.0

        logger.info(epoch_msg + f'{"| ": >4}')
        # early-stopping only after all components have been trained
        if v_loss < best_loss:
            e = 0
            best_loss = v_loss
            torch.save(model, args.snap_dir + 'model.pt')
        elif (args.early_stopping_epochs > 0) and (model.component == model.num_components and beta == 1.0):
            e += 1
            if e > args.early_stopping_epochs:
                break

        # save log g(z|x) to check for convergence of new component
        prev_tr_fixed = tr_fixed

    train_loss = np.hstack(train_loss)
    train_rec = np.hstack(train_rec)
    train_kl = np.hstack(train_kl)
    val_loss = np.array(val_loss)
    val_rec = np.array(val_rec)
    val_kl = np.array(val_kl)
    train_times = np.array(train_times)
    return train_loss, train_rec, train_kl, val_loss, val_rec, val_kl, train_times


def train_epoch_boosted(epoch, train_loader, model, optimizer, scheduler, beta, args):
    model.train()

    total_batches = len(train_loader)
    train_loss = np.zeros(total_batches)
    train_bpd = np.zeros(total_batches)
    train_rec = np.zeros(total_batches)
    train_kl = np.zeros(total_batches)
    train_fixed = np.zeros(total_batches)

    for batch_id, (x, _) in enumerate(train_loader):

        x = x.to(args.device)

        if args.dynamic_binarization:
            x = torch.bernoulli(x)

        optimizer.zero_grad()
        x_recon, z_mu, z_var, z_g, g_ldj, z_G, G_ldj = model(x)
        is_first_component = model.component == 0 and not model.all_trained
        loss, rec, kl, log_G_z = calculate_boosted_loss(x_recon, x, z_mu, z_var, z_g, g_ldj, z_G, G_ldj, args, is_first_component, beta)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
        optimizer.step()
        if not args.no_lr_schedule:
            scheduler.step(loss)


        train_loss[batch_id] = loss.item()
        train_rec[batch_id] = rec.item()
        train_kl[batch_id] = kl.item()
        train_fixed[batch_id] = log_G_z.item()

    return train_loss, train_rec, train_kl, train_fixed


def kl_annealing_rate(epoch, component, args):
    if component == 0:
        epochs_per_component = max(args.annealing_schedule + args.burnin, 1)
        zero_offset = 1.0 / epochs_per_component  # don't want annealing rate to start at zero
        beta = (((epoch - 1) % epochs_per_component) / epochs_per_component) + zero_offset
        beta = min(beta, args.max_beta)
        beta = max(beta, args.min_beta)
    else:
        beta = 1.0

    return beta


def sample_from_all_prob(epoch, converged_epoch, current_component, args):
    """
    Want to occasionally sample from all components so decoder doesn't solely focus on new component
    """
    if current_component == 0:
        # first component runs longer than the rest
        epochs_per_component = max(args.annealing_schedule + args.burnin, 1)
        epoch_offset = 1
    else:
        epochs_per_component = max(args.annealing_schedule, 1)
        epoch_offset = (args.burnin if converged_epoch == 0 else converged_epoch) + 1
        
    non_zero_offset = 1.0 / epochs_per_component
    max_prob_all = 1.0 - (1.0 / (current_component + 1.0))
    
    prob_all = (((epoch - epoch_offset) % epochs_per_component) / epochs_per_component) + non_zero_offset
    prob_all = min(prob_all, max_prob_all)

    if current_component == args.num_components:
        # all components trained and rho updated for all components, make sure annealing rate doesn't continue to cycle
        prob_all = 1.0

    return prob_all


def check_convergence(tr_entropy, prev_tr_entropy, component_threshold, current_component, num_components, epoch, burnin):
    converged = abs(tr_entropy - prev_tr_entropy) < component_threshold
    performance_improved = prev_tr_entropy > tr_entropy
    
    # don't worry about convergence on the last component, we'll just let that one finish the annealing cycle
    not_last_component = current_component < (num_components - 1)

    # must be past to the burnin period to be considered converged
    past_burnin = epoch > burnin

    converged_flag = converged and performance_improved and not_last_component and past_burnin
    if converged_flag:
        logger.info(f"\tComponent {current_component} (out of {num_components}) converged at epoch {epoch} ({abs(tr_entropy - prev_tr_entropy)} < {component_threshold}).")
        
    return converged_flag


def check_time_to_update(epoch, annealing_schedule, burnin, current_component, converged_epoch):
    # has the anneal schedule concluded for this component
    if current_component == 0:
        annealing_schedule += burnin
        offset = 0
    else:
        offset = burnin if converged_epoch == 0 else converged_epoch

    # must be past to the burnin period to update
    past_burnin = epoch > burnin

    time_to_update = ((epoch - offset) % annealing_schedule == 0) and past_burnin
    return time_to_update

