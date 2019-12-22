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
    header_msg = f'| Epoch |  TRAIN{"Loss": >12}{"Reconstruction": >18}'
    header_msg += f'{"Log G": >12}{"Prior": >12}{"Entropy": >12}{"Log Ratio": >12}{"| ": >4}' if args.flow == "boosted" else f'{"KL": >12}{"| ": >4}'
    header_msg += f'{"VALIDATION": >11}{"Loss": >12}{"Reconstruction": >18}{"KL": >12}{"| ": >4}{"Annealing": >12}{"P(c in 1:C)": >16}{"|": >3}'
    logger.info('|' + "-"*(len(header_msg)-2) + '|')
    logger.info(header_msg)
    logger.info('|' + "-"*(len(header_msg)-2) + '|')

    if args.flow == "boosted":
        t_loss, t_rec, t_G, t_p, t_entropy, v_loss, v_rec, v_kl, t_times  = train_boosted(
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
        if args.flow == "boosted":
            np.savetxt(args.snap_dir + '/train_loss.csv', t_loss, fmt='%f', delimiter=',')
            np.savetxt(args.snap_dir + '/train_rec.csv', t_rec, fmt='%f', delimiter=',')
            np.savetxt(args.snap_dir + '/train_log_G_z.csv', t_G, fmt='%f', delimiter=',')
            np.savetxt(args.snap_dir + '/train_log_p_zk.csv', t_p, fmt='%f', delimiter=',')
            np.savetxt(args.snap_dir + '/train_entropy.csv', t_entropy, fmt='%f', delimiter=',')
        else:
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
        epoch_msg += f'{"| ": >4}{v_loss:22.3f}{v_rec:19.3f}{v_kl:12.3f}{"| ": >4}{beta:12.3f}{"| ": >20}'
        logger.info(epoch_msg)

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

    return train_loss, train_rec, train_kl


def train_boosted(train_loader, val_loader, model, optimizer, scheduler, args):
    train_times = []
    train_loss = []
    train_rec = []
    train_G = []
    train_p = []
    train_entropy = []
    
    val_loss = []
    val_rec = []
    val_kl = []

    # for early stopping
    best_loss = np.inf
    prev_tr_ratio = np.inf
    component_threshold = 0.0 #0.0001
    e = 0
    epoch = 0
    converged_epoch = 0  # corrects the annealing schedule when a component converges early
    converged = False

    model.component = 0
    for c in range(args.num_components):
        optimizer.param_groups[c]['lr'] = args.learning_rate if c == model.component else 0.0

    for epoch in range(1, args.epochs + 1):

        # compute annealing rate for KL loss term
        beta = kl_annealing_rate(epoch, model.component, model.all_trained, args)

        # occasionally sample from all components to keep decoder from focusing solely on new component
        prob_all = sample_from_all_prob(epoch, converged_epoch, model.component, model.all_trained, args)
            
        t_start = time.time()
        tr_loss, tr_rec, tr_G, tr_p, tr_entropy, tr_ratio = train_epoch_boosted(
            epoch, train_loader, model, optimizer, scheduler, beta, prob_all, args)
        train_times.append(time.time() - t_start)
        
        train_loss.append(tr_loss)
        train_rec.append(tr_rec)
        train_G.append(tr_G)
        train_p.append(tr_p)
        train_entropy.append(tr_entropy)

        v_loss, v_rec, v_kl = evaluate(val_loader, model, args, epoch=epoch)
        val_loss.append(v_loss)
        val_rec.append(v_rec)
        val_kl.append(v_kl)

        # new component performance converged?
        converged = check_convergence(tr_ratio, prev_tr_ratio, component_threshold, model.component, model.num_components, epoch, args.burnin)
        if converged:
            converged_epoch = epoch

        epoch_msg = f'| {epoch: <6}|{tr_loss.mean():19.3f}{tr_rec.mean():18.3f}{tr_G.mean():12.3f}{tr_p.mean():12.3f}{tr_entropy.mean():12.3f}{tr_ratio:12.3f}'
        epoch_msg += f'{"| ": >4}{v_loss:23.3f}{v_rec:18.3f}{v_kl:12.3f}{"| ": >4}{beta:12.3f}{prob_all:16.2f}  |  c={model.component}'

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
        prev_tr_ratio = tr_ratio

    train_loss = np.hstack(train_loss)
    train_rec = np.hstack(train_rec)
    train_G = np.hstack(train_G)
    train_p = np.hstack(train_p)
    train_entropy = np.hstack(train_entropy)
    
    val_loss = np.array(val_loss)
    val_rec = np.array(val_rec)
    val_kl = np.array(val_kl)
    train_times = np.array(train_times)
    return train_loss, train_rec, train_G, train_p, train_entropy, val_loss, val_rec, val_kl, train_times


def train_epoch_boosted(epoch, train_loader, model, optimizer, scheduler, beta, prob_all, args):
    model.train()
    is_first_component = model.component == 0 and not model.all_trained
    
    total_batches = len(train_loader)
    total_samples = len(train_loader.sampler)
    train_loss = np.zeros(total_batches)
    train_rec = np.zeros(total_batches)
    train_p = np.zeros(total_batches)
    train_entropy = np.zeros(total_batches)
    train_G = []
    train_ratio = []

    for batch_id, (x, _) in enumerate(train_loader):
        x = x.to(args.device)

        if args.dynamic_binarization:
            x = torch.bernoulli(x)

        optimizer.zero_grad()
        x_recon, z_mu, z_var, z_g, g_ldj, z_G, G_ldj = model(x, prob_all=prob_all)
        loss, rec, log_G, log_p, entropy, log_ratio = calculate_boosted_loss(
            x_recon, x, z_mu, z_var, z_g, g_ldj, z_G, G_ldj, args, is_first_component, beta)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
        optimizer.step()
        if not args.no_lr_schedule:
            scheduler.step(loss)

        train_loss[batch_id] = loss.item()
        train_rec[batch_id] = rec.item()
        train_p[batch_id] = log_p.item()
        train_entropy[batch_id] = entropy.item()

        # ignore the boosting terms if we sampled from all (to alleviate decoder shock)
        if z_G is not None and G_ldj is not None:
            train_G.append(log_G.item())
            train_ratio.append(log_ratio.item())

    train_G = np.array(train_G) if len(train_G) > 0 else np.zeros(1)
    train_ratio = np.array(train_ratio) if len(train_ratio) > 0 else np.zeros(1)
    return train_loss, train_rec, train_G, train_p, train_entropy, train_ratio.mean()


def kl_annealing_rate(epoch, component, all_trained, args):
    if args.annealing_schedule == 1:
        beta = epoch / (100.0 * args.num_components) if args.flow == "boosted" else epoch / 100.0
    else:
        if all_trained:
            beta = 1.0
        else:
            epochs_per_component = max(args.annealing_schedule + args.burnin if component == 0 else args.annealing_schedule, 1)
            zero_offset = 1.0 / epochs_per_component  # don't want annealing rate to start at zero
            beta = (((epoch - 1) % epochs_per_component) / epochs_per_component) + zero_offset
            
    beta = min(beta, args.max_beta)
    beta = max(beta, args.min_beta)
    return beta


def sample_from_all_prob(epoch, converged_epoch, current_component, all_trained, args):
    """
    Want to occasionally sample from all components so decoder doesn't solely focus on new component
    """
    max_prob_all = 1.0 - (1.0 / (current_component + 1.0))
    
    if all_trained:
        # all components trained and rho updated for all components, make sure annealing rate doesn't continue to cycle
        prob_all = 1.0

    else:
        if current_component == 0:
            # first component runs longer than the rest
            epochs_per_component = max(args.annealing_schedule + args.burnin, 1)
            epoch_offset = 1
        else:
            epochs_per_component = max(args.annealing_schedule, 1)
            epoch_offset = (args.burnin if converged_epoch == 0 else converged_epoch) + 1
            
        non_zero_offset = 1.0 / epochs_per_component
        prob_all = (((epoch - epoch_offset) % epochs_per_component) / epochs_per_component) + non_zero_offset
        
    prob_all = min(prob_all, max_prob_all)
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

