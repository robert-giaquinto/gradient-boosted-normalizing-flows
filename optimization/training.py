import torch
from functools import reduce
import numpy as np
import random
import datetime
import time
import logging
from shutil import copyfile

from optimization.loss import calculate_loss, calculate_loss_array, calculate_boosted_loss
from utils.plotting import plot_training_curve
from scipy.special import logsumexp
from optimization.evaluation import evaluate
from utils.utilities import save, load

logger = logging.getLogger(__name__)


def train(train_loader, val_loader, model, optimizer, scheduler, args):
    header_msg = f'| Epoch |  TRAIN{"Loss": >12}{"Reconstruction": >18}'
    header_msg += f'{"Log G": >12}{"Prior": >12}{"Entropy": >12}{"Log Ratio": >12}{"| ": >4}' if args.flow == "boosted" else f'{"KL": >12}{"| ": >4}'
    header_msg += f'{"VALIDATION": >11}{"Loss": >12}{"Reconstruction": >18}{"KL": >12}{"| ": >4}{"Annealing": >12}'
    header_msg += f'{"P(c in 1:C)": >16}{"|": >3}' if args.flow == "boosted" else f'{"|": >3}'
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
        epoch_msg += f'{"| ": >4}{v_loss:23.3f}{v_rec:18.3f}{v_kl:12.3f}{"| ": >4}{beta:12.3f}{"| ": >4}'
        logger.info(epoch_msg)

        # early-stopping: does adding a new component help?
        if v_loss < best_loss:
            e = 0
            best_loss = v_loss
            save(model, optimizer, args.snap_dir + 'model.pt', scheduler)
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
    beta = max(min([(epoch * 1.) / max([args.annealing_schedule, 1.]), args.max_beta]), args.min_beta) if args.load is not None else 1.0

    for batch_id, (x, _) in enumerate(train_loader):
        x = x.to(args.device)

        if args.dynamic_binarization:
            x = torch.bernoulli(x)

        if args.vae_layers == 'linear':
            x = x.view(-1, np.prod(args.input_size))
        else:
            x = x.view(-1, *args.input_size)

        optimizer.zero_grad()
        x_mean, z_mu, z_var, ldj, z0, zk = model(x)
        loss, rec, kl = calculate_loss(x_mean, x, z_mu, z_var, z0, zk, ldj, args, beta=beta)
        loss.backward()
        #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
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
    best_loss = np.array([np.inf] * args.num_components)
    best_tr_ratio = np.array([-np.inf] * args.num_components)
    early_stop_count = 0
    converged_epoch = 0  # corrects the annealing schedule when a component converges early

    # initialize learning rates for boosted components
    prev_lr = []
    for c in range(args.num_components):
        prev_lr.append(args.learning_rate)
        optimizer.param_groups[c]['lr'] = args.learning_rate if c == model.component else 0.0
    for n, param in model.named_parameters():
        param.requires_grad = True if n.startswith(f"flow_param.{model.component}") or not n.startswith("flow_param") else False

    for epoch in range(args.init_epoch, args.epochs + 1):

        # compute annealing rate for KL loss term
        beta = kl_annealing_rate(epoch - converged_epoch, model.component, model.all_trained, args)

        # occasionally sample from all components to keep decoder from focusing solely on new component
        prob_all = sample_from_all_prob(epoch - converged_epoch, model.component, model.all_trained, args)

        # Train model
        t_start = time.time()
        tr_loss, tr_rec, tr_G, tr_p, tr_entropy, tr_ratio = train_epoch_boosted(
            epoch, train_loader, model, optimizer, scheduler, beta, prob_all, args)
        train_times.append(time.time() - t_start)
        train_loss.append(tr_loss)
        train_rec.append(tr_rec)
        train_G.append(tr_G)
        train_p.append(tr_p)
        train_entropy.append(tr_entropy)

        # Evaluate model
        v_loss, v_rec, v_kl = evaluate(val_loader, model, args, epoch=epoch)
        val_loss.append(v_loss)
        val_rec.append(v_rec)
        val_kl.append(v_kl)

        epoch_msg = f'| {epoch: <6}|{tr_loss.mean():19.3f}{tr_rec.mean():18.3f}{tr_G.mean():12.3f}{tr_p.mean():12.3f}{tr_entropy.mean():12.3f}{tr_ratio:12.3f}'
        epoch_msg += f'{"| ": >4}{v_loss:23.3f}{v_rec:18.3f}{v_kl:12.3f}{"| ": >4}{beta:12.3f}{prob_all:16.2f}  |  c={model.component} | AT={str(model.all_trained)}'

        # Assess convergence
        component_converged, model_improved, early_stop_count, best_loss, best_tr_ratio = check_convergence(
            early_stop_count, v_loss, best_loss, tr_ratio, best_tr_ratio, epoch - converged_epoch, model, args)

        if model_improved:
            epoch_msg += f' Improved'
            save(model, optimizer, args.snap_dir + f'model_c{model.component}.pt', scheduler)

        #epoch_msg += ' | Flow LRs: ' + ' '.join([f"{optimizer.param_groups[c]['lr']:6.5f}" for c in range(model.num_components)]) + f", VAE LRs: {optimizer.param_groups[args.num_components]['lr']:6.5f}"
        if component_converged:
            logger.info(epoch_msg + f'{"| ": >4}')
            
            converged_epoch = epoch
            prev_lr[model.component] = optimizer.param_groups[model.component]['lr']  # save LR for LR scheduler in case we train this component again

            # revert back to the last best version of the model and update rho
            load(model, optimizer, args.snap_dir + f'model_c{model.component}.pt', args)
            model.update_rho(train_loader)
            logger.info('Rho Updated: ' + ' '.join([f"{val:1.2f}" for val in model.rho.data]))

            train_components_once = args.epochs <= (args.epochs_per_component * args.num_components)
            if model.component == (args.num_components - 1) and (model.all_trained or train_components_once):
                # stop the full model after all components have been trained
                logger.info(f"Model converged, stopping training and saving final model to: {args.snap_dir + 'model.pt'}")
                model.all_trained = True
                save(model, optimizer, args.snap_dir + f'model.pt', scheduler)
                break

            # else if not done training:
            # save model with updated rho
            save(model, optimizer, args.snap_dir + f'model_c{model.component}.pt', scheduler)
            # reset early_stop_count and train the next component
            model.increment_component()
            early_stop_count = 0
            # freeze all but the new component being trained
            for c in range(args.num_components):
                optimizer.param_groups[c]['lr'] = prev_lr[c] if c == model.component else 0.0
            # reset VAE's learning rate too since it may have been reduced
            optimizer.param_groups[args.num_components]['lr'] = max(optimizer.param_groups[args.num_components]['lr'], args.learning_rate / 2.0)
            for n, param in model.named_parameters():
                param.requires_grad = True if n.startswith(f"flow_param.{model.component}") or not n.startswith("flow_param") else False
            logger.info('New Learning Rates. Flows: ' + ' '.join([f"{optimizer.param_groups[c]['lr']:8.6f}" for c in range(model.num_components)]) + f", VAE: {optimizer.param_groups[args.num_components]['lr']:8.6f}")
        else:
            logger.info(epoch_msg + f'{"| ": >4}')
            if epoch == args.epochs:
                # Save the best version of the model trained up to the current component with filename model.pt
                # This is to protect against times when the model is trained/re-trained but doesn't run long enough
                #   for all components to converge / train completely
                copyfile(args.snap_dir + f'model_c{model.component}.pt', args.snap_dir + 'model.pt')
                logger.info(f"Resaving last improved version of {f'model_c{model.component}.pt'} as 'model.pt' for future testing") 
        
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
    total_samples = len(train_loader.sampler) * 1.0
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

        if args.vae_layers == 'linear':
            x = x.view(-1, np.prod(args.input_size))
        else:
            x = x.view(-1, *args.input_size)

        optimizer.zero_grad()
        x_recon, z_mu, z_var, z_g, g_ldj, z_G, G_ldj = model(x, prob_all=prob_all)

        loss, rec, log_G, log_p, entropy, log_ratio = calculate_boosted_loss(
            x_recon, x, z_mu, z_var, z_g, g_ldj, z_G, G_ldj, args, is_first_component, beta)
        loss.backward()
        #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
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
    return train_loss, train_rec, train_G, train_p, train_entropy, (train_ratio.sum() / total_samples)


def kl_annealing_rate(epochs_since_prev_convergence, component, all_trained, args):
    """
    TODO need to adjust this for when an previous component converged early
    """
    past_warmup =  ((epochs_since_prev_convergence - 1) % args.epochs_per_component) >= args.annealing_schedule
    if all_trained or past_warmup:
        # all trained or past the first args.annealing_schedule epochs of training this component, so no annealing
        beta = args.max_beta
    else:
        # within the first args.annealing_schedule epochs of training this component
        beta = (((epochs_since_prev_convergence - 1) % args.annealing_schedule) / args.annealing_schedule) * args.max_beta
        beta += 1.0 / args.annealing_schedule  # don't want annealing rate to start at zero
            
    beta = min(beta, args.max_beta)
    beta = max(beta, args.min_beta)
    return beta


def sample_from_all_prob(epochs_since_prev_convergence, current_component, all_trained, args):
    """
    Want to occasionally sample from all components so decoder doesn't solely focus on new component
    """
    max_prob_all = min(0.5, 1.0 - (1.0 / (args.num_components)))
    if all_trained:
        # all components trained and rho updated for all components, make sure annealing rate doesn't continue to cycle
        return max_prob_all

    else:
        if current_component == 0:
            return 0.0
        else:
            pct_trained = ((epochs_since_prev_convergence - 1) % args.epochs_per_component) / args.epochs_per_component
            pct_trained += (1.0 / args.epochs_per_component)  # non-zero offset (don't start at zero)
            prob_all = min(pct_trained, 1.0) * max_prob_all
            
        return prob_all


def check_convergence(early_stop_count, v_loss, best_loss, tr_ratio, best_tr_ratio, epochs_since_prev_convergence, model, args):
    """
    Verify if a boosted component has converged
    """
    c = model.component
    first_component_trained = model.component > 0 or model.all_trained
    model_improved = v_loss < best_loss[c]
    early_stop_flag = False
    if first_component_trained and v_loss < best_loss[c]: # tried also checking: tr_ratio > best_tr_ratio[c]), but simpler is better
        # already trained more than one component, boosted component improved
        early_stop_count = 0
        best_loss[c] = v_loss
        best_tr_ratio[c] = tr_ratio
    elif not first_component_trained and v_loss < best_loss[c]:
        # training only the first component (for the first time), and it improved
        early_stop_count = 0
        best_loss[c] = v_loss
    elif args.early_stopping_epochs > 0:
        # model didn't improve, do we consider it converged yet?
        early_stop_count += 1        
        early_stop_flag = early_stop_count > args.early_stopping_epochs

    # Lastly, we consider the model converged if a pre-set number of epochs have elapsed
    time_to_update = epochs_since_prev_convergence % args.epochs_per_component == 0

    # But, model must have exceeded the warmup period before "converging"
    past_warmup = (epochs_since_prev_convergence >= args.annealing_schedule) or model.all_trained
    
    converged = (early_stop_flag or time_to_update) and past_warmup
    return converged, model_improved, early_stop_count, best_loss, best_tr_ratio

