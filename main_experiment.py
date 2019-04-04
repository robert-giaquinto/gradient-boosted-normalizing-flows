from __future__ import print_function
import argparse
import time
import torch
import torch.utils.data
import torch.optim as optim
import numpy as np
import math
import random
import os
import datetime

import models.VAE as VAE
from optimization.training import train, evaluate
from utils.load_data import load_dataset
from utils.plotting import plot_training_curve


parser = argparse.ArgumentParser(description='PyTorch Sylvester Normalizing flows')

parser.add_argument('-d', '--dataset', type=str, default='mnist',
    choices=['mnist', 'freyfaces', 'omniglot', 'caltech', 'cifar10'],
    metavar='DATASET',
    help='Dataset choice.')

parser.add_argument('-freys', '--freyseed', type=int, default=123,
    metavar='FREYSEED',
    help="""Seed for shuffling frey face dataset for test split. Ignored for other datasets.
    Results in paper are produced with seeds 123, 321, 231""")

parser.add_argument('-nc', '--no_cuda', action='store_true', default=False,
    help='disables CUDA training')

parser.add_argument('--manual_seed', type=int,
    help='manual seed, if not given resorts to random seed.')

parser.add_argument('-li', '--log_interval', type=int, default=10, metavar='LOG_INTERVAL',
    help='how many batches to wait before logging training status')
parser.add_argument('-pi', '--plot_interval', type=int, default=10, metavar='PLOT_INTERVAL',
    help='how many batches to wait before creating reconstruction plots')

parser.add_argument('-od', '--out_dir', type=str, default='snapshots', metavar='OUT_DIR',
    help='output directory for model snapshots etc.')

fp = parser.add_mutually_exclusive_group(required=False)
fp.add_argument('-te', '--testing', action='store_true', dest='testing',
help='evaluate on test set after training')
fp.add_argument('-va', '--validation', action='store_false', dest='testing',
help='only evaluate on validation set')
parser.set_defaults(testing=True)

# optimization settings
parser.add_argument('-e', '--epochs', type=int, default=2000, metavar='EPOCHS',
    help='number of epochs to train (default: 2000)')
parser.add_argument('-es', '--early_stopping_epochs', type=int, default=100, metavar='EARLY_STOPPING',
    help='number of early stopping epochs')

parser.add_argument('-bs', '--batch_size', type=int, default=100, metavar='BATCH_SIZE',
    help='input batch size for training (default: 100)')
parser.add_argument('-lr', '--learning_rate', type=float, default=0.0005, metavar='LEARNING_RATE',
    help='learning rate')

parser.add_argument('-w', '--warmup', type=int, default=100, metavar='N',
    help='number of epochs for warm-up. Set to 0 to turn warmup off.')
parser.add_argument('--max_beta', type=float, default=1., metavar='MB',
    help='max beta for warm-up')
parser.add_argument('--min_beta', type=float, default=0.0, metavar='MB',
    help='min beta for warm-up')

parser.add_argument('-f', '--flow', type=str, default='no_flow',
choices=['planar', 'radial', 'iaf', 'householder', 'orthogonal', 'triangular', 'no_flow', 'boosted'],
    help="""Type of flows to use, no flows can also be selected""")
parser.add_argument('-nf', '--num_flows', type=int, default=4,
    metavar='NUM_FLOWS', help='Number of flow layers, ignored in absence of flows')
parser.add_argument('-nv', '--num_ortho_vecs', type=int, default=8, metavar='NUM_ORTHO_VECS',
    help=""" For orthogonal flow: How orthogonal vectors per flow do you need.
    Ignored for other flow types.""")
parser.add_argument('-nh', '--num_householder', type=int, default=8, metavar='NUM_HOUSEHOLDERS',
    help=""" For Householder Sylvester flow: Number of Householder matrices per flow.
    Ignored for other flow types.""")
parser.add_argument('-mhs', '--made_h_size', type=int, default=320,
    metavar='MADEHSIZE', help='Width of mades for iaf. Ignored for all other flows.')
parser.add_argument('--z_size', type=int, default=64, metavar='ZSIZE',
    help='how many stochastic hidden units')

# Bagging/Boosting parameters
parser.add_argument('--num_learners', type=int, default=8, metavar='NUM_LEARNERS',
    help='How many weak learners are combined to form the flow')
parser.add_argument('-l', '--learner_type', type=str, default='planar',
    choices=['planar', 'radial', 'iaf', 'householder', 'orthogonal', 'triangular', 'random'],
    metavar='FLOW_TYPE',
    help='When flow is boosted -- what type of flow should each weak learner implement.')

# gpu/cpu
parser.add_argument('--gpu_num', type=int, default=0, metavar='GPU', help='choose GPU to run on.')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
args.device = torch.device("cuda" if args.cuda else "cpu")

if args.manual_seed is None:
    args.manual_seed = random.randint(1, 100000)
random.seed(args.manual_seed)
torch.manual_seed(args.manual_seed)
np.random.seed(args.manual_seed)

if args.cuda:
    torch.cuda.set_device(args.gpu_num)
else:
    num_cpus = max(1, os.cpu_count() - 1)
    torch.set_num_threads(num_cpus)


kwargs = {'num_workers': 0, 'pin_memory': True} if args.cuda else {}



def run(args, kwargs):

    print('\nMODEL SETTINGS: \n', args, '\n')
    print("Random Seed: ", args.manual_seed)

    # ====================================================
    # SNAPSHOTS
    # ====================================================
    args.model_signature = str(datetime.datetime.now())[0:19].replace(' ', '_')
    args.model_signature = args.model_signature.replace(':', '_')

    snapshots_path = os.path.join(args.out_dir, 'vae_' + args.dataset + '_')
    snap_dir = snapshots_path + args.flow + '_gpunum_' + str(args.gpu_num)

    if args.flow != 'no_flow':
        snap_dir += '_' + 'num_flows_' + str(args.num_flows)
    if args.flow == 'orthogonal':
        snap_dir = snap_dir + '_num_vectors_' + str(args.num_ortho_vecs)
    elif args.flow == 'householder':
        snap_dir = snap_dir + '_num_householder_' + str(args.num_householder)
    elif args.flow == 'iaf':
        snap_dir = snap_dir + '_madehsize_' + str(args.made_h_size)

    snap_dir = snap_dir + '__' + args.model_signature + '/'

    args.snap_dir = snap_dir

    if not os.path.exists(snap_dir):
        os.makedirs(snap_dir)

    # SAVING
    torch.save(args, snap_dir + args.flow + '.config')


    # ====================================================
    # LOAD DATA
    # ====================================================
    train_loader, val_loader, test_loader, args = load_dataset(args, **kwargs)


    # ====================================================
    # SELECT MODEL
    # ====================================================
    # flow parameters and architecture choice are passed on to model through args
    if args.flow == 'no_flow':
        model = VAE.VAE(args).to(args.device)
    elif args.flow == 'boosted':
        model = VAE.BoostedVAE(args).to(args.device)
    elif args.flow == 'planar':
        model = VAE.PlanarVAE(args).to(args.device)
    elif args.flow == 'radial':
        model = VAE.RadialVAE(args).to(args.device)
    elif args.flow == 'iaf':
        model = VAE.IAFVAE(args).to(args.device)
    elif args.flow == 'orthogonal':
        model = VAE.OrthogonalSylvesterVAE(args).to(args.device)
    elif args.flow == 'householder':
        model = VAE.HouseholderSylvesterVAE(args).to(args.device)
    elif args.flow == 'triangular':
        model = VAE.TriangularSylvesterVAE(args).to(args.device)
    else:
        raise ValueError('Invalid flow choice')

    print(model)


    # group model parameters to more easily modify learning rates of weak learners (flow parameters)
    if args.flow == 'boosted':

        param_labels = []
        previous_learner_id = "0"
        vae_params = torch.nn.ParameterList()
        learner_params = torch.nn.ParameterList()
        all_params = []  # contains both vae and flow learner parameters

        for name, param in model.named_parameters():
            if name.startswith("amor_u") or name.startswith("amor_w") or name.startswith("amor_b"):
                learner_id = name[7]

                if learner_id != previous_learner_id:
                    # save parameters for this learner to the collection
                    all_params.append(learner_params)
                    param_labels.append("Weak Learner {}".format(previous_learner_id))

                    # re-init params for the next weak learner
                    learner_params = torch.nn.ParameterList()

                # else: accumulate the remaining u, w, or b terms for this learner
                learner_params.append(param)

                previous_learner_id = learner_id
            else:
                vae_params.append(param)

        all_params.append(learner_params)
        param_labels.append(previous_learner_id)
        all_params.append(vae_params)
        param_labels.append("VAE")

        optimizer = optim.Adamax([{'params': p} for p in all_params], lr=args.learning_rate, eps=1.e-7)
    else:
        optimizer = optim.Adamax(model.parameters(), lr=args.learning_rate, eps=1.e-7)


    # ====================================================
    # TRAINING
    # ====================================================
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
        tr_loss, tr_rec, tr_kl = train(epoch, train_loader, model, optimizer, args)
        train_times.append(time.time() - t_start)

        train_loss.append(tr_loss)
        train_rec.append(tr_rec)
        train_kl.append(tr_kl)

        v_loss, v_rec, v_kl = evaluate(val_loader, model, args, save_plots=True, epoch=epoch)
        val_loss.append(v_loss)
        val_rec.append(v_rec)
        val_kl.append(v_kl)

        # early-stopping
        if v_loss < best_loss:
            e = 0
            best_loss = v_loss
            torch.save(model, snap_dir + args.flow + '.model')
        elif (args.early_stopping_epochs > 0) and (epoch >= args.warmup):
            e += 1
            if e > args.early_stopping_epochs:
                break

    # save training and validation results
    train_loss = np.hstack(train_loss)
    train_rec = np.hstack(train_rec)
    train_kl = np.hstack(train_kl)
    np.savetext(snap_dir + '/train_loss.csv', train_loss, fmt='%10.5f', delimiter=',')
    np.savetext(snap_dir + '/train_rec.csv', train_rec, fmt='%10.5f', delimiter=',')
    np.savetext(snap_dir + '/train_kl.csv', train_kl, fmt='%10.5f', delimiter=',')
    val_loss = np.array(val_loss)
    val_rec = np.array(val_rec)
    val_kl = np.array(val_kl)
    np.savetext(snap_dir + '/val_loss.csv', val_loss, fmt='%10.5f', delimiter=',')
    np.savetext(snap_dir + '/val_rec.csv', val_rec, fmt='%10.5f', delimiter=',')
    np.savetext(snap_dir + '/val_kl.csv', val_kl, fmt='%10.5f', delimiter=',')

    # plot training and validation loss curves
    plot_training_curve(train_loss, val_loss, fname=snap_dir + '/training_curve_%s.pdf' % args.flow)

    # training time per epoch
    train_times = np.array(train_times)
    mean_train_time = np.mean(train_times)
    std_train_time = np.std(train_times, ddof=1)
    # print('Average train time per epoch: %.2f +/- %.2f' % (mean_train_time, std_train_time))
    with open('experiment_log.txt', 'a') as ff:
        print("\n", str(datetime.datetime.now())[0:19].replace(' ', '_'), file=ff)
        print(args, file=ff)
        print('Stopped after %d epochs' % epoch, file=ff)
        print('Average train time per epoch: %.2f +/- %.2f' % (mean_train_time, std_train_time), file=ff)


    # ====================================================
    # EVALUATION
    # ====================================================
    final_model = torch.load(snap_dir + args.flow + '.model')
    validation_loss = evaluate(val_loader, final_model, args, save_plots=True)
    with open('experiment_log.txt', 'a') as ff:
        print('FINAL EVALUATION ON VALIDATION SET\n'
              'ELBO (VAL): {:.4f}\n'.format(validation_loss), file=ff)

        if args.input_type != 'binary':
            validation_bpd = (validation_loss / (np.prod(args.input_size) * np.log(2.))) / len(val_loader)
            print('FINAL EVALUATION ON VALIDATION SET\n'
                'ELBO (VAL) BPD : {:.4f}\n'.format(validation_bpd), file=ff)

    if args.testing:
        test_nll = evaluate_likelihood(test_loader, final_model, args)
        with open('experiment_log.txt', 'a') as ff:
            print('FINAL EVALUATION ON TEST SET\n'
                  'NLL (TEST): {:.4f}\n'.format(test_nll), file=ff)

            if args.input_type != 'binary':
                test_bpd = test_nll / (np.prod(args.input_size) * np.log(2.))
                print('FINAL EVALUATION ON TEST SET\n'
                      'NLL (TEST) BPD: {:.4f}\n'.format(test_bpd), file=ff)


if __name__ == "__main__":
    run(args, kwargs)


