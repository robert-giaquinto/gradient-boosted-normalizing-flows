import argparse
import datetime
import torch
import torch.utils.data
import torch.optim as optim
import numpy as np
import math
import random
import os

import models.VAE as VAE
from optimization.training import train, evaluate, evaluate_likelihood
from utils.load_data import load_dataset
from utils.plotting import plot_training_curve, plot_decoded_manifold, plot_data_manifold, plot_decoded_random_sample


parser = argparse.ArgumentParser(description='PyTorch Ensemble Normalizing flows')

parser.add_argument('-d', '--dataset', type=str, default='mnist', help='Dataset choice.',
					choices=['mnist', 'freyfaces', 'omniglot', 'caltech', 'cifar10'])

# seeds
parser.add_argument('--manual_seed', type=int, default=123,
					help='manual seed, if not given resorts to random seed.')
parser.add_argument('-freys', '--freyseed', type=int, default=123,
					help="Seed for shuffling frey face dataset for test split. Ignored for other datasets.")

# gpu/cpu
parser.add_argument('--gpu_id', type=int, default=0, metavar='GPU', help='choose GPU to run on.')
parser.add_argument('--num_workers', type=int, default=1,
					help='How many CPU cores to run on. Setting to 0 uses os.cpu_count() - 1.')
parser.add_argument('-nc', '--no_cuda', action='store_true', default=False,
					help='disables CUDA training')

# Reporting
parser.add_argument('--log_interval', type=int, default=0, metavar='LOG_INTERVAL',
					help='how many batches to wait before logging training status. Set to <0 to turn off.')
parser.add_argument('--plot_interval', type=int, default=10, metavar='PLOT_INTERVAL',
					help='how many batches to wait before creating reconstruction plots. Set to <0 to turn off.')

parser.add_argument('--experiment_name', type=str, default=None,
					help="A name to help identify the experiment being run when training this model.")

parser.add_argument('--out_dir', type=str, default='./results/snapshots',
					help='Output directory for model snapshots etc.')
parser.add_argument('--data_dir', type=str, default='./data/raw/',
					help="Where raw data is saved.")
parser.add_argument('--exp_log', type=str, default='./results/experiment_log.txt',
					help='File to save high-level results from each run of an experiment.')

sr = parser.add_mutually_exclusive_group(required=False)
sr.add_argument('--save_results', action='store_true', dest='save_results', help='Save results from experiments.')
sr.add_argument('--discard_results', action='store_false', dest='save_results', help='Do NOT save results from experiments.')
parser.set_defaults(save_results=True)


# testing vs. just validation
fp = parser.add_mutually_exclusive_group(required=False)
fp.add_argument('-te', '--testing', action='store_true', dest='testing',
				help='evaluate on test set after training')
fp.add_argument('-va', '--validation', action='store_false', dest='testing',
				help='only evaluate on validation set')
parser.set_defaults(testing=True)

# optimization settings
parser.add_argument('-e', '--epochs', type=int, default=2000,
					help='number of epochs to train (default: 2000)')
parser.add_argument('-es', '--early_stopping_epochs', type=int, default=20,
					help='number of early stopping epochs')

parser.add_argument('-bs', '--batch_size', type=int, default=64,
					help='input batch size for training (default: 64)')
parser.add_argument('-lr', '--learning_rate', type=float, default=0.0005,
					help='learning rate')
parser.add_argument('--regularization_rate', type=float, default=0.01, help='Regularization penalty for boosting.')


parser.add_argument('-w', '--warmup', type=int, default=100,
					help='number of epochs for warm-up. Set to 0 to turn warmup off.')
parser.add_argument('--max_beta', type=float, default=1.0,
					help='max beta for warm-up')
parser.add_argument('--min_beta', type=float, default=0.0,
					help='min beta for warm-up')
parser.add_argument('--burnin', type=int, default=3,
					help='number of epochs to run with no kl term.')


# flow parameters
parser.add_argument('-f', '--flow', type=str, default='no_flow',
					choices=['planar', 'radial', 'iaf', 'householder', 'orthogonal', 'triangular', 'no_flow', 'boosted', 'bagged'],
					help="""Type of flows to use, no flows can also be selected""")
parser.add_argument('-nf', '--num_flows', type=int, default=2,
					help='Number of flow layers, ignored in absence of flows')

# Sylvester parameters
parser.add_argument('-nv', '--num_ortho_vecs', type=int, default=8,
					help=" For orthogonal flow: How orthogonal vectors per flow do you need. Ignored for other flow types.")
parser.add_argument('-nh', '--num_householder', type=int, default=8,
					help="For Householder Sylvester flow: Number of Householder matrices per flow. Ignored for other flow types.")
parser.add_argument('-mhs', '--made_h_size', type=int, default=320,
					help='Width of mades for iaf. Ignored for all other flows.')
parser.add_argument('--z_size', type=int, default=64, help='how many stochastic hidden units')

# Bagging/Boosting parameters
parser.add_argument('--num_learners', type=int, default=8,
					help='How many weak learners are combined to form the flow')
parser.add_argument('-l', '--learner_type', type=str, default='planar',
					choices=['planar', 'radial', 'iaf', 'householder', 'orthogonal', 'triangular', 'random'],
					help='When flow is bagged or boosted -- what type of flow should each weak learner implement.')



def parse_args(main_args=None):
	"""
	Parse command line arguments and compute number of cores to use
	"""
	args = parser.parse_args(main_args)
	args.cuda = not args.no_cuda and torch.cuda.is_available()
	args.device = torch.device("cuda" if args.cuda else "cpu")

	# Set a random seed if not given one
	if args.manual_seed is None:
		args.manual_seed = random.randint(1, 100000)

		
	random.seed(args.manual_seed)
	torch.manual_seed(args.manual_seed)
	np.random.seed(args.manual_seed)
	print("Random Seed: ", args.manual_seed)

	# Set up multiple CPU/GPUs
	print("\nCOMPUTATION SETTINGS:")
	if args.cuda:
		print("\tUsing CUDA GPU")
		torch.cuda.set_device(args.gpu_id)
	else:
		print("\tUsing CPU")
		if args.num_workers > 0:
			num_workers = args.num_workers
		else:
			num_workers = max(1, os.cpu_count() - 1)

		print("\tCores available: {} (only requesting {})".format(os.cpu_count(), num_workers))
		torch.set_num_threads(num_workers)
		print("\tConfirmed Number of CPU threads: {}".format(torch.get_num_threads()))

	# SETUP SNAPSHOTS DIRECTORY FOR SAVING MODELS
	args.model_signature = str(datetime.datetime.now())[0:19].replace(' ', '_').replace(':', '_').replace('-', '_')
	args.experiment_name = args.experiment_name + "_" if args.experiment_name is not None else ""
	args.snap_dir = os.path.join(args.out_dir, args.experiment_name + args.flow + '_')

	if args.flow != 'no_flow':
		args.snap_dir += 'num_flows_' + str(args.num_flows) + '_'
	if args.flow == 'orthogonal':
		args.snap_dir += 'num_vectors_' + str(args.num_ortho_vecs)
	elif args.flow == 'householder':
		args.snap_dir += 'num_householder_' + str(args.num_householder)
	elif args.flow == 'iaf':
		args.snap_dir += 'madehsize_' + str(args.made_h_size)
	elif args.flow in ['boosted', 'bagged']:
		args.snap_dir += args.learner_type + '_num_learners_' + str(args.num_learners)

	args.snap_dir += args.model_signature + '/'

	kwargs = {'num_workers': 0, 'pin_memory': True} if args.cuda else {}
	return args, kwargs


def init_model(args):
	if args.flow == 'no_flow':
		model = VAE.VAE(args).to(args.device)
	elif args.flow == 'boosted':
		model = VAE.BoostedVAE(args).to(args.device)
	elif args.flow == 'bagged':
		model = VAE.BaggedVAE(args).to(args.device)
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

	return model


def init_optimizer(model, args, debug_param_groups=False):
	"""
	group model parameters to more easily modify learning rates of weak learners (flow parameters)
	"""
	if args.flow in ['boosted', 'bagged']:
		param_labels = []
		debug_arr = []
		previous_learner_id = "0"
		vae_params = torch.nn.ParameterList()
		learner_params = torch.nn.ParameterList()
		all_params = []	 # contains both vae and flow learner parameters

		for name, param in model.named_parameters():
			if name.startswith("amor_u") or name.startswith("amor_w") or name.startswith("amor_b"):
				learner_id = name[7:name.find(".")]

				if learner_id != previous_learner_id:
					# save parameters for this learner to the collection
					all_params.append(learner_params)
					param_labels.append("Weak Learner {}".format(previous_learner_id))
					if debug_param_groups:
						print("Appending layers [{}] to group for Weak Learner {}".format(
							', '.join(debug_arr), previous_learner_id))

				# re-init params for the next weak learner
				learner_params = torch.nn.ParameterList()
				debug_arr = []

				# else: accumulate the remaining u, w, or b terms for this learner
				learner_params.append(param)
				debug_arr.append(name)

				previous_learner_id = learner_id
			else:
				vae_params.append(param)

		all_params.append(learner_params)
		param_labels.append(previous_learner_id)

		if debug_param_groups:
			print("Appending layers [{}] to group for Weak Learner {}".format(
				', '.join(debug_arr), previous_learner_id))

		all_params.append(vae_params)
		param_labels.append("VAE")

		optimizer = optim.Adamax([{'params': p} for p in all_params], lr=args.learning_rate, eps=1.e-7)
	else:
		optimizer = optim.Adamax(model.parameters(), lr=args.learning_rate, eps=1.e-7)

	return optimizer



def main(main_args=None):
	"""
	use main_args to run this script as function in another script
	"""

	# =========================================================================
	# PARSE EXPERIMENT SETTINGS, SETUP SNAPSHOTS DIRECTORY
	# =========================================================================
	args, kwargs = parse_args(main_args)
	if not os.path.exists(args.snap_dir):
		os.makedirs(args.snap_dir)


	# =========================================================================
	# LOAD DATA
	# =========================================================================
	train_loader, val_loader, test_loader, args = load_dataset(args, **kwargs)


	# =========================================================================
	# SAVE EXPERIMENT SETTINGS
	# =========================================================================
	print('\nEXPERIMENT SETTINGS: \n', args)
	torch.save(args, os.path.join(args.snap_dir, 'config.pt'))


	# =========================================================================
	# INITIALIZE MODEL AND OPTIMIZATION
	# =========================================================================
	model = init_model(args)
	optimizer = init_optimizer(model, args, debug_param_groups=False)
	num_params = sum([param.nelement() for param in model.parameters()])
	print(f"\nMODEL:\nNumber of model parameters={num_params}\n", model)


	# =========================================================================
	# TRAINING
	# =========================================================================
	print('\nTRAINING:\n')
	train_loss, val_loss = train(train_loader, val_loader, model, optimizer, args)
	plot_training_curve(train_loss, val_loss, fname=args.snap_dir + 'training_curve.pdf')


	# =========================================================================
	# VALIDATION
	# =========================================================================
	print('\nVALIDATION:\n')
	final_model = torch.load(args.snap_dir + 'model.pt')
	plot_decoded_random_sample(args, final_model, size_x=5, size_y=5)

	if final_model.z_size == 2:
		plot_decoded_manifold(final_model, args)
		plot_data_manifold(final_model, test_loader, args)

	# Compute losses on validaion set
	validation_loss, validation_rec, validation_kl = evaluate(val_loader, final_model, args, save_plots=True)
	with open('experiment_log.txt', 'a') as ff:
		print('FINAL EVALUATION ON VALIDATION SET', file=ff)
		print('ELBO (VAL): {:.4f}\n'.format(validation_loss), file=ff)
		print('RECON (VAL): {:.4f}\n'.format(validation_rec), file=ff)
		print('KL (VAL): {:.4f}\n'.format(validation_kl), file=ff)
		if args.input_type != 'binary':
			validation_bpd = (validation_loss / (np.prod(args.input_size) * np.log(2.))) / len(val_loader)
			print('ELBO (VAL) BPD : {:.4f}\n'.format(validation_bpd), file=ff)


	# =========================================================================
	# TESTING
	# =========================================================================
	if args.testing:
		print("\nTESTING:")
		test_nll = evaluate_likelihood(test_loader, final_model, args)

		with open('experiment_log.txt', 'a') as ff:
			print('FINAL EVALUATION ON TEST SET', file=ff)
			print('NLL (TEST): {:.4f}\n'.format(test_nll), file=ff)
			if args.input_type != 'binary':
				test_bpd = test_nll / (np.prod(args.input_size) * np.log(2.))
				print('NLL (TEST) BPD: {:.4f}\n'.format(test_bpd), file=ff)


if __name__ == "__main__":
	main()


