import torch
from functools import reduce
import numpy as np
import random
import datetime
import time

from optimization.loss import calculate_loss, calculate_loss_array
from utils.plotting import plot_reconstructions
from scipy.misc import logsumexp


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
