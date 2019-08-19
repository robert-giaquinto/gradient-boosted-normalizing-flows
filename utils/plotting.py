import numpy as np
import matplotlib
import os
import torch
import torch.distributions as D
# noninteractive background
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def plot_training_curve(train_loss, validation_loss, fname='training_curve.png', labels=None):
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


def plot_decoded_random_sample(args, model, size_x=3, size_y=3):
	"""
	Use model to decode a random sample of the latent space
	Save in plot
	"""
	# decode random sample
	z_rs = torch.normal(torch.from_numpy(np.zeros((25, model.z_size))).float(), 1.).to(args.device)
	x_rs_recon = model.decode(z_rs)
	samples = x_rs_recon.data.cpu().numpy()[:size_x*size_y]

	fig = plt.figure(figsize=(size_x, size_y))
	gs = gridspec.GridSpec(size_x, size_y)
	gs.update(wspace=0.05, hspace=0.05)

	for i, sample in enumerate(samples):
		ax = plt.subplot(gs[i])
		plt.axis('off')
		ax.set_xticklabels([])
		ax.set_yticklabels([])
		ax.set_aspect('equal')
		plt.imshow(sample.reshape(args.input_size[1], args.input_size[2]), cmap='Greys_r')

	plt.savefig(args.snap_dir + 'random_sample_decoded.png', bbox_inches='tight')
	plt.close(fig)


def plot_decoded_manifold(model, args, x_lim=5, y_lim=5, nx = 25):
	"""
	Use model to decode a range of values within the latent space
	and plot.
	"""
	#visualize 2D latent space
	ny = nx
	x_values = np.linspace(-x_lim, x_lim, nx)
	y_values = np.linspace(-y_lim, y_lim, ny)
	canvas = np.empty((args.input_size[1]*ny, args.input_size[2]*nx))

	for i, yi in enumerate(x_values):
		for j, xi in enumerate(y_values):
			zz = np.array( [[xi], [yi]], dtype='float32' ).T
			z_rs = torch.from_numpy(zz).to(args.device)
			x_rs_recon = model.decode(z_rs)
			x = x_rs_recon.data.cpu().numpy().flatten().reshape(args.input_size[1], args.input_size[2])

			xi0 = (nx-i-1) * args.input_size[1]
			xi1 = (nx-i) * args.input_size[1]
			yi0 = j * args.input_size[2]
			yi1 = (j+1) * args.input_size[2]
			canvas[xi0:xi1, yi0:yi1] = x

	fig = plt.figure(figsize=(8, 10))
	plt.imshow(canvas, origin="upper", cmap='Greys_r')
	plt.axis('off')
	plt.tight_layout()
	plt.savefig(os.path.join(args.snap_dir, 'latentSpace2D.png'), bbox_inches='tight')
	plt.close(fig)


def plot_data_manifold(model, data_loader, args, limit=None):
	"""
	Pass data through encoder (and flows if possible) and then plot
	latent dimension.

	ASSUMES args.z_size == 2
	"""
	if model.z_size > 2:
		raise ValueError("Can only plot latent dimension of data when z_size==2")

	X = torch.cat([x for x, y in list(data_loader)], 0).to(args.device)
	# MNIST should have just integer labels in a vector
	Y = torch.cat([y for x, y in list(data_loader)], 0).data.cpu().numpy().astype(int)[:, 0]

	if args.flow == "no_flow":
		z_mean, z_var = model.encode(X)
		Z = model.reparameterize(z_mean, z_var).data.cpu().numpy()
	else:
		# not efficient, but much easier to just pass model through entire network
		_, _, _, _, _, Z = model(X)
		Z = Z.data.cpu().numpy()

	if len( np.shape(Y) ) > 2:
		# For one-hot vectors
		Y_label = np.argmax(Y, 1)
	else:
		Y_label = Y

	fig = plt.figure()
	plt.scatter(Z[:, 0], Z[:, 1], c=Y_label,
				alpha=0.5, edgecolors='k', cmap='gist_ncar')
	plt.colorbar()

	if limit != None:
		# set axes range
		limit = np.abs(limit)
		plt.xlim(-limit, limit)
		plt.ylim(-limit, limit)

	plt.savefig(args.snap_dir + 'latent_manifold_of_data.png', bbox_inches='tight')
	plt.close(fig)


def make_target_density(args):
	w1 = lambda z: torch.sin(2 * math.pi * z[:,0] / 4)
	w2 = lambda z: 3 * torch.exp(-0.5 * ((z[:,0] - 1)/0.6)**2)
	w3 = lambda z: 3 * torch.sigmoid((z[:,0] - 1) / 0.3)

	if args.target_density == "u_z1":
		u_z = lambda z: 0.5 * ((torch.norm(z, p=2, dim=1) - 2) / 0.4)**2 - \
			torch.log(torch.exp(-0.5*((z[:,0] - 2) / 0.6)**2) + torch.exp(-0.5*((z[:,0] + 2) / 0.6)**2) + 1e-10)
	elif args.target_density == "u_z2":
		u_z = lambda z: 0.5 * ((z[:,1] - w1(z)) / 0.4)**2
	elif args.target_density == "u_z3":
		u_z = lambda z: - torch.log(torch.exp(-0.5*((z[:,1] - w1(z))/0.35)**2) + torch.exp(-0.5*((z[:,1] - w1(z) + w2(z))/0.35)**2) + 1e-10)
	elif args.target_density == "u_z4":
		u_z = lambda z: - torch.log(torch.exp(-0.5*((z[:,1] - w1(z))/0.4)**2) + torch.exp(-0.5*((z[:,1] - w1(z) + w3(z))/0.35)**2) + 1e-10)
	else:
		raise ValueError("Unacceptable choice of target density")



def plot_flows(base_dist, flow, filename, args):
	"""
	sigma = 4.0
	base_dist = D.MultivariateNormal(torch.zeros(2).to(args.device), sigma * torch.eye(2).to(args.device))
	"""

	n = 200
	lim = 4

	fig, axs = plt.subplots(2, 2, subplot_kw={'aspect': 'equal'})

	# plot target density we're trying to approx
	plot_target_density(u_z, axs[0,0], lim, n)

	# plot posterior approx density
	plot_flow_density(base_dist, flow, axs[0,1], lim, n)

	# plot flow-transformed base dist sample and histogram
	z = base_dist.sample((10000,))
	zk, _ = flow(z)
	zk = zk.cpu().data.numpy()
	axs[1,0].scatter(zk[:,0], zk[:,1], s=10, alpha=0.4)
	axs[1,1].hist2d(zk[:,0], zk[:,1], bins=lim*50, cmap=plt.cm.jet)

	for ax in plt.gcf().axes:
		ax.get_xaxis().set_visible(True)
		ax.get_yaxis().set_visible(True)
		ax.invert_yaxis()

	plt.tight_layout()
	plt.savefig(filename)
	plt.close()


def plot_target_density(u_z, ax, range_lim=4, n=200, output_dir=None):
	x = torch.linspace(-range_lim, range_lim, n)
	xx, yy = torch.meshgrid((x, x))
	zz = torch.stack((xx.flatten(), yy.flatten()), dim=-1).squeeze().to(args.device)

	ax.pcolormesh(xx, yy, torch.exp(-u_z(zz)).view(n,n).data, cmap=plt.cm.jet)

	for ax in plt.gcf().axes:
		ax.set_xlim(-range_lim, range_lim)
		ax.set_ylim(-range_lim, range_lim)
		ax.get_xaxis().set_visible(False)
		ax.get_yaxis().set_visible(False)
		ax.invert_yaxis()

	if output_dir:
		plt.tight_layout()
		plt.savefig(os.path.join(output_dir, 'target_potential_density.png'))
		plt.close()


def plot_flow_density(base_dist, flow, ax, range_lim=4, n=200, output_dir=None):
	x = torch.linspace(-range_lim, range_lim, n)
	xx, yy = torch.meshgrid((x, x))
	zz = torch.stack((xx.flatten(), yy.flatten()), dim=-1).squeeze().to(args.device)

	# plot posterior approx density
	zzk, sum_log_abs_det_jacobians = flow(zz)
	log_q0 = base_dist.log_prob(zz)
	log_qk = log_q0 - sum_log_abs_det_jacobians
	qk = log_qk.exp().cpu()
	zzk = zzk.cpu()
	ax.pcolormesh(zzk[:,0].view(n,n).data, zzk[:,1].view(n,n).data, qk.view(n,n).data, cmap=plt.cm.jet)
	ax.set_facecolor(plt.cm.jet(0.))

	for ax in plt.gcf().axes:
		ax.set_xlim(-range_lim, range_lim)
		ax.set_ylim(-range_lim, range_lim)
		ax.get_xaxis().set_visible(False)
		ax.get_yaxis().set_visible(False)
		ax.invert_yaxis()

	if output_dir:
		plt.tight_layout()
		plt.savefig(os.path.join(output_dir, 'flow_k{}_density.png'.format(len(flow)-1)))
		plt.close()
