import torch
import torch.utils.data as data_utils
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.functional as F
import torch.distributions as D
import math
import pickle
from sklearn.utils import shuffle as util_shuffle
from scipy.io import loadmat

import numpy as np
import logging
import os
import sklearn
import sklearn.datasets

from utils.gas import GAS
from utils.bsds300 import BSDS300
from utils.hepmass import HEPMASS
from utils.miniboone import MINIBOONE
from utils.power import POWER


logger = logging.getLogger(__name__)


def load_density_dataset(args):
    if args.dataset == 'gas':
        dataset = GAS('data/gas/ethylene_CO.pickle')
    elif args.dataset == 'bsds300':
        dataset = BSDS300('data/BSDS300/BSDS300.hdf5')
    elif args.dataset == 'hepmass':
        dataset = HEPMASS('data/hepmass')
    elif args.dataset == 'miniboone':
        dataset = MINIBOONE('data/miniboone/data.npy')
    elif args.dataset == 'power':
        dataset = POWER('data/power/data.npy')
    else:
        raise RuntimeError()
    
    # idle y's
    y_train = np.zeros((dataset.trn.x.shape[0], 1))
    y_val = np.zeros((dataset.val.x.shape[0], 1))
    y_test = np.zeros((dataset.tst.x.shape[0], 1))
    args.y_classes = 1

    train_dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(dataset.trn.x).float().to(args.device), torch.from_numpy(y_train))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    val_dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(dataset.val.x).float().to(args.device), torch.from_numpy(y_val))
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.eval_batch_size, shuffle=False)

    test_dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(dataset.tst.x).float().to(args.device), torch.from_numpy(y_test))
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.eval_batch_size, shuffle=False)
    
    args.z_size = dataset.n_dims
    args.train_size = len(train_loader)
    args.input_size = [dataset.n_dims]
    if args.h_size is None:
        args.h_size = args.h_size_factor * dataset.n_dims

    logger.info(f"Dataset: {args.dataset} has {len(train_loader)}, {len(val_loader)}, and {len(test_loader)} minibatches of size {args.batch_size} in train, validation, and test sets.")
    logger.info(f"Dimension={dataset.n_dims}. Total samples: {len(train_loader.sampler)}, {len(val_loader.sampler)}, and {len(test_loader.sampler)} in train, validation, and test sets.\n")
    return train_loader, val_loader, test_loader, args


def load_image_dataset(args, **kwargs):
    if args.dataset == 'mnist':
        train_loader, val_loader, test_loader, args = load_static_mnist(args, **kwargs)
    elif args.dataset == 'caltech':
        train_loader, val_loader, test_loader, args = load_caltech101silhouettes(args, **kwargs)
    elif args.dataset == 'freyfaces':
        train_loader, val_loader, test_loader, args = load_freyfaces(args, **kwargs)
    elif args.dataset == 'omniglot':
        train_loader, val_loader, test_loader, args = load_omniglot(args, **kwargs)
    elif args.dataset == 'cifar10':
        train_loader, val_loader, test_loader, args = load_cifar10(args, **kwargs)
    elif args.dataset == 'celeba':
        train_loader, val_loader, test_loader, args = load_celeba(args, **kwargs)
    else:
        raise Exception('Wrong name of the dataset!')

    args.train_size = len(train_loader)
    logger.info(f"Dataset: {args.dataset} has {len(train_loader)}, {len(val_loader)}, and {len(test_loader)} minibatches of size {args.batch_size} in train, validation, and test sets.")
    logger.info(f"Total samples: {len(train_loader.sampler)}, {len(val_loader.sampler)}, and {len(test_loader.sampler)} in train, validation, and test sets.\n")
    return train_loader, val_loader, test_loader, args


def make_toy_density(args):
    w1 = lambda z: torch.sin(2 * math.pi * z[:,0] / 4)
    w2 = lambda z: 3 * torch.exp(-0.5 * ((z[:,0] - 1)/0.6)**2)
    w3 = lambda z: 3 * torch.sigmoid((z[:,0] - 1) / 0.3)

    if args.dataset  == "u0":
        u_z = lambda z: 0.5 * ((torch.norm(z, p=2, dim=1) - 2) / 0.4)**2 - \
            torch.log(torch.exp(-0.5*((z[:,0] - 2) / 0.6)**2) + 1e-10)
    elif args.dataset == "u1":
        u_z = lambda z: 0.5 * ((torch.norm(z, p=2, dim=1) - 2) / 0.4)**2 - \
            torch.log(torch.exp(-0.5*((z[:,0] - 2) / 0.6)**2) + \
                      torch.exp(-0.5*((z[:,0] + 2) / 0.6)**2) + 1e-10)
    elif args.dataset == "u2":
        u_z = lambda z: 0.5 * ((z[:,1] - w1(z)) / 0.4)**2
    elif args.dataset == "u3":
        u_z = lambda z: - torch.log(torch.exp(-0.5*((z[:,1] - w1(z))/0.35)**2) + \
                                    torch.exp(-0.5*((z[:,1] - w1(z) + w2(z))/0.35)**2) + 1e-10)
    elif args.dataset == "u4":
        u_z = lambda z: - torch.log(torch.exp(-0.5*((z[:,1] - w1(z))/0.4)**2) + \
                                    torch.exp(-0.5*((z[:,1] - w1(z) + w3(z))/0.35)**2) + 1e-10)
    elif args.dataset == "u5":
        num_clusters = args.mog_clusters
        mix_props = np.random.dirichlet([10.0] * num_clusters).astype("float32")
        mu = torch.from_numpy(np.random.normal(loc=[0.0, 0.0], scale=args.mog_sigma, size=[num_clusters, 2]).astype("float32"))
        sigma = np.repeat(np.eye(2)[None], num_clusters, axis=0).astype("float32") * 1.1
        sigma[:, 1, 0] = np.random.uniform(low=0.0, high=0.7, size=[num_clusters]).astype("float32") *\
            np.random.choice([1, -1], size=[num_clusters])
        mix_props = torch.from_numpy(mix_props)
        sigma = torch.from_numpy(sigma)

        u_z = lambda z: -1.0 * torch.log(sum(
            torch.exp(D.MultivariateNormal(mu_i, sigma_i).log_prob(z)) * mix_props_i \
            for (mix_props_i, mu_i, sigma_i) in zip(mix_props, mu, sigma)))

        return u_z
    elif args.dataset == 'u6':
        # like the two moons, but less curvature
        u_z = lambda z: 0.5 * ((torch.norm(z, p=2, dim=1) - 1.0) / 0.5)**2 - \
            torch.log(torch.exp(-0.5*((z[:,0] - 1.0) / 0.2)**2) + \
                      torch.exp(-0.5*((z[:,0] + 1.0) / 0.2)**2) + 1e-10)


    else:
        raise ValueError("Unacceptable choice of target density to sample from")

    return u_z


def make_toy_sampler(args):
    
    def data_sampler(batch_size):
        if args.dataset == "swissroll":
            data = sklearn.datasets.make_swiss_roll(n_samples=batch_size, noise=1.0)[0]
            data = data.astype("float32")[:, [0, 2]]
            data /= 5
            data = torch.from_numpy(data)

        elif args.dataset == "circles":
            data = sklearn.datasets.make_circles(n_samples=batch_size, factor=.5, noise=0.08)[0]
            data = data.astype("float32")
            data *= 3
            data = torch.from_numpy(data)
            
        elif args.dataset == "rings":
            n_samples4 = n_samples3 = n_samples2 = batch_size // 4
            n_samples1 = batch_size - n_samples4 - n_samples3 - n_samples2

            # so as not to have the first point = last point, we set endpoint=False
            linspace4 = np.linspace(0, 2 * np.pi, n_samples4, endpoint=False)
            linspace3 = np.linspace(0, 2 * np.pi, n_samples3, endpoint=False)
            linspace2 = np.linspace(0, 2 * np.pi, n_samples2, endpoint=False)
            linspace1 = np.linspace(0, 2 * np.pi, n_samples1, endpoint=False)

            circ4_x = np.cos(linspace4)
            circ4_y = np.sin(linspace4)
            circ3_x = np.cos(linspace4) * 0.75
            circ3_y = np.sin(linspace3) * 0.75
            circ2_x = np.cos(linspace2) * 0.5
            circ2_y = np.sin(linspace2) * 0.5
            circ1_x = np.cos(linspace1) * 0.25
            circ1_y = np.sin(linspace1) * 0.25

            data = np.vstack([
                np.hstack([circ4_x, circ3_x, circ2_x, circ1_x]),
                np.hstack([circ4_y, circ3_y, circ2_y, circ1_y])
            ]).T * 3.0
            data = util_shuffle(data)

            # Add noise
            data = data.astype("float32")
            data += np.random.randn(*data.shape) * 0.1
            data = torch.from_numpy(data)

        elif args.dataset == "moons":
            data = sklearn.datasets.make_moons(n_samples=batch_size, noise=0.1)[0]
            data = data.astype("float32")
            data = data * 2 + np.array([-1, -0.2])
            data = data.astype("float32")
            data = torch.from_numpy(data)

        elif args.dataset == "pinwheel":
            radial_std = 0.3
            tangential_std = 0.1
            num_classes = 5
            num_per_class = batch_size // 5
            rate = 0.25
            rads = np.linspace(0, 2 * np.pi, num_classes, endpoint=False)

            features = np.random.randn(num_classes*num_per_class, 2) \
                * np.array([radial_std, tangential_std])
            features[:, 0] += 1.
            labels = np.repeat(np.arange(num_classes), num_per_class)

            angles = rads[labels] + rate * np.exp(features[:, 0])
            rotations = np.stack([np.cos(angles), -np.sin(angles), np.sin(angles), np.cos(angles)])
            rotations = np.reshape(rotations.T, (-1, 2, 2))
            data = 2 * np.random.permutation(np.einsum("ti,tij->tj", features, rotations)).astype("float32")
            data = torch.from_numpy(data)

        elif args.dataset == "2spirals":
            n = np.sqrt(np.random.rand(batch_size // 2, 1)) * 540 * (2 * np.pi) / 360
            d1x = -np.cos(n) * n + np.random.rand(batch_size // 2, 1) * 0.5
            d1y = np.sin(n) * n + np.random.rand(batch_size // 2, 1) * 0.5
            data = np.vstack((np.hstack((d1x, d1y)), np.hstack((-d1x, -d1y)))) / 3
            data += np.random.randn(*data.shape) * 0.1
            data = data.astype("float32")
            data = torch.from_numpy(data)

        elif args.dataset == "checkerboard":
            x1 = np.random.rand(batch_size) * 4 - 2
            x2_ = np.random.rand(batch_size) - np.random.randint(0, 2, batch_size) * 2
            x2 = x2_ + (np.floor(x1) % 2)
            data = np.concatenate([x1[:, None], x2[:, None]], 1).astype("float32") * 2
            data = torch.from_numpy(data)

        elif args.dataset == "line":
            x = np.random.rand(batch_size)
            x = x * 5 - 2.5
            y = x + np.random.randn(batch_size)
            data = np.stack((x, y), 1).astype("float32")
            data = torch.from_numpy(data)
            
        elif args.dataset == "cos":
            x = np.random.rand(batch_size) * 5 - 2.5
            y = np.sin(x) * 2.5
            data = np.stack((x, y), 1).astype("float32")
            data = torch.from_numpy(data)
            
        elif args.dataset == "joint_gaussian":
            x2 = torch.distributions.Normal(0., 4.).sample((batch_size, 1))
            x1 = torch.distributions.Normal(0., 1.).sample((batch_size, 1)) + (x2**2)/4
            data = torch.cat((x1, x2), 1)
            
        elif args.dataset == "8gaussians":
            scale = 4.0
            sq2 = 1.0 / np.sqrt(2)
            centers = [(1,0), (-1,0), (0,1), (0,-1), (sq2,sq2), (-sq2,sq2), (sq2,-sq2), (-sq2,-sq2)]
            centers = torch.tensor([(scale * x, scale * y) for x,y in centers]).float()
            noise = torch.randn(batch_size, 2)
            data = sq2 * (0.5 * noise + centers[torch.randint(8, size=(batch_size,))])

        elif args.dataset == "1gaussian":
            scale = 4.0
            sq2 = 1.0 / np.sqrt(2)
            centers = [(1,0), (-1,0)]
            centers = torch.tensor([(scale * x, scale * y) for x,y in centers]).float()
            noise = torch.randn(batch_size, 2)
            data = sq2 * (0.5 * noise + centers[torch.randint(1, size=(batch_size,))])

        elif args.dataset == "2gaussians":
            scale = 4.0
            sq2 = 1.0 / np.sqrt(2)
            centers = [(1,0), (-1,0)]
            centers = torch.tensor([(scale * x, scale * y) for x,y in centers]).float()
            noise = torch.randn(batch_size, 2)
            data = sq2 * (0.5 * noise + centers[torch.randint(2, size=(batch_size,))])

        elif args.dataset == "mog":
            num_clusters = args.mog_clusters
            mix_props = np.random.dirichlet([10.0] * num_clusters).astype("float32")
            mu = torch.from_numpy(np.random.normal(loc=[0.0, 0.0], scale=args.mog_sigma, size=[num_clusters, 2]).astype("float32"))
            sigma = np.repeat(np.eye(2)[None], num_clusters, axis=0).astype("float32") * 0.8
            sigma[:, 1, 0] = np.random.uniform(low=0.0, high=0.8, size=[num_clusters]).astype("float32") *\
                np.random.choice([1, -1], size=[num_clusters])
            mix_props = torch.from_numpy(mix_props)
            sigma = torch.from_numpy(sigma)
            
            u_z = lambda z: -1.0 * torch.log(sum(
                torch.exp(D.MultivariateNormal(mu_i, sigma_i).log_prob(z)) * mix_props_i \
                for (mix_props_i, mu_i, sigma_i) in zip(mix_props, mu, sigma)))

            data = []
            for (mix_props_i, mu_i, sigma_i) in zip(mix_props, mu, sigma):
                data.extend(np.random.multivariate_normal(mu_i, sigma_i, size=[int(batch_size * mix_props_i)]))
                
            data = torch.from_numpy(np.array(data).reshape([batch_size, 2]))
            
        else:
            raise ValueError(f"The toy dataset {args.dataset} hasn't been defined!")

        return data

    return data_sampler


def load_celeba(args, **kwargs):
    """
    data_train = np.load('../celeba_full_64x64_3bit.npy')
    data_train = np.load('../celeba_full_64x64_5bit.npy')
    """
    args.dynamic_binarization = False
    args.input_type = 'multinomial'
    #args.input_size = [3, 64, 64]
    args.input_size = [3, 32, 32]
    args.y_classes = 40

    if args.augment_images:
        transformations = [transforms.RandomAffine(0, translate=(0.1, 0.1)),
                           transforms.RandomHorizontalFlip()]
    else:
        transformations = []

    transformations.extend([transforms.Resize(args.input_size[1:]), transforms.ToTensor()])
    train_transform = transforms.Compose(transformations)
    test_transform = transforms.Compose([transforms.Resize(args.input_size[1:]), transforms.ToTensor()])

    # Create the datasets
    data_dir = './data/'
    train_data = torchvision.datasets.CelebA(root=data_dir, split='train', target_type='attr', download=True,
                                             transform=train_transform)
    val_data = torchvision.datasets.CelebA(root=data_dir, split='valid', target_type='attr',
                                           transform=test_transform)
    test_data = torchvision.datasets.CelebA(root=data_dir, split='test', target_type='attr',
                                            transform=test_transform)

    
    # Create the dataloaders
    train_loader = data_utils.DataLoader(train_data, batch_size=args.batch_size, shuffle=args.shuffle, **kwargs)
    val_loader = data_utils.DataLoader(val_data, batch_size=args.batch_size, shuffle=False, **kwargs)
    test_loader = data_utils.DataLoader(test_data, batch_size=args.batch_size, shuffle=False, **kwargs)

    return train_loader, val_loader, test_loader, args


def load_cifar10(args, **kwargs):
    args.dynamic_binarization = False
    args.input_type = 'multinomial'
    args.y_classes = 10
    args.input_size = [3, 32, 32]
    valid_size = 0.1

    if args.augment_images:
        transformations = [transforms.RandomAffine(0, translate=(0.1, 0.1)),
                           transforms.RandomHorizontalFlip()]
    else:
        transformations = []

    transformations.extend([transforms.ToTensor()])
    train_transform = transforms.Compose(transformations)
    test_transform = transforms.Compose([transforms.ToTensor()])

    one_hot_encode = lambda target: F.one_hot(torch.tensor(target), args.y_classes)

    # load / download the data
    data_dir = './data/CIFAR10/'
    train_data = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_transform, target_transform=one_hot_encode)
    val_data = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=test_transform, target_transform=one_hot_encode)
    test_data = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=test_transform, target_transform=one_hot_encode)

    # split training and validation sets
    num_train = len(train_data)
    indices = list(range(num_train))
    num_val = int(np.floor(valid_size * num_train))
    if args.shuffle:
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[:-num_val], indices[-num_val:]
    train_sampler = data_utils.sampler.SubsetRandomSampler(train_idx)
    valid_sampler = data_utils.sampler.SubsetRandomSampler(valid_idx)

    train_loader = data_utils.DataLoader(train_data, batch_size=args.batch_size, sampler=train_sampler, **kwargs)
    val_loader = data_utils.DataLoader(val_data, batch_size=args.batch_size, sampler=valid_sampler, **kwargs)
    test_loader = data_utils.DataLoader(test_data, batch_size=args.batch_size, **kwargs)

    return train_loader, val_loader, test_loader, args


def load_static_mnist(args, **kwargs):
    """
    Dataloading function for static mnist. Outputs image data in vectorized form: each image is a vector of size 784
    """
    args.dynamic_binarization = False
    args.input_type = 'binary'

    args.input_size = [1, 28, 28]

    # start processing
    def lines_to_np_array(lines):
        return np.array([[int(i) for i in line.split()] for line in lines])

    with open(os.path.join('data', 'MNIST_static', 'binarized_mnist_train.amat')) as f:
        lines = f.readlines()
    x_train = lines_to_np_array(lines).astype('float32')
    
    with open(os.path.join('data', 'MNIST_static', 'binarized_mnist_valid.amat')) as f:
        lines = f.readlines()
    x_val = lines_to_np_array(lines).astype('float32')
    
    with open(os.path.join('data', 'MNIST_static', 'binarized_mnist_test.amat')) as f:
        lines = f.readlines()
    x_test = lines_to_np_array(lines).astype('float32')

    # shuffle train data
    np.random.shuffle(x_train)

    # idle y's
    y_train = np.zeros((x_train.shape[0], 1))
    y_val = np.zeros((x_val.shape[0], 1))
    y_test = np.zeros((x_test.shape[0], 1))

    # pytorch data loader
    train = data_utils.TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    train_loader = data_utils.DataLoader(train, batch_size=args.batch_size, shuffle=args.shuffle, **kwargs)

    validation = data_utils.TensorDataset(torch.from_numpy(x_val).float(), torch.from_numpy(y_val))
    val_loader = data_utils.DataLoader(validation, batch_size=args.batch_size, shuffle=False, **kwargs)

    test = data_utils.TensorDataset(torch.from_numpy(x_test).float(), torch.from_numpy(y_test))
    test_loader = data_utils.DataLoader(test, batch_size=args.batch_size, shuffle=False, **kwargs)

    return train_loader, val_loader, test_loader, args


def load_freyfaces(args, **kwargs):
    # set args
    args.input_size = [1, 28, 20]
    args.input_type = 'multinomial'
    args.dynamic_binarization = False

    TRAIN = 1565
    VAL = 200
    TEST = 200

    # start processing
    with open('data/Freyfaces/freyfaces.pkl', 'rb') as f:
        data = pickle.load(f, encoding='latin1')[0]

    data = data/ 255.

    # NOTE: shuffling is done before splitting into train and test set, so test set is different for every run!
    # shuffle data:
    np.random.seed(args.freyseed)

    np.random.shuffle(data)

    # train images
    x_train = data[0:TRAIN].reshape(-1, 28 * 20)
    # validation images
    x_val = data[TRAIN:(TRAIN + VAL)].reshape(-1, 28 * 20)
    # test images
    x_test = data[(TRAIN + VAL):(TRAIN + VAL + TEST)].reshape(-1, 28 * 20)

    # idle y's
    y_train = np.zeros((x_train.shape[0], 1))
    y_val = np.zeros((x_val.shape[0], 1))
    y_test = np.zeros((x_test.shape[0], 1))

    # pytorch data loader
    train = data_utils.TensorDataset(torch.from_numpy(x_train).float(), torch.from_numpy(y_train))
    train_loader = data_utils.DataLoader(train, batch_size=args.batch_size, shuffle=args.shuffle, **kwargs)

    validation = data_utils.TensorDataset(torch.from_numpy(x_val).float(), torch.from_numpy(y_val))
    val_loader = data_utils.DataLoader(validation, batch_size=args.batch_size, shuffle=False, **kwargs)

    test = data_utils.TensorDataset(torch.from_numpy(x_test).float(), torch.from_numpy(y_test))
    test_loader = data_utils.DataLoader(test, batch_size=args.batch_size, shuffle=False, **kwargs)

    return train_loader, val_loader, test_loader, args


def load_omniglot(args, **kwargs):
    num_validation = 1345

    # set args
    args.input_size = [1, 28, 28]
    args.input_type = 'binary'
    args.dynamic_binarization = True

    # start processing
    def reshape_data(data):
        return data.reshape((-1, 28, 28)).reshape((-1, 28 * 28), order='F')

    omni_raw = loadmat(os.path.join('data', 'OMNIGLOT', 'chardata.mat'))

    # train and test data
    train_data = reshape_data(omni_raw['data'].T.astype('float32'))
    x_test = reshape_data(omni_raw['testdata'].T.astype('float32'))

    # shuffle train data
    np.random.shuffle(train_data)

    # set train and validation data
    x_train = train_data[:-num_validation]
    x_val = train_data[-num_validation:]

    # binarize
    if args.dynamic_binarization:
        args.input_type = 'binary'
        np.random.seed(777)
        x_val = np.random.binomial(1, x_val)
        x_test = np.random.binomial(1, x_test)
    else:
        args.input_type = 'gray'

    # idle y's
    y_train = np.zeros((x_train.shape[0], 1))
    y_val = np.zeros((x_val.shape[0], 1))
    y_test = np.zeros((x_test.shape[0], 1))

    # pytorch data loader
    train = data_utils.TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    train_loader = data_utils.DataLoader(train, batch_size=args.batch_size, shuffle=args.shuffle, **kwargs)

    validation = data_utils.TensorDataset(torch.from_numpy(x_val).float(), torch.from_numpy(y_val))
    val_loader = data_utils.DataLoader(validation, batch_size=args.batch_size, shuffle=False, **kwargs)

    test = data_utils.TensorDataset(torch.from_numpy(x_test).float(), torch.from_numpy(y_test))
    test_loader = data_utils.DataLoader(test, batch_size=args.batch_size, shuffle=False, **kwargs)

    return train_loader, val_loader, test_loader, args


def load_caltech101silhouettes(args, **kwargs):
    # set args
    args.input_size = [1, 28, 28]
    args.input_type = 'binary'
    args.dynamic_binarization = False

    # start processing
    def reshape_data(data):
        return data.reshape((-1, 28, 28)).reshape((-1, 28 * 28), order='F')

    caltech_raw = loadmat(os.path.join('data', 'Caltech101Silhouettes', 'caltech101_silhouettes_28_split1.mat'))

    # train, validation and test data
    x_train = 1. - reshape_data(caltech_raw['train_data'].astype('float32'))
    np.random.shuffle(x_train)
    x_val = 1. - reshape_data(caltech_raw['val_data'].astype('float32'))
    np.random.shuffle(x_val)
    x_test = 1. - reshape_data(caltech_raw['test_data'].astype('float32'))

    y_train = caltech_raw['train_labels']
    y_val = caltech_raw['val_labels']
    y_test = caltech_raw['test_labels']

    # pytorch data loader
    train = data_utils.TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    train_loader = data_utils.DataLoader(train, batch_size=args.batch_size, shuffle=args.shuffle, **kwargs)

    validation = data_utils.TensorDataset(torch.from_numpy(x_val).float(), torch.from_numpy(y_val))
    val_loader = data_utils.DataLoader(validation, batch_size=args.batch_size, shuffle=False, **kwargs)

    test = data_utils.TensorDataset(torch.from_numpy(x_test).float(), torch.from_numpy(y_test))
    test_loader = data_utils.DataLoader(test, batch_size=args.batch_size, shuffle=False, **kwargs)

    return train_loader, val_loader, test_loader, args

