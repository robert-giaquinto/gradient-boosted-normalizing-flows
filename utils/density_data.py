import torch
import torch.distributions as D
import numpy as np
import math
import sklearn
import sklearn.datasets
from sklearn.utils import shuffle as util_shuffle


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
        num_clusters = 6
        mix_props = np.random.dirichlet([10.0] * num_clusters).astype("float32")
        mu = torch.from_numpy(np.random.normal(loc=[0.0, 0.0], scale=1.0, size=[num_clusters, 2]).astype("float32"))
        sigma = np.repeat(np.eye(2)[None], num_clusters, axis=0).astype("float32") * 0.8
        sigma[:, 1, 0] = np.random.uniform(low=0.0, high=0.8, size=[num_clusters]).astype("float32") *\
            np.random.choice([1, -1], size=[num_clusters])
        mix_props = torch.from_numpy(mix_props)
        sigma = torch.from_numpy(sigma)

        print(mix_props, "\n", mu, "\n", sigma)

        u_z = lambda z: -1.0 * torch.log(sum(
            torch.exp(D.MultivariateNormal(mu_i, sigma_i).log_prob(z)) * mix_props_i \
            for (mix_props_i, mu_i, sigma_i) in zip(mix_props, mu, sigma)))

        return u_z
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
            data = torch.from_numpy(data) + torch.randn(data.shape)
            data = data.astype("float32")

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
            num_clusters = 4
            mix_props = np.random.dirichlet([18.5] * num_clusters)
            mu = np.random.normal(loc=[0.0, 0.0], scale=1.0, size=[n_clust, 2])
            sigma = np.random.gamma(5.5, scale=0.05, size=[n_clust, 2, 2]) 
            sigma[:, 0, 0] = np.random.uniform(low=0.7, high=1.4, size=[num_clusters]) * sigma[:, 1, 1]
            sigma[:, 1, 0] = np.random.uniform(low=0.2, high=0.4, size=[num_clusters]) * sigma[:, 1, 1]

            sigma[:,0,1] = sigma[:,1,0]
            Y = []
            for (mix_props_i, mu_i, cov_mat_i) in zip(mix_props, mu, sigma):
                Y.extend(np.random.multivariate_normal(mu_i, cov_mat_i, size=[int(batch_size * mix_props_i)]))
                
            Y = np.array(Y).reshape([batch_size, 2])
            data = torch.from_numpy(Y)
            
        else:
            raise ValueError(f"The toy dataset {args.dataset} hasn't been defined!")

        return data

    return data_sampler
