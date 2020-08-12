<img src="../master/docs/images/stew.png" width="100">

# Gradient Boosted Normalizing Flows

## Introduction
The trend in normalizing flow (NF) literature has been to devise deeper, more complex transformations to achieve greater flexibility.

We propose an alternative: Gradient Boosted Normalizing Flows (GBNF) model a density by successively adding new NF components with gradient boosting. Under the boosting framework, each new NF component optimizes a sample weighted likelihood objective, resulting in new components that are fit to the residuals of the previously trained components.

The GBNF formulation results in a mixture model structure, whose flexibility increases as more components are added. Moreover, GBNFs offer a wider, as opposed to strictly deeper, approach that improves existing NFs at the cost of additional training---not more complex transformations.


Link to paper:

[Gradient Boosted Flows](https://arxiv.org/abs/2002.11896) (under review) by Robert Giaquinto and Arindam Banerjee




## Requirements
The code is compatible with:

  * `pytorch 1.1.0`
  * `python 3.6+` (should work fine with python 2.7 though if you include print_function)

It is recommended that you create a virtual environment with the correct
python version and dependencies. After cloning the repository, change directories
and run the following codes to create a virtual environment:

```bash
python -m venv ./venv
source ./venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

(code assumes `python` refers to python 3.6+, if not use `python3`)


## Data
The experiments can be run on the following images datasets:


* static MNIST: dataset is in data folder;
* OMNIGLOT: the dataset can be downloaded from [link](https://github.com/yburda/iwae/blob/master/datasets/OMNIGLOT/chardata.mat);
* Caltech 101 Silhouettes: the dataset can be downloaded from [link](https://people.cs.umass.edu/~marlin/data/caltech101_silhouettes_28_split1.mat).
* Frey Faces: the dataset can be downloaded from [link](https://github.com/y0ast/Variational-Autoencoder/blob/master/freyfaces.pkl).
* CIFAR10: from Torchvision library
* CelebA: from Torchvision library

Additionally, density estimation experiments can be run on datasets from the UCI repository, which can be downloaded by:

```
./download_datasets.sh
```



## Project Structure

* [main_experiment.py](https://github.com/robert-giaquinto/gradient-boosted-flows/blob/master/main_experiment.py): Run experiments for generative modeling with variational autoencoders on image datasets.
* [density_experiment.py](https://github.com/robert-giaquinto/gradient-boosted-flows/blob/master/density_experiment): Run experiments for density estimation on real datasets.
* [toy_experiment.py](https://github.com/robert-giaquinto/gradient-boosted-flows/blob/master/toy_experiment.py): Run experiments for the toy datasets for density estimation and matching.
* [image_experiment.py](https://github.com/robert-giaquinto/gradient-boosted-flows/blob/master/image_experiment.py): Run experiments for image modeling with only flows (no VAE).
* [models](https://github.com/robert-giaquinto/gradient-boosted-flows/tree/master/models): Collection of models implemented in experiments
* [optimization](https://github.com/robert-giaquinto/gradient-boosted-flows/tree/master/optimization): Training, evaluation, and loss functions used in main experiment.
* [scripts](https://github.com/robert-giaquinto/gradient-boosted-flows/tree/master/scripts): Bash scripts for running experiments, along with default configurations used in experiments.
* [utils](https://github.com/robert-giaquinto/gradient-boosted-flows/tree/master/utils): Utility functions, plotting, and data preparation.
* [data](https://github.com/robert-giaquinto/gradient-boosted-flows/tree/master/data): Folder containing raw data.





## Getting Started

The scripts folder includes examples for running the GBF model on the
Caltech 101 Silhouettes dataset and a density estimation experiment.

**Toy problem: match 2-moons energy function with Boosted Real-NVPs**<br/>
```bash
./scripts/getting_started_toy_matching_gbnf.sh &
```

**Toy problem: density estimation on the 8-Gaussians with Boosted Real-NVPs**<br/>
```bash
./scripts/getting_started_toy_estimation_gbnf.sh &
```

**Density estimation of MINIBOONE dataset with Boosted Glow**<br/>
```bash
./scripts/getting_started_density_estimation_gbnf.sh &
```

**Generative modeling of Caltech 101 Silhouettes images with Boosted Real-NVPs**<br/>
```bash
./scripts/getting_started_vae_gbnf.sh &
```


<img src="../master/docs/images/8gaussians_boosted_K1_bs64_C8_reg80_realnvp_tanh1_hsize256.gif" width="400" height="400" />

<br/>
More information about additional argument options can be found by running ```python main_experiment.py -h```









