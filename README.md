<img src="../master/docs/images/stew.png" width="100">

# Ensemble Methods for Normalizing Flows

Under construction! Experiments based off Sylvester Normalizing Flows ([code](https://github.com/riannevdberg/sylvester-flows).

## Requirements
The code is compatible with:

  * `pytorch 1.0.0`
  * `python 2.7` (should work fine with python 3 though)


## Data
The experiments can be run on the following datasets:
* static MNIST: dataset is in data folder;
* OMNIGLOT: the dataset can be downloaded from [link](https://github.com/yburda/iwae/blob/master/datasets/OMNIGLOT/chardata.mat);
* Caltech 101 Silhouettes: the dataset can be downloaded from [link](https://people.cs.umass.edu/~marlin/data/caltech101_silhouettes_28_split1.mat).
* Frey Faces: the dataset can be downloaded from [link](https://github.com/y0ast/Variational-Autoencoder/blob/master/freyfaces.pkl).
* CIFAR10: from Torch Vision library



## Project Structure
TBD



## Getting Started

Below, example commands are given for running experiments on static MNIST with different types of normalizing flows, for 4 flows:

**Factorized Gaussian posterior**<br/>
```bash
python main_experiment.py -d mnist --flow no_flow
```

**Planar flows**<br/>
```bash
python main_experiment.py -d mnist -nf 4 --flow planar
```

**Inverse Autoregressive flows**<br/>
This examples uses MADEs with 320 hidden units.
```bash
python main_experiment.py -d mnist -nf 4 --flow iaf --made_h_size 320
```

<br/>
More information about additional argument options can be found by running ```python main_experiment.py -h```




