# Realisitic Out-of-Distribution (OOD) Detection Benchmark

Welcome to the official repository for reproducing the results and accessing the dataset associated with our Out-of-Distribution (OOD) Detection Benchmark. This README will guide you through the necessary steps to set up the environment, download the required packages, and run the code to replicate the experiments.

For more details on our evaluation framework, you can refer to our paper titled "[Towards Realistic Out-of-Distribution Detection: A Novel Evaluation Framework for Improving Generalization in OOD Detection](https://arxiv.org/abs/2211.10892)".

## Table of Contents

- [Installation](#installation)
- [Environment](#environment)
- [Dataset](#dataset)
- [Usage](#usage)
  - [Creating Modified Datasets](#creating-modified-datasets)
  - [Feature Extraction](#feature-extraction)
  - [OOD Detection Performance](#ood-detection-performance)

## Installation

Before you begin, ensure you have the necessary packages installed by running the following commands:

```bash
pip install faiss-gpu
pip install git+https://github.com/rwightman/pytorch-image-models
pip install Wand
apt-get install libmagickwand-dev
```

You also need to have PyTorch, scikit-learn (sklearn), scipy, and scikit-image installed. Visit the respective websites to find installation commands based on your operating system.

## Environment
This codebase has been extensively tested on Unix-based systems. If you prefer a hassle-free experience, you can also run the code on Google Colab. Most required packages are already pre-installed. On Google Colab, you only need to install the packages mentioned above, as the rest are readily available. (Successfully tested on Colab Pro+)


## Dataset

### ImageNet-30

Download the ImageNet-30 dataset from this [link](https://drive.google.com/file/d/13xzVuQMEhSnBRZr-YaaO08coLU2dxAUq/view).

### CIFAR-10 and CIFAR-100

The CIFAR-10 and CIFAR-100 datasets will be automatically downloaded using PyTorch's data loader.

If you want to skip creating the datasets and directly download these datasets, use these links: [CIFAR-10-R]([https://drive.google.com/file/d/13xzVuQMEhSnBRZr-YaaO08coLU2dxAUq/view](https://zenodo.org/record/8316298/files/CIFAR-10-R.zip)), [CIFAR-100-R](https://zenodo.org/record/8316429/files/CIFAR-100-R.zip)

## Usage

### Creating Modified Datasets
Run the following commands to create modified datasets:

```bash
python create_CIFAR-10-R.py
python create_CIFAR-100-R.py
python create_ImageNet-30-R.py
```


### Feature Extraction
Extract features from the modified datasets with these commands:

```bash
python extract_features_CIFAR-10-R.py
python extract_features_CIFAR-100-R.py
python extract_features_ImageNet-30-R.py
```


### OOD Detection Performance
To calculate OOD detection performance on different datasets, run:

```bash
python calculate_OOD_performance_CIFAR-10-R.py
python calculate_OOD_performance_CIFAR-100-R.py
python calculate_OOD_performance_ImageNet-30-R.py
```

You can preview the results before applying adaptation by commenting out the sections in the code indicated by "############ adaptation".


## How to Cite

If you find our work or dataset helpful in your research, please consider citing:

```
@article{khazaie2022out,
  title={Towards Realistic Out-of-Distribution Detection: A Novel Evaluation Framework for Improving Generalization in OOD Detection},
  author={Khazaie, Vahid Reza and Wong, Anthony and Sabokrou, Mohammad},
  journal={arXiv preprint arXiv:2211.10892},
  year={2023}
}
```
