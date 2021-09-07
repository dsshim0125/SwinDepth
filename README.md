# Unsupervised Depth Estimation Using Monocular Sequences via Hierarchical Transformer and Densely Cascaded Network

This is the PyTorch implementation of the paraper submitted in AAAI 22.

We provide pre-trained weights and evaluation codes for a simple visualization of depth estimation  results on KITTI dataset.

## Setup


```shell
conda create -n ht_dcmnet python=3.8.5
conda activate ht_dcmnet
conda install pytorch torchvision cudatoolkit=11.1 -c pytorch -c nvidia
```
Our experiments has been done with PyTorch 1.9.0, CUDA 11.2, Python 3.8.5 and Ubuntu 18.04.
