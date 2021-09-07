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

## Simple Prediction

You can simply visualize the depth estimation results on some images from KITTI with:

```shell
python test_simple.py --image_path=./test_images/
```

You can check depth estimation results with other images from KITTI or your own datasets by adding test images on the folder named "test_images".

## KITTI Dataset

You can download the entire [raw KITTI dataset](http://www.cvlibs.net/datasets/kitti/raw_data.php) by running:
```shell
wget -i splits/kitti_archives_to_download.txt -P /YOUR/DATA/PATH/
```

```shell
find /YOUR/DATA/PATH/ -name '*.png' | parallel 'convert -quality 92 -sampling-factor 2x2,1x1,1x1 {.}.png {.}.jpg && rm {}'
```

## Training

```shell
python train.py --data_path=/YOUR/DATA/PATH --log_dir=./checkpoints --model_name=ht_dcmnet --num_epochs=40 --batch_size=12
```

## Evaluation

To prepare the ground truth depth maps run:
```shell
python export_gt_depth.py --data_path /YOUR/DATA/PATH --split eigen
```

The following example command evaluates best weights:
```shell
python evaluate_depth.py --data_path=/YOUR/DATA/PATH --load_weights_folder ./checkpoints/ht_dcmnet/models/best/
```
