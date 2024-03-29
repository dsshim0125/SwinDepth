# SwinDepth

This is the PyTorch implementation of the paper "SwinDepth: Unsupervised Depth Estimation using Monocular Sequences via Swin Transformer and Densely Cascaded Network" (ICRA 2023). [[paper](https://arxiv.org/abs/2301.06715)]

We provide pre-trained weights and evaluation codes for a simple visualization of depth estimation results on KITTI dataset.

Download the pre-trained weights from [here](https://drive.google.com/drive/folders/1I3E3qLFoYeDw8pmbFbr4TNpElnAaxdtO?usp=sharing) and place them in "./checkpoints/best" folder
## Setup


```shell
conda create -n ht_dcmnet python=3.8.5
conda activate ht_dcmnet
conda install pytorch torchvision cudatoolkit=11.1 -c pytorch -c nvidia
pip install -r requirements.txt
```
Our experiments has been done with PyTorch 1.9.0, CUDA 11.2, Python 3.8.5 and Ubuntu 18.04. We use 4 NVIDIA RTX 3090 GPUs for training, but you can still run our code with GPUs which have smaller memory by reducing the batch_size. A simpel visualziation can be done by GPUs with 3GB of memory use or CPU only is also functional.

## Simple Prediction

You can simply visualize the depth estimation results on some images from KITTI with:

```shell
python test_simple.py --image_path=./test_images/
```

You can check depth estimation results with other images from KITTI or your own datasets by adding test images on the folder named "test_images". You can run the code without GPU by using --no_cuda flag.

## KITTI Dataset

You can download the entire [raw KITTI dataset](http://www.cvlibs.net/datasets/kitti/raw_data.php) by running:
```shell
wget -i splits/kitti_archives_to_download.txt -P /YOUR/DATA/PATH/
```

KITTI images are converted from `.png` to `.jpg` extension with this command for fast load times during training:

```shell
find /YOUR/DATA/PATH/ -name '*.png' | parallel 'convert -quality 92 -sampling-factor 2x2,1x1,1x1 {.}.png {.}.jpg && rm {}'
```

The commands above results in the data_path:
```
/YOUR/DATA/PATH
  |----2011_09_26
      |----2011_09_26_drive_0001_sync  
          |-----.......  
          |----image_02
              |-----data
                  |-----0000000000.jpg
                  |-----.......
              |-----timestamps.txt
          |-----.......
      |----.........        
  |----2011_09_28        
  |----.........        
```

## Training

For training, you have to pre-train Swin Transformer encoder in ImageNet-1k dataset.

You can either simply download ImageNet-pretrained encoder weight [here](https://drive.google.com/drive/folders/1I3E3qLFoYeDw8pmbFbr4TNpElnAaxdtO?usp=sharing) named '104checkpoint.pth' or train Swin Transformer yourself with PyTorch offical [code](https://github.com/pytorch/examples/tree/main/imagenet).
Then, you place the pretrained weight in ./checkpoints/imagenet folder.

The depth estimation network is trained by running:
```shell
python train.py --data_path=/YOUR/DATA/PATH --log_dir=./checkpoints --model_name=ht_dcmnet --num_epochs=40 --batch_size=12
```

## Evaluation

Before evaluation, you should prepare ground truth depth maps by running:

```shell
python export_gt_depth.py --data_path /YOUR/DATA/PATH --split eigen
```

The following example command evaluates best weights:

```shell
python evaluate_depth.py --data_path=/YOUR/DATA/PATH --load_weights_folder ./checkpoints/best/
```

## Reference

1. Monodepth2 - https://github.com/nianticlabs/monodepth2
2. timm - https://github.com/rwightman/pytorch-image-models
3. mmsegmentation - https://github.com/open-mmlab/mmsegmentation
