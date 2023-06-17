# Setup

This document describes the necessary steps for setting the Pascal VOC 2012 and MS COCO 2014 datasets
for evaluation. We utilize them in their original format/setting and no specific alterations are performed.
Thus, these steps may overlap with other repositories.

## Common

```shell
export PYTHONPATH=$(pwd)

PY=python3.9     # path to python
PIP=pip3.9       # path to PIP
DEVICES=0,1,2,3  # the GPUs used.
WORKERS=24       # number of workers spawn during dCRF refinement and evaluation.

$PIP install torch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu111
$PIP install -r requirements.txt

# WANDB_PROJECT=some-project-id  # Specify project to export training reports
# wandb disabled                 # Otherwise, no metrics exported.
```

## Pascal VOC 2012

```shell
DATA_DIR=/datasets
cd $DATA_DIR

# Download images and labels from http://host.robots.ox.ac.uk/pascal/VOC/voc2012/#devkit:
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
tar -xf VOCtrainval_11-May-2012.tar

# Download the augmented segmentation maps from http://home.bharathh.info/pubs/codes/SBD
# *only if* you are planning on training the fully-suppervised models (for comparison purposes):
wget http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz
tar -xzf benchmark.tgz

# Merge them into the VOCdevkit/VOC2012/SegmentationClass folder:
mv aug_seg/* VOCdevkit/VOC2012/SegmentationClass/
```

## MS COCO 2014

```shell
DATA_DIR=/datasets/coco14
cd $DATA_DIR

# Download MS COCO images and labels:
wget http://images.cocodataset.org/zips/train2014.zip
wget http://images.cocodataset.org/zips/val2014.zip
wget https://github.com/jbeomlee93/RIB/raw/main/coco14/cls_labels.npy
wget https://github.com/jbeomlee93/RIB/raw/main/coco14/cls_labels_coco.npy

# Download segmentation labels in VOC format from
# https://drive.google.com/file/d/1pRE9SEYkZKVg0Rgz2pi9tg48j7GlinPV/view
gdown 1pRE9SEYkZKVg0Rgz2pi9tg48j7GlinPV

# Unzip everything:
unzip train2014.zip
unzip val2014.zip
unzip coco_annotations_semantic.zip
mv coco_annotations_semantic/coco_seg_anno .
rm coco_annotations_semantic -r
```
