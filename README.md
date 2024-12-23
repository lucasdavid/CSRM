# CSRM

## Introduction

This repository contains the official implementation for the paper "Learning Weakly Supervised Semantic Segmentation Through
Cross-Supervision and Contrasting of Pixel-Level Pseudo-Labels", accepted for publishing in VISAPP 2025.

<p align="center" style="text-align:center;">
  <img src="https://github.com/lucasdavid/wsss-csrm/blob/master/assets/wsss-csrm-diagram.png"
       alt="Diagram for the proposed method CSRM." />
</p>

## Results

### Pascal VOC 2012 (test)

| bg | a.plane | bike | bird  | boat  | bottle | bus   | car   | cat   | chair | cow   | d.table | dog   | horse | m.bike | person | p.plant | sheep | sofa  | train | tv | Overall |
|--------------------------------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|
| 92.3 | 92.0 | 43.9 | 90.1 | 66.4 | 75.0 | 93.3    | 87.1 | 86.8    | 41.4 | 89.7 | 49.6    | 88.7    | 87.9 | 85.1 | 77.9    | 72.2 | 91.5 | 46.5    | 70.1    | 47.8    | 75.0   |


### MS COCO 2014 (val)

| bg | person | bicycle | car | motorcycle | airplane | bus | train | truck | boat | traffic light | fire hydrant | stop sign | parking meter | bench | bird | cat | dog | horse | sheep | cow | elephant | bear | zebra | giraffe | backpack | umbrella | handbag | tie | suitcase | frisbee | skis | snowboard | sports ball | kite | baseball bat | baseball glove | skateboard | surfboard | tennis racket | bottle | wine glass | cup | fork | knife | spoon | bowl | banana | apple | sandwich | orange | broccoli | carrot | hot dog | pizza | donut | cake | chair | couch | potted plant | bed | dining table | toilet | tv | laptop | mouse | remote | keyboard | cell phone | microwave | oven | toaster | sink | refrigerator | book | clock | vase | scissors | teddy bear | hair drier | toothbrush | Overall |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 75.4 | 46.6 | 49.8 | 50.7 | 73.9 | 53.3 | 80.1 | 52.5 | 59.6 | 49.7 | 33.3 | 80.2 | 66.3 | 51.0 | 44.9 | 53.4 | 76.2 | 75.4 | 66.3 | 76.9 | 81.6 | 82.8 | 85.1 | 85.9 | 82.4 | 33.8 | 66.0 | 22.0 | 35.0 | 59.5 | 74.6 | 24.2 | 39.6 | 25.4 | 40.8 | 21.2 | 7.6 | 35.0 | 42.1 | 36.2 | 42.8 | 45.0 | 40.0 | 26.2 | 32.7 | 23.7 | 21.2 | 64.8 | 58.3 | 40.2 | 62.2 | 54.2 | 41.8 | 52.5 | 61.0 | 56.3 | 52.4 | 29.1 | 46.6 | 29.6 | 58.1 | 13.0 | 67.4 | 42.8 | 66.6 | 31.0 | 58.2 | 66.9 | 69.4 | 55.8 | 39.7 | 46.7 | 35.0 | 50.4 | 46.4 | 12.5 | 35.0 | 54.8 | 73.0 | 52.4 | 35.0 | 50.5 |

## Setup
Check the [SETUP.md](SETUP.md) file for information regarding the setup of the Pascal VOC 2012 and MS COCO 2014 datasets.

## Experiments

The scripts used for training P-NOC are available in the [runners](runners) folder.
Generally, they will run the following scripts, in this order:

```shell
./runners/0-setup.sh
./runners/1-csrm.sh
./runners/3-sam-refinement.sh
./runners/4-segmentation.sh
```

## Acknowledgements

Much of the code here was borrowed from psa, OC-CSE, Puzzle-CAM and CCAM repositories.
We thank the authors for their considerable contributions and efforts.
