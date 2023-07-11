# Dataset

if [[ $DATASET == voc12 ]]; then
  DOMAIN_TRAIN=train_aug
  DOMAIN_VALID=train
  DOMAIN_VALID_SEG=val
  DATA_DIR=$DATASETS_DIR/VOCdevkit/VOC2012

  IMAGE_SIZE=512
  MIN_IMAGE_SIZE=320
  MAX_IMAGE_SIZE=640

  VALIDATE_MAX_STEPS=0
  VALIDATE_THRESHOLDS=0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45

  LR=0.1
fi
if [[ $DATASET == coco14 ]]; then
  DOMAIN_TRAIN=train2014
  DOMAIN_VALID=train2014
  DOMAIN_VALID_SEG=valid2014
  DATA_DIR=$DATASETS_DIR/coco14

  IMAGE_SIZE=640
  MIN_IMAGE_SIZE=400
  MAX_IMAGE_SIZE=800

  VALIDATE_MAX_STEPS=620  # too many valid samples
  VALIDATE_THRESHOLDS=0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45

  LR=0.05
fi
if [[ $DATASET == deepglobe ]]; then
  DOMAIN_TRAIN=train75
  DOMAIN_VALID=train75
  DOMAIN_VALID_SEG=test
  DATA_DIR=$DATASETS_DIR/DGdevkit

  IMAGE_SIZE=2048
  MIN_IMAGE_SIZE=2048
  MAX_IMAGE_SIZE=2048

  VALIDATE_MAX_STEPS=0
  # single th because the bg class is not added based on thresholding.
  # We assume bg=agriculture_land (most frequent).
  VALIDATE_THRESHOLDS=0.5

  LR=0.1
fi
