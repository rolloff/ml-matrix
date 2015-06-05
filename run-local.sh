#!/bin/bash

SPARK_MEM=8g
SPARK_MASTER="local"
DATA_DIR="imagenet-linear-solver-data/"
PARTS=1
LAMBDA=0.1
CLASS="CheckQR"
THRESH=1e-8
DATASET="daisy"

export SPARK_MEM

ID=$CLASS-$DATASET-$PARTS-$LAMBDA-$THRESH-`date +"%Y_%m_%d_%H_%M_%S"`

./run-main.sh edu.berkeley.cs.amplab.mlmatrix.$CLASS $SPARK_MASTER $DATA_DIR $PARTS $LAMBDA $THRESH $DATASET 2>$ID.stderr 1>$ID.stdout
