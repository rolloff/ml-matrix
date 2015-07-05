#!/bin/bash

SPARK_MEM=8g
SPARK_MASTER="local"
DATA_DIR="imagenet-linear-solver-data/"
PARTS=2
CLASS="CheckQR"
DATASET="gaussian-500-100"
NUMROWS=500
NUMCOLS=100

export SPARK_MEM

for PARTS in 2
do
  ID=$CLASS-$DATASET-$PARTS-$NUMROWS-$NUMCOLS-`date +"%Y_%m_%d_%H_%M_%S"`
  ./run-main.sh edu.berkeley.cs.amplab.mlmatrix.$CLASS $SPARK_MASTER $DATA_DIR $PARTS $DATASET $NUMROWS $NUMCOLS 2>$ID.stderr 1>$ID.stdout
done
