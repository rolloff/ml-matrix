#!/bin/bash

SPARK_MEM=8g
SPARK_MASTER="local"
DATA_DIR="imagenet-linear-solver-data/"
PARTS=2
LAMBDA=0.1
CLASS="CheckQR"
DATASET="gaussian-10000-10"

export SPARK_MEM

#for PARTS in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 22 24 26 28 30 32 64
#for PARTS in 64 256
#for PARTS in 2
for PARTS in 128 256  
do
  ID=$CLASS-$DATASET-$PARTS-$LAMBDA-`date +"%Y_%m_%d_%H_%M_%S"`
  ./run-main.sh edu.berkeley.cs.amplab.mlmatrix.$CLASS $SPARK_MASTER $DATA_DIR $PARTS $LAMBDA $DATASET 2>$ID.stderr 1>>results.stdout
done
