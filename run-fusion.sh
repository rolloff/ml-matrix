#!/usr/bin/env bash
set -x #echo on

SPARK_MEM=8g
SPARK_MASTER="spark://Beccas-MacBook-Pro.local:7077"
DATA_DIR="imagenet-linear-solver-data/"
PARTS=8
LAMBDA=0.1
SOLVER="tsqr"

# Parameters for SGD
#SGD_STEP_SIZE=0.1
#SGD_ITERATIONS=10
#SGD_BATCH_FRACTION=1.0

export SPARK_MEM

./run-main.sh edu.berkeley.cs.amplab.mlmatrix.Fusion $SPARK_MASTER $DATA_DIR $PARTS $SOLVER $LAMBDA
#>& logs-$PARTS-$SOLVER-$LAMBDA.txt
