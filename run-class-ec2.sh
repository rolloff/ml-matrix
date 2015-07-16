#!/bin/bash

DRIVER_MEM="240g"
FAT_JAR="/root/ml-matrix/target/scala-2.10/mlmatrix-assembly-0.1.jar"
DATA_DIR="/"
CLASS="CheckSolvers"
DATASET="imagenet-fv-4k"
PARTS=128
SOLVER="tsqr"
SPARK_MASTER=`cat /root/spark-ec2/cluster-url`

for LAMBDA in 0.1 0.01 0.001 0.0001 0.000075 0.00005 0.000025 0.000001 0.0000001
do
ID=$CLASS-EC2-$DATASET-$PARTS-$SOLVER-$LAMBDA-`date +"%Y_%m_%d_%H_%M_%S"`
  /root/spark/bin/spark-submit \
    --class edu.berkeley.cs.amplab.mlmatrix.$CLASS \
    --driver-class-path $FAT_JAR \
    --driver-memory $DRIVER_MEM \
    --master $SPARK_MASTER \
    $FAT_JAR \
    $SPARK_MASTER $DATA_DIR $DATASET $PARTS $SOLVER $LAMBDA \
    2>$ID.stderr \
    1>$ID.stdout
done
