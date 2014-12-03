#!/bin/bash

DRIVER_MEM="40g"
FAT_JAR="/root/ml-matrix/target/scala-2.10/mlmatrix-assembly-0.1.jar"
DATA_DIR="/"
PARTS=224
SOLVER=tsqr
LAMBDA=0.1

SPARK_MASTER=`cat /root/spark-ec2/cluster-url`

ID=fusion-imagenet-$PARTS-$SOLVER-$LAMBDA-`date +"%Y_%m_%d_%H_%M_%S"`

/root/spark/bin/spark-submit \
  --class edu.berkeley.cs.amplab.mlmatrix.Fusion \
  --driver-class-path $FAT_JAR \
  --driver-memory $DRIVER_MEM \
  --master $SPARK_MASTER \
  $FAT_JAR \
  $SPARK_MASTER $DATA_DIR $PARTS $SOLVER $LAMBDA \
  2>$ID.stderr \
  1>$ID.stdout
