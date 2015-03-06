#!/bin/bash

DRIVER_MEM="40g"
FAT_JAR="/root/ml-matrix/target/scala-2.10/mlmatrix-assembly-0.1.jar"
DATA_DIR="/"
PARTS=32
SOLVER=normal
LAMBDA=0.1
CLASS="CheckQR"
THRESH=1e-8
DATASET="daisy"

SPARK_MASTER=`cat /root/spark-ec2/cluster-url`

ID=$CLASS-EC2-$DATASET-$PARTS-$LAMBDA-$THRESH-`date +"%Y_%m_%d_%H_%M_%S"`

/root/spark/bin/spark-submit \
  --class edu.berkeley.cs.amplab.mlmatrix.$CLASS \
  --driver-class-path $FAT_JAR \
  --driver-memory $DRIVER_MEM \
  --master $SPARK_MASTER \
  $FAT_JAR \
  $SPARK_MASTER $DATA_DIR $PARTS $SOLVER $LAMBDA $THRESH $DATASET \
  2>$ID.stderr \
  1>$ID.stdout
