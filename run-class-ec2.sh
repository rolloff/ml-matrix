#!/bin/bash

DRIVER_MEM="240g"
FAT_JAR="/root/ml-matrix/target/scala-2.10/mlmatrix-assembly-0.1.jar"
DATA_DIR="/"
#PARTS=128
LAMBDA=0.1
CLASS="CheckQR"
DATASET="gaussian-10000-10"

SPARK_MASTER=`cat /root/spark-ec2/cluster-url`


for PARTS in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 22 24 26 28 30 32 64
do
ID=$CLASS-EC2-$DATASET-$PARTS-$LAMBDA-`date +"%Y_%m_%d_%H_%M_%S"`
  /root/spark/bin/spark-submit \
    --class edu.berkeley.cs.amplab.mlmatrix.$CLASS \
    --driver-class-path $FAT_JAR \
    --driver-memory $DRIVER_MEM \
    --master $SPARK_MASTER \
    $FAT_JAR \
    $SPARK_MASTER $DATA_DIR $PARTS $LAMBDA $DATASET \
    2>$ID.stderr \
    1>$ID.stdout
done
