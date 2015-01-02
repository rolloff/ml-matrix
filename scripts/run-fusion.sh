#!/usr/bin/env bash

#SPARK_MASTER=`cat /root/spark-ec2/cluster-url`
SPARK_MASTER="local"

# Parameters for SGD
#SGD_STEP_SIZE=0.1
#SGD_ITERATIONS=10
#SGD_BATCH_FRACTION=1.0

if [[ $# -ne 1 ]]; then
  echo "Usage: run-fusion.sh <suffix>"
  exit 0;
fi

./run-main.sh edu.berkeley.cs.amplab.mlmatrix.Fusion $SPARK_MASTER tsqr >& logs-tsqr-$1.txt

#./run-main.sh edu.berkeley.cs.amplab.mlmatrix.Fusion $SPARK_MASTER sgd $SGD_STEP_SIZE $SGD_ITERATIONS $SGD_BATCH_FRACTION >& logs-sgd-$1.txt
