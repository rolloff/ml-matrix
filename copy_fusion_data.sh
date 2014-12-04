#!/bin/bash

/root/ephemeral-hdfs/bin/hadoop distcp s3n://imagenet-linear-solver-data/daisy-aPart1-1 /daisy-aPart1-1
/root/ephemeral-hdfs/bin/hadoop distcp s3n://imagenet-linear-solver-data/daisy-testFeatures-test-1 /daisy-testFeatures-test-1
/root/ephemeral-hdfs/bin/hadoop distcp s3n://imagenet-linear-solver-data/daisy-null-labels /daisy-null-labels

/root/ephemeral-hdfs/bin/hadoop distcp s3n://imagenet-linear-solver-data/lcs-aPart1-1 /lcs-aPart1-1
/root/ephemeral-hdfs/bin/hadoop distcp s3n://imagenet-linear-solver-data/lcs-testFeatures-test-1 /lcs-testFeatures-test-1
/root/ephemeral-hdfs/bin/hadoop distcp s3n://imagenet-linear-solver-data/lcs-null-labels /lcs-null-labels

/root/ephemeral-hdfs/bin/hadoop distcp s3n://imagenet-linear-solver-data/imagenet-test-actual /imagenet-test-actual

for i in `seq 2 5`
do
  /root/ephemeral-hdfs/bin/hadoop distcp s3n://imagenet-linear-solver-data/daisy-aPart$i-$i /daisy-aPart$i-$i
  /root/ephemeral-hdfs/bin/hadoop distcp s3n://imagenet-linear-solver-data/daisy-testFeatures-test-$i /daisy-testFeatures-test-$i

  /root/ephemeral-hdfs/bin/hadoop distcp s3n://imagenet-linear-solver-data/lcs-aPart$i-$i /lcs-aPart1-$i
  /root/ephemeral-hdfs/bin/hadoop distcp s3n://imagenet-linear-solver-data/lcs-testFeatures-test-$i /lcs-testFeatures-test-$i
done
