#!/bin/bash

/root/ephemeral-hdfs/bin/hadoop distcp s3n://imagenet-linear-solver-data/daisy-aPart1-1 /daisy-aPart1-1
/root/ephemeral-hdfs/bin/hadoop distcp s3n://imagenet-linear-solver-data/daisy-testFeatures-test-1 /daisy-testFeatures-test-1
/root/ephemeral-hdfs/bin/hadoop distcp s3n://imagenet-linear-solver-data/daisy-null-labels /daisy-null-labels

/root/ephemeral-hdfs/bin/hadoop distcp s3n://imagenet-linear-solver-data/lcs-aPart1-1 /lcs-aPart1-1
/root/ephemeral-hdfs/bin/hadoop distcp s3n://imagenet-linear-solver-data/lcs-testFeatures-test-1 /lcs-testFeatures-test-1
/root/ephemeral-hdfs/bin/hadoop distcp s3n://imagenet-linear-solver-data/lcs-null-labels /lcs-null-labels

/root/ephemeral-hdfs/bin/hadoop distcp s3n://imagenet-linear-solver-data/imagenet-test-actual /imagenet-test-actual
