#!/bin/bash

/root/ephemeral-hdfs/bin/hadoop distcp s3n://timit-linear-solver-data/timit-fft-aPart1-1 /timit-fft-aPart1-1
/root/ephemeral-hdfs/bin/hadoop distcp s3n://timit-linear-solver-data/timit-fft-testRPM-test-1 /timit-fft-testRPM-test-1
/root/ephemeral-hdfs/bin/hadoop distcp s3n://timit-linear-solver-data/timit-fft-null-labels /timit-fft-null-labels

/root/ephemeral-hdfs/bin/hadoop distcp s3n://timit-linear-solver-data/timit-actual /timit-actual

for i in `seq 2 5`
do
  /root/ephemeral-hdfs/bin/hadoop distcp s3n://timit-linear-solver-data/timit-fft-aPart$i-$i /timit-fft-aPart$i-$i
  /root/ephemeral-hdfs/bin/hadoop distcp s3n://timit-linear-solver-data/timit-fft-testRPM-test-$i /timit-fft-testRPM-test-$i
done
