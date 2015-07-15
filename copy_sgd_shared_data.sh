#!/bin/bash

/root/ephemeral-hdfs/bin/hadoop distcp s3n://sgd-shared-data/imagenet-fv-4k/trainFeatures-all.csv /imagenet-fv-4k/trainFeatures-all.csv
/root/ephemeral-hdfs/bin/hadoop distcp s3n://sgd-shared-data/imagenet-fv-4k/trainLabels-all.csv /imagenet-fv-4k/trainLabels-all.csv
