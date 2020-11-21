#!/usr/bin/env bash

#please download dataset from tianchi 
tar -zxf tianchi_datasets.tgz
rm *.tgz

for TASK_NAME in TNEWS OCEMOTION OCNLI
do
  echo " task name is $TASK_NAME"
  python convert_csv_to_tfrecords.py --mode preprocess --config config/${TASK_NAME}_preprocess_train.json
  python convert_csv_to_tfrecords.py --mode preprocess --config config/${TASK_NAME}_preprocess_dev.json
done

ls -d $PWD/tianchi_datasets/TNEWS/train.tfrecord > train.list_tfrecord
ls -d $PWD/tianchi_datasets/OCEMOTION/train.tfrecord >> train.list_tfrecord
ls -d $PWD/tianchi_datasets/OCNLI/train.tfrecord >> train.list_tfrecord

ls -d $PWD/tianchi_datasets/TNEWS/dev.tfrecord > dev.list_tfrecord
ls -d $PWD/tianchi_datasets/OCEMOTION/dev.tfrecord >> dev.list_tfrecord
ls -d $PWD/tianchi_datasets/OCNLI/dev.tfrecord >> dev.list_tfrecord



