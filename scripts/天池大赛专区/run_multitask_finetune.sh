#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES="1"
python multitask_finetune.py --workerGPU=1 \
  --mode=train  \
  --train_input_fp=train.list_tfrecord  \
  --predict_input_fp=dev.list_tfrecord  \
  --pretrain_model_name_or_path=google-bert-base-zh  \
  --train_batch_size=16  \
  --num_epochs=1  \
  --model_dir=multitask_model_dir  \
  --learning_rate=3e-5  \


