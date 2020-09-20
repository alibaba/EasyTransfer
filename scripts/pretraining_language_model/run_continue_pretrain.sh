#!/usr/bin/env bash
# coding=utf-8
# Copyright (c) 2019 Alibaba PAI team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#Warning: If you want to distributed training using multiple gpus, please upgrade your tensorflow >=1.14 or using PAITF in Alibaba Cloud
export CUDA_VISIBLE_DEVICES="1"
python pretrain_main.py \
  --workerGPU=1  \
  --mode=train_and_evaluate \
  --train_input_fp=train_list.list_tfrecord  \
  --eval_input_fp=dev_list.list_tfrecord  \
  --pretrain_model_name_or_path=google-bert-base-zh  \
  --train_batch_size=4  \
  --num_epochs=2  \
  --model_dir=plm_model_dir_continue_pretrain  \
  --learning_rate=1e-4  \
  --vocab_fp=./vocab.txt \
  --input_sequence_length=128  \
  --input_schema="input_ids:int:128,input_mask:int:128,segment_ids:int:128,masked_lm_positions:int:20,masked_lm_ids:int:20,masked_lm_weights:float:20,next_sentence_labels:int:1"  \
  --loss="mlm+nsp"  \
  --model_type="bert"  \
  --hidden_size=768  \
  --intermediate_size=3072  \
  --num_hidden_layers=12  \
  --num_attention_heads=12