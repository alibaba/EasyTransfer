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

wget https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/tutorial/fashion_bert/fashionbert_fashiongen_patch_train_1K

ls -d $PWD/fashionbert_fashiongen_patch_train_1K > train_list.list_csv
ls -d $PWD/fashionbert_fashiongen_patch_train_1K > dev_list.list_csv

export CUDA_VISIBLE_DEVICES="0"

python pretrain_main.py \
  --workerGPU=1 \
  --mode=train_and_evaluate \
  --train_input_fp=train_list.list_csv  \
  --eval_input_fp=dev_list.list_csv  \
  --pretrain_model_name_or_path=pai-imagebert-base-en  \
  --input_sequence_length=64  \
  --train_batch_size=4  \
  --num_epochs=1  \
  --model_dir=./fashionbert_model_dir  \
  --learning_rate=1e-4  \
  --image_feature_size=131072  \
  --input_schema="image_feature:float:131072,image_mask:int:64,masked_patch_positions:int:5,input_ids:int:64,input_mask:int:64,segment_ids:int:64,masked_lm_positions:int:10,masked_lm_ids:int:10,masked_lm_weights:float:10,nx_sent_labels:int:1"  \


