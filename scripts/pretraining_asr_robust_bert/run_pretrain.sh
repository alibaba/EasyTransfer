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

input_sequence_length=512
model_type="bert"
python pretrain_main.py \
  --workerGPU=1 \
  --mode=train  \
  --train_input_fp=train_list_${input_sequence_length}.list_tfrecord  \
  --train_batch_size=8  \
  --num_epochs=1000  \
  --model_dir=model_dir_${model_type}_${input_sequence_length}  \
  --learning_rate=1e-4  \
  --vocab_fp=vocab.txt \
  --loss=mlm  \
  --input_sequence_length=${input_sequence_length}  \
  --input_schema="input_ids:int:${input_sequence_length},input_mask:int:${input_sequence_length},segment_ids:int:${input_sequence_length},masked_lm_positions:int:20,masked_lm_ids:int:20,masked_lm_weights:float:20,sparse_idxs:int:400,sparse_values:float:400,sparse_idxs_supervised:int:400,sparse_values_supervised:float:400"  \
  --model_type=${model_type}  \
  --hidden_size=768  \
  --intermediate_size=3072 \
  --num_hidden_layers=12  \
  --num_attention_heads=6  \
  --attention_head_size=128
#> logs/co_${input_sequence_length}.log 2>&1 &