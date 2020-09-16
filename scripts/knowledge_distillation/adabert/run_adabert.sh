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
#
# Easy Transfer KD: AdaBERT KD Example
# We only contain a case of basic usage, please refer to the documentation for more details.

export MODE=$1 # choices in ["search", "finetune"]

if [ ! -f ./mrpc ]; then
    wget http://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/tutorial/knowledge_distillation/adabert/mrpc.zip
    unzip mrpc.zip
    rm -f mrpc.zip
fi


if [ $MODE == "search" ]
then
  # ** Search mode (Search for the best arch) **
  #
  # Arguments:
  #   --model_dir: Directory for saving the searched model.
  #   --train_file: input source
  #   --num_core_per_host: the number of GPUs used.
  #   --train_batch_size: batch size for training
  #   --temp_decay_steps: Number of steps for gumbel annealing temperature.
  #   --train_steps: Number of training steps
  #   --save_steps: If None, not to save any model.
  #   --is_pair_task: single sentence or paired sentences.
  #   --model_opt_lr: learning rate for updating model parameters
  #   --emb_pathes: given embeddings
  #   --distribution_strategy: distribution strategy
  #   --max_save: Max number of checkpoints to save.

  python main_adabert.py \
    --model_dir ./adabert_models/search \
    --train_file ./mrpc/train_mrpc_output_logits.txt,./mrpc/dev_mrpc_output_logits.txt \
    --num_core_per_host 1 \
    --train_batch_size 128 \
    --temp_decay_steps 1568 \
    --train_steps 1960 \
    --save_steps 28 \
    --is_pair_task 1 \
    --model_opt_lr 5e-5 \
    --emb_pathes ./mrpc/word_embedding.npy,./mrpc/position_embedding.npy \
    --distribution_strategy None \
    --max_save 100

elif [ $MODE == "finetune" ]
then
  # ** Fine-tuning mode (fine-tuning the given arch) **
  #
  # Arguments:
  #   --model_dir: Directory for saving the fine-tuned model.
  #   --train_file: input source
  #   --num_core_per_host: the number of GPUs used.
  #   --train_batch_size: batch size for training
  #   --train_steps: Number of training steps
  #   --save_steps: If None, not to save any model.
  #   --is_pair_task: single sentence or paired sentences.
  #   --model_opt_lr: learning rate for updating model parameters
  #   --searched_model: checkpoints directories for searched model
  #   --emb_pathes: given embeddings by searched models
  #   --distribution_strategy: distribution strategy
  #   --max_save: Max number of checkpoints to save.

  rm -rf ./adabert_models/finetune/
  python main_adabert.py \
    --model_dir ./adabert_models/finetune/ \
    --train_file ./mrpc/train_mrpc_output_logits.txt,./mrpc/dev_mrpc_output_logits.txt \
    --num_core_per_host 1 \
    --train_batch_size 32 \
    --train_steps=30 \
    --save_steps=30 \
    --is_pair_task 1 \
    --model_opt_lr 5e-6 \
    --searched_model=./adabert_models/search/best \
    --emb_pathes=./adabert_models/search/best/wemb.npy,./adabert_models/search/best/pemb.npy \
    --arch_path=./adabert_models/search/best/arch.json \
    --distribution_strategy None \
    --max_save 1

  python main_adabert.py \
    --model_dir ./adabert_models/finetune/ \
    --train_file ./mrpc/dev_mrpc_output_logits.txt \
    --outputs ./dev.preds.tsv \
    --arch_path=./adabert_models/search/best/arch.json \
    --is_training=false

else

  echo "invalid mode ${MODE}"

fi
