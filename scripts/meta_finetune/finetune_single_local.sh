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

if [ ! -f ./amazon_dev.tsv ]; then
    wget http://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/tutorial/meta_finetune/amazon/amazon_dev.tsv
fi

domain_name="books"
domain_column_index=1
config="./config/finetune_single.json"

echo "create fine-tune datasets..."
python create_finetune_datasets_local.py \
     --input_file="amazon_train.tsv" \
     --output_file="amazon_train_books.tsv" \
     --domain_name=${domain_name} \
     --domain_column_index=${domain_column_index}

python create_finetune_datasets_local.py \
     --input_file="amazon_dev.tsv" \
     --output_file="amazon_dev_books.tsv" \
     --domain_name=${domain_name} \
     --domain_column_index=${domain_column_index}

echo "fine-tune..."
python finetune.py \
     --config=${config} \
     --mode="train_and_evaluate_on_the_fly"


