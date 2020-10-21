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

if [ ! -f ./amazon_train.tsv ]; then
    wget http://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/tutorial/meta_finetune/amazon/amazon_train.tsv
fi

if [ ! -f ./google-bert-base-en/config.json ]; then
   wget https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/eztransfer_modelzoo/bert/google-bert-base-en.tgz
   tar zxvf google-bert-base-en.tgz -C ./
   rm -f google-bert-base-en.tgz
fi

outputs="amazon_train_weights.tsv"
domains="books,dvd,electronics,kitchen"
classes="POS,NEG"
do_sent_pair=False
config="./config/preprocess_single.json"

echo "preprocess..."
python preprocess.py \
     --outputs=${outputs} \
     --do_sent_pair=${do_sent_pair} \
     --domains=${domains} \
     --classes=${classes} \
     --config=${config} \
     --mode="predict_on_the_fly"

