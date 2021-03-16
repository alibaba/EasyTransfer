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

if [ ! -f ./mnli_train.tsv ]; then
    wget http://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/tutorial/meta_finetune/mnli/mnli_train.tsv
fi

outputs="mnli_train_weights.tsv"
domains="telephone,government,slate,travel,fiction"
classes="entailment,neutral,contradiction"
do_sent_pair=True
config="./config/preprocess_pair.json"

echo "preprocess..."
python preprocess.py \
     --outputs=${outputs} \
     --do_sent_pair=${do_sent_pair} \
     --domains=${domains} \
     --classes=${classes} \
     --config=${config} \
     --mode="predict_on_the_fly"

