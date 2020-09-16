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
# Easy Transfer AppZoo: Text Classification (CNN) Example
# We only contain a case of basic usage, please refer to the documentation for more details.

if [ ! -f ./train.csv ]; then
  wget http://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/tutorial/ez_text_classify/mr_sample/train.csv
fi

if [ ! -f ./dev.csv ]; then
  wget http://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/tutorial/ez_text_classify/mr_sample/dev.csv
fi

if [ ! -f ./glove.840B.300d.mr.txt ]; then
  wget http://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/tutorial/ez_text_classify/mr_sample/glove.840B.300d.mr.txt
fi

# ** Training **
# We only show the example of training, evaluate/predict is similar as text_classify_bert
#
# Arguments:
#   --mode=train
#   --inputTable: input source, format is ${train_file},${dev_file}
#   --inputSchema: input columns schema, format is column_name:column_type:field_length
#   --firstSequence: to be classified column name
#   --labelName: ground truth label column name
#   --labelEnumerateValues: unique values of ground truth labels
#   --sequenceLength: length of input_ids, truncate/padding
#   --checkpointDir: where the model is saved
#   --modelName: the type of app model
#   --advancedParameters: more parameters for cnn model, please refer to the documentation
easy_transfer_app \
  --mode=train \
  --inputTable=./train.csv,./dev.csv \
  --inputSchema=example_id:int:1,content:str:1,label:str:1 \
  --firstSequence=content \
  --labelName=label \
  --labelEnumerateValues=0,1 \
  --sequenceLength=64 \
  --checkpointDir=./cnn_classify_models \
  --batchSize=50 \
  --numEpochs=1 \
  --optimizerType=adadelta \
  --learningRate=0.95 \
  --modelName=text_classify_cnn \
  --advancedParameters=' \
        pretrain_word_embedding_name_or_path=./glove.840B.300d.mr.txt
        fix_embedding=false
        max_vocab_size=20000
        num_filters=100,100,100
        filter_sizes=3,4,5
        dropout_rate=0.5

        lr_decay=none
        warmup_ratio=0
        weight_decay_ratio=0
        gradient_clip=true
        clip_norm_value=3.0
        throttle_secs=10
  '