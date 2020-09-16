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
# Easy Transfer AppZoo: Text Match (HCNN) Example
# We only contain a case of basic usage, please refer to the documentation for more details.

if [ ! -f ./train_tokenized.csv ]; then
  wget http://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/tutorial/ez_text_match/afqmc_public/train_tokenized.csv
fi

if [ ! -f ./dev_tokenized.csv ]; then
  wget http://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/tutorial/ez_text_match/afqmc_public/dev_tokenized.csv
fi

if [ ! -f ./cc.zh.300.vec.afqmc.txt ]; then
  wget http://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/tutorial/ez_text_match/afqmc_public/cc.zh.300.vec.afqmc.txt
fi

# ** Training mode **
# We only show the example of training, evaluate/predict is similar as text_classify_bert
#
# Arguments:
#   --mode=train
#   --inputTable: input source, format is ${train_file},${dev_file}
#   --inputSchema: input columns schema, format is column_name:column_type:field_length
#   --firstSequence: column name of the first sequence to match
#   --secondSequence: column name of the second sequence to match
#   --labelName: ground truth label column name
#   --labelEnumerateValues: unique values of ground truth labels
#   --sequenceLength: length of input_ids, truncate/padding
#   --checkpointDir: where the model is saved
#   --modelName: the type of app model
#   --advancedParameters: more parameters for text_match_bert model, please refer to the documentation
easy_transfer_app \
  --mode=train \
  --inputTable=./train_tokenized.csv,./dev_tokenized.csv \
  --inputSchema=query1_words:str:1,query2_words:str:1,label:str:1 \
  --firstSequence=query1_words \
  --secondSequence=query2_words \
  --labelName=label \
  --labelEnumerateValues=0,1 \
  --checkpointDir=./hcnn_match_models \
  --batchSize=32 \
  --numEpochs=1 \
  --optimizerType=adagrad \
  --learningRate=0.025 \
  --modelName=text_match_hcnn \
  --advancedParameters='
        first_sequence_length=64
        second_sequence_length=64
        pretrain_word_embedding_name_or_path=./cc.zh.300.vec.afqmc.txt
        fix_embedding=true
        max_vocab_size=30000
        embedding_size=300
        hidden_size=300
    '