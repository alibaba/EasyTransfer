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
# Easy Transfer AppZoo: Text Match / Vector retrieving Example
# We only contain a case of basic usage, please refer to the documentation for more details.


export mode=$1 # choices from ["train", "export", "extract_vector"]

if [ ! -f ./train.csv ]; then
  wget http://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/tutorial/ez_text_match/afqmc_public/train.csv
fi

if [ ! -f ./dev.csv ]; then
  wget http://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/tutorial/ez_text_match/afqmc_public/dev.csv
fi

if [ $mode == "train" ]
then
    # ** Training mode **
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
    #   --modelZooBasePath: where the model zoo is
    #   --advancedParameters: more parameters for text_match_bert model, please refer to the documentation
    easy_transfer_app \
      --mode=train \
      --inputSchema=example_id:int:1,query1:str:1,query2:str:1,label:str:1,category:str:1,score:float:1 \
      --inputTable=./train.csv,./dev.csv \
      --firstSequence=query1 \
      --secondSequence=query2 \
      --labelName=label \
      --labelEnumerateValues=0,1 \
      --sequenceLength=32 \
      --checkpointDir=./bert_two_tower_match_models \
      --batchSize=32 \
      --numEpochs=1 \
      --optimizerType=adam \
      --learningRate=2e-5 \
      --modelName=text_match_bert_two_tower \
      --advancedParameters=' \
          pretrain_model_name_or_path=google-bert-base-zh
        '

elif [ $mode == "export" ]
then
    # ** Evaluation mode **
    #
    # Arguments:
    #   --mode=evaluate
    #   --checkpointPath: which checkpoint you want to export
    #   --exportType=ez_bert_feat
    #   --exportDirBase: the exported saved model dir
    #   --modelZooBasePath: where the model zoo is
    easy_transfer_app \
        --mode=export \
        --checkpointPath=./bert_two_tower_match_models/model.ckpt-32 \
        --exportType=ez_bert_feat \
        --exportDirBase=./bert_two_tower_match_models/feature_model \

elif [ $mode == "extract_vector" ]
then
    # ** Extraction vector mode **
    #
    # Arguments:
    #   --inputTable: input source, only the ${test_file},
    #               this file do not need to have the same input schema as ${train_file}
    #   --inputSchema: input columns schema, format is column_name:column_type:field_length
    #   --outputTable: output table path
    #   --firstSequence: column name of the first sequence to extract the features
    #   --appendCols: which columns of input table you want to append to the output
    #   --outputSchema: choice of prediction results. The final schema of output file is
    #                 outputSchema + appendCols
    #   --modelName: here is the exported saved model dir
    ez_bert_feat \
      --inputTable=./dev.csv \
      --inputSchema=example_id:int:1,query1:str:1,query2:str:1,label:str:1,category:str:1,score:float:1 \
      --outputTable=./dev.feats.csv \
      --firstSequence=query1 \
      --appendCols=example_id \
      --outputSchema=pool_output \
      --modelName=./bert_two_tower_match_models/feature_model \
      --sequenceLength=32 \
      --batchSize=32
fi