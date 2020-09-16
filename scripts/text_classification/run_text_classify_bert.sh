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
# Easy Transfer AppZoo: Text Classification (BERT) Example
# We only contain a case of basic usage, please refer to the documentation for more details.

export mode=$1 # choices from ["train", "evaluate", "predict"]

if [ ! -f ./train.csv ]; then
  wget http://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/tutorial/ez_text_classify/tnews_sample/train.csv
fi

if [ ! -f ./dev.csv ]; then
  wget http://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/tutorial/ez_text_classify/tnews_sample/dev.csv
fi

if [ $mode == "train" ]
then
    # ** Training mode **
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
    #   --modelZooBasePath: where the model zoo is
    #   --advancedParameters: more parameters for text_classify_bert model, please refer to the documentation

    easy_transfer_app \
      --mode=train \
      --inputTable=./train.csv,./dev.csv \
      --inputSchema=example_id:int:1,content:str:1,label:str:1,label_str:str:1 \
      --firstSequence=content \
      --labelName=label \
      --labelEnumerateValues=100,101,102,103,104,105,106,107,108,109,110,112,113,114,115,116 \
      --sequenceLength=64 \
      --checkpointDir=./bert_classify_models \
      --batchSize=32 \
      --numEpochs=1 \
      --optimizerType=adam \
      --learningRate=2e-5 \
      --modelName=text_classify_bert \
      --advancedParameters='
        pretrain_model_name_or_path=google-bert-base-zh
        '

elif [ $mode == "evaluate" ]
then
    # ** Evaluation mode **
    #
    # Arguments:
    #   --mode=evaluate
    #   --inputTable: input source, only the ${dev_file},
    #               this file should have the same input schema as ${train_file}
    #   --checkpointPath: which checkpoint you want to evaluate
    #   --modelZooBasePath: where the model zoo is

    easy_transfer_app \
      --mode=evaluate \
      --inputTable=./dev.csv \
      --checkpointPath=./bert_classify_models/model.ckpt-32 \
      --batchSize=10

elif [ $mode == "predict" ]
then
    # ** Predict mode **
    #
    # Arguments:
    #   --mode=predict
    #   --inputTable: input source, only the ${test_file},
    #               this file do not need to have the same input schema as ${train_file}
    #   --inputSchema: input columns schema, format is column_name:column_type:field_length
    #   --outputTable: output table path
    #   --firstSequence: to be classified column name
    #   --appendCols: which columns of input table you want to append to the output
    #   --outputSchema: choice of prediction results. The final schema of output file is
    #                 outputSchema + appendCols
    #   --checkpointPath: the directory of the saved model,
    #   --modelZooBasePath: where the model zoo is
    easy_transfer_app \
      --mode=predict \
      --inputTable=dev.csv \
      --inputSchema=example_id:int:1,content:str:1,label:str:1,label_str:str:1 \
      --outputTable=dev.pred.csv \
      --firstSequence=content \
      --appendCols=example_id \
      --outputSchema=predictions,probabilities,logits \
      --checkpointPath=./bert_classify_models/ \
      --batchSize=100

else
  echo "invalid mode"
fi