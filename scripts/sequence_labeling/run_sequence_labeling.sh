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
# Easy Transfer AppZoo: Sequence Labeling (BERT) Example
# We only contain a case of basic usage, please refer to the documentation for more details.

export mode=$1 # choices from ["train", "predict"]


if [ ! -f ./train.csv ]; then
  wget http://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/tutorial/ez_sequence_labeling/renming_news/train.csv
fi

if [ ! -f ./dev.csv ]; then
  wget http://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/tutorial/ez_sequence_labeling/renming_news/dev.csv
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
    easy_transfer_app \
      --mode=train \
      --inputSchema=content:str:1,ner_tags:str:1 \
      --inputTable=./train.csv,./dev.csv \
      --firstSequence=content \
      --labelName=ner_tags \
      --labelEnumerateValues=B-LOC,B-ORG,B-PER,I-LOC,I-ORG,I-PER,O \
      --sequenceLength=128 \
      --checkpointDir=./sequence_labeling_models \
      --batchSize=32 \
      --numEpochs=1 \
      --optimizerType=adam \
      --learningRate=2e-5 \
      --modelName=sequence_labeling_bert \
      --advancedParameters='
          pretrain_model_name_or_path=google-bert-base-zh
        '

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
    #   --firstSequence: to be labeled column name
    #   --appendCols: which columns of input table you want to append to the output
    #   --outputSchema: choice of prediction results. The final schema of output file is
    #                 outputSchema + appendCols
    #   --checkpointPath: the directory of the saved model,
    #   --modelZooBasePath: where the model zoo is
    easy_transfer_app \
      --mode=predict \
      --inputTable=./dev.csv \
      --inputSchema=content:str:1,ner_tags:str:1 \
      --outputTable=./dev.pred.csv \
      --firstSequence=content \
      --appendCols=ner_tags \
      --outputSchema=predictions \
      --checkpointPath=./sequence_labeling_models/ \
      --batchSize=32

else
  echo "invalid mode"
fi