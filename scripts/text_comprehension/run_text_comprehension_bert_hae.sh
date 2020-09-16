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
# Easy Transfer AppZoo: Text Comprehension (BERT-HAE) Example
# We only contain a case of basic usage, please refer to the documentation for more details.

export mode=$1 # choices from ["preprocess", "predict"]

if [ ! -f ./train.csv ]; then
  wget http://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/tutorial/ez_text_comprehension/quac/train.csv
fi

if [ ! -f ./dev.csv ]; then
  wget http://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/tutorial/ez_text_comprehension/quac/dev.csv
fi

if [ $mode == "preprocess" ]
then
    # ** Preprocess mode **
    #
    # Arguments:
    #   --mode=preprocess
    #   --inputTable: input source, only the ${test_file},
    #               this file do not need to have the same input schema as ${train_file}
    #   --inputSchema: input columns schema, format is column_name:column_type:field_length
    #   --outputTable: output table path
    #   --firstSequence: paragraph text column name
    #   --secondSequence: question text column name, format is q_text1||q_id1||q_text2||q_id2
    #   --labelName: answer text column name, format is a_text1||a_char_start1||a_text2||a_char_start2
    #   --appendCols: which columns of input table you want to append to the output
    #   --outputSchema: you do need to change this
    #   --modelZooBasePath: where the model zoo is

    easy_transfer_app \
      --mode=preprocess \
      --inputTable=train.csv \
      --inputSchema=context:str:1,question:str:1,answer:str:1 \
      --outputTable=train.serialized.csv \
      --firstSequence=context \
      --secondSequence=question \
      --labelName=answer \
      --sequenceLength=384 \
      --modelName=text_comprehension_bert_hae \
      --outputSchema=input_ids,input_mask,segment_ids,history_answer_marker,start_position,end_position \
      --batchSize=32 \
      --workerCount=16 \
      --advancedParameters='
        tokenizer_name_or_path=google-bert-base-en
        max_query_length=64
        doc_stride=128
      '

    easy_transfer_app \
      --mode=preprocess \
      --inputTable=dev.csv \
      --inputSchema=context:str:1,question:str:1,answer:str:1 \
      --outputTable=dev.serialized.csv \
      --firstSequence=context \
      --secondSequence=question \
      --labelName=answer \
      --sequenceLength=384 \
      --modelName=text_comprehension_bert_hae \
      --outputSchema=input_ids,input_mask,segment_ids,history_answer_marker,start_position,end_position \
      --batchSize=32 \
      --workerCount=16 \
      --advancedParameters='
        tokenizer_name_or_path=google-bert-base-en
        max_query_length=64
        doc_stride=128
      '

elif [ $mode == "train" ]
then
    # ** Training mode **
    #
    # Arguments:
    #   --mode=train
    #   --inputTable: input source, format is ${train_file},${dev_file}
    #   --inputSchema: input columns schema, format is column_name:column_type:field_length
    #   --sequenceLength: length of input_ids, truncate/padding
    #   --checkpointDir: where the model is saved
    #   --modelName: the type of app model
    #   --modelZooBasePath: where the model zoo is
    #   --advancedParameters: more parameters for text_comprehension_bert model, please refer to the documentation

    easy_transfer_app \
      --mode=train \
      --inputSchema=input_ids:int:384,input_mask:int:384,segment_ids:int:384,history_answer_marker:int:384,start_position:int:1,end_position:int:1 \
      --inputTable=./train.serialized.csv,./dev.serialized.csv \
      --sequenceLength=384 \
      --checkpointDir=./text_comprehension_bert_hae_models \
      --batchSize=32 \
      --numEpochs=1 \
      --optimizerType=adam \
      --learningRate=3e-5 \
      --modelName=text_comprehension_bert_hae \
      --advancedParameters='
          pretrain_model_name_or_path=google-bert-base-en
          max_query_length=64
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
    #   --firstSequence: paragraph text column name
    #   --secondSequence: question text column name, format is q_text1||q_id1||q_text2||q_id2
    #   --appendCols: which columns of input table you want to append to the output
    #   --outputSchema: choice of prediction results. The final schema of output file is
    #                 outputSchema + appendCols
    #   --checkpointPath: the directory of the saved model,
    #   --modelZooBasePath: where the model zoo is

    easy_transfer_app \
      --mode=predict \
      --inputTable=./dev.csv \
      --inputSchema=context:str:1,question:str:1,answer:str:1 \
      --outputTable=./dev.pred.csv \
      --firstSequence=context \
      --secondSequence=question \
      --labelName=answer \
      --sequenceLength=384 \
      --appendCols=qas_id \
      --outputSchema=predictions \
      --checkpointPath=./text_comprehension_bert_hae_models/ \
      --batchSize=12 \
      --advancedParameters='
          max_answer_length=40
        '
else
    echo "invalid mode"
fi