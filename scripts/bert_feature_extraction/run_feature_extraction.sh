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
# Easy Transfer AppZoo: Feature Extraction (BERT) Example
if [ ! -f ./test.csv ]; then
  wget http://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/tutorial/ez_bert_feat/test.csv
fi

# Arguments:
#   --inputTable: input source, only the ${test_file},
#               this file do not need to have the same input schema as ${train_file}
#   --inputSchema: input columns schema, format is column_name:column_type:field_length
#   --outputTable: output table path
#   --firstSequence: column name of the first sequence to extract the features
#   --appendCols: which columns of input table you want to append to the output
#   --outputSchema: choice of prediction results. The final schema of output file is
#                 outputSchema + appendCols
#   --modelName: choice of pre-train language models, can also be a path
#   --modelZooBasePath: where the model zoo is
ez_bert_feat \
  --inputTable=./test.csv \
  --inputSchema=example_id:int:1,query1:str:1,query2:str:1,category:str:1,score:float:1 \
  --outputTable=./test.feats.csv \
  --firstSequence=query1 \
  --appendCols=example_id,category,score,query2 \
  --outputSchema=pool_output \
  --modelName=google-bert-base-zh \
  --sequenceLength=32 \
  --batchSize=100