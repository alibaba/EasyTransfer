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

set -e
export CUDA_VISIBLE_DEVICES=6
DIR="output"
if [ -d "$DIR" ]; then
  rm -rf $DIR
fi
mkdir $DIR
echo "================== test_train_and_eval_on_the_fly_classification =================="
python test_train_and_eval_on_the_fly_classification.py
echo "================== test_predict_on_the_fly_classification =================="
python test_predict_on_the_fly_classification.py
rm -rf cache_train_and_eval_on_the_fly_classification
echo "================== test_train_and_eval_on_the_fly_regression =================="
python test_train_and_eval_on_the_fly_regression.py
rm -rf cache_train_and_eval_on_the_fly_regression

echo "================== test_preprocess_for_classification =================="
python test_preprocess_for_classification.py
echo "================== test_preprocess_for_predict_classification =================="
python test_preprocess_for_predict_classification.py
echo "================== test_distributed_preprocess_for_classification =================="
python test_distributed_preprocess_for_classification.py
rm $DIR/preprocess_output_for_finetune.csv
rm $DIR/preprocess_output_for_predict.csv

echo "================== test_preprocess_for_pretrain =================="
python test_preprocess_for_pretrain.py
echo "================== test_distributed_preprocess_for_pretrain =================="
python test_distributed_preprocess_for_pretrain.py
rm $DIR/preprocess_output_for_pretrain.csv

echo "================== test_export =================="
python test_export.py
echo "================== test_distributed_feat_ext =================="
python test_distributed_feat_ext.py
rm $DIR/test_dist_pred_out.csv

echo "================== test_train_classification =================="
python test_train_classification.py
echo "================== test_eval_classification =================="
python test_eval_classification.py
rm -rf cache_train
echo "================== test_train_and_evaluate_classification =================="
python test_train_and_evaluate_classification.py
echo "================== test_predict_classification =================="
python test_predict_classification.py
rm -rf cache_train_eval
rm $DIR/test_train_eval_predict_dev_features_pred_out.csv
rm -rf $DIR








