#!/usr/bin/env bash
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

echo "================== Test ez_bert_feat =================="
if [ ! -f ./google-bert-base-zh.tgz ]; then
  wget http://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/eztransfer_modelzoo/bert/google-bert-base-zh.tgz
  tar -zxvf google-bert-base-zh.tgz
fi
python test_ez_bert_feat.py
rm -f google-bert-base-zh.tgz
rm -rf ./google-bert-base-zh

echo "================== Test ez_serialization =================="
python test_ez_serialization.py

echo "================== Test ez_conversion =================="
if [ ! -f ./pai-bert-base-zh.tgz ]; then
  wget http://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/eztransfer_modelzoo/bert/pai-bert-base-zh.tgz
  tar -zxvf pai-bert-base-zh.tgz
fi
echo "model_checkpoint_path: \"model.ckpt\"" > ./pai-bert-base-zh/checkpoint
python test_ez_conversion.py
rm -f pai-bert-base-zh.tgz
rm -rf ./pai-bert-base-zh

echo "================== Test ez_text_match =================="
python test_ez_text_match.py

echo "================== Test ez_text_classify =================="
python test_ez_text_classify.py

echo "================== Test ez_text_comprehension =================="
python test_ez_text_comprehension.py

echo "================== Test ez_sequence_labeling =================="
python test_ez_sequence_labeling.py

