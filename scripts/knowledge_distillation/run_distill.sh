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
# Easy Transfer KD: Vanilla/Probes KD Example
# We only contain a case of basic usage, please refer to the documentation for more details.

export model_type=$1 # choices from ["vanilla", "probes"]
export mode=$2       # choices from ["teacher_train", "teacher_predict", "student_train"]

if [ ! -f ./sst2_train.tsv ]; then
    wget http://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/tutorial/knowledge_distillation/SST-2/sst2_train.tsv
fi

if [ ! -f ./sst2_dev.tsv ]; then
    wget http://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/tutorial/knowledge_distillation/SST-2/sst2_dev.tsv
fi

if [ ! -f ./google-bert-small-en/config.json ]; then
   wget http://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/tutorial/knowledge_distillation/SST-2/google-bert-small-en.zip
   unzip google-bert-small-en.zip
   rm -f google-bert-small-en.zip
fi

if [ $mode == "teacher_train" ]
then
    # ** Training teacher **
    python main_teacher.py \
      --mode=train_and_evaluate_on_the_fly \
      --config=./config/${model_type}_teacher_config.json

elif [ $mode == "teacher_predict" ]
then
    # ** Predict teacher logits **
    # Predict dev set logits
    python main_teacher.py \
      --mode=predict_on_the_fly \
      --config=./config/${model_type}_teacher_config.json

    # Predict train set logits
    sed "s/sst2_dev/sst2_train/g" ./config/${model_type}_teacher_config.json > ./config/tmp_config.json
    python main_teacher.py \
      --mode=predict_on_the_fly \
      --config=./config/tmp_config.json

elif [ $mode == "student_train" ]
then
    # ** Training student **
    python main_student.py \
    --mode=train_and_evaluate \
    --config=./config/${model_type}_student_config.json

elif [ $mode == "student_predict" ]
then
    # ** Predict student **
    python main_student.py \
    --mode=predict \
    --config=./config/${model_type}_student_config.json

else
  echo "invalid mode"
fi