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
set -e

odpscmd="/Users/wangchengyu/AliDocuments/odps/odps_console/bin/odpscmd"
config="/Users/wangchengyu/AliDocuments/odps/odps_console/conf/odps_config.ini"

# create odps table and training data
if [ ! -f ./amazon_train.tsv ]; then
    wget http://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/tutorial/meta_finetune/amazon/amazon_train.tsv
fi

command="drop table if exists amazon_train;"
${odpscmd} --config="${config}" -e "${command}"

command="create table amazon_train(text STRING, domain STRING, label STRING);"
${odpscmd} --config="${config}" -e "${command}"

command="tunnel upload amazon_train.tsv amazon_train -fd '\t';"
${odpscmd} --config="${config}" -e "${command}"

command="create table if not exists amazon_train_weights(text STRING, domain STRING, label STRING, score DOUBLE);"
${odpscmd} --config="${config}" -e "${command}"

rm -f ./amazon_train.tsv

cur_path=/Users/wangchengyu/AliDocuments/easytransfer-internal
cd ${cur_path}
rm -f metaft.tar.gz
tar -zcf metaft.tar.gz scripts/meta_finetune/

job_path='file://'${cur_path}'/metaft.tar.gz'

command="
pai -name easytransfer
-project algo_platform_dev
-Dmode=predict_on_the_fly
-Dconfig=scripts/meta_finetune/config/preprocess_single.json
-Dtables='odps://sre_mpi_algo_dev/tables/amazon_train'
-Doutputs='odps://sre_mpi_algo_dev/tables/amazon_train_weights'
-Dscript=${job_path}
-DentryFile='scripts/meta_finetune/preprocess.py'
-Dbuckets=\"oss://pai-wcy/?role_arn=xxx&host=cn-zhangjiakou.oss-internal.aliyun-inc.com\"
-DuserDefinedParameters='
--do_sent_pair=False
--domains=books,dvd,electronics,kitchen
--classes=POS,NEG
'
-DworkerGPU=1
-DworkerCount=1;
"

echo "${command}"
${odpscmd} --config="${config}" -e "${command}"

command="drop table if exists amazon_eval_weights;"
${odpscmd} --config="${config}" -e "${command}"

command="create table amazon_eval_weights like amazon_train_weights;"
${odpscmd} --config="${config}" -e "${command}"

command="insert into amazon_eval_weights select * from amazon_train_weights;"
${odpscmd} --config="${config}" -e "${command}"

echo "finish..."