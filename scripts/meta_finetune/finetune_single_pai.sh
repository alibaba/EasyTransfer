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

if [ ! -f ./amazon_dev.tsv ]; then
    wget http://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/tutorial/meta_finetune/amazon/amazon_dev.tsv
fi

command="drop table if exists amazon_train_books;"
${odpscmd} --config="${config}" -e "${command}"

command="create table amazon_train_books(text STRING, domain STRING, label STRING);"
${odpscmd} --config="${config}" -e "${command}"

command="insert into amazon_train_books select * from amazon_train where domain='books';"
${odpscmd} --config="${config}" -e "${command}"

command="drop table if exists amazon_eval_books;"
${odpscmd} --config="${config}" -e "${command}"

command="create table amazon_eval_books(text STRING, domain STRING, label STRING);"
${odpscmd} --config="${config}" -e "${command}"

command="tunnel upload amazon_dev.tsv amazon_eval_books -fd '\t';"
${odpscmd} --config="${config}" -e "${command}"

command="insert overwrite table amazon_eval_books select * from amazon_eval_books where domain='books';"
${odpscmd} --config="${config}" -e "${command}"

rm -f amazon_dev.tsv

cur_path=/Users/wangchengyu/AliDocuments/easytransfer-internal
cd ${cur_path}
rm -f metaft.tar.gz
tar -zcf metaft.tar.gz scripts/meta_finetune/

job_path='file://'${cur_path}'/metaft.tar.gz'

command="
pai -name easytransfer
-project algo_platform_dev
-Dmode=train_and_evaluate_on_the_fly
-Dconfig=scripts/meta_finetune/config/finetune_single.json
-Dtables='odps://sre_mpi_algo_dev/tables/amazon_train_books,odps://sre_mpi_algo_dev/tables/amazon_eval_books'
-Dscript=${job_path}
-DentryFile='scripts/meta_finetune/finetune.py'
-Dbuckets=\"oss://pai-wcy/?role_arn=xxx&host=cn-zhangjiakou.oss-internal.aliyun-inc.com\"
-DworkerGPU=1
-DworkerCount=1;
"

echo "${command}"
${odpscmd} --config="${config}" -e "${command}"
echo "finish..."

