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

if [ ! -f ./mnli_dev.tsv ]; then
    wget http://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/tutorial/meta_finetune/mnli/mnli_dev.tsv
fi

command="drop table if exists mnli_train_telephone;"
${odpscmd} --config="${config}" -e "${command}"

command="create table mnli_train_telephone(text1 STRING, text2 STRING, domain STRING, label STRING);"
${odpscmd} --config="${config}" -e "${command}"

command="insert into mnli_train_telephone select * from mnli_train where domain='telephone';"
${odpscmd} --config="${config}" -e "${command}"

command="drop table if exists mnli_eval_telephone;"
${odpscmd} --config="${config}" -e "${command}"

command="create table mnli_eval_telephone(text1 STRING, text2 STRING, domain STRING, label STRING);"
${odpscmd} --config="${config}" -e "${command}"

command="tunnel upload mnli_dev.tsv mnli_eval_telephone -fd '\t';"
${odpscmd} --config="${config}" -e "${command}"

command="insert overwrite table mnli_eval_telephone select * from mnli_eval_telephone where domain='telephone';"
${odpscmd} --config="${config}" -e "${command}"

rm -f mnli_dev.tsv

cur_path=/Users/wangchengyu/AliDocuments/easytransfer-internal
cd ${cur_path}
rm -f metaft.tar.gz
tar -zcf metaft.tar.gz scripts/meta_finetune/

job_path='file://'${cur_path}'/metaft.tar.gz'

command="
pai -name easytransfer_opensource
-project algo_platform_dev
-Dmode=train_and_evaluate_on_the_fly
-Dconfig=scripts/meta_finetune/config/finetune_pair.json
-Dtables='odps://sre_mpi_algo_dev/tables/mnli_train_telephone,odps://sre_mpi_algo_dev/tables/mnli_eval_telephone'
-Dscript=${job_path}
-DentryFile='scripts/meta_finetune/finetune.py'
-Dbuckets=\"oss://pai-wcy/?role_arn=xxx&host=cn-zhangjiakou.oss-internal.aliyun-inc.com\"
-DworkerGPU=1
-DworkerCount=1;
"

echo "${command}"
${odpscmd} --config="${config}" -e "${command}"
echo "finish..."

