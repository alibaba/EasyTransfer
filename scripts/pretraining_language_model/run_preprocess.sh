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

wget https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/tutorial/pretrain_lauguage_model/train_dir.tgz
wget https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/tutorial/pretrain_lauguage_model/dev_dir.tgz
wget https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/tutorial/pretrain_lauguage_model/vocab.txt

tar -zxf train_dir.tgz
tar -zxf dev_dir.tgz
rm train_dir.tgz dev_dir.tgz

python pretrain_main.py  \
  --mode=preprocess \
  --num_threads=2 \
  --input_dir=train_dir \
  --output_dir=train_dir_tfrecords \
  --tokenizer=wordpiece  \
  --vocab_fp=vocab.txt \
  --data_format=segment_pair \
  --dupe_factor=1 \
  --max_seq_length=128 \
  --do_chinese_whole_word_mask=True

python pretrain_main.py  \
  --mode=preprocess \
  --num_threads=2 \
  --input_dir=dev_dir \
  --output_dir=dev_dir_tfrecords \
  --tokenizer=wordpiece  \
  --vocab_fp=vocab.txt \
  --data_format=segment_pair \
  --dupe_factor=1 \
  --max_seq_length=128 \
  --do_chinese_whole_word_mask=True

ls -d $PWD/train_dir_tfrecords/* > train_list.list_tfrecord
ls -d $PWD/dev_dir_tfrecords/* > dev_list.list_tfrecord

rm -rf train_dir dev_dir