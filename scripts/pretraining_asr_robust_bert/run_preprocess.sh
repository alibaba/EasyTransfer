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

#!/usr/bin/env bash
seq_length=512
nohup python pretrain_main.py  \
  --mode=preprocess \
  --num_threads=5 \
  --input_dir=processed_txt \
  --output_dir=processed_txt_tfr \
  --tokenizer=wordpiece  \
  --vocab_fp=vocab.txt \
  --data_format=full_sentences \
  --dupe_factor=10 \
  --max_seq_length=${seq_length} >logs/asr_process.log 2>&1 &