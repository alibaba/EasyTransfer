#!/usr/bin/env bash

wget https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/tutorial/clue_glue_superglue_benchmark/clue_datasets.tgz
wget https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/tutorial/clue_glue_superglue_benchmark/glue_datasets.tgz
wget https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/tutorial/clue_glue_superglue_benchmark/superglue_datasets.tgz
tar -zxf clue_datasets.tgz
tar -zxf glue_datasets.tgz
tar -zxf superglue_datasets.tgz
mkdir datasets
mv clue_datasets/* datasets
mv glue_datasets/* datasets
mv superglue_datasets/* datasets
rm -rf *_datasets
rm *.tgz

export CUDA_VISIBLE_DEVICES="0"
#CLUE---> AFQMC, CMNLI, CSL, IFLYTEK, TNEWS
#GLUE---> CoLA, MRPC, QQP, RTE, SST-2
#SuperGLUE---> BoolQ, CB, COPA, WiC, WSC
task_name=CoLA

python main_finetune.py --workerGPU=1 \
  --task_name=${task_name}  \
  --train_input_fp=datasets/${task_name}/train.csv  \
  --eval_input_fp=datasets/${task_name}/dev.csv  \
  --pretrain_model_name_or_path=google-bert-base-en  \
  --train_batch_size=16  \
  --num_epochs=2.5  \
  --model_dir=${task_name}_model_dir  \
  --learning_rate=1e-5  \


