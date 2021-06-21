#!/usr/bin/env bash
wget https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/tutorial/clue_glue_superglue_benchmark/clue_datasets.tgz
#wget https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/tutorial/clue_glue_superglue_benchmark/glue_datasets.tgz
#wget https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/tutorial/clue_glue_superglue_benchmark/superglue_datasets.tgz
tar -zxf clue_datasets.tgz
#tar -zxf glue_datasets.tgz
#tar -zxf superglue_datasets.tgz
mkdir datasets
mv clue_datasets/* datasets
#mv glue_datasets/* datasets
#mv superglue_datasets/* datasets
rm -rf *_datasets
rm *.tgz
task_name=CLUEWSC

# change the main_finetune.py path according to your sever folder.
# Train mode
python /root/data/easytransfer/CLUE_GLUE_SuperGLUE_benchmark/main_finetune.py --workerGPU=1 \
  --mode=train_and_evaluate_on_the_fly  \
  --task_name=${task_name}  \
  --train_input_fp=/root/data/easytransfer/CLUE_GLUE_SuperGLUE_benchmark/datasets/${task_name}/train.csv  \
  --eval_input_fp=/root/data/easytransfer/CLUE_GLUE_SuperGLUE_benchmark/datasets/${task_name}/dev.csv  \
  --pretrain_model_name_or_path=pai-bert-tiny-zh  \
  --train_batch_size=16  \
  --num_epochs=1  \
  --model_dir=${task_name}_model_dir_2  \
  --learning_rate=3e-5

# Train and Predict mode
python /root/data/easytransfer/CLUE_GLUE_SuperGLUE_benchmark/main_finetune.py --workerGPU=1 \
  --mode=predict_on_the_fly  \
  --task_name=${task_name}  \
  --train_input_fp=/root/data/easytransfer/CLUE_GLUE_SuperGLUE_benchmark/datasets/${task_name}/train.csv  \
  --eval_input_fp=/root/data/easytransfer/CLUE_GLUE_SuperGLUE_benchmark/datasets/${task_name}/dev.csv  \
  --predict_input_fp=/root/data/easytransfer/CLUE_GLUE_SuperGLUE_benchmark/datasets/${task_name}/test.csv  \
  --predict_checkpoint_path=${task_name}_model_dir/model.ckpt-0 \
  --pretrain_model_name_or_path=hit-roberta-large-zh  \
  --train_batch_size=16  \
  --num_epochs=10  \
  --model_dir=${task_name}_model_dir  \
  --learning_rate=3e-5
