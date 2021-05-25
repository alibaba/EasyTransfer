export CUDA_VISIBLE_DEVICES=$1

# Training
# Download `bert-base-uncased`(https://huggingface.co/bert-base-uncased) to ./
python meta_teacher.py --pretrain_model_name_or_path ./bert-base-uncased \
                       --data_dir ./data/SENTI \
                       --task_name SENTI \
                       --output_dir meta_teacher_model \
                       --max_seq_length 128 \
                       --train_batch_size 32 \
                       --learning_rate 2e-5  \
                       --num_train_epochs 3.0 \
                       --do_lower_case \
                       --eval_step 50 \
                       --domain books,dvd,electronics,kitchen \
                       --use_sample_weights 1 \
                       --domain_loss_weight 0.5

# Evaluation
python task_distill.py --do_eval \
                       --student_model ./meta_teacher_model \
                       --data_dir ./data/SENTI \
                       --task_name SENTI \
                       --output_dir ./meta_teacher_model/tmp \
                       --genre kitchen \
                       --do_lower_case \
                       --eval_batch_size 128 \
                       --max_seq_length 128