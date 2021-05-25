export CUDA_VISIBLE_DEVICES=$1
export GENRE=$2

# Download TinyBERT https://drive.google.com/uc?export=download&id=1dDigD7QBv1BmE6pWU71pFYPgovvEqOOj to ./
mkdir ./students/
mkdir ./students/${GENRE}/

python meta_distill.py --teacher_model ./meta_teacher_model \
                       --student_model ./General_TinyBERT_4L_312D/ \
                       --data_dir ./data/SENTI \
                       --task_name SENTI \
                       --output_dir ./students/${GENRE}/tmp \
                       --max_seq_length 128 \
                       --train_batch_size 32 \
                       --learning_rate 5e-5 \
                       --num_train_epochs 10 \
                       --do_lower_case \
                       --domain ${GENRE} \
                       --domain_loss_weight 0.5

python task_distill.py --pred_distill  \
                       --teacher_model ./meta_teacher_model \
                       --student_model ./students/${GENRE}/tmp \
                       --data_dir ./data/SENTI \
                       --task_name SENTI \
                       --output_dir ./students/${GENRE} \
                       --do_lower_case \
                       --learning_rate 3e-5  \
                       --num_train_epochs  3  \
                       --eval_step 50 \
                       --max_seq_length 128 \
                       --train_batch_size 32 \
                       --genre ${GENRE}

python task_distill.py --do_eval \
                       --student_model ./meta_teacher_model \
                       --data_dir ./data/SENTI \
                       --task_name SENTI \
                       --output_dir ./meta_teacher_model/tmp \
                       --genre ${GENRE} \
                       --do_lower_case \
                       --eval_batch_size 128 \
                       --max_seq_length 128