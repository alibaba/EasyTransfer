AdaBERT PyTorch Version

Config Example

python search.py --gpus=0 --model_type bert --model_name_or_path /home/daoyuanchen.cdy/work/bert_with_NAS/MRPC/output/checkpoint-450 --config_name /home/daoyuanchen.cdy/work/bert_with_NAS/useful_codes/PyTransformer/examples/pre_trained_bert_models/bert_config.json --tokenizer_name /home/daoyuanchen.cdy/work/bert_with_NAS/useful_codes/PyTransformer/examples/pre_trained_bert_models/bert-base-uncased-vocab.txt --task_name MRPC --do_lower_case --data_dir /home/daoyuanchen.cdy/work/bert_with_NAS/GLUE/MRPC --max_seq_length 128 --batch_size 64 --epochs 70 --output_dir /home/daoyuanchen.cdy/work/bert_with_NAS/MRPC/output --init_channels 1 --layers 8 --workers 1 --teacher_probe_train_epochs 5 --data_aug False --fp16 --fp16_opt_level O2
