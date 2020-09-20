CLUE
# EasyTransfer基准评测

# 评测方法
采用统一的finetune子网络结构，在超参组合空间中启动训练，搜索最佳超参组合。每个训练任务采用early stopping，选择出验证集上最好的accuracy的ckpt然后终止。


seq_len=128

warmup_ratio = 0.1

weight_decay_ratio=0.01

dropout=0.1

batch_size=[8, 16, 32]

num_epochs=5

learning_rate=[1e-5，3e-5，5e-5]


## CLUE
|  | TNEWS | AFQMC | IFLYTEK | CMNLI | CSL | **Avg** |
| --- | --- | --- | --- | --- | --- | --- |
| google-bert-base-zh | 0.6673 | 0.7375 | 0.5968 | 0.7981 | 0.7976 | 0.7194 |
| pai-bert-base-zh | 0.6694 | 0.7412 | 0.6114 | 0.7967 | 0.7993 | 0.7236 |
| hit-roberta-base-zh | 0.6734 | 0.7418 | 0.6052 | 0.8010 | 0.8010 | 0.7245 |
| hit-roberta-large-zh | 0.6742 | 0.7521 | 0.6052 | 0.8231 | 0.8100 | 0.7329 |
| google-albert-xxlarge-zh | 0.6253 | 0.6899 | 0.5017 | 0.7721 | 0.7106 | 0.6599 |
| pai-albert-xxlarge-zh | 0.6809 | 0.7525 | 0.6118 | 0.8284 | 0.8137 | 0.7375 |

## GLUE
| 预训练模型 | QQP | SST-2 | CoLA | MRPC | RTE | Avg |
| --- | --- | --- | --- | --- | --- | --- |
| google-bert-base-en | 0.9086 | 0.9243 | 0.6103 | 0.8848 | 0.722 | 0.81 |
| google-bert-large-en | 0.9154 | 0.9346 | 0.8293 | 0.8627 | 0.7284 | 0.8541 |
| google-albert-large-en | 0.8957 | 0.9404 | 0.7967 | 0.8824 | 0.7076 | 0.8446 |
| google-albert-xxlarge-en | 0.9082 | 0.9655 | 0.8313 | 0.8625 | 0.7159 | 0.8567 |
| pai-albert-xxlarge-en | 0.9136 | 0.9633 | 0.8428 | 0.9093 | 0.7076 | 0.8673 |

## SuperGLUE
| 预训练模型 | CB | COPA | BoolQ | WiC | WSC | Avg |
| --- | --- | --- | --- | --- | --- | --- |
| google-bert-base-en | 0.75 | 0.61 | 0.7453 | 0.6912 | 0.6346 | 0.6862 |
| google-bert-large-en | 0.7321 | 0.62 | 0.7911 | 0.6975 | 0.6538 | 0.6989 |
| google-albert-large-en | 0.8571 | 0.68 | 0.7920 | 0.7273 | 0.6346 | 0.7382 |
| google-albert-xxlarge-en | 0.8393 | 0.85 | 0.8459 | 0.7524 | 0.6346 | 0.7844 |
| pai-albert-xxlarge-en | 0.8571 | 0.84 | 0.8535 | 0.7461 | 0.6346 | 0.7863 |

