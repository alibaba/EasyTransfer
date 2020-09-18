<p align="center">
    <br>
    <img src="https://cdn.nlark.com/yuque/0/2020/png/2480469/1600401425964-828d6ffe-90d7-4cda-9b76-b9f17e35f11f.png#align=left&display=inline&height=188&margin=%5Bobject%20Object%5D&name=image.png&originHeight=608&originWidth=649&size=41423&status=done&style=none&width=201" width="200"/>
    <br>
<p>

<p align="center"> <b> EasyTransfer is designed to make the development of transfer learning in NLP applications easier. </b> </p>
<p align="center">
    <a href="https://www.yuque.com/easytransfer/itfpm9/ah0z6o">
        <img src="https://cdn.nlark.com/yuque/0/2020/svg/2480469/1600310258840-bfe6302e-d934-409d-917c-8eab455675c1.svg" height="24">
    </a>
    <a href="https://dsw-dev.data.aliyun.com/#/?fileUrl=https://pai-public-data.oss-cn-beijing.aliyuncs.com/easytransfer/easytransfer-quick_start.ipynb&fileName=easytransfer-quick_start.ipynb">
        <img src="https://cdn.nlark.com/yuque/0/2020/svg/2480469/1600310258886-ad896af5-b7da-4ca6-8369-4b14c23cb7a3.svg" height="24">
    </a>
</p>


The literature has witnessed the success of applying deep Transfer Learning (TL) for many NLP applications, yet it not easy to build a simple and easy-to-use TL toolkit to achieve this goal. To bridge this gap, EasyTransfer is designed to make it easy to build deep TL for NLP applications. It was developed in Alibaba in early 2017, and has been in the major BUs in Alibaba group and achieved very good results in 20+ business scenarios. It supports the mainstream pre-trained ModelZoo, including pre-trained language models (PLMs) and multi-modal models on the PAI platform, integrates the SOTA model for the mainstream NLP applications in AppZoo, and supports knowledge distillation for PLMs. The toolkit is convenient for users to quickly start model training, evaluation, offline prediction, and online deployment. It provides rich APIs to make the development of NLP and transfer learning easier.

# Main Features

- **Language model pre-training tool:** it supports a comprehensive pre-training tool for users to pre-train language models such as T5 and BERT. Based on the tool, the user can easily train a model to achieve great results in the benchmark leaderboards such as CLUE, GLUE, and SuperGLUE;
- **ModelZoo with rich and high-quality pre-trained models:** supports the Continual Pre-training and Fine-tuning of mainstream LM models such as BERT, ALBERT, RoBERTa, T5, etc. It also supports a multi-modal model FashionBERT developed using the fashion domain data in Alibaba;
- **AppZoo with rich and easy-to-use applications:** supports mainstream NLP applications and those models developed inside of Alibaba, e.g.: HCNN for text matching, and BERT-HAE for MRC.
- **Automatic knowledge distillation:** supports task-adaptive knowledge distillation to distill knowledge from a teacher model to a small task-specific student model. The resulting method is AdaBERT, which uses a neural architecture search method to find a task-specific architecture to compress the original BERT model. The compressed models are 12.7x to 29.3x faster than BERT in inference time and 11.5x to 17.0x smaller in terms of parameter size and with comparable performance.
- **Easy-to-use and high-performance distributed strategy:** based on the in-house PAI features, it provides easy-to-use and high-performance distributed strategy for multiple CPU/GPU training.

# Architecture
![image.png](https://cdn.nlark.com/yuque/0/2020/png/2480469/1600310258839-04837b68-ef37-449d-8ff4-02dbd8dcef9e.png#align=left&display=inline&height=357&margin=%5Bobject%20Object%5D&name=image.png&originHeight=713&originWidth=1492&size=182794&status=done&style=none&width=746)

# Installation

You can either install from pip 

```bash
$ pip install easytransfer
```

or setup from the source：

```bash
$ git clone https://github.com/alibaba/EasyTransfer.git
$ cd EasyTransfer
$ python setup.py install
```
This repo is tested on Python3.6/2.7, tensorflow 1.12.3


# Quick Start
Now let's show how to use 30 lines to build bert-based text classification. 

```python
from easytransfer import base_model, layers, model_zoo, preprocessors
from easytransfer.datasets import CSVReader, CSVWriter
from easytransfer.losses import softmax_cross_entropy
from easytransfer.evaluators import classification_eval_metrics

class TextClassification(base_model):
    def __init__(self, **kwargs):
        super(TextClassification, self).__init__(**kwargs)
	self.pretrained_model_name = "google-bert-base-en"
        self.num_labels = 2
        
    def build_logits(self, features, mode=None):
        preprocessor = preprocessors.get_preprocessor(self.pretrained_model_name)
        model = model_zoo.get_pretrained_model(self.pretrained_model_name)
        dense = layers.Dense(self.num_labels)
        input_ids, input_mask, segment_ids, label_ids = preprocessor(features)
        _, pooled_output = model([input_ids, input_mask, segment_ids], mode=mode)
        return dense(pooled_output), label_ids

    def build_loss(self, logits, labels):
        return softmax_cross_entropy(labels, self.num_labels, logits)
    
    def build_eval_metrics(self, logits, labels):
        return classification_eval_metrics(logits, labels, self.num_labels)
        
app = TextClassification()

train_reader = CSVReader(input_glob=app.train_input_fp, is_training=True, batch_size=app.train_batch_size)
eval_reader = CSVReader(input_glob=app.eval_input_fp, is_training=False, batch_size=app.eval_batch_size)              
app.run_train_and_evaluate(train_reader=train_reader, eval_reader=eval_reader)
```
You can find more details or play with codes in our Jupyter/Notebook [PAI-DSW](https://dsw-dev.data.aliyun.com/#/?fileUrl=https://pai-public-data.oss-cn-beijing.aliyuncs.com/easytransfer/easytransfer-quick_start.ipynb&fileName=easytransfer-quick_start.ipynb). 


You can also use AppZoo Command Line Tools to quickly train an App model. Take text classification on SST-2 dataset as an example. First you can download the [train.tsv](http://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/tutorial/glue/SST-2/train.tsv), [dev.tsv](http://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/tutorial/glue/SST-2/dev.tsv) and [test.tsv](http://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/tutorial/glue/SST-2/test.tsv), then start training: 

```bash
$ easy_transfer_app --mode train \
    --inputTable=./train.tsv,./dev.tsv \
    --inputSchema=content:str:1,label:str:1 \
    --firstSequence=content \
    --sequenceLength=128 \
    --labelName=label \
    --labelEnumerateValues=0,1 \
    --checkpointDir=./sst2_models/\
    --numEpochs=3 \
    --batchSize=32 \
    --optimizerType=adam \
    --learningRate=2e-5 \
    --modelName=text_classify_bert \
    --advancedParameters='pretrain_model_name_or_path=google-bert-base-en'
```

And then predict:

```bash
$ easy_transfer_app --mode predict \
    --inputTable=./test.tsv \
    --outputTable=./test.pred.tsv \
    --inputSchema=id:str:1,content:str:1 \
    --firstSequence=content \
    --appendCols=content \
    --outputSchema=predictions,probabilities,logits \
    --checkpointPath=./sst2_models/ 
```
To learn more about the usage of AppZoo, please refer to our [documentation](https://www.yuque.com/easytransfer/itfpm9/ky6hky).


# Tutorials

- [PAI-ModelZoo (20+ pretrained models)](https://www.yuque.com/easytransfer/itfpm9/geiy58)
- [FashionBERT-cross-modality pretrained model](https://www.yuque.com/easytransfer/itfpm9/nm3mxu)
- [Knowledge Distillation including vanilla KD, Probes KD, AdaBERT](https://www.yuque.com/easytransfer/itfpm9/kp1dtx)
- [BERT Feature Extraction](https://www.yuque.com/easytransfer/itfpm9/blz7k6)
- [Text Matching including BERT, BERT Two Tower, DAM, HCNN](https://www.yuque.com/easytransfer/itfpm9/xfe19v)
- [Text Classification including BERT, TextCNN](https://www.yuque.com/easytransfer/itfpm9/rypc5x)
- [Machine Reading Comprehesion including BERT, BERT-HAE](https://www.yuque.com/easytransfer/itfpm9/qrvqco)
- [Sequence Labeling including BERT](https://www.yuque.com/easytransfer/itfpm9/we0go2)



# CLUE Benchmark



|  | TNEWS | AFQMC | IFLYTEK | CMNLI | CSL | Average |
| --- | --- | --- | --- | --- | --- | --- |
| google-bert-base-zh | 0.6673 | 0.7375 | 0.5968 | 0.7981 | 0.7976 | 0.7194 |
| pai-bert-base-zh | 0.6694 | 0.7412 | 0.6114 | 0.7967 | 0.7993 | 0.7236 |
| hit-roberta-base-zh | 0.6734 | 0.7418 | 0.6052 | 0.8010 | 0.8010 | 0.7245 |
| hit-roberta-large-zh | 0.6742 | 0.7521 | 0.6052 | 0.8231 | 0.8100 | 0.7329 |
| google-albert-xxlarge-zh | 0.6253 | 0.6899 | 0.5017 | 0.7721 | 0.7106 | 0.6599 |
| pai-albert-xxlarge-zh | 0.6809 | 0.7525 | 0.6118 | 0.8284 | 0.8137 | 0.7375 |



You can find more benchmarks in [https://www.yuque.com/easytransfer/cn/rkm4p7](https://www.yuque.com/easytransfer/itfpm9/rkm4p7)


# Links

Tutorials：[https://www.yuque.com/easytransfer/itfpm9/qtzvuc](https://www.yuque.com/easytransfer/itfpm9/qtzvuc)

ModelZoo：[https://www.yuque.com/easytransfer/itfpm9/oszcof](https://www.yuque.com/easytransfer/itfpm9/oszcof)

AppZoo：[https://www.yuque.com/easytransfer/itfpm9/ky6hky](https://www.yuque.com/easytransfer/itfpm9/ky6hky)

API docs：[http://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/eztransfer_docs/html/index.html](http://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/eztransfer_docs/html/index.html)


# Contact Us
Scan the following QR codes to join Dingtalk discussion group. The group discussions are most in Chinese, but English is also welcomed.

<img src="https://cdn.nlark.com/yuque/0/2020/png/2480469/1600310258842-d7121051-32f1-494b-a7a5-a35ede74b6c4.png#align=left&display=inline&height=352&margin=%5Bobject%20Object%5D&name=image.png&originHeight=1178&originWidth=1016&size=312154&status=done&style=none&width=304" width="300"/>
