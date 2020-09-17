# PAI-ModelZoo实践

# 背景介绍


自然语言处理的一大热点工作就是预训练语言模型比方说BERT，ALBERT等，这类模型在各大自然语言处理场景都取得了非常不错的效果。基于EasyTransfer平台的预训练模型，PAI团队分别在中文CLUE刷榜上取得过第一名，在超难度NLP新基准（SuperGLUE）公开评测榜单上取得：全球第二名，国内第一名。


![20200829170203.jpg](https://intranetproxy.alipay.com/skylark/lark/0/2020/jpeg/226643/1598691739123-7825d354-83e0-4c1a-a40f-6ab70a32caca.jpeg#align=left&display=inline&height=1066&margin=%5Bobject%20Object%5D&name=20200829170203.jpg&originHeight=1066&originWidth=2164&size=211035&status=done&style=none&width=2164#align=left&display=inline&height=1066&margin=%5Bobject%20Object%5D&originHeight=1066&originWidth=2164&status=done&style=none&width=2164)
![20200829170240.jpg](https://intranetproxy.alipay.com/skylark/lark/0/2020/jpeg/226643/1598691775133-0c5bcbb2-5b91-45c6-8046-953d196d7baf.jpeg#align=left&display=inline&height=950&margin=%5Bobject%20Object%5D&name=20200829170240.jpg&originHeight=950&originWidth=2142&size=270066&status=done&style=none&width=2142#align=left&display=inline&height=950&margin=%5Bobject%20Object%5D&originHeight=950&originWidth=2142&status=done&style=none&width=2142)


为了更好的支持用户使用预训练语言模型，我们在迁移学习框架EasyTransfer里植入了一套预训练语言模型的标准范式和预训练语言模型库ModelZoo供用户直接调用。按照下图所示的标准使用范式从左到右走一遍，一般都会在业务上取得不错的效果。
# ![20200527233954.jpg](https://cdn.nlark.com/yuque/0/2020/jpeg/2556223/1600351891389-636fda1e-748b-4056-8485-e3d2eeb3d103.jpeg#align=left&display=inline&height=726&margin=%5Bobject%20Object%5D&name=20200527233954.jpg&originHeight=726&originWidth=1514&size=145142&status=done&style=none&width=1514)


我们以Bert为例，它Google提出的‘11项全能模型’，在很多NLP场景里取得了非常好的结果。BERT的idea和Elmo，GPT都非常接近，网络架构如下。但是BERT可以在很大的非监督的语料里pretrain，速度比ELMo快很多，效果也比GPT好。
![](https://cdn.nlark.com/lark/0/2018/png/34639/1544363524186-5fed06a7-ac00-44e9-aece-3ef0f593cb6a.png#align=left&display=inline&height=222&margin=%5Bobject%20Object%5D&originHeight=222&originWidth=702&status=done&style=none&width=702#align=left&display=inline&height=222&margin=%5Bobject%20Object%5D&originHeight=222&originWidth=702&status=done&style=none&width=702)


## 通用领域的预训练强化


PAI团队&ICBU团队收集了大量的非监督数据，拥有100+G的中文数据，150+T的英文C4数据，还有整套分布式数据预处理流程。为了加速这样大规模的数据训练，PAI团队单独优化了LAMB优化器，改进的模型网络结构，同时结合强大的底层多机多卡分布式技术，对谷歌开源的模型进行继续预训练，效果远超现有的开源的模型。


## 垂直领域的继续预训练
基础模型吸收了语言学中的通用知识，要想在特定领域（新闻信息流，电商，对话，社交等领域）发挥更大的作用，需进一步将特定领域的知识迁移进基础模型中。什么是最好的Continue Pretrain模型？根据我们大量实验得出的经验，选择在验证集上mask_lm_loss最小的模型，往往在下游finetune中效果相对较好。

# 组件结构示意图
## Bert


![20200528205701.jpg](https://cdn.nlark.com/yuque/0/2020/jpeg/2556223/1600351891395-e0635893-2cb7-4d11-8e0d-f4c00d6908d9.jpeg#align=left&display=inline&height=1060&margin=%5Bobject%20Object%5D&name=20200528205701.jpg&originHeight=1060&originWidth=1770&size=225043&status=done&style=none&width=1770)


## Albert
![20200528205610.jpg](https://cdn.nlark.com/yuque/0/2020/jpeg/2556223/1600351891389-76b194a4-9a09-402d-80a2-ca17be7453fa.jpeg#align=left&display=inline&height=1060&margin=%5Bobject%20Object%5D&name=20200528205610.jpg&originHeight=1060&originWidth=1770&size=227341&status=done&style=none&width=1770)


## ImageBert
![20200527150730.jpg](https://cdn.nlark.com/yuque/0/2020/jpeg/2556223/1600351891373-846f85ea-482a-41ed-9ba9-e302b2e2d49f.jpeg#align=left&display=inline&height=1008&margin=%5Bobject%20Object%5D&name=20200527150730.jpg&originHeight=1008&originWidth=1646&size=219944&status=done&style=none&width=1646)


# 内部运行机制解析
```python
def build_logits(self, features, mode=None):
    preprocessor = preprocessors.get_preprocessor(self.pretrain_model_name_or_path)
    model = model_zoo.get_pretrained_model(self.pretrain_model_name_or_path)

    dense = layers.Dense(self.num_labels,
                         kernel_initializer=layers.get_initializer(0.02),
                         name='dense')

    input_ids, input_mask, segment_ids, label_ids = preprocessor(features)
    outputs = model([input_ids, input_mask, segment_ids], mode=mode)
    pooled_output = outputs[1]

    logits = dense(pooled_output)
    return logits, label_ids
```


我们回顾一下在[Quick Start](https://yuque.antfin-inc.com/pai-innovative-algo/ctrdil/golsh8)章节中用户控制的前端代码构图部分的逻辑。在Finetune阶段，用户通过model_zoo.get_pretrained_model 方法创建了预训练模型对象，然后通过outputs = model([input_ids, input_mask, segment_ids], mode=mode)，进行实际的前向推理。下面是Bert模型的代码，用于展示内部的调用逻辑。
```python
class BertPreTrainedModel(PreTrainedModel):
    def __init__(self, config, **kwargs):
        super(BertPreTrainedModel, self).__init__(config, **kwargs)

        self.bert = BertBackbone(config, name="bert")
        self.mlm = layers.MLMHead(config, self.bert.embeddings, name="cls/predictions")
        self.nsp = layers.NSPHead(config, name="cls/seq_relationship")

    def call(self, inputs, masked_lm_positions=None， **kwargs):
        training = kwargs['mode'] == tf.estimator.ModeKeys.TRAIN
        outputs = self.bert(inputs, training=training)
        sequence_output = outputs[0]
        pooled_output = outputs[1]
        return sequence_output, pooled_output

```


# 预训练模型列表
| Model | Parameters |
| --- | --- |
| **RoBERTa** |  |
| hit-roberta-base-zh | L=12,H=768,A=12 |
| hit-roberta-large-zh | L=24,H=1024,A=16 |
| brightmart-roberta-small-zh | L=6, H=768,A=12 |
| brightmart-roberta-large-zh | L=24,H=1024,A=16 |
| **ALBERT** |  |
| google-albert-base-zh/en | L=12,H=768,A=12 |
| google-albert-large-zh/en | L=12,H=1024,A=16 |
| google-albert-xlarge-zh/en | L=24,H=2048,A=32 |
| google-albert-xxlarge-zh/en | L=12,H=4096,A=64 |
| pai-albert-base-zh/en | L=12,H=768,A=12 |
| pai-albert-large-zh/en | L=12,H=1024,A=16 |
| pai-albert-xlarge-zh/en | L=24,H=2048,A=32 |
| pai-albert-xxlarge-zh/en | L=12,H=4096,A=64 |
| **BERT** |  |
| google-bert-base-zh | L=12,H=768,A=12 |
| google-bert-base-en | L=12,H=768,A=12 |
| google-bert-large-en | L=24,H=1024,A=16 |
| google-bert-small-en | L=6,H=768,A=12 |
| google-bert-tiny-en | L=2,H=128,A=2 |
| pai-bert-large-zh | L=24,H=1024,A=16 |
| pai-bert-base-zh | L=12,H=768,A=12 |
| pai-bert-small-zh | L=6,H=768,A=12 |
| pai-bert-tiny-zh | L=2,H=128,A=2 |
| [pai-bert-tiny-zh-L2-H768-A12]() | [L=2,H768,A=12]() |
| **Cross-modal** |  |
| icbu-imagebert-small-en | L=6,H=128,A=2 |
| pai-imagebert-base-en | L=12,H=768,A=12 |
| pai-imagebert-base-zh | L=12,H=768,A=12 |



同时您还可以在EasyTransfer里面使用谷歌提供的24种小模型：[https://github.com/google-research/bert](https://github.com/google-research/bert)
![20200528203013.jpg](https://cdn.nlark.com/yuque/0/2020/jpeg/2556223/1600351891380-c99cfebf-43de-44b2-83de-da1504624182.jpeg#align=left&display=inline&height=338&margin=%5Bobject%20Object%5D&name=20200528203013.jpg&originHeight=338&originWidth=968&size=72570&status=done&style=none&width=968)
具体加载方法如下：


下面我们详细介绍下这个模型以及如何在EasyTransfer平台上使用BERT。


# 建模思路


BERT模型是基于Google的Neural Machine Translation模型的encoder，架构如下：
![](https://cdn.nlark.com/lark/0/2018/png/34639/1544363640198-cc8c5e81-b963-4570-b741-1ce99539ea33.png#align=left&display=inline&height=391&margin=%5Bobject%20Object%5D&originHeight=391&originWidth=559&status=done&style=none&width=559#align=left&display=inline&height=391&margin=%5Bobject%20Object%5D&originHeight=391&originWidth=559&status=done&style=none&width=559)
在encoder部分，模型对输入的sequence建模，相当于学一个language model，学到一个sequence representation。BERT的一个核心idea是把很多NLP任务看成是一个language model，用BERT pretrain的模型继续finetune，就可以提升很多NLP任务。


# 快速上手


下面我们以一个简单的新闻标题分类任务熟悉一下基于EasyTransfer的ModelZoo中的bert模型用法。下面是标签到ID的一个映射表，总共有15个类别


```bash
{"label": "100", "label_desc": "news_story"}
{"label": "101", "label_desc": "news_culture"}
{"label": "102", "label_desc": "news_entertainment"}
{"label": "103", "label_desc": "news_sports"}
{"label": "104", "label_desc": "news_finance"}
{"label": "106", "label_desc": "news_house"}
{"label": "107", "label_desc": "news_car"}
{"label": "108", "label_desc": "news_edu"}
{"label": "109", "label_desc": "news_tech"}
{"label": "110", "label_desc": "news_military"}
{"label": "112", "label_desc": "news_travel"}
{"label": "113", "label_desc": "news_world"}
{"label": "114", "label_desc": "news_stock"}
{"label": "115", "label_desc": "news_agriculture"}
{"label": "116", "label_desc": "news_game"}
```


![19FCE7D7-FF68-4CA2-BC53-43A0A38941AA.png](https://intranetproxy.alipay.com/skylark/lark/0/2020/png/226643/1590393917768-0cf30eac-181b-47ce-b2a3-39bcf6997e0c.png#align=left&display=inline&height=357&margin=%5Bobject%20Object%5D&name=19FCE7D7-FF68-4CA2-BC53-43A0A38941AA.png&originHeight=782&originWidth=1080&size=656598&status=done&style=none&width=493#align=left&display=inline&height=386&margin=%5Bobject%20Object%5D&originHeight=782&originWidth=1080&status=done&style=none&width=533#align=left&display=inline&height=782&margin=%5Bobject%20Object%5D&originHeight=782&originWidth=1080&status=done&style=none&width=1080)


## 编辑配置文件


```json
{
  "preprocess_config": {
    "input_schema": "content:str:1,label:str:1",
    "first_sequence": "content",
    "second_sequence": null,
    "sequence_length":64,
    "label_name": "label",
    "label_enumerate_values": "100,101,102,103,104,106,107,108,109,110,112,113,114,115,116",
    "output_schema": "content,label,predictions"
  },
  "model_config": {
    "pretrain_model_name_or_path": "google-bert-base-zh",
    "num_labels": 15
  },
  "train_config": {
    "train_input_fp": null,
    "train_batch_size": 32,
    "save_steps": 100,
    "num_epochs": 1,
    "model_dir": "oss://tongruntwzjk/test_quick_start",
    "optimizer_config": {
      "optimizer": "adam",
      "weight_decay_ratio": 0,
      "warmup_ratio": 0.1,
      "learning_rate": 1e-5
    },
    "distribution_config": {
      "distribution_strategy": "MirroredStrategy"
    }
  },
  "evaluate_config": {
    "eval_input_fp": null,
    "eval_batch_size": 8
  },
  "predict_config": {
    "predict_checkpoint_path": "oss://tongruntwzjk/test_quick_start/model.ckpt-200",
    "predict_input_fp": null,
    "predict_batch_size": 32,
    "predict_output_fp": null
  },
  "export_config": {
    "input_tensors_schema": "input_ids:int:64,input_mask:int:64,segment_ids:int:64,label_ids:int:1",
    "receiver_tensors_schema": "input_ids:int:64,input_mask:int:64,segment_ids:int:64",
    "export_dir_base": "oss://tongruntwzjk/test_quick_start/cache_export",
    "checkpoint_path": "oss://tongruntwzjk/test_quick_start/model.ckpt-200"
  }
}
```


## 搭建网络


该前端代码展示了仅用一个文件就可以完成训练/评估/预测/导出这个四个功能，用户在使用的时候可以以此为基础进行修改。


```python
from ez_transfer import base_model
from ez_transfer import preprocessors
from ez_transfer import model_zoo
from ez_transfer.datasets import OdpsTableReader,OdpsTableWriter
from ez_transfer.losses import softmax_cross_entropy
from ez_transfer import FLAGS
from ez_transfer import layers
from easytransfer.evaluators import classification_eval_metrics
import tensorflow as tf

class Application(base_model):
    def __init__(self, **kwargs):
        super(Application, self).__init__(**kwargs)

    def build_logits(self, features, mode=None):
        preprocessor = preprocessors.get_preprocessor(self.pretrain_model_name_or_path)
        model = model_zoo.get_pretrained_model(self.pretrain_model_name_or_path)

        dense = layers.Dense(self.num_labels,
                             kernel_initializer=layers.get_initializer(0.02),
                             name='dense')

        input_ids, input_mask, segment_ids, label_ids = preprocessor(features)
        outputs = model([input_ids, input_mask, segment_ids], mode=mode)
        pooled_output = outputs[1]

        logits = dense(pooled_output)
        return logits, label_ids


    def build_loss(self, logits, labels):
        return softmax_cross_entropy(labels, self.num_labels, logits)

    def build_eval_metrics(self, logits, labels):
        return classification_eval_metrics(logits, labels, self.num_labels)
    
    def build_predictions(self, output):
        logits, _ = output
        predictions = dict()
        predictions["predictions"] = tf.argmax(logits, axis=-1, output_type=tf.int32)
        return predictions

def train_and_evaluate_on_the_fly():
    app = Application()
    train_reader = OdpsTableReader(input_glob=app.train_input_fp,
                                   is_training=True,
                                   input_schema=app.input_schema,
                                   batch_size=app.train_batch_size)

    eval_reader = OdpsTableReader(input_glob=app.eval_input_fp,
                                  input_schema=app.input_schema,
                                  batch_size=app.eval_batch_size)

    app.run_train_and_evaluate(train_reader=train_reader, eval_reader=eval_reader)

def predict_on_the_fly():
    app = Application()
    pred_reader = OdpsTableReader(input_glob=app.predict_input_fp,
                                  input_schema=app.input_schema,
                                  batch_size=app.predict_batch_size)

    pred_writer = OdpsTableWriter(output_glob=app.predict_output_fp,
                                  output_schema=app.output_schema)

    app.run_predict(reader=pred_reader, writer=pred_writer, checkpoint_path=app.predict_checkpoint_path)

def export():
    app = Application()
    app.export_model()

if __name__ == "__main__":
    if FLAGS.mode == "train_and_evaluate_on_the_fly":
        train_and_evaluate_on_the_fly()
    elif FLAGS.mode == "predict_on_the_fly":
        predict_on_the_fly()
    elif FLAGS.mode == "export":
        export()
    else:
        raise RuntimeError("Run mode input")
```


#负责对原始数据进行预处理，生成模型需要的特征，比如：input_ids, input_mask, segment_ids等
preprocessor = preprocessors.get_preprocessor(self.pretrain_model_name_or_path)


#负责构建网络的backbone
model = model_zoo.get_pretrained_model(self.pretrain_model_name_or_path)


## 训练/评估


```bash
#!/usr/bin/env bash
set -e

odpscmd="/Users/jerry/Develops/odpscmd/bin/odpscmd"
config="/Users/jerry/Develops/odpscmd/conf/odps_config_pai_exp_dev_tn_hz.ini"
ini_path='/Users/jerry/Develops/odpscmd/conf/pai_exp_dev_zjk_tw.ini'
role_arn_and_host=`cat ${ini_path}`
oss_dir=tongruntwzjk


cur_path=/Users/jerry/Develops/ATP-tutorials/tutorials-easytransfer
cd ${cur_path}
rm -f proj.tar.gz
tar -zcf proj.tar.gz quick_start

job_path='file://'${cur_path}'/proj.tar.gz'

command="
pai -name easytransfer
-project algo_platform_dev
-Dmode=train_and_evaluate_on_the_fly
-Dconfig=quick_start/config.json
-Dtables='odps://pai_exp_dev/tables/clue_tnews_train,odps://pai_exp_dev/tables/clue_tnews_dev'
-Dscript=${job_path}
-DentryFile='quick_start/main_finetune.py'
-Dbuckets=\"oss://tongruntwzjk/?${role_arn_and_host}\"
-DworkerGPU=2
-DworkerCount=1
"

echo "${command}"
${odpscmd} --config="${config}" -e "${command}"
echo "finish..."
```


在logview中会呈现参数配置，训练进度，评估指标输出，loss等信息，如下图所示：
![20200526222411.jpg](https://intranetproxy.alipay.com/skylark/lark/0/2020/jpeg/226643/1590503063283-f8c0d836-ff9f-4b67-944b-b5e75e48de1c.jpeg#align=left&display=inline&height=656&margin=%5Bobject%20Object%5D&name=20200526222411.jpg&originHeight=656&originWidth=1740&size=375069&status=done&style=none&width=1740#align=left&display=inline&height=656&margin=%5Bobject%20Object%5D&originHeight=656&originWidth=1740&status=done&style=none&width=1740#align=left&display=inline&height=656&margin=%5Bobject%20Object%5D&originHeight=656&originWidth=1740&status=done&style=none&width=1740)


![20200526223540.jpg](https://intranetproxy.alipay.com/skylark/lark/0/2020/jpeg/226643/1590503756019-6780715a-78ea-4993-b3d1-857074974d32.jpeg#align=left&display=inline&height=388&margin=%5Bobject%20Object%5D&name=20200526223540.jpg&originHeight=388&originWidth=1876&size=200246&status=done&style=none&width=1876#align=left&display=inline&height=388&margin=%5Bobject%20Object%5D&originHeight=388&originWidth=1876&status=done&style=none&width=1876#align=left&display=inline&height=388&margin=%5Bobject%20Object%5D&originHeight=388&originWidth=1876&status=done&style=none&width=1876)


## 离线预测


在往odps表中写预测输出前，先在odps上创建写入的表：
注意：创建的column的名字一定要和配置文件中的output_schema保持一致,output_schema中的column名字只能来自于input_schema或者main_finetune.py中的buidl_predictions中的名字


```json
#!/usr/bin/env bash
  
odpscmd="/Users/jerry/Develops/odpscmd/bin/odpscmd"
config="/Users/jerry/Develops/odpscmd/conf/odps_config_pai_exp_dev_tn_hz.ini"

${odpscmd} --config="${config}" -e "drop table if exists clue_tnews_pred_out;"
${odpscmd} --config="${config}" -e "create table clue_tnews_pred_out(
    content STRING, label STRING, predictions STRING) lifecycle 365;"
```


```bash
#!/usr/bin/env bash
set -e

odpscmd="/Users/jerry/Develops/odpscmd/bin/odpscmd"
config="/Users/jerry/Develops/odpscmd/conf/odps_config_pai_exp_dev_tn_hz.ini"
ini_path='/Users/jerry/Develops/odpscmd/conf/pai_exp_dev_zjk_tw.ini'
role_arn_and_host=`cat ${ini_path}`
oss_dir=tongruntwzjk


cur_path=/Users/jerry/Develops/ATP-tutorials/tutorials-easytransfer
cd ${cur_path}
rm -f proj.tar.gz
tar -zcf proj.tar.gz quick_start

job_path='file://'${cur_path}'/proj.tar.gz'

command="
pai -name easytransfer
-project algo_platform_dev
-Dmode=predict_on_the_fly
-Dconfig=quick_start/config.json
-Dtables='odps://pai_exp_dev/tables/clue_tnews_dev'
-Doutputs='odps://pai_exp_dev/tables/clue_tnews_pred_out'
-Dscript=${job_path}
-DentryFile='quick_start/main_finetune.py'
-Dbuckets=\"oss://tongruntwzjk/?${role_arn_and_host}\"
-DworkerGPU=1
-DworkerCount=1
"

echo "${command}"
${odpscmd} --config="${config}" -e "${command}"
echo "finish..."
```


输出预测结果支持append输入表内容的功能，只要在output_schema中配置一下就可以了，预测结果如下图所示：


# ![F8CAC60A-4039-43D3-91C2-0A8E38100BCE.png](https://intranetproxy.alipay.com/skylark/lark/0/2020/png/226643/1590502497037-f3658e67-ad98-49ec-b6db-e265b0fb77c6.png#align=left&display=inline&height=298&margin=%5Bobject%20Object%5D&name=F8CAC60A-4039-43D3-91C2-0A8E38100BCE.png&originHeight=764&originWidth=1204&size=554171&status=done&style=none&width=470#align=left&display=inline&height=375&margin=%5Bobject%20Object%5D&originHeight=764&originWidth=1204&status=done&style=none&width=591#align=left&display=inline&height=764&margin=%5Bobject%20Object%5D&originHeight=764&originWidth=1204&status=done&style=none&width=1204)


## 在线Serving


如果需要部署模型到线上做线上的serving，可以修改配置文件导出savedmodel，然后对接EAS serving。
详细的操作流程参考：[https://yuque.antfin-inc.com/pai-innovative-algo/ctrdil/ll00gn](https://yuque.antfin-inc.com/pai-innovative-algo/ctrdil/ll00gn)
