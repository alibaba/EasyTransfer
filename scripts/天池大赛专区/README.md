# 阿里云天池人工智能大赛-中文预训练模型泛化能力挑战赛
## 背景介绍    
中文预训练模型的质量会影响以AI技术驱动的企业中核心业务算法的性能。比如智能客服问答，搜索推荐，安全风控，舆情分析，文字识别，信息抽取，智能打标等等，这些业务或产品都会或多或少依赖于预训练模型提供的通用语言学知识来做迁移学习。因此打造高质量的预训练模型是当前产业界和学术界的热点。
 
自从2017年具有划时代意义的Transformer模型问世以来，短短两年多的时间内，如雨后春笋般的出现了大量的预训练模型，比如：Bert，Albert，ELECTRA，RoBERta，T5，GPT3等等。然而当前的基准评测体系存在两个缺陷：评测强度不够，模型不通用。评测强度不够指的是选手只提交结果，不提交inference的代码。模型不通用指的是预训练模型不能保证在相同超参数情况下在所有任务上都获得比较好的性能。以上两点极大限制了预训练技术的应用和发展。
 
如果我们能通过算法实现泛化能力强的中文预训练模型，这将提高下游业务的准确性，从而提升企业的核心竞争力，并为企业创造更多的价值。
 
阿里云计算平台PAI团队联合CLUE中文语言理解评测组织和上海乐言信息科技有限公司，共同推出中文预训练模型泛化能力挑战赛，邀请业内算法高手、爱好者一起促进自然语言处理预训练技术的发展。


## 赛题背景    
本次比赛是CLUE与阿里云平台、乐言科技联合发起的第一场针对中文预训练模型泛化能力的挑战赛。    
    
赛题以自然语言处理为背景，要求选手通过算法实现泛化能力强的中文预训练模型。通过这道赛题可以引导大家更好地理解预训练模型的运作机制，探索深层次的模型构建和模型训练，而不仅仅是针对特定任务进行简单微调。    


## 数据说明
本赛题精选了以下3个具有代表性的任务，要求选手提交的模型能够同时预测每个任务对应的标签：
>1. OCNLI，是第一个非翻译的、使用原生汉语的大型中文自然语言推理数据集；
>1. OCEMOTION，是包含7个分类的细粒度情感性分析数据集；
>1. TNEWS，来源于今日头条的新闻版块，共包含15个类别的新闻；

## 数据格式  
任务1:  OCNLI--中文原版自然语言推理 

```
0	一月份跟二月份肯定有一个月份有.	肯定有一个月份有	0
1	一月份跟二月份肯定有一个月份有.	一月份有	1
2	一月份跟二月份肯定有一个月份有.	一月二月都没有	2
3	一点来钟时,张永红却来了	一点多钟,张永红来了	0
4	不讲社会效果,信口开河,对任何事情都随意发议论,甚至信谣传谣,以讹传讹,那是会涣散队伍、贻误事业的	以讹传讹是有害的	0

（注：id  句子1  句子2  标签）
```

任务2:  OCEMOTION--中文情感分类    
```
0	你知道多伦多附近有什么吗?哈哈有破布耶...真的书上写的你听哦...你家那块破布是世界上最大的破布,哈哈,骗你的啦它是说尼加拉瓜瀑布是世界上最大的瀑布啦...哈哈哈''爸爸,她的头发耶!我们大扫除椅子都要翻上来我看到木头缝里有头发...一定是xx以前夹到的,你说是不是?[生病]	sadness
1	平安夜,圣诞节,都过了,我很难过,和妈妈吵了两天,以死相逼才终止战争,现在还处于冷战中。	sadness
2	我只是自私了一点,做自己想做的事情!	sadness
3	让感动的不仅仅是雨过天晴,还有泪水流下来的迷人眼神。	happiness
4	好日子	happiness

（注：id  句子  标签）
```

任务3：TNEWS--今日头条新闻标题分类
```
0	上课时学生手机响个不停,老师一怒之下把手机摔了,家长拿发票让老师赔,大家怎么看待这种事?	108
1	商赢环球股份有限公司关于延期回复上海证券交易所对公司2017年年度报告的事后审核问询函的公告	104
2	通过中介公司买了二手房,首付都付了,现在卖家不想卖了。怎么处理?	106
3	2018年去俄罗斯看世界杯得花多少钱?	112
4	剃须刀的个性革新,雷明登天猫定制版新品首发	109

（注：id  句子  标签）
```
  
## 评测标准    
依照参赛选手提交的模型，求出每个任务的macro f1，然后在四个任务上取平均值，macro f1具体计算公式请参考sklearn上的定义：
https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score


 
## 提交说明
测试数据提供txt文本，类训练数据，选手需要为每一个txt文本输出对应的ann文本。ann文件存放目录压缩成zip格式进行提交。
【待确定】提交的文件结构描述

## 比赛规则
1. 本次挑战可以使用公开数据资源（如使用开源数据或者代码模型，需要在最终提交文件中说明外部公开的资源，并在提交审核的代码文档中打包提供）；
2. 禁止任何形式人工标注；
3. 禁止仅使用开源的模型和代码，没有算法贡献的情况出现。

---
# 示例教程

## 环境配置
以下教程是在linux系统中运行，其中涉及到模型下载，所以需要联网并安装有wget，其他python依赖包如下：    
- tensorflow-gpu  1.12.3
- easytransfer  0.1.2

## 文本目录结构
```
├── config    #配置文件目录
│   ├── OCEMOTION_preprocess_dev.json
│   ├── OCEMOTION_preprocess_train.json
│   ├── OCNLI_preprocess_dev.json
│   ├── OCNLI_preprocess_train.json
│   ├── TNEWS_preprocess_dev.json
│   └── TNEWS_preprocess_train.json
├── convert_csv_to_tfrecords.py    #csv转tfrecord代码
├── multitask_finetune.py    #多任务训练代码
├── run_convert_csv_to_tfrecords.sh     #数据转换脚本
├── run_multitask_finetune.sh    #训练脚本
└── tianchi_datasets    #数据文件目录
    ├── OCEMOTION
    │   ├── dev.csv
    │   ├── dev.tfrecord
    │   ├── train.csv
    │   └── train.tfrecord
    ├── OCNLI
    │   ├── dev.csv
    │   ├── dev.tfrecord
    │   ├── train.csv
    │   └── train.tfrecord
    └── TNEWS
        ├── dev.csv
        ├── dev.tfrecord
        ├── train.csv
        └── train.tfrecord
```

## 数据处理

#### csv转tfrecord    
1. 配置文件示例    
以OCNLI--中文原版自然语言推理任务为例：
```
{
  "preprocess_config": {
    "preprocess_input_fp": "tianchi_datasets/OCNLI/train.csv",
    "preprocess_output_fp": "tianchi_datasets/OCNLI/train.tfrecord",
    "preprocess_batch_size": 16,
    "input_schema": "idx:str:1,sentence1:str:1,sentence2:str:1,label:str:1",
    "tokenizer_name_or_path": "google-bert-base-zh",
    "first_sequence": "sentence1",
    "second_sequence": "sentence2",
    "label_name":"label",
    "label_enumerate_values": "0,1,2",
    "sequence_length": 128,
    "output_schema": "input_ids,input_mask,segment_ids,label_id"
  }
}
```
2. 数据转换成tfrecord    
然后运行下面脚本，即可将数据转换成tfrecord格式，每个样本将包含input_ids/input_mask/segment_ids/label_id这四个特征，并会将生成的数据链接到train.list_tfrecord上。    
```
sh run_convert_csv_to_tfrecords.sh
```

## 模型训练与评估
根据上面方法可以生成训练与验证数据，运行以下脚本，就可以实现基于bert-base模型的数据训练与评估，因为评估中是按照顺序判断任务类型的，所以每个任务的评估数量需要一致。    
```
sh run_multitask_finetune.sh
```
也可以通过jupyter ipython notebook来实现，训练与评估的具体细节，则可以参考[【使用EasyTransfer快速搭建天池大赛Baseline】](https://github.com/alibaba/EasyTransfer/blob/master/scripts/%E5%A4%A9%E6%B1%A0%E5%A4%A7%E8%B5%9B%E4%B8%93%E5%8C%BA/tianchi-quick_start.ipynb)


## 提交结果
【待确定】

## 答疑群

下载钉钉，扫码如下二维码，进入天池大赛答疑群

<img src="https://cdn.nlark.com/yuque/0/2020/png/2556223/1605033894968-73f08826-fefd-4ee3-a542-cb61dfba407c.png#align=left&display=inline&height=352&margin=%5Bobject%20Object%5D&name=image.png&originHeight=1178&originWidth=1016&size=312154&status=done&style=none&width=304" width="300"/>
