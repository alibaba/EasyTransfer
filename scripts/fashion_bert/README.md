# Fashionbert工具使用指南
### 预训练fashionbert
```bash
wget https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/tutorial/fashion_bert/fashionbert_fashiongen_patch_train_1K

ls -d $PWD/fashionbert_fashiongen_patch_train_1K > train_list.list_csv
ls -d $PWD/fashionbert_fashiongen_patch_train_1K > dev_list.list_csv

export CUDA_VISIBLE_DEVICES="0"

python pretrain_main.py \
  --workerGPU=1 \
  --mode=train_and_evaluate \
  --train_input_fp=train_list.list_csv  \
  --eval_input_fp=dev_list.list_csv  \
  --pretrain_model_name_or_path=pai-imagebert-base-en  \
  --input_sequence_length=64  \
  --train_batch_size=4  \
  --num_epochs=1  \
  --model_dir=./fashionbert_model_dir  \
  --learning_rate=1e-4  \
  --image_feature_size=131072  \
  --input_schema="image_feature:float:131072,image_mask:int:64,masked_patch_positions:int:5,input_ids:int:64,input_mask:int:64,segment_ids:int:64,masked_lm_positions:int:10,masked_lm_ids:int:10,masked_lm_weights:float:10,nx_sent_labels:int:1"  \

```

### 评估fashionbert

img2text:

| ACC | 0.913441733253 |
| --- | --- |
| Rank @ 1 | 0.243154434594 |
| Rank @ 5 | 0.462760128901 |
| Rank @ 10 | 0.515881705828 |
| Rank @ 100 | 0.547645122959 |



text2img：

| ACC | 0.912456911831 |
| --- | --- |
| Rank @ 1 | 0.28947368259 |
| Rank @ 5 | 0.479843222397 |
| Rank @ 10 | 0.521276592826 |
| Rank @ 100 | 0.559910411199 |



```bash
wget https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/tutorial/fashion_bert/fashionbert-fashiongen-patch-eval-img2txt__097eaabf2d1e4464b88453bc7dfc8878
wget https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/tutorial/fashion_bert/fashionbert-fashiongen-patch-eval-img2txt__1b7320f883b6453e8922f520bac18e84
wget https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/tutorial/fashion_bert/fashionbert-fashiongen-patch-eval-img2txt__770ef9af0ab246dfb2269b9e008bc144
wget https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/tutorial/fashion_bert/fashionbert-fashiongen-patch-eval-txt2img__9d7082f64d0346fea770b66cdba0fcd2
wget https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/tutorial/fashion_bert/fashionbert-fashiongen-patch-eval-txt2img__c4aff1da32324da081af6324570c0bda
wget https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/tutorial/fashion_bert/fashionbert-fashiongen-patch-eval-txt2img__e928ee31b75940e88f1da64f133d9c4d
wget https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/tutorial/fashion_bert/fashionbert_pretrain_model_fin.tar.gz

tar -zxf fashionbert_pretrain_model_fin.tar.gz

mkdir eval_img2txt eval_txt2img
mv fashionbert-fashiongen-patch-eval-img2txt* eval_img2txt
mv fashionbert-fashiongen-patch-eval-txt2img* eval_txt2img
ls -d $PWD/eval_img2txt/* > eval_img2txt_list.list_csv
ls -d $PWD/eval_txt2img/* > eval_txt2img_list.list_csv

export CUDA_VISIBLE_DEVICES="0"

python pretrain_main.py \
  --workerGPU=1 \
  --type=img2txt  \
  --mode=predict \
  --predict_input_fp=eval_img2txt_list.list_csv  \
  --predict_batch_size=64  \
  --output_dir=./fashionbert_out  \
  --pretrain_model_name_or_path=pai-imagebert-base-en \
  --image_feature_size=131072  \
  --predict_checkpoint_path=./fashionbert_pretrain_model_fin/model.ckpt-54198  \
  --input_schema="image_feature:float:131072,image_mask:int:64,input_ids:int:64,input_mask:int:64,segment_ids:int:64,nx_sent_labels:int:1,prod_desc:str:1,text_prod_id:str:1,image_prod_id:str:1,prod_img_id:str:1"  \

python pretrain_main.py \
  --workerGPU=1 \
  --type=txt2img  \
  --mode=predict \
  --predict_input_fp=eval_txt2img_list.list_csv  \
  --predict_batch_size=64  \
  --output_dir=./fashionbert_out  \
  --pretrain_model_name_or_path=./pai-imagebert-base-en/model.ckpt  \
  --predict_checkpoint_path=./fashionbert_pretrain_model_fin/model.ckpt-54198  \
  --image_feature_size=131072  \
  --input_schema="image_feature:float:131072,image_mask:int:64,input_ids:int:64,input_mask:int:64,segment_ids:int:64,nx_sent_labels:int:1,prod_desc:str:1,text_prod_id:str:1,image_prod_id:str:1,prod_img_id:str:1"  \

```


# FashionGEN 特征提取流程
## Train&Valid数据提取
由于Train&Valid数据较大，无法开源。数据提取过程如下

1. <图像，文本> Pair提取: 分别从FashionGEN的Train和Valid数据提取图像和文本信息，组成正样本Pair

```bash
{"sentids": [32108], 
"text": "Cropped skinny-fit jeans in blue 'Sanoma' wash Fading throughout Five-pocket styling Embroidered logo at back pocket Contrast stitching in golden yellow Zip-fly"，
"image":"193603_3.png"，
"split": "train",
"label":"true"}
```

2. 针对Train和Valid数据中任意产品id，在相同的sub-category采样另外一个产品作为负样本

```bash
{"sentids": [32104], 
"text": "Lightweight knit long sleeve t-shirt in ivory white Ribbed knit trim at scoopneck collar and armscyes Raw edge at hem Tonal stitching"，
"image":"193603_3.png"，
"split": "train",
"label":"false"}
```

3. 这样可以得到Train和Valid的数据，数据正负样本比例基本为1:1
3. 针对Train和Valid的文本和图像做特征提取
   1. 文本侧采用和BERT一样的Masking策略
   1. 图像侧: 先将图像切分成8*8的Patch，然后每个Patch用预训练的ResNet50提取图像2048维特征。
(**图像特征建议用Tensorflow(tf-slim)进行特征的提取。提取AvgPool2D层(2048维度)输出作为图像特征。**[https://github.com/google-research/tf-slim](https://github.com/google-research/tf-slim))。
**注：对于图像Patch，选定被Mask的图像Patch所有像素置白(0)**
5. 提取好图文特征作为Train和Valid数据进入EasyTransfer训练，得到预训练模型。(FashionGEN Train 约为26W，Valid 约为3.2W)。

预训练模型地址: https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/tutorial/fashion_bert/fashionbert_pretrain_model_fin.tar.gz
## Eval 数据提取
Image2Text(或者Text2Image)任务

1. 在Valid数据上抽取100个产品，然后将产品的Image(或者Text)做为Query，Text(或者Image)作为Ground True
1. 在subcategory下随机抽取100条其他产品的Text(或者 Image) 和 Image (或者Text)Query 组成Negative 样本
1. 针对每个产品的正负样本打分(使用FashionBERT [CLS]的输出)
1. 根据打分做排序，检验Ground True排序位置，评估Rank@K
```bash
# Eval 地址
wget https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/tutorial/fashion_bert/fashionbert-fashiongen-patch-eval-img2txt__097eaabf2d1e4464b88453bc7dfc8878
wget https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/tutorial/fashion_bert/fashionbert-fashiongen-patch-eval-img2txt__1b7320f883b6453e8922f520bac18e84
wget https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/tutorial/fashion_bert/fashionbert-fashiongen-patch-eval-img2txt__770ef9af0ab246dfb2269b9e008bc144
wget https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/tutorial/fashion_bert/fashionbert-fashiongen-patch-eval-txt2img__9d7082f64d0346fea770b66cdba0fcd2
wget https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/tutorial/fashion_bert/fashionbert-fashiongen-patch-eval-txt2img__c4aff1da32324da081af6324570c0bda
wget https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/tutorial/fashion_bert/fashionbert-fashiongen-patch-eval-txt2img__e928ee31b75940e88f1da64f133d9c4d
```

## Reference

``` bash
@inproceedings{10.1145/3397271.3401430,
author = {Gao, Dehong and Jin, Linbo and Chen, Ben and Qiu, Minghui and Li, Peng and Wei, Yi and Hu, Yi and Wang, Hao},
title = {FashionBERT: Text and Image Matching with Adaptive Loss for Cross-Modal Retrieval},
year = {2020},
publisher = {Association for Computing Machinery},
booktitle = {Proceedings of the 43rd International ACM SIGIR Conference on Research and Development in Information Retrieval},
pages = {2251–2260},
numpages = {10},
location = {Virtual Event, China},
series = {SIGIR '20}
}
```