# Fashionbert Tutorial

## Input Schema
``` bash 
image_feature: image patch features, (image_feature size) = (patch_feature size) * (image_patch number). For example, given on image, we equally split it into 64 patches, and for each patch we extract 2048 embedding. Thus, the image_feature size is equal to 131072(2048*64) in our paper.
image_mask: image mask to indicate there exists padding image_feature(s). In FashionBERT, all image are equally splitted into patches with same number. Thus, image_mask are 1.
masked_patch_positions: the masked position of the patch sequence, which is randomly selected from the patch sequence.
input_ids: input tokens ids, which is the same with BERT.
input_mask: input mask of tokens, which is the same with BERT.
segment_ids: input segment ids, which is used to distinct from image patch input. We use 0 to indicate text token input in our paper, which is the same with BERT.
masked_lm_positions: the masked lm positions in input token sequence, which is the same with BERT.
masked_lm_weights: the masked token ids at the masked positions, respectively, which is the same with BERT.
```

## Pretrain
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

## Evaluation

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


# Dataset Preparation (FashionGEN)
## Train & Valid Data 
Due to the large scale, it is really hard to release train&valid datasets directly.
The procedure is as following:

1. Positive <Image, Text> Pair Extraction: for one product, the text and its image are extract to compose of one <Image, Text> Pair. 

```bash
{"sentids": [32108], 
"text": "Cropped skinny-fit jeans in blue 'Sanoma' wash Fading throughout Five-pocket styling Embroidered logo at back pocket Contrast stitching in golden yellow Zip-fly"，
"image":"193603_3.png"，
"split": "train",
"label":"true"}
```

2. Negative <Image, Text>: for one product, we randomly select one product under its sub-category as one negative product pair, e.g. <product_id, negative_product_id>. In sequence, the text and image from these two products are composed of one negative <Image, Text> pair.

```bash
{"sentids": [32104], 
"text": "Lightweight knit long sleeve t-shirt in ivory white Ribbed knit trim at scoopneck collar and armscyes Raw edge at hem Tonal stitching"，
"image":"193603_3.png"，
"split": "train",
"label":"false"}
```

3. Now, we obtain the train and valid <Image, Text> pairs, which include both positive and negative pairs.

4. Extract Text and Image Features
   1. Text: The procedure of text features extraction is exactly same to that of the BERT model. refer **[google create pretrain data](https://github.com/google-research/bert/blob/master/create_pretraining_data.py)**
   2. Image: The image is equally splited into 8*8 image patches, and for each patch we extract 2048 features with pretrained ResNet50 model.
 ```bash
 sh run_image_feature_extract.sh # refer this script
 ```
(** Note 1: Leverage Tensorflow(tf-slim) to extract the image features. In detail, we extract AvgPool2D (2048) as the patch features。**[https://github.com/google-research/tf-slim](https://github.com/google-research/tf-slim)。
** Note 2: For the masked image patches, the image featuers are set to zero**)

5. The text token features and image patch features are composed the input of FashionBERT.

Pretrained Model: https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/tutorial/fashion_bert/fashionbert_pretrain_model_fin.tar.gz

## Eval Extraction
Taking the Image2Text Task as example:

1. From the valid dataset, we randomly select 1000 products. For one product, we regard the image as the QUERY and regard its text as the GROUND TRUTH.
2. For one selected product, we randomly sample 100 other products under its sub-category. These texts from these sampled products are composed of the negative samples. Finally, for the image query of the selected product, we collect one ground truth text and 100 negative texts.
3. For one selected product, we score each <Text, Image> pair, including the ground truth text and sampled texts, with FashionBERT. We finally rank the texts of different products according the scores for each selected image query.
4.  According to the ranking, calculate the metrics.

```bash
# Eval Websites
wget https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/tutorial/fashion_bert/fashionbert-fashiongen-patch-eval-img2txt__097eaabf2d1e4464b88453bc7dfc8878
wget https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/tutorial/fashion_bert/fashionbert-fashiongen-patch-eval-img2txt__1b7320f883b6453e8922f520bac18e84
wget https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/tutorial/fashion_bert/fashionbert-fashiongen-patch-eval-img2txt__770ef9af0ab246dfb2269b9e008bc144
wget https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/tutorial/fashion_bert/fashionbert-fashiongen-patch-eval-txt2img__9d7082f64d0346fea770b66cdba0fcd2
wget https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/tutorial/fashion_bert/fashionbert-fashiongen-patch-eval-txt2img__c4aff1da32324da081af6324570c0bda
wget https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/tutorial/fashion_bert/fashionbert-fashiongen-patch-eval-txt2img__e928ee31b75940e88f1da64f133d9c4d
```

# Reference

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
