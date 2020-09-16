![image.png](https://intranetproxy.alipay.com/skylark/lark/0/2020/png/34639/1600266657075-c04a81af-c21a-4846-80ef-d97b1b7f9d6f.png#align=left&display=inline&height=159&margin=%5Bobject%20Object%5D&name=image.png&originHeight=608&originWidth=702&size=42626&status=done&style=none&width=184)
**EasyTransfer is designed to make the development of transfer learning in NLP applications easier.**
[![](https://intranetproxy.alipay.com/skylark/lark/0/2020/svg/34639/1600266548658-bed57cc0-4570-4083-b372-c981b648a549.svg#align=left&display=inline&height=24&margin=%5Bobject%20Object%5D&originHeight=20&originWidth=96&size=0&status=done&style=none&width=115)](https://huggingface.co/transformers/index.html)[![](https://intranetproxy.alipay.com/skylark/lark/0/2020/svg/34639/1600266553304-e6c2c66a-871e-4e12-b85f-bd051f342914.svg#align=left&display=inline&height=24&margin=%5Bobject%20Object%5D&originHeight=24&originWidth=137&size=0&status=done&style=none&width=137)](https://dsw-dev.data.aliyun.com/#/?fileUrl=https://pai-public-data.oss-cn-beijing.aliyuncs.com/easytransfer/easytransfer-quick_start.ipynb&fileName=easytransfer-quick_start.ipynb)


The literature has witnessed the success of applying deep Transfer Learning (TL) for many NLP applications, yet it not easy to build a simple and easy-to-use TL toolkit to achieve this goal. To bridge this gap, EasyTransfer is designed to make it easy to apply deep TL for NLP applications. It was developed in Alibaba in early 2017, and has been in the major BUs in Alibaba group and achieved very good results in 20+ businesses scenarios. It supports the mainstream pre-trained ModelZoo, including pre-trained language models (PLMs) and multi-modal models on the PAI platform, integrates the SOTA model for the mainstream NLP applications in AppZoo, and supports knowledge distillation for PLMs. The toolkit is convenient for users to quickly start model training, evaluation, offline prediction, and online deployment. It provides rich APIs to make the development of NLP and transfer learning easier.

# Main Features

- **Language model pre-training tool: **it supports a comprehensive pre-training tool for users to pre-train language models such as T5 and BERT. Based on the tool, the user can easily train a model to achieve great results in the benchmark leaderboards such as CLUE, GLUE, and SuperGLUE;
- **ModelZoo with rich and high-quality pre-trained models: **supports the Continual Pre-training and Fine-tuning of mainstream LM models such as BERT, ALBERT, RoBERTa, T5, etc. It also supports a multi-modal model FashionBERT developed using the fashion domain data in Alibaba;
- **AppZoo with rich and easy-to-use applications:** supports mainstream NLP applications and those models developed inside of Alibaba, e.g.: HCNN for text matching, and BERT-HAE for MRC.
- **Automatic knowledge distillation: **supports task-adaptive knowledge distillation to distill knowledge from a teacher model to a small task-specific student model. The resulting method is AdaBERT, which uses a neural architecture search method to find a task-specific architecture to compress the original BERT model. The compressed models are 12.7x to 29.3x faster than BERT in inference time and 11.5x to 17.0x smaller in terms of parameter size, and with comparable performance.
- **Easy-to-use and high-performance distributed strategy: **based on the in-house PAI features, it provides easy-to-use and high-performance distributed strategy for multiple CPU/GPU training.

# Architecture
![image.png](https://intranetproxy.alipay.com/skylark/lark/0/2020/png/34639/1597657331054-5bb31845-7d46-4cef-8518-f5576783fdf5.png#align=left&display=inline&height=357&margin=%5Bobject%20Object%5D&name=image.png&originHeight=713&originWidth=1492&size=182794&status=done&style=none&width=746)
# Quick Start


# Tutorials

- ModelZoo fine-tuning  (20+ pretrained models)
- FashionBERT-cross-modality pretrained model
- [Knowledge distillation including vanilla KD, BERT PKD, AdaBERT](https://www.yuque.com/easytransfer/itfpm9/kp1dtx)

# Reference
Tutorials：[https://yuque.antfin-inc.com/pai-innovative-algo/apx4dp/ig3owr](https://yuque.antfin-inc.com/pai-innovative-algo/apx4dp/ig3owr)
ModelZoo：[https://yuque.antfin-inc.com/pai-innovative-algo/apx4dp/yh01gk](https://yuque.antfin-inc.com/pai-innovative-algo/apx4dp/yh01gk)
AppZoo：[https://yuque.antfin-inc.com/pai-innovative-algo/apx4dp/gcaiqq](https://yuque.antfin-inc.com/pai-innovative-algo/apx4dp/gcaiqq)
API docs：[https://atp-modelzoo.oss-cn-hangzhou.aliyuncs.com/easy_transfer_docs/html/index.html](https://atp-modelzoo.oss-cn-hangzhou.aliyuncs.com/easy_transfer_docs/html/index.html)


# Contact Us
Scan the following QR codes to join Dingtalk discussion group. The group discussions are most in Chinese, but English is also welcomed.
![image.png](https://intranetproxy.alipay.com/skylark/lark/0/2020/png/34639/1600273487223-23b2d405-07b0-40d5-8c1f-14135c18720c.png#align=left&display=inline&height=352&margin=%5Bobject%20Object%5D&name=image.png&originHeight=1178&originWidth=1016&size=312154&status=done&style=none&width=304)




