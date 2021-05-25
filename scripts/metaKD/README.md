# Meta-KD: A Meta Knowledge Distillation Framework for Language Model Compression across Domains


## How to build the dataset
1. The MNLI dataset can be found in this [link](https://cims.nyu.edu/~sbowman/multinli/) 
2. The Amazon Review dataset can be found in this [link](https://www.cs.jhu.edu/~mdredze/datasets/sentiment/index2.html)
one can randomly split the dataset as the paper mentions.


## How to run the code
1. Preprocess the data
```bash
$ sh run_meta_preprocess.sh
```

2. Meta-teacher learning
```bash
$ sh run_meta_teacher.sh
```

3. Meta-distillation
```bash
$ sh run_meta_distill.sh
```


If you use this code, please cite the following paper. Thanks.

```
@article{pan2021metakd,
  author    = {Haojie Pan and
               Chengyu Wang and
               Minghui Qiu and
               Yichang Zhang and
               Yaliang Li and
               Jun Huang},
  title     = {Meta-KD: A Meta Knowledge Distillation Framework for Language Model
               Compression across Domains},
  journal   = {ACL},
  year      = {2021}
}
```