# coding=utf-8
# Copyright (c) 2019 Alibaba PAI team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
from collections import namedtuple
from sklearn import metrics
import tensorflow as tf
from easytransfer import FLAGS, Config

_app_flags = tf.app.flags
_app_flags.DEFINE_string("input_dir", default=None, help='')
_app_flags.DEFINE_string("output_dir", default=None, help='')
_app_flags.DEFINE_integer("num_threads", default=None, help='')
_app_flags.DEFINE_string("data_format", default=None, help='')
_app_flags.DEFINE_string("tokenizer", default="wordpiece", help='')
_app_flags.DEFINE_string("spm_model_fp", None, "The model file for sentence piece tokenization.")
_app_flags.DEFINE_string("vocab_fp", None, "The model file for word piece tokenization.")
_app_flags.DEFINE_bool("do_whole_word_mask", True,
                       "Whether to use whole word masking rather than per-WordPiece masking.")
_app_flags.DEFINE_bool("do_chinese_whole_word_mask", False,
                       "Whether to use whole word masking rather than per-WordPiece masking.")
_app_flags.DEFINE_bool("random_next_sentence", False, "")
_app_flags.DEFINE_integer("dupe_factor", 40, "Number of times to duplicate the input data (with different masks).")
_app_flags.DEFINE_integer("max_seq_length", 512, "Maximum sequence length.")
_app_flags.DEFINE_integer("max_predictions_per_seq", 20, "Maximum number of masked LM predictions per sequence.")
_app_flags.DEFINE_float("masked_lm_prob", 0.15, "Masked LM probability.")
_app_flags.DEFINE_float("short_seq_prob", 0.1,
                        "Probability of creating sequences which are shorter than the maximum length.")

_app_flags.DEFINE_string("loss", None, "loss")
_app_flags.DEFINE_string("model_type", None, "model_type")
_app_flags.DEFINE_string("input_schema", default=None, help='input_schema')
_app_flags.DEFINE_string("pretrain_model_name_or_path", default=None, help='pretrain_model_name_or_path')
_app_flags.DEFINE_string("train_input_fp", default=None, help='train_input_fp')
_app_flags.DEFINE_string("eval_input_fp", default=None, help='eval_input_fp')
_app_flags.DEFINE_string("predict_input_fp", default=None, help='predict_input_fp')
_app_flags.DEFINE_string("predict_checkpoint_path", default=None, help='predict_checkpoint_path')
_app_flags.DEFINE_integer("train_batch_size", default=128, help='train_batch_size')
_app_flags.DEFINE_integer("predict_batch_size", default=128, help='predict_batch_size')
_app_flags.DEFINE_integer("num_epochs", default=1, help='num_epochs')
_app_flags.DEFINE_string("model_dir", default='', help='model_dir')
_app_flags.DEFINE_float("learning_rate", 1e-4, "learning_rate")
_app_flags.DEFINE_integer("hidden_size", default=None, help='')
_app_flags.DEFINE_integer("intermediate_size", default=None, help='')
_app_flags.DEFINE_integer("num_hidden_layers", default=None, help='')
_app_flags.DEFINE_integer("num_attention_heads", default=None, help='')
_APP_FLAGS = _app_flags.FLAGS

class PretrainConfig(Config):
    def __init__(self):

        if _APP_FLAGS.pretrain_model_name_or_path is not None:
            pretrain_model_name_or_path = _APP_FLAGS.pretrain_model_name_or_path
        else:
            pretrain_model_name_or_path = _APP_FLAGS.model_dir + "/model.ckpt"

        config_json = {
            "preprocess_config": {
                "input_schema": _APP_FLAGS.input_schema,
                "output_schema": None
            },
            'model_config': {
                'pretrain_model_name_or_path': pretrain_model_name_or_path
            },
            'train_config': {
                "train_input_fp": _APP_FLAGS.train_input_fp,
                "train_batch_size": _APP_FLAGS.train_batch_size,
                "num_epochs": _APP_FLAGS.num_epochs,
                "model_dir": _APP_FLAGS.model_dir,
                "save_steps": None,
                "optimizer_config": {
                    "optimizer": "adam",
                    "weight_decay_ratio": 0,
                    "warmup_ratio": 0.1,
                    "learning_rate": _APP_FLAGS.learning_rate,
                },
                "distribution_config": {
                    "distribution_strategy": "MirroredStrategy"
                }
            },
            "evaluate_config": {
                "eval_input_fp": _APP_FLAGS.eval_input_fp,
                "eval_batch_size": 16,
                "num_eval_steps": 1000
            },
            "predict_config": {
                "predict_input_fp": _APP_FLAGS.predict_input_fp,
                "predict_output_fp": None,
                "predict_checkpoint_path":_APP_FLAGS.predict_checkpoint_path,
                "predict_batch_size": _APP_FLAGS.predict_batch_size,
                "output_schema":None
            },

        }
        config_json["worker_hosts"] = FLAGS.worker_hosts
        config_json["task_index"] = FLAGS.task_index
        config_json["job_name"] = FLAGS.job_name
        config_json["num_gpus"] = FLAGS.workerGPU
        config_json["num_workers"] = FLAGS.workerCount
        super(PretrainConfig, self).__init__(mode=FLAGS.mode, config_json=config_json)


Doc = namedtuple('Doc', ['id', 'score', 'label']) 

def read_batch_result_file(filename, type):
    # for accuracy
    example_cnt = 0
    true_pred_cnt = 0
    # for rank@k
    query_dict = {}
    # for AUC
    y_preds = []
    y_trues = []
    with open(filename, 'r') as fin:
        while True:
            line = fin.readline()
            if not line:
                break
            items = line.split('\001')
            if len(items) != 5:
                print("Line errors {}".format(line))
            
            # text_prod_ids, imag_prod_ids, prod_img_ids, labels, nsp_logits
            text_prod_ids  = items[0].replace('[', '').replace(']', '').split(',')
            image_prod_ids = items[1].replace('[', '').replace(']', '').split(',')
            prod_img_ids   = items[2].replace('[', '').replace(']', '').split(',')
            labels         = np.array([int(x) for x in items[3].replace('[', '').replace(']', '').split(',')], dtype=np.int32)
            predictions    = np.array([float(x) for x in items[4].replace('[', '').replace(']', '').split(',')], dtype=np.float32).reshape((-1,2))
            # print(predictions.shape, len(text_prod_ids))

            assert len(text_prod_ids) == len(image_prod_ids), len(text_prod_ids) == len(prod_img_ids)
            assert len(text_prod_ids) == labels.shape[0], len(text_prod_ids) == predictions.shape[0]
            example_cnt = example_cnt + len(text_prod_ids)

            # step 1: accuracy
            pred_labels = np.argmax(predictions, axis=1)
            true_pred_cnt += np.sum(pred_labels == labels)

            # step 2: rank@K
            for idx in range(len(text_prod_ids)):
                query_id = text_prod_ids[idx] if type == 'txt2img' else image_prod_ids[idx]
                doc_id   = image_prod_ids[idx] if type == 'txt2img' else text_prod_ids[idx]
                dscore   = predictions[idx, 1]
                dlabel   = labels[idx]
                doc      = Doc(id=doc_id, score = dscore, label=dlabel)
                if query_id in query_dict:
                    query_dict[query_id].append(doc)
                else:
                    docs = []
                    docs.append(doc)
                    query_dict[query_id] = docs
            
            # step 3: AUC
            for idx in range(len(text_prod_ids)):
                y_preds.append(predictions[idx, 1])
                y_trues.append(labels[idx])
    
    return example_cnt, true_pred_cnt, query_dict, y_preds, y_trues
        
def prediction_summary(results):
    if results is None:
        return None
    
    example_cnt, true_pred_cnt, query_dict, y_preds, y_trues = results
    # step 1: accuracy
    print("Accuracy: ", float(true_pred_cnt) / (float(example_cnt) + 1e-5))

    # step 2: rank @ K
    query_sorted_dict = {}
    for query_id in query_dict.keys():
        query_dict[query_id].sort(key=lambda x: x.score, reverse=True)
    # for query_id in query_dict.keys():
    #     print("Query_id, ", query_id, " after sort: ", query_dict[query_id])
    Ks = [1, 5, 10, 100]
    for k in Ks:
        print("========== Rank @ {} evaluation ============".format(k))
        fount_at_top_k = 0
        for query_id in query_dict.keys():
            query_sorted_docs = query_dict[query_id]
            tmp_range = k if k < len(query_sorted_docs) else len(query_sorted_docs)
            for idx in range(tmp_range):
                if query_sorted_docs[idx].label:
                    fount_at_top_k += 1
                    break

        print("========== Rank @ {} is {} ============".format(k, float(fount_at_top_k)/float(len(query_dict.keys()) + 1e-5)))
    
    # step 3: AUC
    test_auc = metrics.roc_auc_score(y_trues,y_preds) #验证集上的auc值
    print("==== AUC {} ====".format(test_auc))

'''
    filename: filename
    type    : img2txt or txt2img
'''
def prediction_analysis(filename, type='img2txt'):
    file_name = './logs_bak/fashiongen-tb/eval_txt2img_results.txt'
    print("To analysis file : ", file_name)
    results = read_batch_result_file(filename, type)
    # print(results)
    prediction_summary(results)

def append_to_file(filename, content):
    with open(filename, 'a') as fout:
        fout.write(content)

def delete_exists_file(filename):
    if tf.gfile.Exists(filename):
        print("file {} found, and deleted".format(filename))
        tf.gfile.remove(filename)
