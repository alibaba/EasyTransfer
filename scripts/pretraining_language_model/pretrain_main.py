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

import json
import multiprocessing
import random
import shutil
from glob import glob
import os
import tensorflow as tf
from tqdm import tqdm

from easytransfer import base_model, FLAGS
from easytransfer import model_zoo
from easytransfer import layers
from easytransfer.losses import softmax_cross_entropy
from easytransfer import preprocessors
from easytransfer.datasets import BundleTFRecordReader, OdpsTableReader
from easytransfer.evaluators import masked_language_model_eval_metrics, next_sentence_prediction_eval_metrics
from easytransfer.losses import masked_language_model_loss, next_sentence_prediction_loss
from easytransfer.preprocessors.tokenization import FullTokenizer
from pretrain_utils import create_training_instances, write_instance_to_file, PretrainConfig, _APP_FLAGS

class Pretrain(base_model):
    def __init__(self, **kwargs):
        super(Pretrain, self).__init__(**kwargs)
        self.user_defined_config = kwargs.get("user_defined_config", None)

    def build_logits(self, features, mode=None):
        bert_preprocessor = preprocessors.get_preprocessor(self.pretrain_model_name_or_path,
                                                           app_model_name="pretrain_language_model",
                                                           user_defined_config=self.user_defined_config)

        if _APP_FLAGS.distribution_strategy == "WhaleStrategy" or \
                self.config.distribution_strategy == "WhaleStrategy":
            tf.logging.info("*********Calling Whale Encoder***********")
            model = model_zoo.get_pretrained_model(self.pretrain_model_name_or_path, enable_whale=True,
                                                   input_sequence_length=_APP_FLAGS.input_sequence_length)
        else:
            model = model_zoo.get_pretrained_model(self.pretrain_model_name_or_path,
                                                   input_sequence_length=_APP_FLAGS.input_sequence_length)

        if _APP_FLAGS.loss == "mlm+nsp" or _APP_FLAGS.loss == "mlm+sop":
            input_ids, input_mask, segment_ids, masked_lm_positions, \
            masked_lm_ids, masked_lm_weights, next_sentence_labels = bert_preprocessor(features)

            lm_logits, nsp_logits, _ = model([input_ids, input_mask, segment_ids],
                                             masked_lm_positions=masked_lm_positions,
                                             output_features=False,
                                             mode=mode)

            return (lm_logits, nsp_logits), (masked_lm_ids, masked_lm_weights, next_sentence_labels)

        elif _APP_FLAGS.loss == "mlm":
            input_ids, input_mask, segment_ids, masked_lm_positions, \
            masked_lm_ids, masked_lm_weights = bert_preprocessor(features)

            lm_logits, _, _ = model([input_ids, input_mask, segment_ids],
                                    masked_lm_positions=masked_lm_positions,
                                    output_features=False,
                                    mode=mode)

            return lm_logits, (masked_lm_ids, masked_lm_weights)

    def build_loss(self, logits, labels):
        if _APP_FLAGS.loss == "mlm":
            lm_logits = logits
            masked_lm_ids, masked_lm_weights = labels
            masked_lm_loss = masked_language_model_loss(lm_logits, masked_lm_ids, masked_lm_weights,
                                                        _APP_FLAGS.vocab_size)
            return masked_lm_loss

        elif _APP_FLAGS.loss == "mlm+nsp" or _APP_FLAGS.loss == "mlm+sop":

            lm_logits, nsp_logits = logits
            masked_lm_ids, masked_lm_weights, nx_sent_labels = labels

            masked_lm_loss = masked_language_model_loss(lm_logits, masked_lm_ids, masked_lm_weights,
                                                        _APP_FLAGS.vocab_size)
            nsp_loss = next_sentence_prediction_loss(nsp_logits, nx_sent_labels)

            return masked_lm_loss + nsp_loss

    def build_eval_metrics(self, logits, labels):
        if _APP_FLAGS.loss == "mlm+nsp" or _APP_FLAGS.loss == "mlm+sop":
            lm_logits, nsp_logits = logits
            masked_lm_ids, masked_lm_weights, next_sentence_labels = labels
            mlm_metrics = masked_language_model_eval_metrics(lm_logits, masked_lm_ids, masked_lm_weights,
                                                             _APP_FLAGS.vocab_size)
            nsp_metrics = next_sentence_prediction_eval_metrics(nsp_logits, next_sentence_labels)
            return mlm_metrics.update(nsp_metrics)

        elif _APP_FLAGS.loss == "mlm":
            lm_logits = logits
            masked_lm_ids, masked_lm_weights = labels
            mlm_metrics = masked_language_model_eval_metrics(lm_logits, masked_lm_ids, masked_lm_weights,
                                                             _APP_FLAGS.vocab_size)
            return mlm_metrics

class PretrainMultitask(base_model):
    def __init__(self, **kwargs):
        super(PretrainMultitask, self).__init__(**kwargs)
        self.user_defined_config = kwargs.get("user_defined_config", None)

    def build_logits(self, features, mode=None):
        bert_preprocessor = preprocessors.get_preprocessor(self.pretrain_model_name_or_path,
                                                           app_model_name="pretrain_language_model",
                                                           user_defined_config=self.user_defined_config)

        if _APP_FLAGS.distribution_strategy == "WhaleStrategy" or \
                self.config.distribution_strategy == "WhaleStrategy":
            tf.logging.info("*********Calling Whale Encoder***********")
            model = model_zoo.get_pretrained_model(self.pretrain_model_name_or_path, enable_whale=True,
                                                   input_sequence_length=_APP_FLAGS.input_sequence_length)
        else:
            model = model_zoo.get_pretrained_model(self.pretrain_model_name_or_path,
                                                   input_sequence_length=_APP_FLAGS.input_sequence_length)

        if _APP_FLAGS.loss == "mlm+nsp" or _APP_FLAGS.loss == "mlm+sop":
            input_ids, input_mask, segment_ids, masked_lm_positions, \
            masked_lm_ids, masked_lm_weights, next_sentence_labels = bert_preprocessor(features)

            lm_logits, nsp_logits, _ = model([input_ids, input_mask, segment_ids],
                                             masked_lm_positions=masked_lm_positions,
                                             output_features=False,
                                             mode=mode)

            return (lm_logits, nsp_logits), (masked_lm_ids, masked_lm_weights, next_sentence_labels)

        elif _APP_FLAGS.loss == "mlm":

            task_1_dense = layers.Dense(2,
                                  kernel_initializer=layers.get_initializer(0.02),
                                  name='task_1_dense')

            input_ids, input_mask, segment_ids, masked_lm_positions, \
            masked_lm_ids, masked_lm_weights, task_1_label = bert_preprocessor(features)

            lm_logits, _, pooled_output = model([input_ids, input_mask, segment_ids],
                                    masked_lm_positions=masked_lm_positions,
                                    output_features=False,
                                    mode=mode)

            task_1_logits = task_1_dense(pooled_output)

            return (lm_logits, task_1_logits), (masked_lm_ids, masked_lm_weights, task_1_label)

    def build_loss(self, logits, labels):
        if _APP_FLAGS.loss == "mlm":
            lm_logits, task_1_logits = logits
            masked_lm_ids, masked_lm_weights, task_1_label = labels
            masked_lm_loss = masked_language_model_loss(lm_logits, masked_lm_ids, masked_lm_weights,
                                                        _APP_FLAGS.vocab_size)

            task_1_loss = softmax_cross_entropy(task_1_label, 2, task_1_logits)

            return masked_lm_loss + task_1_loss

        elif _APP_FLAGS.loss == "mlm+nsp" or _APP_FLAGS.loss == "mlm+sop":

            lm_logits, nsp_logits = logits
            masked_lm_ids, masked_lm_weights, nx_sent_labels = labels

            masked_lm_loss = masked_language_model_loss(lm_logits, masked_lm_ids, masked_lm_weights,
                                                        _APP_FLAGS.vocab_size)
            nsp_loss = next_sentence_prediction_loss(nsp_logits, nx_sent_labels)

            return masked_lm_loss + nsp_loss

    def build_eval_metrics(self, logits, labels):
        if _APP_FLAGS.loss == "mlm+nsp" or _APP_FLAGS.loss == "mlm+sop":
            lm_logits, nsp_logits = logits
            masked_lm_ids, masked_lm_weights, next_sentence_labels = labels
            mlm_metrics = masked_language_model_eval_metrics(lm_logits, masked_lm_ids, masked_lm_weights,
                                                             _APP_FLAGS.vocab_size)
            nsp_metrics = next_sentence_prediction_eval_metrics(nsp_logits, next_sentence_labels)
            return mlm_metrics.update(nsp_metrics)

        elif _APP_FLAGS.loss == "mlm":
            lm_logits = logits
            masked_lm_ids, masked_lm_weights = labels
            mlm_metrics = masked_language_model_eval_metrics(lm_logits, masked_lm_ids, masked_lm_weights,
                                                             _APP_FLAGS.vocab_size)
            return mlm_metrics


def run(mode):
    if FLAGS.config is None:
        config_json = {
            "model_type": _APP_FLAGS.model_type,
            "vocab_size": _APP_FLAGS.vocab_size,
            "hidden_size": _APP_FLAGS.hidden_size,
            "intermediate_size": _APP_FLAGS.intermediate_size,
            "num_hidden_layers": _APP_FLAGS.num_hidden_layers,
            "max_position_embeddings": 512,
            "num_attention_heads": _APP_FLAGS.num_attention_heads,
            "type_vocab_size": 2
        }

        if not tf.gfile.Exists(_APP_FLAGS.model_dir):
            tf.gfile.MkDir(_APP_FLAGS.model_dir)

        # Pretrain from scratch
        if _APP_FLAGS.pretrain_model_name_or_path is None:
            if not tf.gfile.Exists(_APP_FLAGS.model_dir + "/config.json"):
                with tf.gfile.GFile(_APP_FLAGS.model_dir + "/config.json", mode='w') as f:
                    json.dump(config_json, f)

            shutil.copy2(_APP_FLAGS.vocab_fp, _APP_FLAGS.model_dir)
            if _APP_FLAGS.spm_model_fp is not None:
                shutil.copy2(_APP_FLAGS.spm_model_fp, _APP_FLAGS.model_dir)

        config = PretrainConfig()
        if _APP_FLAGS.do_multitaks_pretrain:
            app = PretrainMultitask(user_defined_config=config)
        else:
            app = Pretrain(user_defined_config=config)
    else:
        if _APP_FLAGS.do_multitaks_pretrain:
            app = PretrainMultitask()
        else:
            app = Pretrain()

    if "train" in mode:
        if _APP_FLAGS.data_reader == 'tfrecord':
            train_reader = BundleTFRecordReader(input_glob=app.train_input_fp,
                                                is_training=True,
                                                shuffle_buffer_size=1024,
                                                worker_hosts=FLAGS.worker_hosts,
                                                task_index=FLAGS.task_index,
                                                input_schema=app.input_schema,
                                                batch_size=app.train_batch_size)
        elif _APP_FLAGS.data_reader == 'odps':
            tf.logging.info("***********Reading Odps Table *************")
            worker_id = FLAGS.task_index
            num_workers = len(FLAGS.worker_hosts.split(","))

            train_reader = OdpsTableReader(input_glob=app.train_input_fp,
                                           is_training=True,
                                           shuffle_buffer_size=1024,
                                           input_schema=app.input_schema,
                                           slice_id=worker_id,
                                           slice_count=num_workers,
                                           batch_size=app.train_batch_size)

    if mode == "train_and_evaluate":
        if _APP_FLAGS.data_reader == 'tfrecord':
            eval_reader = BundleTFRecordReader(input_glob=app.eval_input_fp,
                                               is_training=False,
                                               shuffle_buffer_size=1024,
                                               worker_hosts=FLAGS.worker_hosts,
                                               task_index=FLAGS.task_index,
                                               input_schema=app.input_schema,
                                               batch_size=app.eval_batch_size)

        elif _APP_FLAGS.data_reader == 'odps':
            eval_reader = OdpsTableReader(input_glob=app.train_input_fp,
                                          is_training=False,
                                          shuffle_buffer_size=1024,
                                          input_schema=app.input_schema,
                                          slice_id=worker_id,
                                          slice_count=num_workers,
                                          batch_size=app.train_batch_size)

        app.run_train_and_evaluate(train_reader=train_reader, eval_reader=eval_reader)

    elif mode == "train":
        app.run_train(reader=train_reader)

    elif mode == "evaluate":
        if _APP_FLAGS.data_reader == 'tfrecord':
            eval_reader = BundleTFRecordReader(input_glob=app.eval_input_fp,
                                               is_training=False,
                                               shuffle_buffer_size=1024,
                                               worker_hosts=FLAGS.worker_hosts,
                                               task_index=FLAGS.task_index,
                                               input_schema=app.input_schema,
                                               batch_size=app.eval_batch_size)

        elif _APP_FLAGS.data_reader == 'odps':
            eval_reader = OdpsTableReader(input_glob=app.train_input_fp,
                                          is_training=False,
                                          shuffle_buffer_size=1024,
                                          input_schema=app.input_schema,
                                          slice_id=worker_id,
                                          slice_count=num_workers,
                                          batch_size=app.train_batch_size)

        ckpts = set()
        with tf.gfile.GFile(os.path.join(app.config.model_dir, "checkpoint"), mode='r') as reader:
            for line in reader:
                line = line.strip()
                line = line.replace("oss://", "")
                ckpts.add(int(line.split(":")[1].strip().replace("\"", "").split("/")[-1].replace("model.ckpt-", "")))

        ckpts.remove(0)
        writer = tf.summary.FileWriter(os.path.join(app.config.model_dir, "eval_output"))
        for ckpt in sorted(ckpts):
            checkpoint_path = os.path.join(app.config.model_dir, "model.ckpt-" + str(ckpt))
            tf.logging.info("checkpoint_path is {}".format(checkpoint_path))
            ret_metrics = app.run_evaluate(reader=eval_reader,
                             checkpoint_path=checkpoint_path)

            global_step = ret_metrics['global_step']

            eval_masked_lm_accuracy = tf.Summary()
            eval_masked_lm_accuracy.value.add(tag='masked_lm_valid_accuracy', simple_value=ret_metrics['eval_masked_lm_accuracy'])

            eval_masked_lm_loss = tf.Summary()
            eval_masked_lm_loss.value.add(tag='masked_lm_valid_loss', simple_value=ret_metrics['eval_masked_lm_loss'])

            writer.add_summary(eval_masked_lm_accuracy, global_step)
            writer.add_summary(eval_masked_lm_loss, global_step)

        writer.close()

def run_preprocess(input_file, output_file):
    rng = random.Random(12345)

    if _APP_FLAGS.tokenizer == "wordpiece":
        tokenizer = FullTokenizer(vocab_file=_APP_FLAGS.vocab_fp)

    elif _APP_FLAGS.tokenizer == "sentencepiece":
        tokenizer = FullTokenizer(spm_model_file=_APP_FLAGS.spm_model_fp)

    instances = create_training_instances(
        input_file, tokenizer, _APP_FLAGS.max_seq_length, _APP_FLAGS.dupe_factor,
        _APP_FLAGS.short_seq_prob, _APP_FLAGS.masked_lm_prob, _APP_FLAGS.max_predictions_per_seq,
        _APP_FLAGS.do_whole_word_mask,
        rng)

    write_instance_to_file(instances, tokenizer, FLAGS.max_seq_length,
                           FLAGS.max_predictions_per_seq, output_file)


def preprocess():
    po = multiprocessing.Pool(_APP_FLAGS.num_threads)
    if not os.path.exists(_APP_FLAGS.output_dir):
        os.makedirs(_APP_FLAGS.output_dir)

    for input_file in tqdm(glob(_APP_FLAGS.input_dir + "/*.txt")):
        file_name = input_file.split("/")[-1].replace(".txt", ".tfrecord")
        output_file = os.path.join(_APP_FLAGS.output_dir, file_name)
        po.apply_async(func=run_preprocess, args=(input_file, output_file))
    po.close()
    po.join()


def main():
    if FLAGS.mode == "train_and_evaluate" or FLAGS.mode == "train" or FLAGS.mode == "evaluate":
        run(FLAGS.mode)
    elif FLAGS.mode == "preprocess":
        preprocess()


if __name__ == "__main__":
    main()
