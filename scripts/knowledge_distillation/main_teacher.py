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

import datetime
import tensorflow as tf

import easytransfer.layers as layers
from easytransfer import preprocessors, model_zoo, FLAGS, base_model
from easytransfer.losses import softmax_cross_entropy
from easytransfer.evaluators import classification_eval_metrics, teacher_probes_eval_metrics
from easytransfer.app_zoo.app_utils import get_reader_fn, get_writer_fn


class TeacherNetwork(base_model):
    def __init__(self, **kwargs):
        """ Teacher Network for KD """
        super(TeacherNetwork, self).__init__(**kwargs)

    def build_logits(self, features, mode=None):
        """ Building graph of KD Teacher

        Args:
            features (`OrderedDict`): A dict mapping raw input to tensors
            mode (`bool): tell the model whether it is under training
        Returns:
            logits (`list`): logits for all the layers, list of shape of [None, num_labels]
            label_ids (`Tensor`): label_ids, shape of [None]
        """
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)
        preprocessor = preprocessors.get_preprocessor(self.config.pretrain_model_name_or_path,
                                                      user_defined_config=self.config)
        bert_backbone = model_zoo.get_pretrained_model(self.config.pretrain_model_name_or_path)

        # Serialize raw text to get input tensors
        input_ids, input_mask, segment_ids, label_id = preprocessor(features)

        if self.config.train_probes:
            # Get BERT all hidden states
            bert_model = bert_backbone.bert
            embedding_output = bert_model.embeddings([input_ids, segment_ids], training=is_training)
            attention_mask = layers.get_attn_mask_bert(input_ids, input_mask)
            all_hidden_outputs, all_att_outputs = bert_model.encoder(
                [embedding_output, attention_mask], training=is_training)

            # Get teacher Probes
            logits = layers.HiddenLayerProbes(self.config.num_labels,
                                              kernel_initializer=layers.get_initializer(0.02),
                                              name="probes")([embedding_output, all_hidden_outputs])
            self.tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                           "probes/")
        else:
            _, pooled_output = bert_backbone([input_ids, input_mask, segment_ids], mode=mode)
            pooled_output = tf.layers.dropout(
                pooled_output, rate=self.config.dropout_rate, training=is_training)
            logits = layers.Dense(self.config.num_labels,
                                  kernel_initializer=layers.get_initializer(0.02),
                                  name='app/ez_dense')(pooled_output)
            logits = [logits]

        if mode == tf.estimator.ModeKeys.PREDICT:
            return {
                "input_ids": input_ids,
                "input_mask": input_mask,
                "segment_ids": segment_ids,
                "label_id": label_id,
                "logits": tf.concat(logits, axis=-1)
            }
        else:
            return logits, label_id

    def build_loss(self, logits, labels):
        """ Building loss for KD Teacher
        """
        loss = 0.0
        for layer_logits in logits:
            loss += softmax_cross_entropy(labels, self.config.num_labels, layer_logits)
        return loss

    def build_eval_metrics(self, logits, labels):
        """ Building evaluation metrics while evaluating

        Args:
            logits (`Tensor`): list of tensors shape of [None, num_labels]
            labels (`Tensor`): shape of [None]
        Returns:
            ret_dict (`dict`): A dict of each layer accuracy tf.metrics op
        """
        if self.config.train_probes:
            return teacher_probes_eval_metrics(logits, labels, self.config.num_labels)
        else:
            return classification_eval_metrics(logits[0], labels, self.config.num_labels)

    def build_predictions(self, predict_output):
        """ Building prediction dict of KD Teacher

        Args:
            predict_output (`dict`): return value of build_logits
        Returns:
            predict_output (`dict`): A dict for the output to the file
        """
        return predict_output


def train_and_evaluate_on_the_fly():
    app = TeacherNetwork()

    train_reader = get_reader_fn()(input_glob=app.config.train_input_fp,
                                   input_schema=app.config.input_schema,
                                   is_training=True,
                                   batch_size=app.config.train_batch_size)

    eval_reader = get_reader_fn()(input_glob=app.config.eval_input_fp,
                                  input_schema=app.config.input_schema,
                                  is_training=False,
                                  batch_size=app.config.eval_batch_size)

    app.run_train_and_evaluate(train_reader=train_reader, eval_reader=eval_reader)

    tf.logging.info("Finished training")


def evaluate_on_the_fly():
    app = TeacherNetwork()

    eval_reader = get_reader_fn()(input_glob=app.config.eval_input_fp,
                                  input_schema=app.config.input_schema,
                                  is_training=False,
                                  batch_size=app.config.eval_batch_size)

    app.run_evaluate(reader=eval_reader, checkpoint_path=app.config.eval_ckpt_path)

    tf.logging.info("Finished training")


def predict_on_the_fly():
    app = TeacherNetwork()
    reader = get_reader_fn()(input_glob=app.config.predict_input_fp,
                             input_schema=app.config.input_schema,
                             is_training=False,
                             batch_size=app.config.predict_batch_size)
    writer = get_writer_fn()(output_glob=app.config.predict_output_fp,
                             output_schema=app.config.output_schema)
    app.run_predict(reader=reader, writer=writer, checkpoint_path=app.config.predict_checkpoint_path)


if __name__ == "__main__":
    starttime = datetime.datetime.now()
    if FLAGS.mode == "train_and_evaluate_on_the_fly":
        train_and_evaluate_on_the_fly()
    elif FLAGS.mode == "evaluate_on_the_fly":
        evaluate_on_the_fly()
    elif FLAGS.mode == "predict_on_the_fly":
        predict_on_the_fly()
    else:
        raise RuntimeError("invalid mode")

    endtime = datetime.datetime.now()
    tf.logging.info("Finished in {} seconds".format((endtime - starttime).seconds))
