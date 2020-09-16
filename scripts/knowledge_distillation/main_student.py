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

from easytransfer import base_model, layers, model_zoo, FLAGS
from easytransfer import preprocessors
from easytransfer.app_zoo.app_utils import get_reader_fn, get_writer_fn
from easytransfer.evaluators import classification_eval_metrics
from easytransfer.losses import build_kd_loss, build_kd_probes_loss, softmax_cross_entropy


class StudentNetwork(base_model):
    def __init__(self, **kwargs):
        """ Student Network for KD """
        super(StudentNetwork, self).__init__(**kwargs)

    def build_logits(self, features, mode=None):
        """ Building graph of KD Student

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

        if mode != tf.estimator.ModeKeys.PREDICT:
            teacher_logits, input_ids, input_mask, segment_ids, label_ids = preprocessor(features)
        else:
            teacher_logits, input_ids, input_mask, segment_ids = preprocessor(features)
            label_ids = None
        import pdb
        pdb.set_trace()

        teacher_n_layers = int(teacher_logits.shape[1]) / self.config.num_labels - 1
        self.teacher_logits = [teacher_logits[:, i * self.config.num_labels: (i + 1) * self.config.num_labels]
                               for i in range(teacher_n_layers + 1)]

        if self.config.train_probes:
            bert_model = bert_backbone.bert
            embedding_output = bert_model.embeddings([input_ids, segment_ids], training=is_training)
            attention_mask = layers.get_attn_mask_bert(input_ids, input_mask)
            all_hidden_outputs, all_att_outputs = bert_model.encoder(
                [embedding_output, attention_mask], training=is_training)

            # Get teacher Probes
            logits = layers.HiddenLayerProbes(self.config.num_labels,
                                              kernel_initializer=layers.get_initializer(0.02),
                                              name="probes")([embedding_output, all_hidden_outputs])
        else:
            _, pooled_output = bert_backbone([input_ids, input_mask, segment_ids], mode=mode)
            pooled_output = tf.layers.dropout(
                pooled_output, rate=self.config.dropout_rate, training=is_training)
            logits = layers.Dense(self.config.num_labels,
                                  kernel_initializer=layers.get_initializer(0.02),
                                  name='app/ez_dense')(pooled_output)
            logits = [logits]

        return logits, label_ids

    def build_loss(self, logits, labels):
        """ Building loss for KD Student
        """
        ce_loss = softmax_cross_entropy(labels, self.config.num_labels, logits[-1])
        last_layer_kd_loss = build_kd_loss(teacher_logits=self.teacher_logits[-1],
                                           student_logits=logits[-1],
                                           task_balance=0.5,
                                           distill_tempreture=2.0,
                                           labels=None,
                                           loss_type='mse')

        total_loss = ce_loss + last_layer_kd_loss

        if self.config.train_probes and len(logits) > 1:
            probes_kd_loss = build_kd_probes_loss(teacher_logits=self.teacher_logits,
                                                  student_logits=logits,
                                                  task_balance=0.5,
                                                  distill_tempreture=2.0,
                                                  labels=None,
                                                  loss_type='mse')
            total_loss += probes_kd_loss

        return total_loss

    def build_eval_metrics(self, logits, labels):
        return classification_eval_metrics(logits[-1], labels, self.config.num_labels)

    def build_predictions(self, output):
        logits = output[0][-1]
        predictions = dict()
        predictions["logits"] = logits
        predictions["predictions"] = tf.argmax(logits, axis=-1, output_type=tf.int32)
        return predictions


def train_and_evaluate():
    app = StudentNetwork()

    train_reader = get_reader_fn()(input_glob=app.config.train_input_fp,
                                   input_schema=app.config.input_schema,
                                   is_training=True,
                                   batch_size=app.config.train_batch_size)

    eval_reader = get_reader_fn()(input_glob=app.config.eval_input_fp,
                                  input_schema=app.config.input_schema,
                                  is_training=False,
                                  batch_size=app.config.eval_batch_size)

    app.run_train_and_evaluate(train_reader=train_reader,
                               eval_reader=eval_reader)

    tf.logging.info("Finished training")


def evaluate():
    app = StudentNetwork()

    eval_reader = get_reader_fn()(input_glob=app.config.eval_input_fp,
                                  input_schema=app.config.input_schema,
                                  is_training=False,
                                  batch_size=app.config.eval_batch_size)

    app.run_evaluate(reader=eval_reader, checkpoint_path=app.config.eval_ckpt_path)

    tf.logging.info("Finished training")


def predict():
    app = StudentNetwork()
    reader = get_reader_fn()(input_glob=app.config.predict_input_fp,
                             input_schema=app.config.input_schema,
                             is_training=False,
                             batch_size=app.config.predict_batch_size)
    writer = get_writer_fn()(output_glob=app.config.predict_output_fp,
                             output_schema=app.config.output_schema)
    app.run_predict(reader=reader, writer=writer, checkpoint_path=app.config.predict_checkpoint_path)


if __name__ == "__main__":
    starttime = datetime.datetime.now()
    if FLAGS.mode == "train_and_evaluate":
        train_and_evaluate()
    elif FLAGS.mode == "evaluate":
        evaluate()
    elif FLAGS.mode == "predict":
        predict()
    else:
        raise RuntimeError("invalid mode")

    endtime = datetime.datetime.now()
    tf.logging.info("Finished in {} seconds".format((endtime - starttime).seconds))
