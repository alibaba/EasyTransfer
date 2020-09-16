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
import tensorflow as tf
from easytransfer import preprocessors, model_zoo
from easytransfer.app_zoo.base import ApplicationModel
from easytransfer.evaluators import sequence_labeling_eval_metrics
from easytransfer.losses import sequence_labeling_loss
import easytransfer.layers as layers


class BaseSequenceLabeling(ApplicationModel):
    def __init__(self, **kwargs):
        """ Basic Sequence Labeling Model """
        super(BaseSequenceLabeling, self).__init__(**kwargs)

    @staticmethod
    def default_model_params():
        """ The default value of the Sequence Labeling Model """
        raise NotImplementedError

    def build_logits(self, features, mode=None):
        """ Building graph of the Sequence Labeling Model
        """
        raise NotImplementedError

    def build_loss(self, logits, labels):
        """ Building loss for training the Sequence Labeling Model
        """
        return sequence_labeling_loss(logits, labels, self.config.num_labels)

    def build_eval_metrics(self, logits, labels):
        """ Building evaluation metrics while evaluating

        Args:
            logits (`Tensor`): shape of [None, seq_length, num_labels]
            labels (`Tensor`): shape of [None, seq_length]
        Returns:
            ret_dict (`dict`): A dict with (`py_accuracy`, `py_micro_f1`, `py_macro_f1`) tf.metrics op
        """
        return sequence_labeling_eval_metrics(logits, labels, self.config.num_labels)

    def build_predictions(self, predict_output):
        """ Building  prediction dict of the Sequence Labeling Model

        Args:
            predict_output (`tuple`): (logits, _)
        Returns:
            ret_dict (`dict`): A dict with (`predictions`, `probabilities`, `logits`)
        """
        logits = predict_output[0]
        probabilities = tf.nn.softmax(logits, axis=-1)
        predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
        return {
            "predictions": predictions,
            "probabilities": probabilities,
            "logits": logits
        }


class BertSequenceLabeling(BaseSequenceLabeling):
    """ BERT Sequence Labeling Model

        .. highlight:: python
        .. code-block:: python

            default_param_dict = {
                "pretrain_model_name_or_path": "pai-bert-base-zh",
                "dropout_rate": 0.1
            }
    """
    def __init__(self, **kwargs):
        super(BertSequenceLabeling, self).__init__(**kwargs)

    @staticmethod
    def get_input_tensor_schema(sequence_length=128):
        return "input_ids:int:{},input_mask:int:{},segment_ids:int:{},label_ids:int:{},tok_to_orig_index:str:1".format(
            sequence_length, sequence_length, sequence_length, sequence_length)

    @staticmethod
    def get_received_tensor_schema(sequence_length=128):
        return "input_ids:int:{},input_mask:int:{},segment_ids:int:{}".format(
            sequence_length, sequence_length, sequence_length)

    @staticmethod
    def default_model_params():
        """ Get default model required parameters

        Returns:
            default_param_dict (`dict`): key/value pair of default model required parameters
        """
        default_param_dict = {
            "pretrain_model_name_or_path": "pai-bert-base-zh",
            "dropout_rate": 0.1
        }
        return default_param_dict

    def build_logits(self, features, mode=None):
        """ Building graph of BERT Sequence Labeling

        Args:
            features (`OrderedDict`): A dict mapping raw input to tensors
            mode (`bool): tell the model whether it is under training
        Returns:
            logits (`Tensor`): The output after the last dense layer. Shape of [None, sequence_length, num_labels]
            label_ids (`Tensor`): label_ids, shape of [None, sequence_length]
        """
        preprocessor = preprocessors.get_preprocessor(self.config.pretrain_model_name_or_path,
                                                      user_defined_config=self.config,
                                                      app_model_name="sequence_labeling_bert")
        input_ids, input_mask, segment_ids, label_ids, _ = preprocessor(features)

        bert_backbone = model_zoo.get_pretrained_model(self.config.pretrain_model_name_or_path)
        sequence_output, _ = bert_backbone([input_ids, input_mask, segment_ids], mode=mode)


        is_training = (mode == tf.estimator.ModeKeys.TRAIN)
        sequence_output = tf.layers.dropout(
            sequence_output, rate=self.config.dropout_rate, training=is_training)

        kernel_initializer = tf.glorot_uniform_initializer(seed=np.random.randint(10000), dtype=tf.float32)
        bias_initializer = tf.zeros_initializer
        logits = layers.Dense(self.config.num_labels,
                              kernel_initializer=kernel_initializer,
                              bias_initializer=bias_initializer,
                              name='app/ez_dense')(sequence_output)

        self.check_and_init_from_checkpoint(mode)
        return logits, label_ids