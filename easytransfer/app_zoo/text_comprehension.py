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


import tensorflow as tf
from easytransfer import preprocessors, model_zoo
from easytransfer.app_zoo.base import ApplicationModel
from easytransfer.evaluators import comprehension_eval_metrics
from easytransfer.losses import comprehension_loss
from easytransfer.model_zoo.modeling_hae import BertHAEPretrainedModel


class BaseTextComprehension(ApplicationModel):
    def __init__(self, **kwargs):
        """ Basic Text Comprehension Model """
        super(BaseTextComprehension, self).__init__(**kwargs)

    @staticmethod
    def default_model_params():
        """ The default value of the Text Comprehension Model """
        raise NotImplementedError

    def build_logits(self, features, mode=None):
        """ Building graph of the Text Comprehension Model
        """
        raise NotImplementedError

    def build_loss(self, logits, labels):
        """ Building loss for training the Text Comprehension Model
        """
        return comprehension_loss(logits, labels)

    def build_eval_metrics(self, logits, labels):
        """ Building evaluation metrics while evaluating

        Args:
            logits (`Tensor`): shape of [None, num_labels]
            labels (`Tensor`): shape of [None]
        Returns:
            ret_dict (`dict`): A dict with (`py_accuracy`, `py_micro_f1`, `py_macro_f1`) tf.metrics op
        """
        return comprehension_eval_metrics(logits, labels)

    def build_predictions(self, predict_output):
        """ Building  prediction dict of the Text Comprehension Model

        Args:
            predict_output (`tuple`): (logits, _)
        Returns:
            ret_dict (`dict`): A dict with (`predictions`, `probabilities`, `logits`)
        """
        (start_logits, end_logits), _ = predict_output
        ret_dict = {
            "start_logits": start_logits,
            "end_logits": end_logits
        }
        return ret_dict


class BERTTextComprehension(BaseTextComprehension):
    """ BERT Text Comprehension Model

        .. highlight:: python
        .. code-block:: python

            default_param_dict = {
                "pretrain_model_name_or_path": "pai-bert-base-zh",
                "multi_label": False,
                "num_labels": 2,
                "max_query_length": 64,
                "doc_stride": 128,
            }
    """
    def __init__(self, **kwargs):
        super(BERTTextComprehension, self).__init__(**kwargs)

    @staticmethod
    def get_input_tensor_schema(sequence_length=64):
        return "input_ids:int:{},input_mask:int:{},segment_ids:int:{}," \
               "start_position:int:1,end_position:int:1".format(
            sequence_length, sequence_length, sequence_length)

    @staticmethod
    def get_received_tensor_schema():
        return "input_ids:int:64,input_mask:int:64,segment_ids:int:64"

    @staticmethod
    def default_model_params():
        """ Get default model required parameters

        Returns:
            default_param_dict (`dict`): key/value pair of default model required parameters
        """
        default_param_dict = {
            "pretrain_model_name_or_path": "pai-bert-base-zh",
            "num_labels": 2,
            "max_query_length": 64,
            "doc_stride": 128,
        }
        return default_param_dict

    def build_logits(self, features, mode=None):
        """ Building graph of BERT Text Comprehension

        Args:
            features (`OrderedDict`): A dict mapping raw input to tensors
            mode (`bool): tell the model whether it is under training
        Returns:
            logits (`tuple`): (start_logits, end_logits), The output after the last dense layer. Two tensor of
            Shape [None, num_labels]
            label_ids (`tuple`): (start_positions, end_positions). Two tensor of shape [None]
        """
        preprocessor = preprocessors.get_preprocessor(self.config.pretrain_model_name_or_path,
                                                      app_model_name="text_comprehension_bert",
                                                      user_defined_config=self.config)
        input_ids, input_mask, segment_ids, start_positions, end_positions = preprocessor(features)

        bert_backbone = model_zoo.get_pretrained_model(self.config.pretrain_model_name_or_path)
        sequence_output, _ = bert_backbone([input_ids, input_mask, segment_ids], mode=mode)

        seq_length = self.config.sequence_length
        hidden_size = int(sequence_output.shape[2])

        output_weights = tf.get_variable(
            "app/output_weights", [2, hidden_size],
            initializer=tf.truncated_normal_initializer(stddev=0.02))

        output_bias = tf.get_variable(
            "app/output_bias", [2], initializer=tf.zeros_initializer())

        final_hidden_matrix = tf.reshape(sequence_output,
                                         [-1, hidden_size])
        logits = tf.matmul(final_hidden_matrix, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)

        logits = tf.reshape(logits, [-1, seq_length, 2])
        logits = tf.transpose(logits, [2, 0, 1])

        unstacked_logits = tf.unstack(logits, axis=0)

        (start_logits, end_logits) = (unstacked_logits[0], unstacked_logits[1])

        self.check_and_init_from_checkpoint(mode)
        return (start_logits, end_logits), (start_positions, end_positions)


class BERTTextHAEComprehension(BaseTextComprehension):
    """ BERT-HAE Text Classification Model

        .. highlight:: python
        .. code-block:: python

            default_param_dict = {
                "pretrain_model_name_or_path": "pai-bert-base-zh",
                "multi_label": False,
                "num_labels": 2,
                "dropout_rate": 0.1,
                "max_query_length": 64,
            }
    """
    def __init__(self, **kwargs):
        super(BERTTextHAEComprehension, self).__init__(**kwargs)

    @staticmethod
    def get_input_tensor_schema(sequence_length=64):
        return "input_ids:int:{},input_mask:int:{},segment_ids:int:{},history_answer_marker:int:{}," \
               "start_position:int:1,end_position:int:1".format(
            sequence_length, sequence_length, sequence_length, sequence_length)

    @staticmethod
    def get_received_tensor_schema():
        return "input_ids:int:64,input_mask:int:64,segment_ids:int:64,history_answer_marker:int:64"

    @staticmethod
    def default_model_params():
        """ Get default model required parameters

        Returns:
            default_param_dict (`dict`): key/value pair of default model required parameters
        """
        default_param_dict = {
            "pretrain_model_name_or_path": "pai-bert-base-zh",
            "max_query_length": 64,
            "doc_stride": 128,
            "max_considered_history_turns": 11
        }
        return default_param_dict

    def build_logits(self, features, mode=None):
        """ Building graph of BERT Text Comprehension

        Args:
            features (`OrderedDict`): A dict mapping raw input to tensors
            mode (`bool): tell the model whether it is under training
        Returns:
            logits (`tuple`): (start_logits, end_logits), The output after the last dense layer. Two tensor of
            Shape [None, num_labels]
            label_ids (`tuple`): (start_positions, end_positions). Two tensor of shape [None]
        """
        preprocessor = preprocessors.get_preprocessor(self.config.pretrain_model_name_or_path,
                                                      app_model_name="text_comprehension_bert_hae",
                                                      user_defined_config=self.config)
        input_ids, input_mask, segment_ids, history_answer_marker, start_positions, end_positions =\
            preprocessor(features)

        bert_backbone = BertHAEPretrainedModel.get(self.config.pretrain_model_name_or_path)

        tmp = [input_ids, input_mask, segment_ids, history_answer_marker]
        sequence_output, _ = bert_backbone(tmp, mode=mode)

        seq_length = self.config.sequence_length
        hidden_size = int(sequence_output.shape[2])

        output_weights = tf.get_variable(
            "app/output_weights", [2, hidden_size],
            initializer=tf.truncated_normal_initializer(stddev=0.02))

        output_bias = tf.get_variable(
            "app/output_bias", [2], initializer=tf.zeros_initializer())

        final_hidden_matrix = tf.reshape(sequence_output,
                                         [-1, hidden_size])
        logits = tf.matmul(final_hidden_matrix, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)

        logits = tf.reshape(logits, [-1, seq_length, 2])
        logits = tf.transpose(logits, [2, 0, 1])

        unstacked_logits = tf.unstack(logits, axis=0)

        (start_logits, end_logits) = (unstacked_logits[0], unstacked_logits[1])

        self.check_and_init_from_checkpoint(mode)
        return (start_logits, end_logits), (start_positions, end_positions)