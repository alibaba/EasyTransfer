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
from easytransfer.app_zoo.base import ApplicationModel
from easytransfer import preprocessors, model_zoo
import easytransfer.layers as layers
from easytransfer.losses import matching_embedding_margin_loss, mean_square_error, softmax_cross_entropy
from easytransfer.evaluators import match_eval_metrics
from easytransfer.preprocessors.deeptext_preprocessor import DeepTextPreprocessor


class BaseTextMatch(ApplicationModel):
    """ Basic Text Match Model """
    def __init__(self, **kwargs):
        super(BaseTextMatch, self).__init__(**kwargs)

    @staticmethod
    def default_model_params():
        """ Get default model required parameters

        Returns:
            default_param_dict (`dict`): key/value pair of default model required parameters
        """
        default_param_dict = {
            "num_labels": 2
        }
        return default_param_dict

    def build_logits(self, features, mode=None):
        """ Building the graph logic of text match model
        """
        raise NotImplementedError

    def build_loss(self, logits, labels):
        """ Building loss for training text match model
        """
        if self.config.num_labels < 2:
            return mean_square_error(labels, logits)
        else:
            return softmax_cross_entropy(labels, depth=self.config.num_labels, logits=logits)

    def build_eval_metrics(self, logits, labels):
        """ Building evaluation metrics while evaluating

        Args:
            predict_output (`tuple`): (logits, _)
        Returns:
            ret_dict (`dict`): A dict with tf.metrics op
                1. (`mse`) for regression
                2. (`accuracy`, `auc`, `f1`) for binary categories
                3. (`accuracy`, `macro-f1`, `micro-f1`) for multiple categories
        """
        return match_eval_metrics(logits, labels, self.config.num_labels)

    def build_predictions(self, predict_output):
        """ Building general text match model prediction dict.

        Args:
            predict_output (`tuple`): (logits, _*)
        Returns:
            ret_dict (`dict`): A dict with (`predictions`, `probabilities`, `logits`)
        """
        logits = predict_output[0]
        if isinstance(logits, list):
            logits = logits[0]
        if len(logits.shape) == 2:
            predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
            probs = tf.nn.softmax(logits, axis=1)
        else:
            probs = (logits + 1.0) / 2.0
            predictions = tf.cast(logits > 0.5, dtype=tf.int32)

        ret_dict = {
            "predictions": predictions,
            "probabilities": probs,
            "logits": logits,
        }
        return ret_dict

    def _add_word_embeddings(self, vocab_size, embed_size, pretrained_word_embeddings=None, trainable=False):
        with tf.name_scope("input_representations"):
            if pretrained_word_embeddings is not None:
                tf.logging.info("Initialize word embedding from pretrained")
                word_embedding_initializer = tf.constant_initializer(pretrained_word_embeddings)
            else:
                word_embedding_initializer = layers.get_initializer(0.02)
            word_embeddings = tf.get_variable("word_embeddings",
                                              [vocab_size, embed_size],
                                              dtype=tf.float32, initializer=word_embedding_initializer,
                                              trainable=trainable)
        return word_embeddings


class BertTextMatch(BaseTextMatch):
    """ Text Match model based on BERT-like pretrained models

        .. highlight:: python
        .. code-block:: python

            default_param_dict = {
                "pretrain_model_name_or_path": "pai-bert-base-zh",
                "num_labels": 2,
                "dropout_rate": 0.1
            }
    """
    def __init__(self, **kwargs):
        super(BertTextMatch, self).__init__(**kwargs)

    @staticmethod
    def get_input_tensor_schema():
        return "input_ids:int:64,input_mask:int:64,segment_ids:int:64,label_id:int:1"

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
            "dropout_rate": 0.1
        }
        return default_param_dict

    def build_logits(self, features, mode=None):
        """ Building BERT text match graph

        Args:
            features (`OrderedDict`): A dict mapping raw input to tensors
            mode (`bool`): tell the model whether it is under training
        Returns:
            logits (`Tensor`): The output after the last dense layer. Shape of [None, num_labels]
            label_ids (`Tensor`): label_ids, shape of [None]
        """
        bert_preprocessor = preprocessors.get_preprocessor(self.config.pretrain_model_name_or_path,
                                                           user_defined_config=self.config)
        input_ids, input_mask, segment_ids, label_ids = bert_preprocessor(features)

        bert_backbone = model_zoo.get_pretrained_model(self.config.pretrain_model_name_or_path)
        _, pool_output = bert_backbone([input_ids, input_mask, segment_ids], mode=mode)

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)
        pool_output = tf.layers.dropout(
            pool_output, rate=self.config.dropout_rate, training=is_training)
        logits = layers.Dense(self.config.num_labels,
                              kernel_initializer=layers.get_initializer(0.02),
                              name='app/ez_dense')(pool_output)

        self.check_and_init_from_checkpoint(mode)
        return logits, label_ids


class BertTextMatchTwoTower(BaseTextMatch):
    """ Text Match model based on BERT-like pretrained models, Two tower for learning embeddings

        .. highlight:: python
        .. code-block:: python

            default_param_dict = {
                "pretrain_model_name_or_path": "pai-bert-base-zh",
                "num_labels": 2
            }
    """
    def __init__(self, **kwargs):
        super(BertTextMatchTwoTower, self).__init__(**kwargs)

    @staticmethod
    def get_input_tensor_schema():
        return "input_ids_a:int:64,input_mask_a:int:64,segment_ids_a:int:64," \
               "input_ids_b:int:64,input_mask_b:int:64,segment_ids_b:int:64,label_ids:int:1"

    @staticmethod
    def get_received_tensor_schema():
        return "input_ids_a:int:64,input_mask_a:int:64,segment_ids_a:int:64," \
               "input_ids_b:int:64,input_mask_b:int:64,segment_ids_b:int:64"

    @staticmethod
    def default_model_params():
        """ Get default model required parameters

        Returns:
            default_param_dict (`dict`): key/value pair of default model required parameters

        """
        default_param_dict = {
            "pretrain_model_name_or_path": "pai-bert-base-zh",
            "num_labels": 2,
            "projection_dim": -1
        }
        return default_param_dict

    def build_logits(self, features, mode=None):
        """ Building BERT Two Tower text match graph

        Args:
            features (`OrderedDict`): A dict mapping raw input to tensors
            mode (`bool`): tell the model whether it is under training
        Returns:
            logits (`Tensor`): The output after the last dense layer. Shape of [None, num_labels]
            label_ids (`Tensor`): label_ids, shape of [None]
        """
        bert_preprocessor = preprocessors.get_preprocessor(self.config.pretrain_model_name_or_path,
                                                           is_paired=True,
                                                           user_defined_config=self.config)


        input_ids_a, input_mask_a, \
        segment_ids_a, input_ids_b, input_mask_b, segment_ids_b, label_id = bert_preprocessor(features)

        with tf.variable_scope('text_match_bert_two_tower', reuse=tf.AUTO_REUSE):
            bert_backbone = model_zoo.get_pretrained_model(self.config.pretrain_model_name_or_path)
            if self.config.projection_dim == -1:
                _, pool_output_a = bert_backbone([input_ids_a, input_mask_a, segment_ids_a], mode=mode)
                _, pool_output_b = bert_backbone([input_ids_b, input_mask_b, segment_ids_b], mode=mode)
            else:
                all_hidden_outputs_a, pool_output_a = bert_backbone([input_ids_a, input_mask_a, segment_ids_a],
                                                                    mode=mode)
                all_hidden_outputs_b, pool_output_b = bert_backbone([input_ids_b, input_mask_b, segment_ids_b],
                                                                    mode=mode)
                first_token_output_a = all_hidden_outputs_a[:, 0, :]
                first_token_output_b = all_hidden_outputs_b[:, 0, :]

                pool_output_a = tf.layers.dense(inputs=first_token_output_a, units=self.config.projection_dim,
                                                     activation=None, name='output_dense_layer')
                pool_output_b = tf.layers.dense(inputs=first_token_output_b, units=self.config.projection_dim,
                                                         activation=None, name='output_dense_layer')

        logits = self._cosine(pool_output_a, pool_output_b)

        self.check_and_init_from_checkpoint(mode)
        return [logits, pool_output_a, pool_output_b], label_id

    @staticmethod
    def _cosine(q, a):
        pooled_len_1 = tf.sqrt(tf.reduce_sum(q * q, 1))
        pooled_len_2 = tf.sqrt(tf.reduce_sum(a * a, 1))
        pooled_mul_12 = tf.reduce_sum(q * a, 1)
        score = tf.div(pooled_mul_12, pooled_len_1 * pooled_len_2 + 1e-8, name="scores")
        return score

    def build_loss(self, outputs, label_id):
        """ Building loss for training two tower text match model
        """
        _, emb1, emb2 = outputs
        return matching_embedding_margin_loss(emb1, emb2)


class DAMTextMatch(BaseTextMatch):
    """ Text Match model based on DAM models

        Ankur P. Parikh, Oscar Tackstrom, Dipanjan Das, Jakob Uszkoreit, et al.
        `A Decomposable Attention Model for Natural Language Inference <https://arxiv.org/abs/1606.01933/>`_
        , *EMNLP*, 2016.

        .. highlight:: python
        .. code-block:: python

            default_param_dict = {
                "max_vocab_size": 20000,
                "embedding_size": 300,
                "hidden_size": 200,
                "num_labels": 2,
                "first_sequence_length": 50,
                "second_sequence_length": 50,
                "pretrain_word_embedding_name_or_path": "",
                "fix_embedding": False
            }
    """
    def __init__(self, **kwargs):
        super(DAMTextMatch, self).__init__(**kwargs)
        self.pre_build_vocab = self.config.mode.startswith("train")

    @staticmethod
    def get_input_tensor_schema():
        return "input_ids_a:int:64,input_mask_a:int:64,input_ids_b:int:64,input_mask_b:int:64,label_id:int:1"

    @staticmethod
    def get_received_tensor_schema():
        return "input_ids_a:int:64,input_mask_a:int:64,input_ids_b:int:64,input_mask_b:int:64"

    @staticmethod
    def default_model_params():
        """ Get default model required parameters

        Returns:
            default_param_dict (`dict`): key/value pair of default model required parameters
        """
        default_param_dict = {
            "max_vocab_size": 20000,
            "embedding_size": 300,
            "hidden_size": 200,
            "num_labels": 2,
            "first_sequence_length": 50,
            "second_sequence_length": 50,
            "pretrain_word_embedding_name_or_path": "",
            "fix_embedding": False
        }
        return default_param_dict

    def build_logits(self, features, mode=None):
        """ Building DAM text match graph

        Args:
            features (`OrderedDict`): A dict mapping raw input to tensors
            mode (`bool`): tell the model whether it is under training
        Returns:
            logits (`Tensor`): The output after the last dense layer. Shape of [None, num_labels]
            label_ids (`Tensor`): label_ids, shape of [None]
        """
        text_preprocessor = DeepTextPreprocessor(self.config, mode=mode)
        text_a_indices, text_a_masks, text_b_indices, text_b_masks, label_ids = text_preprocessor(features)
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        word_embeddings = self._add_word_embeddings(vocab_size=text_preprocessor.vocab.size,
                                                    embed_size=self.config.embedding_size,
                                                    pretrained_word_embeddings=text_preprocessor.pretrained_word_embeddings,
                                                    trainable=not self.config.fix_embedding)
        a_embeds = tf.nn.embedding_lookup(word_embeddings, text_a_indices)
        b_embeds = tf.nn.embedding_lookup(word_embeddings, text_b_indices)

        dam_output_features = layers.DAMEncoder(self.config.hidden_size)(
            [a_embeds, b_embeds, text_a_masks, text_b_masks], training=is_training)

        dam_output_features = tf.layers.dropout(
            dam_output_features, rate=0.2, training=is_training, name='dam_out_features_dropout')
        dam_output_features = layers.Dense(self.config.hidden_size,
                                           activation=tf.nn.relu,
                                           kernel_initializer=layers.get_initializer(0.02),
                                           name='dam_out_features_projection')(dam_output_features)


        logits = layers.Dense(self.config.num_labels,
                              kernel_initializer=layers.get_initializer(0.02),
                              name='output_layer')(dam_output_features)

        self.check_and_init_from_checkpoint(mode)
        return logits, label_ids


class DAMPlusTextMatch(BaseTextMatch):
    """ Text Match model based on DAM Plus model, Alibaba PAI Group

        .. highlight:: python
        .. code-block:: python

            default_param_dict = {
                "max_vocab_size": 20000,
                "embedding_size": 300,
                "hidden_size": 200,
                "num_labels": 2,
                "first_sequence_length": 50,
                "second_sequence_length": 50,
                "pretrain_word_embedding_name_or_path": "",
                "fix_embedding": False
            }
    """
    def __init__(self, **kwargs):
        super(DAMPlusTextMatch, self).__init__(**kwargs)
        self.pre_build_vocab = self.config.mode.startswith("train")

    @staticmethod
    def get_input_tensor_schema():
        return "input_ids_a:int:64,input_mask_a:int:64,input_ids_b:int:64,input_mask_b:int:64,label_id:int:1"

    @staticmethod
    def get_received_tensor_schema():
        return "input_ids_a:int:64,input_mask_a:int:64,input_ids_b:int:64,input_mask_b:int:64"

    @staticmethod
    def default_model_params():
        """ Get default model required parameters

        Returns:
            default_param_dict (`dict`): key/value pair of default model required parameters
        """
        default_param_dict = {
            "max_vocab_size": 20000,
            "embedding_size": 300,
            "hidden_size": 200,
            "num_labels": 2,
            "first_sequence_length": 50,
            "second_sequence_length": 50,
            "pretrain_word_embedding_name_or_path": "",
            "fix_embedding": False
        }
        return default_param_dict

    def build_logits(self, features, mode=None):
        """ Building DAMPlus text match graph

        Args:
            features (`OrderedDict`): A dict mapping raw input to tensors
            mode (`bool`): tell the model whether it is under training
        Returns:
            logits (`Tensor`): The output after the last dense layer. Shape of [None, num_labels]
            label_ids (`Tensor`): label_ids, shape of [None]
        """
        text_preprocessor = DeepTextPreprocessor(self.config, mode=mode)
        text_a_indices, text_a_masks, text_b_indices, text_b_masks, label_ids = text_preprocessor(features)
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        word_embeddings = self._add_word_embeddings(vocab_size=text_preprocessor.vocab.size,
                                                    embed_size=self.config.embedding_size,
                                                    pretrained_word_embeddings=text_preprocessor.pretrained_word_embeddings,
                                                    trainable=not self.config.fix_embedding)
        a_embeds = tf.nn.embedding_lookup(word_embeddings, text_a_indices)
        b_embeds = tf.nn.embedding_lookup(word_embeddings, text_b_indices)

        dam_output_features = layers.DAMEncoder(self.config.hidden_size)(
            [a_embeds, b_embeds, text_a_masks, text_b_masks], training=is_training)
        bcnn_output_features = layers.BiCNNEncoder(self.config.hidden_size // 2)(
            [a_embeds, b_embeds, text_a_masks, text_b_masks])

        dam_output_features = tf.layers.dropout(
            dam_output_features, rate=0.2, training=is_training, name='dam_out_features_dropout')
        dam_output_features = layers.Dense(self.config.hidden_size,
                                           activation=tf.nn.relu,
                                           kernel_initializer=layers.get_initializer(0.02),
                                           name='dam_out_features_projection')(dam_output_features)

        bcnn_output_features = tf.layers.dropout(
            bcnn_output_features, rate=0.2, training=is_training, name='dam_out_features_dropout')
        bcnn_output_features = layers.Dense(self.config.hidden_size,
                                           activation=tf.nn.relu,
                                           kernel_initializer=layers.get_initializer(0.02),
                                           name='dam_out_features_projection')(bcnn_output_features)

        output_features = tf.concat([dam_output_features, bcnn_output_features], axis=1)

        logits = layers.Dense(self.config.num_labels,
                              kernel_initializer=layers.get_initializer(0.02),
                              name='output_layer')(output_features)

        self.check_and_init_from_checkpoint(mode)
        return logits, label_ids


class BiCNNTextMatch(BaseTextMatch):
    """ Text Match model based on BiCNN model, Alibaba PAI Group

        .. highlight:: python
        .. code-block:: python

            default_param_dict = {
                "max_vocab_size": 20000,
                "embedding_size": 300,
                "hidden_size": 200,
                "num_labels": 2,
                "first_sequence_length": 50,
                "second_sequence_length": 50,
                "pretrain_word_embedding_name_or_path": "",
                "fix_embedding": False
            }
    """
    def __init__(self, **kwargs):
        super(BiCNNTextMatch, self).__init__(**kwargs)
        self.pre_build_vocab = self.config.mode.startswith("train")

    @staticmethod
    def get_input_tensor_schema():
        return "input_ids_a:int:64,input_mask_a:int:64,input_ids_b:int:64,input_mask_b:int:64,label_id:int:1"

    @staticmethod
    def get_received_tensor_schema():
        return "input_ids_a:int:64,input_mask_a:int:64,input_ids_b:int:64,input_mask_b:int:64"

    @staticmethod
    def default_model_params():
        """ Get default model required parameters

        Returns:
            default_param_dict (`dict`): key/value pair of default model required parameters
        """
        default_param_dict = {
            "max_vocab_size": 20000,
            "embedding_size": 300,
            "hidden_size": 200,
            "num_labels": 2,
            "first_sequence_length": 50,
            "second_sequence_length": 50,
            "pretrain_word_embedding_name_or_path": "",
            "fix_embedding": False
        }
        return default_param_dict

    def build_logits(self, features, mode=None):
        """ Building BiCNN text match graph

        Args:
            features (`OrderedDict`): A dict mapping raw input to tensors
            mode (`bool`): tell the model whether it is under training
        Returns:
            logits (`Tensor`): The output after the last dense layer. Shape of [None, num_labels]
            label_ids (`Tensor`): label_ids, shape of [None]
        """
        text_preprocessor = DeepTextPreprocessor(self.config, mode=mode)
        text_a_indices, text_a_masks, text_b_indices, text_b_masks, label_ids = text_preprocessor(features)
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        word_embeddings = self._add_word_embeddings(vocab_size=text_preprocessor.vocab.size,
                                                    embed_size=self.config.embedding_size,
                                                    pretrained_word_embeddings=text_preprocessor.pretrained_word_embeddings,
                                                    trainable=not self.config.fix_embedding)
        a_embeds = tf.nn.embedding_lookup(word_embeddings, text_a_indices)
        b_embeds = tf.nn.embedding_lookup(word_embeddings, text_b_indices)

        bcnn_output_features = layers.BiCNNEncoder(self.config.hidden_size)(
            [a_embeds, b_embeds, text_a_masks, text_b_masks])

        bcnn_output_features = tf.layers.dropout(
            bcnn_output_features, rate=0.2, training=is_training, name='dam_out_features_dropout')
        bcnn_output_features = layers.Dense(self.config.hidden_size,
                                           activation=tf.nn.relu,
                                           kernel_initializer=layers.get_initializer(0.02),
                                           name='dam_out_features_projection')(bcnn_output_features)

        logits = layers.Dense(self.config.num_labels,
                              kernel_initializer=layers.get_initializer(0.02),
                              name='output_layer')(bcnn_output_features)

        self.check_and_init_from_checkpoint(mode)
        return logits, label_ids


class HCNNTextMatch(BaseTextMatch):
    """ Text Match model based on Hybrid Context CNN

        Minghui Qiu, Yang Liu, Feng Ji, Wei Zhou, Jun Huang, et al.
        `Transfer Learning for Context-Aware Question Matching in Information-seeking
        Conversation Systems <https://www.aclweb.org/anthology/P18-2034//>`_
        , *ACL* 2018.

        .. highlight:: python
        .. code-block:: python

            default_param_dict = {
                "max_vocab_size": 20000,
                "embedding_size": 300,
                "hidden_size": 300,
                "num_labels": 2,
                "first_sequence_length": 64,
                "second_sequence_length": 64,
                "pretrain_word_embedding_name_or_path": "",
                "fix_embedding": False,
                "l2_reg": 0.0004,
                "filter_size": 4,
            }
    """
    def __init__(self, **kwargs):
        super(HCNNTextMatch, self).__init__(**kwargs)
        self.pre_build_vocab = self.config.mode.startswith("train")

    @staticmethod
    def get_input_tensor_schema():
        return "input_ids_a:int:64,input_mask_a:int:64,input_ids_b:int:64,input_mask_b:int:64,label_id:int:1"

    @staticmethod
    def get_received_tensor_schema():
        return "input_ids_a:int:64,input_mask_a:int:64,input_ids_b:int:64,input_mask_b:int:64"

    @staticmethod
    def default_model_params():
        """ Get default model required parameters

        Returns:
            default_param_dict (`dict`): key/value pair of default model required parameters
        """
        default_param_dict = {
            "max_vocab_size": 20000,
            "embedding_size": 300,
            "hidden_size": 200,
            "num_labels": 2,
            "first_sequence_length": 64,
            "second_sequence_length": 64,
            "pretrain_word_embedding_name_or_path": "",
            "fix_embedding": False,
            "l2_reg": 0.0004,
            "filter_size": 4,

        }
        return default_param_dict

    def build_logits(self, features, mode=None):
        text_preprocessor = DeepTextPreprocessor(self.config, mode=mode)
        text_a_indices, text_a_masks, text_b_indices, text_b_masks, label_ids = text_preprocessor(features)
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        word_embeddings = self._add_word_embeddings(vocab_size=text_preprocessor.vocab.size,
                                                    embed_size=self.config.embedding_size,
                                                    pretrained_word_embeddings=text_preprocessor.pretrained_word_embeddings,
                                                    trainable=not self.config.fix_embedding)
        a_embeds = tf.nn.embedding_lookup(word_embeddings, text_a_indices)
        b_embeds = tf.nn.embedding_lookup(word_embeddings, text_b_indices)

        hcnn_output_features = layers.HybridCNNEncoder(
            num_filters=self.config.hidden_size,
            l2_reg=self.config.l2_reg,
            filter_size=self.config.filter_size)([a_embeds, b_embeds, text_a_masks, text_b_masks])

        hcnn_output_features = tf.layers.dropout(
            hcnn_output_features, rate=0.2, training=is_training, name='dam_out_features_dropout')
        hcnn_output_features = layers.Dense(self.config.hidden_size,
                                            activation=tf.nn.relu,
                                            kernel_initializer=layers.get_initializer(0.02),
                                            name='dam_out_features_projection')(hcnn_output_features)

        logits = layers.Dense(self.config.num_labels,
                              kernel_initializer=layers.get_initializer(0.02),
                              name='output_layer')(hcnn_output_features)

        self.check_and_init_from_checkpoint(mode)
        return logits, label_ids

