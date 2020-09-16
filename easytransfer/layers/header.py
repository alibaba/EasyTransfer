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


from tensorflow.python.layers.base import Layer
import tensorflow as tf
from .core import Dense, LayerNormalization
from .activations import gelu_new
from .utils import get_initializer, gather_indexes

class MLMHead(Layer):
    def __init__(self, config, embeddings, **kwargs):
        super(MLMHead, self).__init__(**kwargs)
        self.vocab_size = config.vocab_size
        self.dense = Dense(
            config.hidden_size,
            kernel_initializer=get_initializer(config.initializer_range),
            activation=gelu_new,
            name="transform/dense",
        )

        self.LayerNorm = LayerNormalization
        self.embeddings = embeddings

    def build(self, input_shape):
        self.bias = self.add_weight(shape=(self.vocab_size,),
                                    initializer="zeros", trainable=True, name="output_bias")
        super(MLMHead, self).build(input_shape)

    def call(self, hidden_states, masked_lm_positions):
        hidden_states = gather_indexes(hidden_states, masked_lm_positions)
        word_embeddings = self.embeddings.word_embeddings
        hidden_states = self.dense(hidden_states)
        hidden_states = self.LayerNorm(hidden_states, name="transform/LayerNorm")
        logits = tf.matmul(hidden_states, word_embeddings, transpose_b=True)
        logits = tf.nn.bias_add(logits, self.bias)
        return logits


class NSPHead(Layer):
    def __init__(self, config, **kwargs):
        super(NSPHead, self).__init__(**kwargs)
        self.hidden_size = config.hidden_size
        self.config = config

    def build(self, input_shape):
        self.output_weights = self.add_weight(shape=[2, self.hidden_size],
                                    initializer=get_initializer(self.config.initializer_range),
                                       trainable=True, name="output_weights")

        self.bias = self.add_weight(shape=(2,),
                                    initializer="zeros", trainable=True, name="output_bias")

        super(NSPHead, self).build(input_shape)

    def call(self, hidden_states):
        logits = tf.matmul(hidden_states, self.output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, self.bias)
        return logits

class AlbertMLMHead(Layer):
    def __init__(self, config, embeddings, **kwargs):
        super(AlbertMLMHead, self).__init__(**kwargs)
        self.vocab_size = config.vocab_size
        self.dense = Dense(
            config.embedding_size,
            kernel_initializer=get_initializer(config.initializer_range),
            activation=gelu_new,
            name="transform/dense",
        )
        self.LayerNorm = LayerNormalization
        self.embeddings = embeddings

    def build(self, input_shape):
        self.bias = self.add_weight(shape=(self.vocab_size,),
                                    initializer="zeros", trainable=True, name="output_bias")
        super(AlbertMLMHead, self).build(input_shape)

    def call(self, hidden_states, masked_lm_positions):
        hidden_states = gather_indexes(hidden_states, masked_lm_positions)
        word_embeddings = self.embeddings.word_embeddings
        hidden_states = self.dense(hidden_states)
        hidden_states = self.LayerNorm(hidden_states, name="transform/LayerNorm")
        logits = tf.matmul(hidden_states, word_embeddings, transpose_b=True)
        logits = tf.nn.bias_add(logits, self.bias)
        return logits

class FactorizedBertMLMHead(Layer):
    def __init__(self, config, embeddings, **kwargs):
        super(FactorizedBertMLMHead, self).__init__(**kwargs)
        self.vocab_size = config.vocab_size
        self.dense = Dense(
            config.factorized_size,
            kernel_initializer=get_initializer(config.initializer_range),
            activation=gelu_new,
            name="transform/dense",
        )
        self.LayerNorm = LayerNormalization
        self.embeddings = embeddings

    def build(self, input_shape):
        self.bias = self.add_weight(shape=(self.vocab_size,),
                                    initializer="zeros", trainable=True, name="output_bias")
        super(FactorizedBertMLMHead, self).build(input_shape)

    def call(self, hidden_states, masked_lm_positions):
        hidden_states = gather_indexes(hidden_states, masked_lm_positions)
        word_embeddings = self.embeddings.word_embeddings
        hidden_states = self.dense(hidden_states)
        hidden_states = self.LayerNorm(hidden_states, name="transform/LayerNorm")
        logits = tf.matmul(hidden_states, word_embeddings, transpose_b=True)
        logits = tf.nn.bias_add(logits, self.bias)
        return logits