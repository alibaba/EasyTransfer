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
from tensorflow.python.layers.base import Layer
from .core import LayerNormalization, Dropout
from .utils import get_initializer, get_shape_list

class BertEmbeddings(Layer):
    """Construct the embeddings from word, position and token_type embeddings.
    """
    def __init__(self, config, **kwargs):
        super(BertEmbeddings, self).__init__(**kwargs)

        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.initializer_range = config.initializer_range
        self.token_type_vocab_size = config.type_vocab_size
        self.max_position_embeddings  = config.max_position_embeddings

        self.LayerNorm = LayerNormalization
        self.dropout = Dropout(config.hidden_dropout_prob)
        self.initializer = get_initializer(self.initializer_range)

    def build(self, input_shape):
        """Build shared word embedding layer """
        self.word_embeddings = self.add_weight(
            "word_embeddings",
            dtype=tf.float32,
            shape=[self.vocab_size, self.hidden_size],
            initializer=self.initializer,
        )

        self.position_embeddings = self.add_weight(
            "position_embeddings",
            dtype=tf.float32,
            shape=[self.max_position_embeddings, self.hidden_size],
            initializer=self.initializer,
        )

        self.token_type_embeddings = self.add_weight(
            "token_type_embeddings",
            dtype=tf.float32,
            shape=[self.token_type_vocab_size, self.hidden_size],
            initializer=self.initializer,
        )
        super(BertEmbeddings,self).build(input_shape)

    def call(self, inputs, training=False):
        input_ids, token_type_ids = inputs

        input_embeddings = tf.gather(self.word_embeddings, input_ids)

        input_shape = get_shape_list(input_embeddings)
        batch_size = input_shape[0]
        seq_length = input_shape[1]
        width = input_shape[2]

        # This vocab will be small so we always do one-hot here, since it is always
        # faster for a small vocabulary.
        flat_token_type_ids = tf.reshape(token_type_ids, [-1])
        one_hot_ids = tf.one_hot(flat_token_type_ids, depth=self.token_type_vocab_size)
        token_type_embeddings = tf.matmul(one_hot_ids, self.token_type_embeddings)
        token_type_embeddings = tf.reshape(token_type_embeddings,
                                           [batch_size, seq_length, width])
        input_embeddings += token_type_embeddings

        position_embeddings = tf.gather(self.position_embeddings, tf.range(0, seq_length))
        position_embeddings = tf.expand_dims(position_embeddings, 0)

        input_embeddings += position_embeddings

        output = self.LayerNorm(input_embeddings, name="LayerNorm")
        output = self.dropout(output, training=training)
        return output


class AlbertEmbeddings(Layer):
    """Construct the embeddings from word, position and token_type embeddings.
    """
    def __init__(self, config, **kwargs):
        super(AlbertEmbeddings, self).__init__(**kwargs)

        self.vocab_size = config.vocab_size
        self.embedding_size = config.embedding_size
        self.initializer_range = config.initializer_range
        self.token_type_vocab_size = config.type_vocab_size
        self.max_position_embeddings  = config.max_position_embeddings

        self.LayerNorm = LayerNormalization
        self.dropout = Dropout(config.hidden_dropout_prob)
        self.initializer = get_initializer(self.initializer_range)

    def build(self, input_shape):
        """Build shared word embedding layer """

        self.word_embeddings = self.add_weight(
            "word_embeddings",
            dtype=tf.float32,
            shape=[self.vocab_size, self.embedding_size],
            initializer=self.initializer,
        )

        self.position_embeddings = self.add_weight(
            "position_embeddings",
            dtype=tf.float32,
            shape=[self.max_position_embeddings, self.embedding_size],
            initializer=self.initializer,
        )

        self.token_type_embeddings = self.add_weight(
            "token_type_embeddings",
            dtype=tf.float32,
            shape=[self.token_type_vocab_size, self.embedding_size],
            initializer=self.initializer,
        )
        super(AlbertEmbeddings, self).build(input_shape)

    def call(self, inputs, training=False):
        input_ids, token_type_ids = inputs

        input_embeddings = tf.gather(self.word_embeddings, input_ids)

        input_shape = get_shape_list(input_embeddings)
        batch_size = input_shape[0]
        seq_length = input_shape[1]
        width = input_shape[2]

        # This vocab will be small so we always do one-hot here, since it is always
        # faster for a small vocabulary.
        flat_token_type_ids = tf.reshape(token_type_ids, [-1])
        one_hot_ids = tf.one_hot(flat_token_type_ids, depth=self.token_type_vocab_size)
        token_type_embeddings = tf.matmul(one_hot_ids, self.token_type_embeddings)
        token_type_embeddings = tf.reshape(token_type_embeddings,
                                           [batch_size, seq_length, width])
        input_embeddings += token_type_embeddings

        position_embeddings = tf.gather(self.position_embeddings, tf.range(0, seq_length))
        position_embeddings = tf.expand_dims(position_embeddings, 0)

        input_embeddings += position_embeddings

        output = self.LayerNorm(input_embeddings, name="LayerNorm")
        output = self.dropout(output, training=training)
        return output