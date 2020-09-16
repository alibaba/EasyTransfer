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
from .core import Dense
import tensorflow as tf


class DAMEncoder(Layer):
    def __init__(self, hidden_size, **kwargs):
        super(DAMEncoder, self).__init__(**kwargs)
        self.hidden_size = hidden_size

    def call(self, inputs, **kwargs):
        a_embeds, b_embeds, a_mask, b_mask = inputs
        training = kwargs.get("training", True)

        a_mask = tf.expand_dims(tf.cast(a_mask, dtype=tf.float32), -1)  # [None, text_a_len, 1]
        a_mask = tf.tile(a_mask, [1, 1, self.hidden_size])
        b_mask = tf.expand_dims(tf.cast(b_mask, dtype=tf.float32), -1)  # [None, text_b_len, 1]
        b_mask = tf.tile(b_mask, [1, 1, self.hidden_size])

        with tf.variable_scope("dam_layer_projection", reuse=tf.AUTO_REUSE):
            # F1a: Shape of [None, text_a_len, hidden_size]
            project_ffn = Dense(self.hidden_size, activation=tf.nn.relu, use_bias=True,
                                kernel_initializer=None, bias_initializer=None,
                                name="projection")
            a_embeds = project_ffn(a_embeds)
            # F1b: Shape of [None, text_b_len, hidden_size]
            b_embeds = project_ffn(b_embeds)

        with tf.variable_scope("dam_layer_attend", reuse=tf.AUTO_REUSE):
            a = tf.layers.dropout(a_embeds, rate=0.1, training=training, name='dam_layer_input_dropout')
            b = tf.layers.dropout(b_embeds, rate=0.1, training=training, name='dam_layer_input_dropout')
            F_ffn = Dense(self.hidden_size, activation=tf.nn.relu, use_bias=True,
                                 kernel_initializer=None, bias_initializer=None,
                                 name="dam_layer_F")
            # Fa: Shape of [None, text_a_len, hidden_size]
            # Fb: Shape of [None, text_b_len, hidden_size]
            Fa, Fb  = F_ffn(a) * a_mask, F_ffn(b) * b_mask

            # attention_weights: Shape of [None, text_a_len, text_b_len]
            attention_weights = tf.matmul(Fa, tf.transpose(Fb, [0, 2, 1]), name='dam_layer_attention_weights')
            attention_weights_transposed = tf.transpose(attention_weights, [0, 2, 1])
            attention_weights1 = attention_weights - \
                                 tf.reduce_max(attention_weights, axis=-1, keep_dims=True)
            attention_weights2 = attention_weights_transposed - \
                                 tf.reduce_max(attention_weights_transposed, axis=-1, keep_dims=True)

            attention_mask = tf.matmul(a_mask, tf.transpose(b_mask, [0, 2, 1]))
            attention_weights_exp1 = tf.exp(attention_weights1) * attention_mask
            attention_soft1 = attention_weights_exp1 / (
                    tf.reduce_sum(attention_weights_exp1, axis=-1, keep_dims=True) + 1e-8)

            attention_weights_exp2 = tf.exp(attention_weights2) * tf.transpose(attention_mask, [0, 2, 1])
            attention_soft2 = attention_weights_exp2 / (
                    tf.reduce_sum(attention_weights_exp2, axis=-1, keep_dims=True) + 1e-8)

            beta = tf.matmul(attention_soft1, b_embeds)
            alpha = tf.matmul(attention_soft2, a_embeds)

        with tf.variable_scope("dam_layer_compare", reuse=tf.AUTO_REUSE):
            a_beta = tf.concat([a_embeds, beta], axis=2)
            b_alpha = tf.concat([b_embeds, alpha], axis=2)
            a_beta = tf.layers.dropout(a_beta, rate=0.2, training=training, name='dam_layer_a_beta_dropout')
            b_alpha = tf.layers.dropout(b_alpha, rate=0.2, training=training, name='dam_layer_b_beta_dropout')

            G_ffn = Dense(self.hidden_size, activation=tf.nn.relu, use_bias=True,
                          kernel_initializer=None, bias_initializer=None,
                          name="dam_layer_G")
            v1i = G_ffn(a_beta) * a_mask
            v2j = G_ffn(b_alpha) * b_mask

        with tf.variable_scope("dam_layer_aggregate", reuse=tf.AUTO_REUSE):
            v1 = tf.reduce_sum(v1i, axis=1) # Shape of [None, hidden_size]
            v2 = tf.reduce_sum(v2j, axis=1) # Shape of [None, hidden_size]
            v1_max = tf.reduce_max(v1i, axis=1)
            v2_max = tf.reduce_max(v2j, axis=1)

            output_features = tf.concat([v1, v2, v1_max, v2_max], axis=1, name="output_features_dam") # Shape of [None, hidden_size * 4]
        return output_features