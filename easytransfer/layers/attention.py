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
from .utils import get_initializer
from .core import Dropout, dense_dropoutput_layernorm

class SelfAttention(Layer):
    def __init__(self, config, **kwargs):
        super(SelfAttention, self).__init__(**kwargs)

        self.hidden_size = config.hidden_size
        self.num_hidden_layers = config.num_hidden_layers
        self.num_attention_heads = config.num_attention_heads
        assert config.hidden_size % config.num_attention_heads == 0
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)

        self.initializer = get_initializer(config.initializer_range)
        self.dropout = Dropout(config.attention_probs_dropout_prob)

    def build(self, input_shape):
        self.q_head_weight = self.add_weight(
            shape=(self.hidden_size, self.hidden_size),
            initializer=self.initializer,
            dtype=tf.float32,
            name='query/kernel',
        )
        self.q_head_bias = self.add_weight(
            shape=(self.hidden_size,),
            initializer=self.initializer,
            dtype=tf.float32,
            name='query/bias',
        )
        self.k_head_weight = self.add_weight(
            shape=(self.hidden_size, self.hidden_size),
            initializer=self.initializer,
            dtype=tf.float32,
            name='key/kernel',
        )
        self.k_head_bias = self.add_weight(
            shape=(self.hidden_size,),
            initializer=self.initializer,
            dtype=tf.float32,
            name='key/bias',
        )
        self.v_head_weight = self.add_weight(
            shape=(self.hidden_size, self.hidden_size),
            initializer=self.initializer,
            dtype=tf.float32,
            name='value/kernel',
        )
        self.v_head_bias = self.add_weight(
            shape=(self.hidden_size,),
            initializer=self.initializer,
            dtype=tf.float32,
            name='value/bias',
        )

        super(SelfAttention, self).build(input_shape)

    def _abs_attn_core(self, q_head, k_head, v_head, attn_mask, training,
                       scale):
        attn_score = tf.einsum('bind,bjnd->bnij', q_head, k_head)
        attn_score = tf.multiply(attn_score, scale)

        attn_mask = tf.expand_dims(attn_mask, axis=[1])
        adder = (1.0 - tf.cast(attn_mask, tf.float32)) * -10000.0
        attn_score += adder

        attn_prob = tf.nn.softmax(attn_score)
        attn_prob = self.dropout(attn_prob, training=training)

        attn_vec = tf.einsum('bnij,bjnd->bind', attn_prob, v_head)
        return attn_vec

    def call(self, attention_input, attention_mask, kv=None, training=False):

        q_input = attention_input
        if kv is None:
            k_input = attention_input
            v_input = attention_input
        else:
            k_input = v_input = kv

        batch_size = tf.shape(attention_mask)[0]
        seq_length = tf.shape(attention_mask)[1]

        q_head_h = tf.einsum('bih,hx->bix', q_input, self.q_head_weight)
        q_head_h = tf.nn.bias_add(q_head_h, self.q_head_bias)

        k_head_h = tf.einsum('bih,hx->bix', k_input, self.k_head_weight)
        k_head_h = tf.nn.bias_add(k_head_h, self.k_head_bias)

        v_head_h = tf.einsum('bih,hx->bix', v_input, self.v_head_weight)
        v_head_h = tf.nn.bias_add(v_head_h, self.v_head_bias)

        q_head_h = tf.reshape(q_head_h, [batch_size, seq_length, self.num_attention_heads, self.attention_head_size])
        k_head_h = tf.reshape(k_head_h, [batch_size, seq_length, self.num_attention_heads, self.attention_head_size])
        v_head_h = tf.reshape(v_head_h, [batch_size, seq_length, self.num_attention_heads, self.attention_head_size])

        scale = 1 / (self.attention_head_size ** 0.5)
        attn_vec = self._abs_attn_core(q_head_h, k_head_h, v_head_h, attention_mask, training, scale)
        attn_vec = tf.reshape(attn_vec, [batch_size, seq_length, self.hidden_size])
        return attn_vec

class Attention(Layer):
    def __init__(self, config, **kwargs):
        super(Attention, self).__init__(**kwargs)
        self.self_attention = SelfAttention(config, name="self")
        self.dense_output = dense_dropoutput_layernorm(config, name="output")

    def call(self, inputs, training=False):
        input_tensor, attention_mask = inputs
        self_outputs = self.self_attention(input_tensor, attention_mask, training=training)
        attention_output = self.dense_output([self_outputs, input_tensor], training=training)
        return attention_output


class CrossAttention(Layer):
    def __init__(self, config, **kwargs):
        super(CrossAttention, self).__init__(**kwargs)
        self.cross_attention = SelfAttention(config, name="cross")
        self.dense_output = dense_dropoutput_layernorm(config, name="output")

    def call(self, inputs, training=False):
        input_tensor, encoder_hidden_states, attention_mask = inputs
        self_outputs = self.cross_attention(input_tensor, attention_mask,
                                           encoder_hidden_states, training=training)
        attention_output = self.dense_output([self_outputs, input_tensor], training=training)
        return attention_output