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
from tensorflow.python.layers.base import Layer
from easytransfer.layers.core import get_initializer


def conv_maxpool(x, filter_sizes=[3, 4, 5], embedding_size=100,
                 num_filters=128, sequence_length=60,
                 name="layer1", reuse=True, print_layer=True):
    # Create a convolution + maxpool layer for each filter size
    pooled_outputs = []
    for i, filter_size in enumerate(filter_sizes):
        with tf.name_scope("%s-maxpool-%s" % (name, filter_size)):
            # Convolution Layer
            filter_shape = [filter_size, embedding_size, 1, num_filters]
            init = tf.contrib.layers.xavier_initializer(uniform=False, seed=None, dtype=tf.float32)
            with tf.variable_scope("foo", reuse=reuse):
                W = tf.get_variable('%s_Wconv%d' % (name, i), filter_shape, initializer=init)
                b = tf.get_variable('%s_bconv%d' % (name, i), [num_filters], initializer=init)

            conv = tf.nn.conv2d(
                x,
                W,
                strides=[1, 1, 1, 1],
                padding="VALID",
                name="%s-conv" % name)

            # Apply nonlinearity
            h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")

            # Maxpooling over the outputs
            pooled = tf.nn.max_pool(
                h,
                ksize=[1, sequence_length - filter_size + 1, 1, 1],
                strides=[1, 1, 1, 1],
                padding='VALID',
                name="%s-pool" % name)
            pooled_outputs.append(pooled)
            if print_layer:
                print('conv shape: {}'.format(conv.shape))
                print('pooled shape: {}'.format(pooled.shape))
                print('skip print other conv and pooled layer shapes')
                print_layer = False

    # Combine all the pooled features
    num_filters_total = num_filters * len(filter_sizes)
    h_pool = tf.concat(pooled_outputs, 3)
    h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])

    h_drop = h_pool_flat

    return h_drop, num_filters_total


class BiCNNEncoder(Layer):
    def __init__(self, num_filters=100, **kwargs):
        super(BiCNNEncoder, self).__init__(**kwargs)
        self.num_filters = num_filters

    def call(self, inputs, **kwargs):
        a_embeds, b_embeds, a_mask, b_mask = inputs
        embed_size, a_length, b_length = int(a_embeds.shape[-1]), \
                                         int(a_embeds.shape[1]), \
                                         int(b_embeds.shape[1])
        a_mask = tf.expand_dims(tf.cast(a_mask, tf.float32), -1)
        b_mask = tf.expand_dims(tf.cast(b_mask, tf.float32), -1)

        a_expanded = tf.expand_dims(a_embeds, -1)
        b_expanded = tf.expand_dims(b_embeds, -1)

        LO_0, _ = conv_maxpool(x=a_expanded,
                               filter_sizes=[2, 3, 4],
                               embedding_size=embed_size,
                               num_filters=self.num_filters,
                               sequence_length=a_length,
                               reuse=False,
                               print_layer=True)
        RO_0, _ = conv_maxpool(x=b_expanded,
                               filter_sizes=[2, 3, 4],
                               embedding_size=embed_size,
                               num_filters=self.num_filters,
                               sequence_length=b_length,
                               reuse=True,
                               print_layer=True)

        a_pooled = tf.reduce_sum(a_embeds, axis=2) / (tf.reduce_sum(a_mask, axis=1) + 1)
        b_pooled = tf.reduce_sum(b_embeds, axis=2) / (tf.reduce_sum(b_mask, axis=1) + 1)

        LO_0 = tf.concat([LO_0, a_pooled], axis=1)
        RO_0 = tf.concat([RO_0, b_pooled], axis=1)
        sims = [LO_0, RO_0, tf.subtract(LO_0, RO_0), tf.multiply(LO_0, RO_0)]
        output_features = tf.concat(sims, axis=1, name="output_features_bcnn")

        return output_features


class HybridCNNEncoder(Layer):
    def __init__(self, num_filters=200, l2_reg=0.01, filter_size=4,
                 attn_filter_size_1=6, attn_filter_size_2=4,
                 attn_num_filters_1=8, attn_num_filters_2=16,
                 **kwargs):
        super(HybridCNNEncoder, self).__init__(**kwargs)
        self.filter_size = filter_size
        self.num_filters = num_filters
        self.attn_filter_size_1 = attn_filter_size_1
        self.attn_filter_size_2 = attn_filter_size_2
        self.attn_num_filters_1 = attn_num_filters_1
        self.attn_num_filters_2 = attn_num_filters_2
        self.l2_reg = l2_reg

    @staticmethod
    def make_attention_mat(x1, x2):
        # x1, x2 = [batch, height, width, 1] = [batch, d, s, 1]
        # x2 => [batch, height, 1, width]
        # [batch, width, wdith] = [batch, s, s])
        dot = tf.reduce_sum(tf.matmul(x1, tf.matrix_transpose(x2)), axis=1, name="att_sim")
        return dot

    @staticmethod
    def wide_convolution(x, filter_size=5, embedding_size=100, num_filters=128,
                         l2_reg=0.01, name_scope="wide_conv", reuse=False):
        """ Wide Convolution for word embeddings

            Args:
                x (`tensor`): shape of [None, embedding_size, seq_length, 1]

            Returns:
                conv_trans (`tensor`): [None, num_filters, seq_length + filter_size - 1, 1]
        """

        def pad_for_wide_conv(x, w):
            return tf.pad(x, np.array([[0, 0], [0, 0], [w - 1, w - 1], [0, 0]]), "CONSTANT", name="pad_wide_conv")

        with tf.variable_scope(name_scope) as scope:
            padded_x = pad_for_wide_conv(x, filter_size)
            # padded_x: shape of [None, embedding_size, seq_length + 2 * w - 1, 1]
            conv = tf.contrib.layers.conv2d(
                inputs=padded_x,
                num_outputs=num_filters,
                kernel_size=(embedding_size, filter_size),
                stride=1,
                padding="VALID",
                activation_fn=tf.nn.tanh,
                weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                weights_regularizer=tf.contrib.layers.l2_regularizer(scale=l2_reg),
                biases_initializer=tf.constant_initializer(1e-04),
                reuse=reuse,
                trainable=True,
                scope=scope
            )
            # Weight: [embedding_size, filter_size, in_channels, num_filters]

            conv_trans = tf.transpose(conv, [0, 3, 2, 1], name="conv_trans")
        return conv_trans

    @staticmethod
    def attention_convolution(x, filter_size, num_filters=128,
                              l2_reg=0.01, stride=1, padding="SAME",
                              name_scope="attn_conv"):
        """ Same Convolution for attention matrix

            Args:
                x (`tensor`): shape of [None, s1_len, s2_len, 1]
            Returns:
                conv (`tensor`): shape of [None, s1_len - kernel_h + 1, s2_len - kernel_w + 1, num_filters]
        """
        with tf.variable_scope(name_scope) as scope:
            # Weight: [kernel_h, kernel_w, 1, num_filters]
            conv = tf.contrib.layers.conv2d(
                inputs=x,
                num_outputs=num_filters,
                kernel_size=(filter_size, filter_size),
                stride=stride,
                padding=padding,
                activation_fn=tf.nn.tanh,
                weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                weights_regularizer=tf.contrib.layers.l2_regularizer(scale=l2_reg),
                biases_initializer=tf.constant_initializer(1e-04),
                reuse=False,
                trainable=True,
                scope=scope
            )

        return conv

    def call(self, inputs, **kwargs):
        a_embeds, b_embeds, a_mask, b_mask = inputs
        embed_size, a_length, b_length = int(a_embeds.shape[-1]), \
                                         int(a_embeds.shape[1]), \
                                         int(b_embeds.shape[1])

        a_mask = tf.expand_dims(tf.cast(a_mask, dtype=tf.float32), -1)  # [None, a_length, 1]
        b_mask = tf.expand_dims(tf.cast(b_mask, dtype=tf.float32), -1)  # [None, b_length, 1]
        a_embeds = tf.transpose(a_embeds * a_mask, [0, 2, 1], name="emb1_trans")
        b_embeds = tf.transpose(b_embeds * b_mask, [0, 2, 1], name="emb2_trans")
        x1 = tf.expand_dims(a_embeds, -1)
        x2 = tf.expand_dims(b_embeds, -1)

        with tf.variable_scope("bicnn"):
                left_conv = self.wide_convolution(x=x1,
                                                  filter_size=self.filter_size,
                                                  embedding_size=embed_size,
                                                  num_filters=self.num_filters,
                                                  l2_reg=self.l2_reg,
                                                  reuse=False)
                # left_avg_pooled: shape of [None, num_filters, 1, 1]
                left_max_pooled = tf.layers.max_pooling2d(inputs=left_conv,
                                                          pool_size=(1, a_length + self.filter_size - 1),
                                                          strides=1,
                                                          padding="VALID",
                                                          name="max_pooled")
                left_max_pooled = tf.squeeze(tf.squeeze(left_max_pooled, axis=-1), axis=-1)
                # left_max_pooled: shape of [None, num_filters, 1, 1]
                right_conv = self.wide_convolution(x=x2,
                                                  filter_size=self.filter_size,
                                                  embedding_size=embed_size,
                                                  num_filters=self.num_filters,
                                                  l2_reg=self.l2_reg,
                                                  reuse=True)
                right_max_pooled = tf.layers.max_pooling2d(inputs=right_conv,
                                                           pool_size=(1, b_length + self.filter_size - 1),
                                                           strides=1,
                                                           padding="VALID",
                                                           name="max_pooled")
                right_max_pooled = tf.squeeze(tf.squeeze(right_max_pooled, axis=-1), axis=-1)

        with tf.variable_scope("pyramid_cnn"):
            attention_matrix = tf.matmul(tf.transpose(a_embeds, [0, 2, 1]), b_embeds,
                                          name='pyramid_cnn_attention_matrix')
            # shape of [None, a_length, b_length]
            attention_mask = tf.matmul(a_mask, tf.transpose(b_mask, [0, 2, 1]))
            attention_matrix = attention_matrix * attention_mask
            attention_matrix = tf.expand_dims(attention_matrix, axis=-1)
            # shape of [None, a_length, b_length, 1]

            attn_conv_1 = self.attention_convolution(x=attention_matrix,
                                                     filter_size=self.attn_filter_size_1,
                                                     num_filters=self.attn_num_filters_1,
                                                     name_scope="attn_conv_1")
            # attn_conv: shape of [None, s1_len - attn_filter_size_1 + 1,
            # s2_len - attn_filter_size_1 + 1, attn_num_filters_1]
            attn_max_pooled_1 = tf.layers.max_pooling2d(inputs=attn_conv_1,
                                                        pool_size=(4, 4),
                                                        strides=4,
                                                        padding="VALID",
                                                        name="attn_max_pooled_1" )

            attn_conv_2 = self.attention_convolution(x=attn_max_pooled_1,
                                                     filter_size=self.attn_filter_size_2,
                                                     num_filters=self.attn_num_filters_2,
                                                     padding="VALID",
                                                     stride=3,
                                                     name_scope="attn_conv_2")
            attn_max_pooled_2 = tf.layers.max_pooling2d(inputs=attn_conv_2,
                                                        pool_size=(2, 2),
                                                        strides=2,
                                                        padding="VALID",
                                                        name="attn_max_pooled_2")
            attn_max_pooled_2 = tf.reshape(attn_max_pooled_2,
                                           [-1, self.attn_num_filters_2 *
                                            (a_length / 4 / 3 / 2) * (b_length / 4 / 3 / 2)])

        bicnn_max_pooled_diff = tf.subtract(left_max_pooled, right_max_pooled)
        bicnn_max_pooled_mul = tf.multiply(left_max_pooled, right_max_pooled)
        output_features = tf.concat([attn_max_pooled_2, left_max_pooled, right_max_pooled,
                                     bicnn_max_pooled_diff, bicnn_max_pooled_mul], axis=1,
                                     name="hcnn_output_features")
        return output_features


class TextCNNEncoder(Layer):
    def __init__(self, num_filters="100,100,100", filter_sizes="3,4,5",
                 embed_size=100, max_seq_len=20, **kwargs):
        super(TextCNNEncoder, self).__init__(**kwargs)
        self.num_filters = [int(t) for t in num_filters.split(",")]
        self.filter_sizes = [int(t) for t in filter_sizes.split(",")]
        self.embed_size = embed_size
        self.max_seq_len = max_seq_len

    def call(self, inputs, **kwargs):
        embeds, mask = inputs
        # mask = tf.expand_dims(tf.cast(mask, dtype=tf.float32), -1)  # [None, b_length, 1]
        # embeds =  tf.expand_dims(embeds * mask, -1)
        embeds =  tf.expand_dims(embeds, -1)
        pooled_outputs = list()
        for i, filter_size in enumerate(self.filter_sizes):
            with tf.variable_scope("Conv-pool-layer-{}".format(filter_size)):
                filter = tf.get_variable('filter-%s' % filter_size,
                                         [filter_size, self.embed_size, 1, self.num_filters[i]],
                                         initializer=get_initializer(0.01))
                conv = tf.nn.conv2d(embeds, filter, strides=[1, 1, 1, 1], padding="VALID", name="conv")
                # [bsize, max_sent_len - filter_size + 1, 1, n_filters]
                # conv = tf.contrib.layers.batch_norm(conv, is_training=self.is_training_flag, scope='cnn_bn_')
                b = tf.get_variable("b-%s" % filter_size, [self.num_filters[i]])  # ADD 2017-06-09
                h = tf.nn.relu(tf.nn.bias_add(conv, b), "relu")
                pooled = tf.nn.max_pool(h, ksize=[1, self.max_seq_len - filter_size + 1, 1, 1],
                                        strides=[1, 1, 1, 1], padding='VALID',
                                        name="pool")
                # [bsize, 1, 1, num_filters]
                pooled = tf.squeeze(tf.squeeze(pooled, axis=1), axis=1)
                pooled_outputs.append(pooled)
        output_features = tf.concat(pooled_outputs, axis=1)  # [bsize, len(filter_sizes) * n_filters]
        return output_features