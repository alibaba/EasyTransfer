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


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.keras import layers as keras_layers
from tensorflow.python.layers import base
from tensorflow.python.ops import init_ops
import tensorflow as tf
from .utils import get_initializer

def LayerNormalization(input_tensor, name=None):
    """Run layer normalization on the last dimension of the tensor."""
    return tf.contrib.layers.layer_norm(
        inputs=input_tensor, begin_norm_axis=-1, begin_params_axis=-1, scope=name)

class Embedding(keras_layers.Embedding, base.Layer):

    def __init__(self,
                 input_dim,
                 output_dim,
                 embeddings_initializer='uniform',
                 embeddings_regularizer=None,
                 activity_regularizer=None,
                 embeddings_constraint=None,
                 mask_zero=False,
                 input_length=None,
                 **kwargs):
        super(Embedding, self).__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            embeddings_initializer=embeddings_initializer,
            embeddings_regularizer=embeddings_regularizer,
            activity_regularizer=activity_regularizer,
            embeddings_constraint=embeddings_constraint,
            mask_zero=mask_zero,
            input_length=input_length,
            **kwargs)


class Dense(keras_layers.Dense, base.Layer):

    def __init__(self, units,
                 activation=None,
                 use_bias=True,
                 kernel_initializer=None,
                 bias_initializer=init_ops.zeros_initializer(),
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 trainable=True,
                 name=None,
                 **kwargs):
        super(Dense, self).__init__(units=units,
                                    activation=activation,
                                    use_bias=use_bias,
                                    kernel_initializer=kernel_initializer,
                                    bias_initializer=bias_initializer,
                                    kernel_regularizer=kernel_regularizer,
                                    bias_regularizer=bias_regularizer,
                                    activity_regularizer=activity_regularizer,
                                    kernel_constraint=kernel_constraint,
                                    bias_constraint=bias_constraint,
                                    trainable=trainable,
                                    name=name,
                                    **kwargs)


class Dropout(keras_layers.Dropout, base.Layer):
    def __init__(self, rate=0.5,
                 noise_shape=None,
                 seed=None,
                 name=None,
                 **kwargs):
        super(Dropout, self).__init__(rate=rate,
                                      noise_shape=noise_shape,
                                      seed=seed,
                                      name=name,
                                      **kwargs)

    def call(self, inputs, training=False):
        return super(Dropout, self).call(inputs, training=training)

class dense_dropoutput_layernorm(base.Layer):
    def __init__(self, config, **kwargs):
        super(dense_dropoutput_layernorm, self).__init__(**kwargs)
        self.dense = Dense(
            config.hidden_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )
        self.LayerNorm = LayerNormalization
        self.dropout = Dropout(config.hidden_dropout_prob)

    def call(self, inputs, training=False):
        hidden_states, input_tensor = inputs
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states, training=training)
        hidden_states = self.LayerNorm(hidden_states + input_tensor, name="LayerNorm")
        return hidden_states