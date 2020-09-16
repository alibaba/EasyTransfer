# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
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
from easytransfer.layers.core import Dense


class HiddenLayerProbes(Layer):
    def __init__(self,
                 num_labels,
                 kernel_initializer=None,
                 name="probes",
                 **kwargs):
        """ Probes for hidden layers KD

        Daoyuan Chen, Yaliang Li, Minghui Qiu, et al.
        `AdaBERT，Task-Adaptive BERT Compression with Differentiable
        Neural Architecture Search <https://arxiv.org/abs/2001.04246//>`_
        , *IJCAI* 2020.

        """
        super(HiddenLayerProbes, self).__init__(name=name, **kwargs)
        self.num_labels = num_labels
        self.kernel_initializer = kernel_initializer

    def call(self, inputs, **kwargs):
        """
        Args:
            inputs (`list`): [embedding_output, all_hidden_outputs]
        Returns:
            layered_logits (`list`): a list of logits
        """
        embedding_output, all_hidden_outputs = inputs

        i = 0
        layered_logits = []
        first_layer_mean_token_output = tf.math.reduce_mean(embedding_output, 1)
        logits_0 = Dense(self.num_labels,
                         kernel_initializer=self.kernel_initializer,
                         name='probe_dense_%d' % i)(first_layer_mean_token_output)
        layered_logits.append(logits_0)
        i += 1
        for layer_output in all_hidden_outputs:
            first_token_output = layer_output[:, 0, :]
            logits_i = Dense(self.num_labels,
                             kernel_initializer=self.kernel_initializer,
                             name='probe_dense_%d' % i)(first_token_output)
            layered_logits.append(logits_i)
            i += 1

        return layered_logits
