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

def softmax_cross_entropy(labels, depth, logits):
    labels = tf.squeeze(labels)
    one_hot_labels = tf.one_hot(labels, depth=depth, dtype=tf.float32)
    loss = tf.losses.softmax_cross_entropy(onehot_labels=one_hot_labels,
                                           logits=logits)
    return loss

def mean_square_error(labels, logits):
    return tf.losses.mean_squared_error(labels, logits)

def multi_label_sigmoid_cross_entropy(labels, depth, logits):
    one_hots = tf.one_hot(labels, depth)
    multi_hots = tf.reduce_max(one_hots, axis=1)
    multi_hots = tf.cast(multi_hots, logits.dtype)
    return tf.losses.sigmoid_cross_entropy(multi_class_labels=multi_hots, logits=logits)