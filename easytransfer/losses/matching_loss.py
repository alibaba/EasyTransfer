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

import easytransfer.layers as layers


def matching_embedding_margin_loss(emb1, emb2):
    margin = 0.3
    batch_size = layers.get_shape_list(emb1)[0]
    emb1_norm = tf.maximum(1e-12, tf.norm(emb1, axis=1))
    tf.summary.scalar('emb1_norm_mean', tf.reduce_mean(emb1_norm))
    emb1_rep = tf.div(tf.transpose(emb1), emb1_norm)
    emb2_norm = tf.maximum(1e-12, tf.norm(emb2, axis=1))
    tf.summary.scalar('emb2_norm_mean', tf.reduce_mean(emb2_norm))
    emb2_rep = tf.div(tf.transpose(emb2), emb2_norm)

    dis = tf.matmul(tf.transpose(emb1_rep), emb2_rep)
    tf.summary.scalar('dis_mean', tf.reduce_mean(dis))

    positive_distance = tf.reshape(tf.diag_part(dis), [batch_size, 1])
    tf.summary.scalar('positive_distance_mean', tf.reduce_mean(positive_distance))

    term1 = tf.reduce_mean(tf.maximum(0., - positive_distance + (
            dis - margin * tf.eye(batch_size)) + margin), axis=1)

    loss = tf.reduce_mean(term1)
    return loss