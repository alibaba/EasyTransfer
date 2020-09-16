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

import math
import tensorflow as tf


def build_kd_loss(teacher_logits,
                  student_logits,
                  task_balance=0.3,
                  distill_tempreture=2.0,
                  labels=None,
                  loss_type='mse'):
    if loss_type == 'mse':
        # mean square error
        return mse_loss(teacher_logits, student_logits)
    elif loss_type == 'xent':
        # cross entropy
        return xent_loss(teacher_logits, student_logits, labels,
                         distill_tempreture, task_balance)
    else:
        # kl divergence
        return kld_loss(teacher_logits, student_logits, labels,
                       distill_tempreture, task_balance)


def mse_loss(teacher_logits, student_logits):
    loss = tf.reduce_mean(tf.nn.l2_loss(teacher_logits - student_logits))
    return loss


def xent_loss(teacher_logits, student_logits, labels, distill_tempreture,
            task_balance):
    student_task_xent = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.squeeze(labels),
                                                       logits=student_logits))
    teacher_targets = tf.nn.softmax(teacher_logits / distill_tempreture)
    student_distill_xent = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        labels=tf.stop_gradient(teacher_targets), logits=student_logits))
    losses = task_balance * student_task_xent
    losses += (1 - task_balance) * student_distill_xent

    return losses


def kld_loss(teacher_logits, student_logits, labels, distill_temperature,
            task_balance):
    student_task_xent = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=tf.squeeze(labels), logits=student_logits)
    student_distill = tf.reduce_sum(tf.nn.softmax(student_logits / distill_temperature) * (
        tf.log(tf.nn.softmax(student_logits / distill_temperature + 1e-5) -
        tf.log(tf.nn.softmax(teacher_logits / distill_temperature + 1e-5)))))
    losses = task_balance * tf.reduce_mean(student_task_xent)
    losses += (1 - task_balance) * tf.reduce_mean(student_distill)

    return losses


def build_kd_probes_loss(teacher_logits,
                  student_logits,
                  task_balance=0.3,
                  distill_tempreture=2.0,
                  labels=None,
                  loss_type='mse'):
    teacher_n_layers = len(teacher_logits) - 1
    student_n_layers = len(student_logits) - 1
    probes_kd_loss = 0.0
    for i in range(student_n_layers):
        proportional_layer_idx = int(math.ceil(i * teacher_n_layers / student_n_layers))

        student_layer_logits = student_logits[i]
        teacher_layer_logits = teacher_logits[proportional_layer_idx]
        probes_kd_loss += build_kd_loss(teacher_logits=teacher_layer_logits,
                                        student_logits=student_layer_logits,
                                        task_balance=task_balance,
                                        distill_tempreture=distill_tempreture,
                                        labels=labels,
                                        loss_type=loss_type)
    return probes_kd_loss