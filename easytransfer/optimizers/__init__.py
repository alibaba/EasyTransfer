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
from tensorflow.python.framework import ops
from .adam_weight_decay_optimizer import AdamWeightDecayOptimizer
from .lamb_weight_decay_optimizer import LambWeightDecayOptimizer
import numpy as np

def get_train_op(learning_rate,
                 weight_decay_ratio,
                 loss,
                 num_towers=1,
                 warmup_ratio=0.1,
                 lr_decay="polynomial",
                 optimizer_name=None,
                 tvars=None,
                 train_steps=None,
                 clip_norm=True,
                 clip_norm_value=1.0,
                 num_freezed_layers=0,
                 ):
    warmup_steps = int(train_steps * warmup_ratio)
    global_step = tf.train.get_or_create_global_step()
    if lr_decay == "polynomial":
        learning_rate = tf.train.polynomial_decay(
            learning_rate,
            global_step,
            train_steps,
            end_learning_rate=0.0,
            power=1.0,
            cycle=False)
    else:
        learning_rate = learning_rate

    if warmup_steps != 0:
        tf.logging.info("*******Warmup {} steps***********".format(warmup_steps))
        global_steps_int = tf.cast(global_step, tf.int32)
        warmup_steps_int = tf.constant(warmup_steps, dtype=tf.int32)

        global_steps_float = tf.cast(global_steps_int, tf.float32)
        warmup_steps_float = tf.cast(warmup_steps_int, tf.float32)

        warmup_percent_done = global_steps_float / warmup_steps_float
        warmup_learning_rate = learning_rate * warmup_percent_done

        is_warmup = tf.cast(global_steps_int < warmup_steps_int, tf.float32)
        learning_rate = (
                (1.0 - is_warmup) * learning_rate + is_warmup * warmup_learning_rate)
    else:
        tf.logging.info("*******Don't warm up, then lr will polynomial_decay only************")

    if optimizer_name == "adam":
        if weight_decay_ratio > 0:
            tf.logging.info("*******Using adamW optimizer************")
            optimizer = AdamWeightDecayOptimizer(
                learning_rate=learning_rate,
                weight_decay_rate=weight_decay_ratio,
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-6,
                exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"])
        else:
            tf.logging.info("*******Using adam optimizer************")
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                               beta1=0.9,
                                               beta2=0.98,
                                               epsilon=1e-6)

    elif optimizer_name == "lamb":
        tf.logging.info("*******Using lamb optimizer************")
        optimizer = LambWeightDecayOptimizer(learning_rate=learning_rate,
                                             weight_decay_rate=weight_decay_ratio,
                                             exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"])
    elif optimizer_name == "adagrad":
        optimizer = tf.train.AdagradOptimizer(learning_rate)
    elif optimizer_name == "adadelta":
        optimizer = tf.train.AdadeltaOptimizer(learning_rate)
    else:
        raise ValueError("Set train op optimizer adam or lamb")

    if tvars is None:
        tvars = tf.trainable_variables()

    grads = tf.gradients(loss, tvars)

    tf.logging.info("*******Num of trainable variables {}************".format(
        np.sum([np.prod(v.get_shape().as_list()) for v in tvars])))

    if clip_norm:
        tf.logging.info("*******Clip Gradients************")
        tf.logging.info("*******Clip Norm Value {}*********".format(clip_norm_value))
        (grads, _) = tf.clip_by_global_norm(grads, clip_norm=clip_norm_value)
    else:
        tf.logging.info("*******Don't Clip Gradients************")

    tf.logging.info("*********Num towers is {} *********".format(num_towers))
    for i in range(len(grads)):
        if grads[i] is not None:
            if isinstance(grads[i], ops.IndexedSlices):
                grads[i] = ops.convert_to_tensor(grads[i])
            grads[i] *= (1. / num_towers)

    if num_freezed_layers > 0:
        tf.logging.info("*******Num Freezed Layers is {} ************".format(num_freezed_layers))
        for i in range(len(grads)):
            freeze = False
            for l in range(num_freezed_layers):
                if "layer_{}/".format(l) in tvars[i].name:
                    freeze = True
            if freeze:
                grads[i] *= 0
                tf.logging.info("Freezing var name is {}".format(tvars[i].name))

    train_op = optimizer.apply_gradients(
        zip(grads, tvars), global_step=global_step)
    tf.summary.scalar("learning_rate", learning_rate)

    return train_op
