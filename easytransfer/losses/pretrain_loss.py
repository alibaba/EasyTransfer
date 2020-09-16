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


def masked_language_model_loss(lm_logits, masked_lm_ids, masked_lm_weights, vocab_size):

    log_probs = tf.nn.log_softmax(lm_logits, axis=-1)
    masked_lm_ids = tf.reshape(masked_lm_ids, [-1])
    label_weights = tf.reshape(masked_lm_weights, [-1])
    one_hot_labels = tf.one_hot(masked_lm_ids, depth=vocab_size, dtype=tf.float32)

    per_example_loss = -tf.reduce_sum(log_probs * one_hot_labels, axis=[-1])
    numerator = tf.reduce_sum(label_weights * per_example_loss)
    denominator = tf.reduce_sum(label_weights) + 1e-5
    masked_lm_loss = numerator / denominator

    tf.summary.scalar("masked_lm_loss", masked_lm_loss)
    return masked_lm_loss


def next_sentence_prediction_loss(nsp_logits, nx_sent_labels):

    log_probs = tf.nn.log_softmax(nsp_logits, axis=-1)
    next_sentence_labels = tf.reshape(nx_sent_labels, [-1])
    one_hot_labels = tf.one_hot(next_sentence_labels, depth=2, dtype=tf.float32)
    per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
    next_sentence_prediction_loss = tf.reduce_mean(per_example_loss)
    tf.summary.scalar("next_sentence_prediction_loss", next_sentence_prediction_loss)

    return next_sentence_prediction_loss


def image_reconstruction_mse_loss(mpm_logits, target_raw_patch_features,
                              masked_image_token_num, patch_feature_size):
    image_pred_probs = tf.nn.log_softmax(mpm_logits)
    image_pred_probs = tf.reshape(image_pred_probs,
                                  (-1, masked_image_token_num,
                                   patch_feature_size))
    image_target_probs = tf.reshape(tf.nn.log_softmax(target_raw_patch_features),
                                    (
                                    -1, masked_image_token_num, patch_feature_size))

    image_loss = tf.keras.losses.mean_squared_error(image_target_probs, image_pred_probs)
    image_loss = tf.reduce_mean(image_loss)
    tf.summary.scalar("image_reconstruction_mse_loss", image_loss)

    return image_loss

def image_reconstruction_kld_loss(mpm_logits, target_raw_patch_features,
                              masked_image_token_num, patch_feature_size):

    image_pred_probs = tf.nn.softmax(mpm_logits)
    image_pred_probs = tf.reshape(image_pred_probs,
                                  (-1, masked_image_token_num, patch_feature_size))
    image_target_probs = tf.reshape(tf.nn.softmax(target_raw_patch_features),
                                    (-1, masked_image_token_num,
                                     patch_feature_size))

    image_loss = tf.keras.losses.KLD(image_target_probs, image_pred_probs)
    image_loss = tf.reduce_mean(image_loss)
    tf.summary.scalar("image_reconstruction_kld_loss", image_loss)

    return image_loss