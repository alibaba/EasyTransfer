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

def masked_language_model_eval_metrics(lm_logits, masked_lm_ids, masked_lm_weights, vocab_size):

    masked_lm_log_probs = tf.nn.log_softmax(lm_logits, axis=-1)
    masked_lm_log_probs = tf.reshape(masked_lm_log_probs,
                                     [-1, masked_lm_log_probs.shape[-1]])

    masked_lm_ids = tf.reshape(masked_lm_ids, [-1])
    one_hot_labels = tf.one_hot(
        masked_lm_ids, depth=vocab_size, dtype=tf.float32)

    masked_lm_example_loss = -tf.reduce_sum(masked_lm_log_probs
                                            * one_hot_labels, axis=[-1])
    masked_lm_predictions = tf.argmax(
        masked_lm_log_probs, axis=-1, output_type=tf.int32)

    masked_lm_example_loss = tf.reshape(masked_lm_example_loss, [-1])
    masked_lm_weights = tf.reshape(masked_lm_weights, [-1])
    masked_lm_accuracy = tf.metrics.accuracy(
        labels=masked_lm_ids,
        predictions=masked_lm_predictions,
        weights=masked_lm_weights)

    masked_lm_mean_loss = tf.metrics.mean(
        values=masked_lm_example_loss, weights=masked_lm_weights)

    metric_dict = {
        "eval_masked_lm_accuracy": masked_lm_accuracy,
        "eval_masked_lm_loss": masked_lm_mean_loss
    }

    return metric_dict

def next_sentence_prediction_eval_metrics(nsp_logits, next_sentence_labels):
    next_sentence_log_probs = tf.nn.log_softmax(nsp_logits, axis=-1)
    next_sentence_log_probs = tf.reshape(
        next_sentence_log_probs, [-1, next_sentence_log_probs.shape[-1]])
    next_sentence_predictions = tf.argmax(
        next_sentence_log_probs, axis=-1, output_type=tf.int32)
    next_sentence_labels = tf.reshape(next_sentence_labels, [-1])
    next_sentence_accuracy = tf.metrics.accuracy(
        labels=next_sentence_labels, predictions=next_sentence_predictions)

    metric_dict = {
        "next_sentence_accuracy": next_sentence_accuracy
    }

    return metric_dict