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
import numpy as np
from .evaluator import Evaluator
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, mean_squared_error


class MatchEvaluator(Evaluator):
    def __init__(self, num_labels):
        """ Evaluator for different situation of text match application

        """
        self.num_labels = num_labels
        if self.num_labels < 2:
            self.metric_names = [
                'mse'
            ]
        elif self.num_labels == 2:
            self.metric_names = [
                'accuracy',
                'auc',
                'f1'
            ]
        else:
            self.metric_names = [
                'accuracy',
                'micro_f1',
                'macro_f1'
            ]
        # pass metric names to base class
        super(MatchEvaluator, self).__init__(self.metric_names)

    def clear(self):
        '''
        clear internal storage
        '''
        self.predictions = []
        self.labels = []

    def add_batch_info(self, predictions, labels):
        '''
        store prediction and labels in a internal list
        Args:
          predictions batched prediction result, numpy array with shape N
          labels batched labels, numpy array with shape N
        '''
        for pred, label in zip(predictions, labels):
            self.predictions.append(pred)
            self.labels.append(label)

    def evaluate(self, labels):
        '''
        python evaluation code which will be run after
        all test batched data are predicted
        '''
        if len(self.predictions) == 0 or len(self.labels) == 0:
            tf.logging.info('empty data to evaluate')
            ret_metric = {key: 0.0 for key in self.metric_names}
            return ret_metric

        self.labels = np.stack(self.labels)
        self.predictions = np.stack(self.predictions)
        if self.num_labels == 1:
            mse = mean_squared_error(self.labels, self.predictions)
            return {'mse': mse}
        elif self.num_labels == 2:
            accuracy = accuracy_score(self.labels, self.predictions)
            auc = roc_auc_score(self.labels, self.predictions)
            f1 = f1_score(self.labels, self.predictions)
            return {'accuracy': accuracy, 'auc': auc, 'f1': f1}
        else:
            micro_f1 = f1_score(
                self.labels, self.predictions, labels=[i for i in range(self.num_labels)], average='micro')
            macro_f1 = f1_score(
                self.labels, self.predictions, labels=[i for i in range(self.num_labels)], average='macro')
            accuracy = accuracy_score(self.labels, self.predictions)
            return {'accuracy': accuracy, 'micro_f1': micro_f1, 'macro_f1': macro_f1}


def match_eval_metrics(logits, labels, num_labels):
    if isinstance(logits, list):
        logits = logits[0]
    if len(logits.shape) == 2:
        predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
    else:
        predictions = tf.cast(logits > 0.5, dtype=tf.int32)
    info_dict = {
        "predictions": predictions,
        "labels": labels,
    }
    label_ids = [i for i in range(num_labels)]

    evaluator = MatchEvaluator(num_labels=num_labels)
    metric_dict = evaluator.get_metric_ops(info_dict, label_ids)
    tf.logging.info(metric_dict)
    ret_metrics = evaluator.evaluate(label_ids)
    tf.logging.info(ret_metrics)
    for key, val in ret_metrics.items():
        tf.summary.scalar("eval_" + key, val)
    return metric_dict