# coding=utf-8
#
# Copyright (c) 2019 Alibaba PAI team.
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
from sklearn.metrics import accuracy_score
import tensorflow as tf
from .evaluator import Evaluator


class PyComprehensionEvaluator(Evaluator):
    def __init__(self):
        # declare metric names this evaluator will return
        metric_names = [
            'start_accuracy',
            'end_accuracy',
            'accuracy'
        ]

        # pass metric names to base class
        super(PyComprehensionEvaluator, self).__init__(metric_names)

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
            self.labels.append(label.astype(np.int32).tolist())

    def get_best_start_end(self, logits):
        n_best_start_indexes = self._get_best_indexes(logits[:, 0])
        n_best_end_indexes = self._get_best_indexes(logits[:, 1])
        best_start, best_end = -1, -1
        best_score = -float('inf')
        for start in n_best_start_indexes:
            for end in n_best_end_indexes:
                if start <= end:
                    score = logits[start][0] + logits[end][1]
                    if score > best_score:
                        best_start = start
                        best_end = end
                        best_score = score
        return best_start, best_end

    @staticmethod
    def _get_best_indexes(logits, n_best_size=20):
        """Get the n-best logits from a list."""
        index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

        best_indexes = []
        for i in range(len(index_and_score)):
            if i >= n_best_size:
                break
            best_indexes.append(index_and_score[i][0])
        return best_indexes

    def evaluate(self, labels):
        '''
        python evaluation code which will be run after
        all test batched data are predicted
        '''
        if len(self.predictions) == 0 or len(self.labels) == 0:
            tf.logging.info('empty data to evaluate')
            return {'start_accuracy': 0.0, 'end_accuracy': 0.0, 'accuracy': 0.0}

        predictions = [self.get_best_start_end(logits) for logits in self.predictions]
        st_preds = [t[0] for t in predictions]
        st_labels = [t[0] for t in self.labels]
        start_accuracy = accuracy_score(st_labels, st_preds)
        end_preds = [t[1] for t in predictions]
        end_labels = [t[1] for t in self.labels]
        end_accuracy = accuracy_score(end_labels, end_preds)
        y_preds = [(t[0], t[1]) for t in predictions]
        y_trues = [(t[0], t[1]) for t in self.labels]
        accuracy = sum([y_pred == y_true for y_pred, y_true in zip(y_preds, y_trues)]) * 1.0 / len(y_preds)
        return {'start_accuracy': start_accuracy, 'end_accuracy': end_accuracy, 'accuracy': accuracy}


def comprehension_eval_metrics(logits, labels):
    start_logits, end_logits = logits
    start_positions, end_positions = labels
    zipped_logits = tf.stack((start_logits, end_logits), axis=2) # [None, seq_len, 2]
    zipped_positions = tf.concat((start_positions, end_positions), axis=1) # [None, 2]
    info_dict = {
        "predictions": zipped_logits,
        "labels": zipped_positions,
    }
    evaluator = PyComprehensionEvaluator()
    label_ids = [i for i in range(2)]
    metric_dict = evaluator.get_metric_ops(info_dict, label_ids)
    ret_metrics = evaluator.evaluate(label_ids)
    tf.summary.scalar("start_accuracy", ret_metrics['start_accuracy'])
    tf.summary.scalar("end_accuracy", ret_metrics['end_accuracy'])
    tf.summary.scalar("accuracy", ret_metrics['accuracy'])
    return metric_dict