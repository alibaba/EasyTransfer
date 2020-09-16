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

import numpy as np
import tensorflow as tf
from .evaluator import Evaluator

class SequenceLablingEvaluator(Evaluator):
    def __init__(self):
        # declare metric names this evaluator will return
        metric_names = [
            'accuracy'
        ]

        # pass metric names to base class
        super(SequenceLablingEvaluator, self).__init__(metric_names)

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
            self.labels.append(label.astype(np.int32))

    def evaluate(self, labels):
        '''
        python evaluation code which will be run after
        all test batched data are predicted
        '''
        if len(self.predictions) == 0 or len(self.labels) == 0:
            tf.logging.info('empty data to evaluate')
            return {'accuracy': 0.0}

        cnt = 0
        hit = 0
        for smp_preds, smp_labels in zip(self.predictions, self.labels):
            for token_pred, token_label in zip(smp_preds, smp_labels):
                if token_label == -1:
                    continue
                cnt += 1
                hit += (token_pred == token_label)
        accuracy = 1.0 * hit / cnt
        return {'accuracy': accuracy}

def sequence_labeling_eval_metrics(logits, labels, num_labels):
        """ Building evaluation metrics while evaluating

        Args:
            logits (`Tensor`): shape of [None, seq_length, num_labels]
            labels (`Tensor`): shape of [None, seq_length]
        Returns:
            ret_dict (`dict`): A dict with (`py_accuracy`, `py_micro_f1`, `py_macro_f1`) tf.metrics op
        """
        predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
        evaluator = SequenceLablingEvaluator()
        info_dict = {
            "predictions": predictions,
            "labels": labels,
        }
        label_ids = [i for i in range(num_labels)]
        metric_dict = evaluator.get_metric_ops(info_dict, label_ids)
        tf.logging.info(metric_dict)
        ret_metrics = evaluator.evaluate(label_ids)
        tf.logging.info(ret_metrics)
        for key, val in ret_metrics.items():
            tf.summary.scalar("eval_" + key, val)
        return metric_dict