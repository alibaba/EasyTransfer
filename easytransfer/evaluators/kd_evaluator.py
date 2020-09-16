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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from collections import OrderedDict
from sklearn.metrics import accuracy_score
import tensorflow as tf
from .evaluator import Evaluator


class ProbesEvaluator(Evaluator):
    def __init__(self, n_layers=12):
        # declare metric names this evaluator will return
        metric_names = ['accuracy_layer_%d' % i for i in range(n_layers + 1)]
        self.n_layers = n_layers
        self.metric_names = metric_names

        # pass metric names to base class
        super(ProbesEvaluator, self).__init__(metric_names)

    def clear(self):
        '''
        clear internal storage
        '''
        self.predictions = [[] for _ in range(self.n_layers + 1)]
        self.labels = []

    def add_batch_info(self, predictions, labels):
        '''
        store prediction and labels in a internal list
        Args:
          predictions batched prediction result, numpy array with shape N
          labels batched labels, numpy array with shape N
        '''
        for i, layer_predictions in enumerate(predictions):
            for pred in layer_predictions:
                self.predictions[i].append(pred)
        for label in labels:
            self.labels.append(label)

    def evaluate(self, labels):
        '''
        python evaluation code which will be run after
        all test batched data are predicted
        '''
        if len(self.predictions) == 0 or len(self.labels) == 0:
            tf.logging.info('empty data to evaluate')
            return {key: 0.0 for key in self.metric_names}

        ret_dict = OrderedDict()
        for i in range(self.n_layers + 1):
            accuracy = accuracy_score(self.labels, self.predictions[i])
            ret_dict['accuracy_layer_%d' % i] = accuracy
        return ret_dict


def teacher_probes_eval_metrics(logits, labels, num_labels):
    """ Building evaluation metrics while evaluating

    Args:
        logits (`Tensor`): list of tensors shape of [None, num_labels]
        labels (`Tensor`): shape of [None]
    Returns:
        ret_dict (`dict`): A dict of each layer accuracy tf.metrics op
    """
    predictions_list = [
        tf.argmax(layer_logits, axis=-1, output_type=tf.int32) for layer_logits in logits]
    info_dict = {
        "predictions": predictions_list,
        "labels": labels,
    }
    evaluator = ProbesEvaluator(n_layers=len(logits) - 1)
    label_ids = [i for i in range(num_labels)]
    metric_dict = evaluator.get_metric_ops(info_dict, label_ids)
    ret_metrics = evaluator.evaluate(label_ids)
    for i in range(len(logits)):
        tf.summary.scalar("eval_accuracy_layer_%d" % i, ret_metrics['accuracy_layer_%d' % i] )
    return metric_dict