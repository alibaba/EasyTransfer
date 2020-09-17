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
from sklearn.metrics import f1_score, accuracy_score, roc_curve, auc, precision_score, recall_score

class ClassificationEvaluator(Evaluator):
    def __init__(self):
        # declare metric names this evaluator will return
        metric_names = [
            'py_accuracy',
            'py_micro_f1',
            'py_macro_f1',
            'py_weighted_f1'

        ]

        # pass metric names to base class
        super(ClassificationEvaluator, self).__init__(metric_names)

    def clear(self):
        self.predictions = []
        self.labels = []

    def add_batch_info(self, predictions, labels):
        for pred, label in zip(predictions, labels):
            self.predictions.append(pred)
            self.labels.append(label)

    def evaluate(self, labels):

        if len(self.predictions) == 0 or len(self.labels) == 0:
            tf.logging.info('empty data to evaluate')
            return {'py_accuracy': 0.0, 'py_micro_f1': 0.0,
                    'py_macro_f1': 0.0, 'py_weighted_f1': 0.0}

        self.labels = np.stack(self.labels)
        self.predictions = np.stack(self.predictions)

        micro_f1 = f1_score(self.labels, self.predictions, labels=labels, average='micro')
        macro_f1 = f1_score(self.labels, self.predictions, labels=labels, average='macro')
        weighted_f1 = f1_score(self.labels, self.predictions, labels=labels, average='weighted')
        accuracy = accuracy_score(self.labels, self.predictions)
        return {'py_accuracy': accuracy, 'py_micro_f1': micro_f1,
                'py_macro_f1': macro_f1, 'py_weighted_f1': weighted_f1}


class MultiLabelEvaluator(Evaluator):
    def __init__(self, num_labels):
        # declare metric names this evaluator will return
        metric_names = [
            'py_micro_f1',
            'py_macro_f1',
            'py_mean_auc'
        ]

        self.num_labels = num_labels
        # pass metric names to base class
        super(MultiLabelEvaluator, self).__init__(metric_names)

    def clear(self):
        '''
        clear internal storage
        '''
        self.predictions = []
        self.probabilities = []
        self.labels = []

    def add_batch_info(self, predictions, labels):
        '''
        store prediction and labels in a internal list
        Args:
          predictions batched prediction result, numpy array with shape N
          labels batched labels, numpy array with shape N
        '''
        for prob, label in zip(predictions, labels):
            self.probabilities.append(prob.tolist())
            self.predictions.append([int(val > 0.5) for val in prob])
            onehot = [0] * self.num_labels
            for cls_idx in label:
                if cls_idx != -1:
                    onehot[cls_idx] = 1
            self.labels.append(onehot)

    @staticmethod
    def macro_f1_score(grt_array, pred_array, n_class):
        # print(f1_score(grt_array, pred_array, average="macro"))
        precs = [precision_score(grt_array[:, i], pred_array[:, i]) for i in range(n_class)]
        recalls = [recall_score(grt_array[:, i], pred_array[:, i]) for i in range(n_class)]
        tf.logging.info("ALL precision scores: {}".format(' | '.join(["{:.4f}".format(t) for t in precs])))
        tf.logging.info("ALL recall scores: {}".format(' | '.join(["{:.4f}".format(t) for t in recalls])))
        prec_ma = np.mean(precs)
        recall_ma = np.mean(recalls)
        if prec_ma == 0 and recall_ma == 0:
            f1_ma = 0
        else:
            f1_ma = 2 * prec_ma * recall_ma / (prec_ma + recall_ma)
        return prec_ma, recall_ma, f1_ma

    @staticmethod
    def micro_f1_score(grt_array, pred_array, n_class):
        # print(f1_score(grt_array, pred_array, average="micro"))
        tp_list = [(grt_array[:, i] & pred_array[:, i]).sum() for i in range(n_class)]
        fp_list = [(grt_array[:, i].sum() - tp_list[i]) for i in range(n_class)]
        tn_list = [((grt_array[:, i] == 0) & (pred_array[:, i] == 0)).sum() for i in range(n_class)]
        fn_list = [((grt_array[:, i] == 0).sum() - tn_list[i]) for i in range(n_class)]
        prec_mi = 1.0 * sum(tp_list) / (sum(tp_list) + sum(fp_list))
        recall_mi = 1.0 * sum(tp_list) / (sum(tp_list) + sum(fn_list))
        if prec_mi == 0 and recall_mi == 0:
            f1_mi = 0
        else:
            f1_mi = 2 * prec_mi * recall_mi / (prec_mi + recall_mi)
        return prec_mi, recall_mi, f1_mi

    @staticmethod
    def mean_auc_score(grt_array, prob_array, n_class):
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_class):
            fpr[i], tpr[i], _ = roc_curve(grt_array[:, i], prob_array[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        fpr["micro"], tpr["micro"], _ = roc_curve(grt_array.ravel(), prob_array.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        tf.logging.info("ALL auc scores: {}".format(
            ' | '.join(["{:.4f}".format(roc_auc[i]) for i in range(n_class)])))
        return roc_auc["micro"]

    def evaluate(self, labels):
        '''
        python evaluation code which will be run after
        all test batched data are predicted
        '''
        if len(self.predictions) == 0 or len(self.labels) == 0:
            tf.logging.info('empty data to evaluate')
            return {'py_micro_f1': 0.0, 'py_macro_f1': 0.0, 'py_mean_auc': 0.0}

        grt_array = np.array(self.labels)
        pred_array = np.array(self.predictions)
        prob_array = np.array(self.probabilities)
        _, _, micro_f1 = self.micro_f1_score(grt_array, pred_array, self.num_labels)
        _, _, macro_f1 = self.macro_f1_score(grt_array, pred_array, self.num_labels)
        mean_auc_score = self.mean_auc_score(grt_array, prob_array, self.num_labels)
        return {'py_micro_f1': micro_f1, 'py_macro_f1': macro_f1, 'py_mean_auc': mean_auc_score}


def classification_eval_metrics(logits, labels, num_labels):
    predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)

    info_dict = {
        "predictions": predictions,
        "labels": labels,
    }


    evaluator = ClassificationEvaluator()

    label_idxs = [i for i in range(num_labels)]
    metric_dict = evaluator.get_metric_ops(info_dict, label_idxs)
    ret_metrics = evaluator.evaluate(label_idxs)

    tf.summary.scalar("accuracy", ret_metrics['py_accuracy'])
    tf.summary.scalar("micro_f1", ret_metrics['py_micro_f1'])
    tf.summary.scalar("macro_f1", ret_metrics['py_macro_f1'])
    tf.summary.scalar("weighted_f1", ret_metrics['py_weighted_f1'])
    return metric_dict

def matthew_corr_metrics(logits, labels):
    predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
    # https://en.wikipedia.org/wiki/Matthews_correlation_coefficient
    tp, tp_op = tf.metrics.true_positives(
        labels=labels, predictions=predictions)
    tn, tn_op = tf.metrics.true_negatives(
        labels=labels, predictions=predictions)
    fp, fp_op = tf.metrics.false_positives(
        labels=labels, predictions=predictions)
    fn, fn_op = tf.metrics.false_negatives(
        labels=labels, predictions=predictions)

    # Compute Matthew's correlation
    mcc = tf.div_no_nan(
        tp * tn - fp * fn,
        tf.pow((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn), 0.5))

    return {"matthew_corr": (mcc, tf.group(tp_op, tn_op, fp_op, fn_op))}


def regression_eval_metrics(logits, labels):
    mse_metric = tf.metrics.mean_squared_error(labels, logits)
    return {"MSE": mse_metric}


def multi_label_eval_metrics(logits, labels, num_labels):
    probabilities = tf.sigmoid(logits)
    info_dict = {
        "predictions": probabilities,
        "labels": labels,
    }
    evaluator = MultiLabelEvaluator(num_labels=num_labels)
    label_ids = [i for i in range(num_labels)]
    metric_dict = evaluator.get_metric_ops(info_dict, label_ids)
    ret_metrics = evaluator.evaluate(label_ids)

    tf.summary.scalar("micro_f1", ret_metrics['py_micro_f1'])
    tf.summary.scalar("macro_f1", ret_metrics['py_macro_f1'])
    tf.summary.scalar("mean_auc", ret_metrics['py_mean_auc'])
    return metric_dict