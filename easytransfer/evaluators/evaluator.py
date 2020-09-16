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
import six
import inspect
from abc import ABCMeta

class Evaluator(six.with_metaclass(ABCMeta, object)):
    def __init__(self, metric_names=[]):
        self.clear()
        self._metric_names = metric_names

    def clear(self):
        raise NotImplementedError("must be implemented in descendants")

    def evaluate(self, labels):
        raise NotImplementedError("must be implemented in descendants")

    def add_batch_info(self, *arg_list):
        raise NotImplementedError("must be implemented in descendants")

    def get_metric_ops(self, tensor_dict, labels):
        '''
        return self-defined metric_ops for tensorflow evaluation
        Args:
          tensor_dict: a dict of tensors for evaluation, each key-value represents a
            param in function `add_batch_sample`, key represents the arg-name, and
            the tensor value will be converted to numpy array through pyfunc
        '''
        if len(self._metric_names) < 1:
            raise ValueError('metric_names should be passed to evaluator.')

        spec = inspect.getargspec(self.add_batch_info)
        func_args = spec.args
        if 'self' in func_args:
            func_args = func_args[1:]
        assert spec.varargs is None and spec.keywords is None, \
            'function add_batch_sample should only have fixed number of args'
        assert len(func_args) > 0, 'function add_batch_sample should have at least one arg'
        feed_list = []
        for arg_name in func_args:
            assert arg_name in tensor_dict, '%s is missing for evaluation' % arg_name
            feed_list.append(tensor_dict[arg_name])

        update_op = tf.py_func(self.add_batch_info, feed_list, [])

        def first_value_func():
            # evaluation will be done only in the first value
            self._metrics = self.evaluate(labels)
            self.clear()
            return np.float32(self._metrics[self._metric_names[0]])

        def value_func_factory(metric_name):
            def value_func():
                return np.float32(self._metrics[metric_name])

            return value_func

        # ensure that the metrics are only evaluated once.
        first_value_op = tf.py_func(first_value_func, [], tf.float32)
        eval_metric_ops = {self._metric_names[0]: (first_value_op, update_op)}
        # because we have done evaluate operation in first_value func, so we need to control the
        # dependency of all value funcs, make sure first value func is running first
        with tf.control_dependencies([first_value_op]):
            for metric_name in self._metric_names[1:]:
                eval_metric_ops[metric_name] = (tf.py_func(
                    value_func_factory(metric_name), [], np.float32), update_op)

        return eval_metric_ops
