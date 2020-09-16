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

"""Base class to make optimizers weight decay ready."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.training import adam


class LambWeightDecayOptimizer(adam.AdamOptimizer):

    def __init__(self,
                 weight_decay_rate,
                 exclude_from_weight_decay=None,
                 exclude_from_layer_adaptation=None,
                 **kwargs):

        self.exclude_from_weight_decay = exclude_from_weight_decay
        self._decay_var_list = []
        self._layer_adaption_var_list = []

        if exclude_from_layer_adaptation:
            self.exclude_from_layer_adaptation = exclude_from_layer_adaptation
        else:
            self.exclude_from_layer_adaptation = exclude_from_weight_decay

        self._weight_decay = weight_decay_rate
        # The tensors are initialized in call to _prepare
        self._weight_decay_tensor = None
        super(LambWeightDecayOptimizer, self).__init__(**kwargs)

    def _do_use_weight_decay(self, param_name):
        """Whether to use L2 weight decay for `param_name`."""
        if self.exclude_from_weight_decay:
            for r in self.exclude_from_weight_decay:
                if re.search(r, param_name) is not None:
                    return False
        return True

    def _do_layer_adaptation(self, param_name):
        """Whether to do layer-wise learning rate adaptation for `param_name`."""
        if self.exclude_from_layer_adaptation:
            for r in self.exclude_from_layer_adaptation:
                if re.search(r, param_name) is not None:
                    return False
        return True

    def _get_variable_name(self, param_name):
        """Get the variable name from the tensor name."""
        m = re.match("^(.*):\\d+$", param_name)
        if m is not None:
            param_name = m.group(1)
        return param_name

    def apply_gradients(self, grads_and_vars, global_step=None, name=None,
                        decay_var_list=None):
        """Apply gradients to variables and decay the variables.

        This function is the same as Optimizer.apply_gradients except that it
        allows to specify the variables that should be decayed using
        decay_var_list. If decay_var_list is None, all variables in var_list
        are decayed.

        For more information see the documentation of Optimizer.apply_gradients.

        Args:
          grads_and_vars: List of (gradient, variable) pairs as returned by
            `compute_gradients()`.
          global_step: Optional `Variable` to increment by one after the
            variables have been updated.
          name: Optional name for the returned operation.  Default to the
            name passed to the `Optimizer` constructor.
          decay_var_list: Optional list of decay variables.

        Returns:
          An `Operation` that applies the specified gradients. If `global_step`
          was not None, that operation also increments `global_step`.
        """
        for _, var in grads_and_vars:
            var_name = self._get_variable_name(var.name)
            if self._do_use_weight_decay(var_name):
                self._decay_var_list.append(var_name)
            if self._do_layer_adaptation(var_name):
                self._layer_adaption_var_list.append(var_name)
        return super(LambWeightDecayOptimizer, self).apply_gradients(
            grads_and_vars, global_step=global_step, name=name)

    def _prepare(self):
        weight_decay = self._weight_decay
        if callable(weight_decay):
            weight_decay = weight_decay()
        self._weight_decay_tensor = ops.convert_to_tensor(
            weight_decay, name="weight_decay")
        # Call the optimizers _prepare function.
        super(LambWeightDecayOptimizer, self)._prepare()

    def _resource_apply_dense(self, grad, var):

        # m_t = beta1 * m + (1 - beta1) * g_t
        m = self.get_slot(var, 'm')
        m_scaled_g_values = tf.multiply(1.0 - self._beta1_t, grad)
        m_t = m * self._beta1_t + m_scaled_g_values
        m_t = m.assign(m_t, use_locking=self._use_locking)
        # v_t = beta2 * v + (1 - beta2) * (g_t * g_t)
        v = self.get_slot(var, 'v')
        v_scaled_g_values = (grad * grad) * (1.0 - self._beta2_t)
        v_t = v * self._beta2_t + v_scaled_g_values
        v_t = v.assign(v_t, use_locking=self._use_locking)

        beta1_power, beta2_power = self._get_beta_accumulators()

        m_t_hat = m_t / (1. - beta1_power)
        v_t_hat = v_t / (1. - beta2_power)

        v_sqrt = tf.sqrt(v_t_hat)
        update = m_t_hat / (v_sqrt + self._epsilon_t)


        var_name = self._get_variable_name(var.name)
        if self._do_use_weight_decay(var_name):
            update += self._weight_decay * var

        ratio = 1.0
        if self._do_layer_adaptation(var_name):
            w_norm = tf.norm(var, ord=2)
            g_norm = tf.norm(update, ord=2)
            ratio = tf.where(
                tf.greater(w_norm, 0),
                tf.where(tf.greater(g_norm, 0), (w_norm / g_norm), 1.0), 1.0)

        var_update = var - ratio * self._lr_t * update
        return var.assign(var_update, use_locking=self._use_locking).op


    def _resource_scatter_add(self, x, i, v, _=None):
        # last argument allows for one overflow argument, to have the same function
        # signature as state_ops.scatter_add
        with ops.control_dependencies(
                [resource_variable_ops.resource_scatter_add(x.handle, i, v)]):
            return x.value()

    def _resource_apply_sparse(self, grad, var, indices):

        # m_t = beta1 * m + (1 - beta1) * g_t
        m = self.get_slot(var, 'm')
        m_scaled_g_values = grad * (1.0 - self._beta1_t)
        m_t = m.assign(
            m * self._beta1_t, use_locking=self._use_locking)
        with tf.control_dependencies([m_t]):
            m_t = self._resource_scatter_add(m, indices, m_scaled_g_values)

        # v_t = beta2 * v + (1 - beta2) * (g_t * g_t)
        v = self.get_slot(var, 'v')
        v_scaled_g_values = (grad * grad) * (1.0 - self._beta2_t)
        v_t = v.assign(
            v * self._beta2_t, use_locking=self._use_locking)
        with tf.control_dependencies([v_t]):
            v_t = self._resource_scatter_add(v, indices, v_scaled_g_values)

        beta1_power, beta2_power = self._get_beta_accumulators()

        m_t_hat = m_t / (1. - beta1_power)
        v_t_hat = v_t / (1. - beta2_power)

        v_sqrt = tf.sqrt(v_t_hat)
        update = m_t_hat / (v_sqrt + self._epsilon_t)

        var_name = self._get_variable_name(var.name)
        if self._do_use_weight_decay(var_name):
            update += self._weight_decay * var

        ratio = 1.0
        if self._do_layer_adaptation(var_name):
            w_norm = tf.norm(var, ord=2)
            g_norm = tf.norm(update, ord=2)
            ratio = tf.where(
                tf.greater(w_norm, 0),
                tf.where(tf.greater(g_norm, 0), (w_norm / g_norm), 1.0), 1.0)

        var_update = var.assign_sub(
            ratio * self._lr_t * update,
            use_locking=self._use_locking)
        return tf.group(*[var_update, m_t, v_t])
