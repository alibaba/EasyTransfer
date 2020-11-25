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

def avgloss_logger_hook(max_steps, loss, model_dir, log_step_count_steps, task_index):
    class _LoggerHook(tf.train.SessionRunHook):
        """Logs loss and runtime."""

        def __init__(self):
            self.avg_loss = None
            self.max_steps = max_steps
            self.decay = 0.99
            self.writer = tf.summary.FileWriter(model_dir+"/avg_loss")
            self.log_step_count_steps  = log_step_count_steps
            self.task_index = task_index

        def begin(self):
            self._step = -1

        def before_run(self, run_context):
            self._step += 1
            return tf.train.SessionRunArgs([loss])

        def after_run(self, run_context, run_values):
            loss_value = run_values.results[0]
            if self.avg_loss is None:
                self.avg_loss = loss_value
            else:
                #Exponential Moving Average
                self.avg_loss = self.avg_loss * self.decay + (1 - self.decay) * loss_value

            if self._step % self.log_step_count_steps == 0 and self.task_index == 0:
                progress = float(self._step) / self.max_steps * 100.0
                summary = tf.Summary()
                summary.value.add(tag='avg_loss', simple_value=self.avg_loss)
                self.writer.add_summary(summary, self._step)
                tf.logging.info(
                    'progress = %.2f%%, avg_loss = %.6f' % (progress, float(self.avg_loss)))

    return _LoggerHook()
