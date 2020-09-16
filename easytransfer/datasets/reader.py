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

import sys
sys.path.append("../..")
import tensorflow as tf
from easytransfer.engines.distribution import Process
from collections import OrderedDict

class Reader(Process):
    def __init__(self,
                 batch_size,
                 is_training,
                 thread_num=1,
                 input_queue=None,
                 output_queue=None,
                 job_name='DISTReader',
                 **kwargs):

        Process.__init__(self, job_name,
                         thread_num,
                         input_queue,
                         output_queue,
                         batch_size=batch_size)

        self.num_train_examples = 0
        self.num_eval_examples = 0

        self.is_training = is_training
        self.num_parallel_batches = kwargs.pop("num_parallel_batches", 1)
        self.shuffle_buffer_size = kwargs.pop("shuffle_buffer_size", None)
        self.prefetch_buffer_size = kwargs.pop("prefetch_buffer_size", 1)
        self.input_schema = kwargs.pop("input_schema", None)
        # for all mode, generate tf.Tensor placeholders
        # all mode need a input_schema, column_name:type:length
        self.input_tensors = OrderedDict()
        self.input_tensor_names = []
        for schema in self.input_schema.split(","):
            name = schema.split(":")[0]
            self.input_tensor_names.append(name)
            type = schema.split(":")[1]
            seq_len = int(schema.split(":")[2])
            if type == "int":
                tensor_type = tf.int64
                default_value = 0
            elif type == "float":
                tensor_type = tf.float32
                default_value = 0.0
            elif type == "str":
                tensor_type = tf.string
                default_value = ''
            elif type == "base64":
                tensor_type = "base64"
                default_value = "base64"
            else:
                raise ValueError("unsupported feature type")

            self.input_tensors[name] = tf.io.FixedLenFeature([seq_len], tensor_type, default_value)
        distribution_strategy = kwargs.pop("distribution_strategy", None)
        num_micro_batches = kwargs.pop("num_micro_batches", 1)
        if distribution_strategy == "ExascaleStrategy":
            self.batch_size = batch_size * num_micro_batches
        else:
            self.batch_size = batch_size

        tf.logging.info("num_parallel_batches {}".format(self.num_parallel_batches))
        tf.logging.info("shuffle_buffer_size {}".format(self.shuffle_buffer_size))
        tf.logging.info("prefetch_buffer_size {}".format(self.prefetch_buffer_size))
        tf.logging.info("batch_size {}".format(self.batch_size))
        tf.logging.info("distribution_strategy {}".format(distribution_strategy))
        tf.logging.info("num_micro_batches {}".format(num_micro_batches))
        tf.logging.info("input_schema {}".format(self.input_schema))

    def _get_data_pipeline(self, dataset, _decode_fn):
        if self.is_training:
            if self.shuffle_buffer_size is None:
                tf.logging.info("Random shuffle on the whole {} training examples".format(self.num_train_examples))
                self.shuffle_buffer_size = self.num_train_examples
            dataset = dataset.repeat()
            dataset = dataset.shuffle(buffer_size=self.shuffle_buffer_size)
        else:
            dataset = dataset.repeat(1)

        return self._map_batch_prefetch(dataset, _decode_fn)

    def _map_batch_prefetch(self, dataset, decode_fn):
        dataset = dataset.apply(
            tf.data.experimental.map_and_batch(
                lambda *record: decode_fn(*record),
                batch_size=self.batch_size,
                num_parallel_batches=self.num_parallel_batches,
                drop_remainder=False))
        dataset = dataset.prefetch(self.prefetch_buffer_size)
        return dataset

    def process(self, input_data):
        raise NotImplementedError("must be implemented in descendants")








