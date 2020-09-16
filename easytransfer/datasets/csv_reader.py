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
from .reader import Reader

class CSVReader(Reader):
    """ Read csv format

    Args:

        input_glob : input file fp
        batch_size : input batch size
        is_training : True or False
        thread_num: thread number

    """

    def __init__(self,
                 input_glob,
                 batch_size,
                 is_training,
                 thread_num=1,
                 input_queue=None,
                 output_queue=None,
                 job_name='DISTCSVReader',
                 **kwargs):

        super(CSVReader, self).__init__(batch_size,
                                        is_training,
                                        thread_num,
                                        input_queue,
                                        output_queue,
                                        job_name,
                                        **kwargs)

        self.input_glob = input_glob
        if is_training:
            with tf.gfile.Open(input_glob, 'r') as f:
                for record in f:
                    self.num_train_examples += 1
            tf.logging.info("{}, total number of training examples {}".format(input_glob, self.num_train_examples))
        else:
            with tf.gfile.Open(input_glob, 'r') as f:
                for record in f:
                    self.num_eval_examples += 1
            tf.logging.info("{}, total number of eval examples {}".format(input_glob, self.num_eval_examples))

        self.csv_reader = tf.gfile.Open(input_glob)

    def get_input_fn(self):
        def input_fn():
            dataset = tf.data.TextLineDataset(self.input_glob)
            return self._get_data_pipeline(dataset, self._decode_csv)

        return input_fn

    def _decode_csv(self, record):
        record_defaults = []
        tensor_names = []
        shapes = []
        for name, feature in self.input_tensors.items():
            default_value = feature.default_value
            shape = feature.shape
            if shape[0] > 1:
                if default_value == 'base64':
                    default_value = 'base64'
                else:
                    default_value = ''
            else:
                default_value = feature.default_value
            record_defaults.append([default_value])
            tensor_names.append(name)
            shapes.append(feature.shape)

        num_tensors = len(tensor_names)

        items = tf.decode_csv(record, field_delim='\t', record_defaults=record_defaults, use_quote_delim=False)
        outputs = dict()
        total_shape = 0
        for shape in shapes:
            total_shape += sum(shape)

        for idx, (name, feature) in enumerate(self.input_tensors.items()):
            # finetune feature_text
            if total_shape != num_tensors:
                input_tensor = items[idx]
                if sum(feature.shape) > 1:
                    default_value = record_defaults[idx]
                    if default_value[0] == '':
                        output = tf.string_to_number(
                            tf.string_split(tf.expand_dims(input_tensor, axis=0), delimiter=",").values,
                            feature.dtype)
                        output = tf.reshape(output, [feature.shape[0], ])
                    elif default_value[0] == 'base64':
                        decode_b64_data = tf.io.decode_base64(tf.expand_dims(input_tensor, axis=0))
                        output = tf.reshape(tf.io.decode_raw(decode_b64_data, out_type=tf.float32),
                                            [feature.shape[0], ])
                else:
                    output = tf.reshape(input_tensor, [1, ])
            elif total_shape == num_tensors:
                # preprocess raw_text
                output = items[idx]
            outputs[name] = output
        return outputs

    def process(self, input_data):
        for line in self.csv_reader:
            line = line.strip()
            segments = line.split("\t")
            output_dict = {}
            for idx, name in enumerate(self.input_tensor_names):
                output_dict[name] = segments[idx]
            self.put(output_dict)
        raise IndexError("Read tabel done")

    def close(self):
        self.csv_reader.close()

class BundleCSVReader(CSVReader):
    """ Read group of csv formats

    Args:

        input_glob : input file fp
        batch_size : input batch size
        worker_hosts: worker hosts
        task_index: task index
        is_training : True or False

    """
    def __init__(self, input_glob, batch_size, worker_hosts, task_index, is_training=False, **kwargs):
        super(BundleCSVReader, self).__init__(input_glob, batch_size, is_training, **kwargs)

        self.input_fps = []
        with tf.gfile.Open(input_glob, 'r') as f:
            for line in f:
                line = line.strip()
                if line == '' or line.isdigit():
                    continue
                self.input_fps.append(line)
        self.worker_hosts = worker_hosts
        self.task_index = task_index

    def get_input_fn(self):
        def input_fn():
            if self.is_training:
                d = tf.data.Dataset.from_tensor_slices(tf.constant(self.input_fps))
                d = d.shard(len(self.worker_hosts.split(',')), self.task_index)
                d = d.repeat()
                d = d.shuffle(buffer_size=len(self.input_fps))

                cycle_length = min(4, len(self.input_fps))
                d = d.apply(
                    tf.data.experimental.parallel_interleave(
                        tf.data.TextLineDataset,
                        sloppy=True,
                        cycle_length=cycle_length))
                d = d.shuffle(buffer_size=self.shuffle_buffer_size)
            else:
                d = tf.data.Dataset.from_tensor_slices(tf.constant(self.input_fps))
                d = d.shard(len(self.worker_hosts.split(',')), self.task_index)
                d = d.repeat(1)
                cycle_length = min(4, len(self.input_fps))
                d = d.apply(
                    tf.data.experimental.parallel_interleave(
                        tf.data.TextLineDataset,
                        sloppy=True,
                        cycle_length=cycle_length))
                # d = tf.data.TextLineDataset(self.input_fps)
                # Since we evaluate for a fixed number of steps we don't want to encounter
                # out-of-range exceptions.
                #d = d.repeat()

            d = self._map_batch_prefetch(d, self._decode_csv)
            return d

        return input_fn
