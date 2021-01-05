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

class TFRecordReader(Reader):
    """ Read tfrecords

    Args:

        input_glob : input file fp
        batch_size : input batch size
        is_training : True or False

    """

    def __init__(self,
                 input_glob,
                 batch_size,
                 is_training,
                 thread_num=1,
                 input_queue=None,
                 output_queue=None,
                 job_name='DISTTFRecordReader',
                 **kwargs):

        super(TFRecordReader, self).__init__(batch_size,
                                             is_training,
                                             thread_num,
                                             input_queue,
                                             output_queue,
                                             job_name,
                                             **kwargs)

        self.input_glob = input_glob
        self.num_train_examples = 0

        if ".list_tfrecord" in self.input_glob:
            if is_training:
                with tf.gfile.Open(input_glob, 'r') as f:
                    for i, line in enumerate(f):
                        if i == 0 and line.strip().isdigit():
                            self.num_train_examples = int(line.strip())
                            break
                        if i % 10 == 0:
                            tf.logging.info("Reading {} files".format(i))
                        fp = line.strip()
                        for record in tf.python_io.tf_record_iterator(fp):
                            self.num_train_examples += 1
                tf.logging.info("{}, total number of training examples {}".format(input_glob, self.num_train_examples))
        else:
            if is_training:
                self.num_train_examples = 0
                for record in tf.python_io.tf_record_iterator(input_glob):
                    self.num_train_examples += 1
                tf.logging.info("{}, total number of training examples {}".format(input_glob, self.num_train_examples))
            else:
                self.num_eval_examples = 0
                for record in tf.python_io.tf_record_iterator(input_glob):
                    self.num_eval_examples += 1
                tf.logging.info("{}, total number of eval examples {}".format(input_glob, self.num_eval_examples))

    def get_input_fn(self):
        def input_fn():
            dataset = tf.data.TFRecordDataset(self.input_glob)
            return self._get_data_pipeline(dataset, self._decode_tfrecord)

        return input_fn

    def _decode_tfrecord(self, record):
        name_to_features = {}
        for name, feature in self.input_tensors.items():
            name_to_features[name] = tf.io.FixedLenFeature(feature.shape, feature.dtype, None)
        example = tf.parse_single_example(record, name_to_features)
        return example


class BundleTFRecordReader(TFRecordReader):
    def __init__(self, input_glob, batch_size, worker_hosts, task_index, distribution_strategy, is_training=False, **kwargs):
        super(BundleTFRecordReader, self).__init__(input_glob, batch_size, is_training, **kwargs)

        self.input_fps = []
        with tf.gfile.Open(input_glob, 'r') as f:
            for line in f:
                line = line.strip()
                if line == '' or line.isdigit():
                    continue
                self.input_fps.append(line)
        self.worker_hosts = worker_hosts
        self.task_index = task_index
        self.distribution_strategy = distribution_strategy
        tf.logging.info("***********Distribution Strategy In Reader is {}***********".format(self.distribution_strategy))
        tf.logging.info("***********Task Index In Reader is {}***********".format(self.task_index))
        tf.logging.info("***********Worker Hosts In Reader is {}***********".format(self.worker_hosts))

    def get_input_fn(self):
        def input_fn():

            if self.is_training:
                d = tf.data.Dataset.from_tensor_slices(tf.constant(self.input_fps))
                if self.distribution_strategy != "WhaleStrategy":
                    d = d.shard(len(self.worker_hosts.split(',')), self.task_index)
                d = d.repeat()
                d = d.shuffle(buffer_size=len(self.input_fps))
                cycle_length = min(4, len(self.input_fps))
                d = d.apply(
                    tf.data.experimental.parallel_interleave(
                        tf.data.TFRecordDataset,
                        sloppy=True,
                        cycle_length=cycle_length))

                d = d.shuffle(buffer_size=self.shuffle_buffer_size)

            else:
                d = tf.data.TFRecordDataset(self.input_fps)
                # Since we evaluate for a fixed number of steps we don't want to encounter
                # out-of-range exceptions.
                d = d.repeat()

            d = self._map_batch_prefetch(d, self._decode_tfrecord)
            return d

        return input_fn