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
from easytransfer.engines.distribution import Process
import collections

class TFRecordWriter(Process):
    """ Writer tfrecords

    Args:

        output_glob : output file fp
        output_schema : output_schema

    """

    def __init__(self, output_glob, output_schema, input_queue=None):

        job_name = 'DistTFRecordWriter'
        super(TFRecordWriter, self).__init__(job_name, 1, input_queue)

        self.writer = tf.python_io.TFRecordWriter(output_glob)
        self.output_schema = output_schema

    def close(self):
        tf.logging.info('Finished writing')
        self.writer.close()

    def create_int_feature(self, values):
        feature = tf.train.Feature(
            int64_list=tf.train.Int64List(value=list(values)))
        return feature

    def create_float_feature(self, values):
        feature = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
        return feature

    def process(self, features):
        ziped_list = []
        for idx, feat_name in enumerate(self.output_schema.split(",")):
            batch_feat_value = features[feat_name]
            curr_list = []
            for feat in batch_feat_value:
                if len(batch_feat_value.shape) == 1:
                    curr_list.append([feat])
                else:
                    curr_list.append(feat.tolist())
            ziped_list.append(curr_list)


        feat_names = self.output_schema.split(",")
        for ele in zip(*ziped_list):
            features = collections.OrderedDict()
            for feat_name, value in zip(feat_names, ele):
                if isinstance(value[0], float):
                    features[feat_name] = self.create_float_feature(value)
                elif isinstance(value[0], int):
                    features[feat_name] = self.create_int_feature(value)
                elif isinstance(value[0], str):
                    new_value = [int(x) for x in value]
                    features[feat_name] = self.create_int_feature(new_value)

            tf_example = tf.train.Example(features=tf.train.Features(feature=features))

            self.writer.write(tf_example.SerializeToString())



