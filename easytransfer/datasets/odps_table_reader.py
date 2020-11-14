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

class OdpsTableReader(Reader):
    """ Read odps table

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
                 slice_id=0,
                 slice_count=1,
                 job_name='DISTOdpsTableReader',
                 **kwargs):

        super(OdpsTableReader, self).__init__(batch_size,
                                              is_training,
                                              thread_num,
                                              input_queue,
                                              output_queue,
                                              job_name,
                                              **kwargs)

        self.input_glob = input_glob

        self.table_reader = tf.python_io.TableReader(
            input_glob,
            selected_cols=','.join(self.input_tensor_names),
            slice_id=slice_id,
            slice_count=slice_count)

        table_schema_list = [item[0] for item in self.table_reader.get_schema().tolist()]

        for input_column_name in self.input_tensor_names:
            if input_column_name not in table_schema_list:
                raise ValueError("{} doesn't appear in odps table schema {}"
                                 .format(input_column_name, ",".join(table_schema_list)))

        if is_training:
            self.num_train_examples = self.table_reader.get_row_count()
            tf.logging.info("{}, total number of training examples {} in slice id {} of slice count {}"
                            .format(input_glob, self.num_train_examples, slice_id, slice_count))
        else:
            self.num_eval_examples = self.table_reader.get_row_count()
            tf.logging.info(
                "{}, total number of eval or predict examples {}".format(input_glob, self.num_eval_examples))

        self.record_defaults = []
        self.feature_types = []
        self.slice_id = slice_id
        self.slice_count = slice_count
        self.shapes = []

        for name, tensor in self.input_tensors.items():
            default_value = tensor.default_value
            shape = tensor.shape
            if shape[0] > 1:
                if default_value == 'base64':
                    default_value = 'base64'
                else:
                    default_value = ''
            self.record_defaults.append([default_value])
            self.shapes.append(tensor.shape)

    def get_input_fn(self):
        def input_fn():
            dataset = tf.data.TableRecordDataset(self.input_glob,
                                                 record_defaults=self.record_defaults,
                                                 selected_cols=','.join(self.input_tensor_names),
                                                 slice_id=self.slice_id,
                                                 slice_count=self.slice_count)

            return self._get_data_pipeline(dataset, self._decode_odps_table)

        return input_fn

    def _decode_odps_table(self, *items):

        num_tensors = len(self.input_tensor_names)
        total_shape = 0
        for shape in self.shapes:
            total_shape += sum(shape)

        ret = dict()
        for idx, (name, feature) in enumerate(self.input_tensors.items()):
            # finetune feature_text
            if total_shape != num_tensors:
                input_tensor = tf.squeeze(items[idx])
                if sum(feature.shape) > 1:
                    default_value = self.record_defaults[idx]
                    if default_value[0] == '':
                        output = tf.string_to_number(
                            tf.string_split(tf.expand_dims(input_tensor, axis=0), delimiter=",").values,
                            feature.dtype)
                        output = tf.reshape(output, [feature.shape[0], ])
                    elif default_value[0] == 'base64':
                        decode_b64_data = tf.io.decode_base64(input_tensor)
                        output = tf.reshape(tf.io.decode_raw(decode_b64_data, out_type=tf.float32),
                                            [feature.shape[0], ])
                else:
                    output = tf.reshape(input_tensor, [1, ])

            elif total_shape == num_tensors:
                # preprocess raw_text
                output = items[idx]
            ret[name] = output
        return ret

    def process(self, input_data):
        while True:
            try:
                batch_records = self.table_reader.read(self.batch_size)
                for _, record in enumerate(batch_records):
                    output_dict = {}
                    for idx, name in enumerate(self.input_tensor_names):
                        output_dict[name] = record[idx]
                    self.put(output_dict)
            except tf.errors.OutOfRangeError:
                raise IndexError('read table data done')
            except tf.python_io.OutOfRangeException:
                raise IndexError('read table data done')
