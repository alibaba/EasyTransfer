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

import unittest
import tensorflow as tf
from easytransfer import base_model
from easytransfer.datasets import CSVWriter, CSVReader
from easytransfer.engines.distribution import ProcessExecutor
from easytransfer import preprocessors


class Serialization(base_model):
    def __init__(self, **kwargs):
        super(Serialization, self).__init__(**kwargs)


class TestDistPreprocess(unittest.TestCase):
    def test_dist_preprocess(self):
        app = Serialization()
        queue_size = 1

        proc_executor = ProcessExecutor(queue_size)

        reader = CSVReader(input_glob=app.preprocess_input_fp,
                           input_schema=app.input_schema,
                           is_training=False,
                           batch_size=app.preprocess_batch_size,
                           output_queue=proc_executor.get_output_queue())

        proc_executor.add(reader)

        feature_process = preprocessors.get_preprocessor('google-bert-base-zh',
                                                         app_model_name="pretrain_language_model",
                                                         thread_num=7,
                                                         input_queue=proc_executor.get_input_queue(),
                                                         output_queue=proc_executor.get_output_queue()
                                                         )
        proc_executor.add(feature_process)
        writer = CSVWriter(output_glob=app.preprocess_output_fp,
                           output_schema=app.output_schema,
                           input_queue=proc_executor.get_input_queue())

        proc_executor.add(writer)
        proc_executor.run()
        proc_executor.wait()


def main(_):
    unittest.main()


if __name__ == '__main__':
    argvs = ['--null', 'None', '--config', 'config/preprocess_for_pretrain.json', '--mode', 'preprocess']
    tf.app.run(main=main, argv=argvs)
