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

import os
import tensorflow as tf

from easytransfer import base_model
from easytransfer import preprocessors
from easytransfer.datasets import CSVReader, CSVWriter


class SerializationPredictModel(base_model):
    def __init__(self, **kwargs):
        super(SerializationPredictModel, self).__init__(**kwargs)

    def build_logits(self, features, mode=None):
        bert_preprocessor = preprocessors.get_preprocessor(self.config.tokenizer_name_or_path)
        input_ids, input_mask, segment_ids, label_id = bert_preprocessor(features)
        return input_ids, input_mask, segment_ids

    def build_predictions(self, predict_output):
        input_ids, input_mask, segment_ids = predict_output
        ret_dict = {
            "input_ids": input_ids,
            "input_mask": input_mask,
            "segment_ids": segment_ids
        }
        return ret_dict


class TestPreprocess(unittest.TestCase):

    def test_preprocess_for_predict(self):
        app = SerializationPredictModel()

        reader = CSVReader(input_glob=app.preprocess_input_fp,
                           is_training=False,
                           input_schema=app.input_schema,
                           batch_size=app.preprocess_batch_size)

        writer = CSVWriter(output_glob=app.preprocess_output_fp,
                           output_schema=app.output_schema)

        app.run_preprocess(reader=reader, writer=writer)

        self.assertTrue(os.path.exists('output/preprocess_output_for_predict.csv'))
        lines = open('output/preprocess_output_for_predict.csv', 'r').readlines()
        pd = lines[0].strip()
        gt = "101,1352,1282,671,5709,1446,2990,7583,102,7027,1377,809,2990,5709,1446,102\t1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1\t0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1"
        self.assertTrue(pd == gt)
        pd = lines[1].strip()
        gt = "101,5709,1446,3118,2898,7770,7188,4873,102,711,784,720,1351,802,2140,102\t1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1\t0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1"
        self.assertTrue(pd == gt)
        pd = lines[2].strip()
        gt = "101,2769,4638,6010,6009,5709,1446,3118,102,2769,1168,3118,802,2140,2141,102\t1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1\t0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1"
        self.assertTrue(pd == gt)


def main(_):
    unittest.main()


if __name__ == '__main__':
    argvs = ['--null', 'None', '--config', 'config/preprocess_for_predict.json', '--mode', 'preprocess']
    tf.app.run(main=main, argv=argvs)
