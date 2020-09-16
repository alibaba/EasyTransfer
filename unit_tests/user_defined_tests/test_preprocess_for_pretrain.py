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
from easytransfer import preprocessors
from easytransfer.datasets import CSVWriter, CSVReader


class PretrainSerialization(base_model):
    def __init__(self, **kwargs):
        super(PretrainSerialization, self).__init__(**kwargs)

    def build_logits(self, features, mode=None):
        preprocessor = preprocessors.get_preprocessor(self.config.tokenizer_name_or_path,
                                                      app_model_name="pretrain_language_model")
        input_ids, input_mask, segment_ids, masked_lm_positions, \
        masked_lm_ids, masked_lm_weights = preprocessor(features)
        return input_ids, input_mask, segment_ids, masked_lm_positions, masked_lm_ids, masked_lm_weights

    def build_predictions(self, predict_output):
        input_ids, input_mask, segment_ids, masked_lm_positions, \
        masked_lm_ids, masked_lm_weights = predict_output
        ret_dict = {
            "input_ids": input_ids,
            "input_mask": input_mask,
            "segment_ids": segment_ids,
            "masked_lm_positions": masked_lm_positions,
            "masked_lm_ids": masked_lm_ids,
            "masked_lm_weights": masked_lm_weights
        }
        return ret_dict


class TestDistPreprocess(unittest.TestCase):
    def test_dist_preprocess(self):
        app = PretrainSerialization()

        reader = CSVReader(input_glob=app.preprocess_input_fp,
                           is_training=False,
                           input_schema=app.input_schema,
                           batch_size=app.preprocess_batch_size)

        writer = CSVWriter(output_glob=app.preprocess_output_fp,
                           output_schema=app.output_schema)

        app.run_preprocess(reader=reader, writer=writer)


def main(_):
    unittest.main()


if __name__ == '__main__':
    argvs = ['--null', 'None', '--config', 'config/preprocess_for_pretrain.json', '--mode', 'preprocess']
    tf.app.run(main=main, argv=argvs)
