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
from easytransfer import model_zoo
from easytransfer import preprocessors

class Application(base_model):
    def __init__(self, **kwargs):
        super(Application, self).__init__(**kwargs)

    def build_logits(self, features, mode=None):
        bert_preprocessor = preprocessors.get_preprocessor(self.pretrain_model_name_or_path)
        model = model_zoo.get_pretrained_model(self.pretrain_model_name_or_path)

        input_ids, input_mask, segment_ids = bert_preprocessor(features)[:3]

        outputs = model([input_ids, input_mask, segment_ids], mode=mode)
        pooled_output = outputs[1]

        if mode == tf.estimator.ModeKeys.PREDICT:
            ret = {
                "pooled_output": pooled_output
            }
            return ret

    def build_predictions(self, output):
        pooled_output = output['pooled_output']
        predictions = dict()
        predictions["pooled_output"] = pooled_output
        return predictions

class TestExport(unittest.TestCase):
    def test_export(self):
        app = Application()
        app.export_model()

def main(_):
    unittest.main()

if __name__ == '__main__':
    argvs = ['--null', 'None', '--config', 'config/export.json', '--mode', 'export']
    tf.app.run(main=main, argv=argvs)
