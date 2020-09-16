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
from easytransfer import layers
from easytransfer import model_zoo
from easytransfer import preprocessors
from easytransfer.datasets import CSVReader, CSVWriter


class Application(base_model):
    def __init__(self, **kwargs):
        super(Application, self).__init__(**kwargs)

    def build_logits(self, features, mode=None):
        preprocessor = preprocessors.get_preprocessor(self.pretrain_model_name_or_path)
        model = model_zoo.get_pretrained_model(self.pretrain_model_name_or_path)

        dense = layers.Dense(self.num_labels,
                             kernel_initializer=layers.get_initializer(0.02),
                             name='dense')

        input_ids, input_mask, segment_ids, label_ids = preprocessor(features)
        outputs = model([input_ids, input_mask, segment_ids], mode=mode)
        pooled_output = outputs[1]
        logits = dense(pooled_output)
        ret = {"logits": logits}
        return ret

    def build_predictions(self, output):
        logits = output['logits']
        predictions = dict()
        predictions["logits"] = logits
        predictions["predictions"] = tf.argmax(logits, axis=-1, output_type=tf.int32)
        return predictions


class TestPredict(unittest.TestCase):

    def test_predict(self):
        app = Application()
        predict_reader = CSVReader(input_glob=app.predict_input_fp,
                                   is_training=False,
                                   input_schema=app.input_schema,
                                   batch_size=app.predict_batch_size)

        predict_writer = CSVWriter(output_glob=app.predict_output_fp,
                                   output_schema=app.output_schema)

        app.run_predict(reader=predict_reader, writer=predict_writer,
                        checkpoint_path=app.predict_checkpoint_path)


def main(_):
    unittest.main()


if __name__ == '__main__':
    argvs = ['--null', 'None', '--config',
             'config/predict_on_the_fly_classification.json', '--mode',
             'predict_on_the_fly']
    tf.app.run(main=main, argv=argvs)
