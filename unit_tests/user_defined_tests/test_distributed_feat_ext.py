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
from easytransfer.engines.distribution import Process, ProcessExecutor
from tensorflow.python.saved_model.signature_constants import DEFAULT_SERVING_SIGNATURE_DEF_KEY


class Serialization(base_model):
    def __init__(self, **kwargs):
        super(Serialization, self).__init__(**kwargs)


class PredViaSavedModel(Process):
    def __init__(self,
                 saved_model_path,
                 thread_num=1,
                 input_queue=None,
                 output_queue=None,
                 output_schema="pool_output",
                 batch_size=1):
        job_name = "PredViaSavedModel"
        super(PredViaSavedModel, self).__init__(
            job_name,
            thread_num,
            input_queue=input_queue,
            output_queue=output_queue,
            batch_size=batch_size)
        self.sess = tf.Session(graph=tf.Graph())
        meta_graph_def = tf.saved_model.loader.load(self.sess,
                                                    [tf.saved_model.tag_constants.SERVING],
                                                    saved_model_path)
        signature = meta_graph_def.signature_def
        signature_key = DEFAULT_SERVING_SIGNATURE_DEF_KEY
        graph = self.sess.graph

        input_ids_tensor_name = signature[signature_key].inputs["input_ids"].name
        input_mask_tensor_name = signature[signature_key].inputs["input_mask"].name
        segment_ids_tensor_name = signature[signature_key].inputs["segment_ids"].name
        pooled_output_tensor_name = signature[signature_key].outputs["pool_output"].name
        self.input_ids_tensor = graph.get_tensor_by_name(input_ids_tensor_name)
        self.input_mask_tensor = graph.get_tensor_by_name(input_mask_tensor_name)
        self.segment_ids_tensor = graph.get_tensor_by_name(segment_ids_tensor_name)
        self.pool_output = graph.get_tensor_by_name(pooled_output_tensor_name)
        self.predictions = {
            "pool_output": self.pool_output
        }
        self.output_schema = output_schema
        print("load saved_model success")

    def process(self, in_data):
        predictions = self.sess.run(
            self.predictions, feed_dict={
                self.input_ids_tensor: in_data["input_ids"].tolist(),
                self.input_mask_tensor: in_data["input_mask"].tolist(),
                self.segment_ids_tensor: in_data["segment_ids"].tolist()
            })
        ret = {}
        for output_col_name in self.output_schema.split(","):
            if output_col_name in in_data:
                ret[output_col_name] = in_data[output_col_name]
            else:
                ret[output_col_name] = predictions[output_col_name]
        return ret

    def destroy(self):
        self.sess.close()


class TestDistFeatExt(unittest.TestCase):
    def test_dist_feat_ext(self):
        app = Serialization()
        queue_size = 2

        proc_executor = ProcessExecutor(queue_size)

        reader = CSVReader(input_glob=app.predict_input_fp,
                           input_schema=app.input_schema,
                           batch_size=app.predict_batch_size,
                           is_training=False,
                           output_queue=proc_executor.get_output_queue())

        proc_executor.add(reader)

        feature_process = preprocessors.get_preprocessor('google-bert-base-zh',
                                                         thread_num=7,
                                                         input_queue=proc_executor.get_input_queue(),
                                                         output_queue=proc_executor.get_output_queue()
                                                         )

        proc_executor.add(feature_process)
        predictor = PredViaSavedModel(
            saved_model_path=app.predict_checkpoint_path,
            thread_num=7,
            input_queue=proc_executor.get_input_queue(),
            output_queue=proc_executor.get_output_queue(),
            output_schema=app.output_schema,
        )
        proc_executor.add(predictor)
        writer = CSVWriter(output_glob=app.predict_output_fp,
                           output_schema=app.output_schema,
                           input_queue=proc_executor.get_input_queue())

        proc_executor.add(writer)

        proc_executor.run()
        proc_executor.wait()


def main(_):
    unittest.main()


if __name__ == '__main__':
    argvs = ['--null', 'None', '--config', 'config/dist_feat_ext.json', '--mode', 'predict_on_the_fly']
    tf.app.run(main=main, argv=argvs)
