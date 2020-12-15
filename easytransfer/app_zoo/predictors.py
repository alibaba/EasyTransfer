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

import sys

if sys.version_info.major == 2:
    import Queue as queue
else:
    import queue
import traceback
import tensorflow as tf
from tensorflow.python.saved_model.signature_constants import DEFAULT_SERVING_SIGNATURE_DEF_KEY
import easytransfer.engines.distribution as distribution
from easytransfer import preprocessors, postprocessors
from easytransfer.app_zoo.app_utils import get_reader_fn, get_writer_fn, get_label_enumerate_values
from easytransfer.preprocessors.deeptext_preprocessor import DeepTextPreprocessor


class PredictProcess(distribution.Process):
    """ Prediction process for tf saved model """

    def __init__(self,
                 saved_model_path,
                 thread_num=1,
                 input_queue=None,
                 output_queue=None,
                 batch_size=1,
                 job_name="ez_transfer_job"):
        super(PredictProcess, self).__init__(job_name,
                                             thread_num,
                                             input_queue=input_queue,
                                             output_queue=output_queue,
                                             batch_size=batch_size)
        self.sess = tf.Session(graph=tf.Graph())
        meta_graph_def = tf.saved_model.loader.load(self.sess,
                                                    [tf.saved_model.tag_constants.SERVING],
                                                    saved_model_path)
        self.signature = meta_graph_def.signature_def
        self.signature_key = DEFAULT_SERVING_SIGNATURE_DEF_KEY
        self.graph = self.sess.graph

    def set_saved_model_io(self, input_keys, output_keys):
        self.input_keys = input_keys
        self.output_keys = output_keys
        for key in input_keys:
            tensor_name = self.signature[self.signature_key].inputs[key].name
            setattr(self, key + '_tensor', self.graph.get_tensor_by_name(tensor_name))
        self.predictions = dict()
        for key in output_keys:
            tensor_name = self.signature[self.signature_key].outputs[key].name
            self.predictions[key] = self.graph.get_tensor_by_name(tensor_name)

    def process(self, in_data):
        predictions = self.sess.run(
            self.predictions, feed_dict={
                getattr(self, key + '_tensor'): in_data[key]
                for key in self.input_keys})
        ret = {}
        for key, val in in_data.items():
            ret[key] = val
        for key, val in predictions.items():
            ret[key] = val
        return ret

    def destroy(self):
        self.sess.close()


class AppPredictor(object):
    """ Application predictor (support distributed predicting) """

    def __init__(self, config, input_keys, output_keys,
                 thread_num=1, queue_size=256, job_name="app_predictor"):

        self.config = config
        self.worker_id = config.task_index
        self.num_workers = len(config.worker_hosts.split(","))
        self.thread_num = thread_num
        self.input_keys = input_keys
        self.output_keys = output_keys
        self.queue_size = queue_size
        self.job_name = job_name

    def get_default_reader(self):
        return get_reader_fn(self.config.predict_input_fp)(input_glob=self.config.predict_input_fp,
                                                           input_schema=self.config.input_schema,
                                                           is_training=False,
                                                           batch_size=self.config.predict_batch_size,
                                                           output_queue=queue.Queue(),
                                                           slice_id=self.worker_id,
                                                           slice_count=self.num_workers)

    def get_default_writer(self):
        return get_writer_fn(self.config.predict_output_fp)(output_glob=self.config.predict_output_fp,
                                                            output_schema=self.config.output_schema,
                                                            slice_id=self.worker_id,
                                                            input_queue=queue.Queue())

    def get_default_preprocessor(self):
        if hasattr(self.config, "model_name"):
            app_model_name = self.config.model_name
        else:
            app_model_name = None
        if app_model_name == "feat_ext_bert":
            app_model_name = "text_classify_bert"
        return preprocessors.get_preprocessor(
            self.config.pretrain_model_name_or_path,
            thread_num=self.thread_num,
            input_queue=queue.Queue(),
            output_queue=queue.Queue(),
            preprocess_batch_size=self.config.predict_batch_size,
            user_defined_config=self.config,
            app_model_name=app_model_name)

    def get_default_postprocessor(self):
        if hasattr(self.config, "label_enumerate_values"):
            label_enumerate_values = get_label_enumerate_values(self.config.label_enumerate_values)
        else:
            label_enumerate_values = None
        if hasattr(self.config, "model_name"):
            app_model_name = self.config.model_name
        else:
            app_model_name = None
        return postprocessors.get_postprocessors(
            label_enumerate_values=label_enumerate_values,
            output_schema=self.config.output_schema,
            thread_num=self.thread_num,
            input_queue=queue.Queue(),
            output_queue=queue.Queue(),
            app_model_name=app_model_name)

    def get_predictor(self):
        predictor = PredictProcess(saved_model_path=self.config.predict_checkpoint_path,
                                   thread_num=self.thread_num,
                                   input_queue=queue.Queue(),
                                   output_queue=queue.Queue(),
                                   job_name=self.job_name)
        predictor.set_saved_model_io(input_keys=self.input_keys, output_keys=self.output_keys)
        return predictor

    def run_predict(self, reader=None, preprocessor=None, postprocessor=None, writer=None):
        self.proc_executor = distribution.ProcessExecutor(self.queue_size)
        reader = reader if reader else self.get_default_reader()
        reader.output_queue = self.proc_executor.get_output_queue()
        self.proc_executor.add(reader)
        preprocessor = preprocessor if preprocessor else self.get_default_preprocessor()
        preprocessor.input_queue = self.proc_executor.get_input_queue()
        preprocessor.output_queue = self.proc_executor.get_output_queue()
        self.proc_executor.add(preprocessor)
        predictor = self.get_predictor()
        predictor.input_queue = self.proc_executor.get_input_queue()
        predictor.output_queue = self.proc_executor.get_output_queue()
        self.proc_executor.add(predictor)
        posprocessor = postprocessor if postprocessor else self.get_default_postprocessor()
        posprocessor.input_queue = self.proc_executor.get_input_queue()
        posprocessor.output_queue = self.proc_executor.get_output_queue()
        self.proc_executor.add(posprocessor)
        writer = writer if writer else self.get_default_writer()
        writer.input_queue = self.proc_executor.get_input_queue()
        self.proc_executor.add(writer)
        self.proc_executor.run()
        self.proc_executor.wait()
        writer.close()


def run_app_predictor(config):
    try:
        if config.model_name == "feat_ext_bert":
            predictor = AppPredictor(config,
                                     input_keys=["input_ids", "input_mask", "segment_ids"],
                                     output_keys=["pool_output", "first_token_output", "all_hidden_outputs"],
                                     job_name="ez_bert_feat")
            predictor.run_predict()
        elif config.model_name in ["text_comprehension_bert", "text_comprehension_bert_hae"]:
            input_keys = ["input_ids", "input_mask", "segment_ids"] if config.model_name == "text_comprehension_bert" \
                else ["input_ids", "input_mask", "segment_ids", "history_answer_marker"]
            predictor = AppPredictor(config,
                                     input_keys=input_keys,
                                     output_keys=["start_logits", "end_logits"],
                                     job_name=config.model_name + "_predictor")
            preprocessor = preprocessors.get_preprocessor(
                config.pretrain_model_name_or_path,
                thread_num=predictor.thread_num,
                input_queue=queue.Queue(),
                output_queue=queue.Queue(),
                preprocess_batch_size=config.predict_batch_size,
                user_defined_config=config,
                app_model_name=config.model_name)
            postprocessor = postprocessors.get_postprocessors(
                n_best_size=int(config.n_best_size) if hasattr(config, "n_best_size") else 20,
                max_answer_length=int(config.max_answer_length) if hasattr(config, "max_answer_length") else 30,
                output_schema=config.output_schema,
                app_model_name=config.model_name,
                thread_num=predictor.thread_num,
                input_queue=queue.Queue(),
                output_queue=queue.Queue())
            predictor.run_predict(preprocessor=preprocessor, postprocessor=postprocessor)
        elif config.model_name in ["text_match_dam", "text_match_damplus", "text_match_bicnn",
                                   "text_match_hcnn", "text_classify_cnn"]:
            predictor = AppPredictor(config,
                                     input_keys=["input_ids_a", "input_mask_a", "input_ids_b", "input_mask_b"],
                                     output_keys=["predictions", "probabilities", "logits"],
                                     job_name=config.model_name + "_predictor")
            preprocessor = DeepTextPreprocessor(config,
                                                thread_num=predictor.thread_num,
                                                input_queue=queue.Queue(),
                                                output_queue=queue.Queue(),
                                                job_name=config.model_name + "_predictor")
            predictor.run_predict(preprocessor=preprocessor)
        elif config.model_name in ["text_match_bert_two_tower"]:
            raise NotImplementedError
        else:
            predictor = AppPredictor(config,
                                     input_keys=["input_ids", "input_mask", "segment_ids"],
                                     output_keys=["predictions", "probabilities", "logits"],
                                     job_name=config.model_name + "_predictor")
            predictor.run_predict()
    except Exception as e:
        traceback.print_exc(file=sys.stderr)
        raise RuntimeError(str(e))
