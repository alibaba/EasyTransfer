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


from easytransfer.app_zoo.app_utils import get_reader_fn, get_writer_fn, log_duration_time
from easytransfer.app_zoo.base import ApplicationModel
import easytransfer.preprocessors as preprocessors
from easytransfer.engines import distribution


class SerializationModel(ApplicationModel):
    def __init__(self, **kwargs):
        """ Bert Serialization model, convert raw text to BERT format
        """
        super(SerializationModel, self).__init__(**kwargs)
        self.queue_size = 256
        self.thread_num = 16

    @log_duration_time
    def run(self):
        self.proc_executor = distribution.ProcessExecutor(self.queue_size)
        worker_id = self.config.task_index
        num_workers = len(self.config.worker_hosts.split(","))
        proc_executor = distribution.ProcessExecutor(self.queue_size)

        reader = get_reader_fn(self.config.preprocess_input_fp)(input_glob=self.config.preprocess_input_fp,
                                                                input_schema=self.config.input_schema,
                                                                is_training=False,
                                                                batch_size=self.config.preprocess_batch_size,
                                                                slice_id=worker_id,
                                                                slice_count=num_workers,
                                                                output_queue=proc_executor.get_output_queue())

        proc_executor.add(reader)
        preprocessor = preprocessors.get_preprocessor(
            self.config.tokenizer_name_or_path,
            thread_num=self.thread_num,
            input_queue=proc_executor.get_input_queue(),
            output_queue=proc_executor.get_output_queue(),
            preprocess_batch_size=self.config.preprocess_batch_size,
            user_defined_config=self.config,
            app_model_name=self.config.app_model_name)
        proc_executor.add(preprocessor)
        writer = get_writer_fn(self.config.preprocess_output_fp)(output_glob=self.config.preprocess_output_fp,
                                                                 output_schema=self.config.output_schema,
                                                                 slice_id=worker_id,
                                                                 input_queue=proc_executor.get_input_queue())

        proc_executor.add(writer)
        proc_executor.run()
        proc_executor.wait()
