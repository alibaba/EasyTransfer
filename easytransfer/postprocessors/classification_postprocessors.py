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

import six
import numpy as np
from easytransfer.engines.distribution import Process


class ClassificationPostprocessor(Process):
    """ Postprocessor for text classification, convert label_id to the label_name

    """
    def __init__(self,
                 label_enumerate_values,
                 output_schema,
                 thread_num=None,
                 input_queue=None,
                 output_queue=None,
                 prediction_colname="predictions",
                 job_name='CLSpostprocessor'):

        super(ClassificationPostprocessor, self).__init__(
            job_name, thread_num, input_queue, output_queue, batch_size=1)
        self.prediction_colname = prediction_colname
        self.label_enumerate_values = label_enumerate_values
        self.output_schema = output_schema
        if label_enumerate_values is not None:
            self.idx_label_map = dict()
            for (i, label) in enumerate(label_enumerate_values.split(",")):
                if six.PY2:
                    self.idx_label_map[i] = label.encode("utf8")
                else:
                    self.idx_label_map[i] = label

    def process(self, in_data):
        """ Post-process the model outputs

        Args:
            in_data (`dict`): a dict of model outputs
        Returns:
            ret (`dict`): a dict of post-processed model outputs
        """
        if self.label_enumerate_values is None:
            return in_data
        tmp = {key: val for key, val in in_data.items()}
        if self.prediction_colname in tmp:
            raw_preds = tmp[self.prediction_colname]
            new_preds = []
            for raw_pred in raw_preds:
                if isinstance(raw_pred, list) or isinstance(raw_pred, np.ndarray):
                    pred = ",".join(
                        [self.idx_label_map[idx] for idx, val
                         in enumerate(raw_pred) if val == 1])
                else:
                    pred = self.idx_label_map[int(raw_pred)]
                new_preds.append(pred)

            tmp[self.prediction_colname] = np.array(new_preds)

        ret = dict()
        for output_col_name in self.output_schema.split(","):
            if output_col_name in tmp:
                ret[output_col_name] = tmp[output_col_name]
        return ret

