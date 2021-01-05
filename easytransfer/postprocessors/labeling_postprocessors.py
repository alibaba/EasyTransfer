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
from easytransfer.engines.distribution import Process
import numpy as np


class LabelingPostprocessor(Process):
    """ Postprocessor for sequence labeling, merge the sub-tokens and output the tag for each word

    """
    def __init__(self,
                 label_enumerate_values,
                 output_schema,
                 thread_num=None,
                 input_queue=None,
                 output_queue=None,
                 prediction_colname="predictions",
                 job_name='LabelingPostprocessor'):
        super(LabelingPostprocessor, self).__init__(
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
            tok_to_orig_indexes = [[int(t) for t in lst.split(",")]
                                   for lst in
                                   tmp["tok_to_orig_index"][0][0].split(" ")]
            new_preds = []
            for idx, (raw_pred, tok_to_orig_index) in enumerate(zip(raw_preds, tok_to_orig_indexes)):
                final_pred = list()
                prev_token_idx = -1
                for k in range(min(len(raw_pred), len(tok_to_orig_index))):
                    token_pred = raw_pred[k]
                    token_orig_idx = tok_to_orig_index[k]
                    if token_orig_idx == -1:
                        continue
                    if token_orig_idx == prev_token_idx:
                        continue
                    if token_pred == -1 or token_pred > len(self.idx_label_map):
                        token_pred = len(self.idx_label_map) - 1
                    if self.idx_label_map[token_pred] == "[CLS]" or self.idx_label_map[token_pred] == "[SEP]":
                        token_pred = len(self.idx_label_map) - 1
                    final_pred.append(self.idx_label_map[token_pred])
                    prev_token_idx = token_orig_idx
                raw_sequence_length = max(tok_to_orig_index) + 1
                while len(final_pred) < raw_sequence_length:
                    final_pred.append(self.idx_label_map[len(self.idx_label_map) - 1])

                new_preds.append(" ".join(final_pred))

            tmp[self.prediction_colname] = np.array(new_preds)

        ret = dict()
        for output_col_name in self.output_schema.split(","):
            if output_col_name in tmp:
                ret[output_col_name] = tmp[output_col_name]
        return ret

