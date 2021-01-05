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


import tensorflow as tf
import traceback
from collections import OrderedDict
from .preprocessor import Preprocessor, PreprocessorConfig
from .tokenization import convert_to_unicode


class SequenceLabelingPreprocessorConfig(PreprocessorConfig):
    def __init__(self, **kwargs):
        super(SequenceLabelingPreprocessorConfig, self).__init__(**kwargs)

        self.input_schema = kwargs.get("input_schema")
        self.sequence_length = kwargs.get("sequence_length")
        self.first_sequence = kwargs.get("first_sequence")
        self.second_sequence = kwargs.get("second_sequence")
        self.label_name = kwargs.get("label_name")
        self.label_enumerate_values = kwargs.get("label_enumerate_values")


class SequenceLabelingPreprocessor(Preprocessor):
    """ Preprocessor for sequence labeling

    """
    config_class = SequenceLabelingPreprocessorConfig

    def __init__(self, config, **kwargs):
        Preprocessor.__init__(self, config, **kwargs)
        self.config = config

        self.input_tensor_names = []
        for schema in config.input_schema.split(","):
            name = schema.split(":")[0]
            self.input_tensor_names.append(name)

        self.label_idx_map = OrderedDict()
        if self.config.label_enumerate_values is not None:
            for (i, label) in enumerate(self.config.label_enumerate_values.split(",")):
                self.label_idx_map[convert_to_unicode(label)] = i

    def set_feature_schema(self):
        if self.mode.startswith("predict") or self.mode == "preprocess":
            self.output_schema = self.config.output_schema
        self.output_tensor_names = ["input_ids", "input_mask", "segment_ids", "label_ids", "tok_to_orig_index"]
        self.seq_lens = [self.config.sequence_length] * 4 + [1]
        self.feature_value_types = [tf.int64] * 4 + [tf.string]

    def convert_example_to_features(self, items):
        """ Convert single example to sequence labeling features

        Args:
            items (`dict`): inputs from the reader
        Returns:
            features (`tuple`): (input_ids, input_mask, segment_ids, label_id, tok_to_orig_index)
        """
        content_text = convert_to_unicode(items[self.input_tensor_names.index(self.config.first_sequence)])
        content_tokens = content_text.split(" ")

        if self.config.label_name is not None:
            label_text = convert_to_unicode(items[self.input_tensor_names.index(self.config.label_name)])
            label_tags = label_text.split(" ")
        else:
            label_tags = None

        all_tokens = ["[CLS]"]
        all_labels = [""]
        tok_to_orig_index = [-1]
        for i, token in enumerate(content_tokens):
            sub_tokens = self.config.tokenizer.tokenize(token)
            if not sub_tokens:
                sub_tokens = ["[UNK]"]
            all_tokens.extend(sub_tokens)
            tok_to_orig_index.extend([i] * len(sub_tokens))
            if label_tags is None:
                all_labels.extend(["" for _ in range(len(sub_tokens))])
            else:
                all_labels.extend([label_tags[i] for _ in range(len(sub_tokens))])
        all_tokens = all_tokens[:self.config.sequence_length - 1]
        all_labels = all_labels[:self.config.sequence_length - 1]
        all_tokens.append("[SEP]")
        all_labels.append("")
        tok_to_orig_index.append(-1)

        input_ids = self.config.tokenizer.convert_tokens_to_ids(all_tokens)
        segment_ids = [0] * len(input_ids)
        input_mask = [1] * len(input_ids)
        label_ids = [self.label_idx_map[label] if label else -1 for label in all_labels]

        while len(input_ids) < self.config.sequence_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            label_ids.append(-1)

        assert len(input_ids) == self.config.sequence_length
        assert len(input_mask) == self.config.sequence_length
        assert len(segment_ids) ==  self.config.sequence_length
        assert len(label_ids) == self.config.sequence_length
        assert max(tok_to_orig_index) == len(content_tokens) - 1, "Abnormal line: {}".format(items)
        return ' '.join([str(t) for t in input_ids]), \
               ' '.join([str(t) for t in input_mask]), \
               ' '.join([str(t) for t in segment_ids]), \
               ' '.join([str(t) for t in label_ids]), \
               ','.join([str(t) for t in tok_to_orig_index])
