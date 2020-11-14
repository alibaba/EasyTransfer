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


from collections import OrderedDict
import tensorflow as tf
from .preprocessor import Preprocessor, PreprocessorConfig, truncate_seq_pair
from .tokenization import convert_to_unicode


class ClassificationRegressionPreprocessorConfig(PreprocessorConfig):
    def __init__(self, **kwargs):
        super(ClassificationRegressionPreprocessorConfig, self).__init__(**kwargs)

        self.input_schema = kwargs.get("input_schema")
        self.output_schema = kwargs.get("output_schema", None)
        self.sequence_length = kwargs.get("sequence_length")
        self.first_sequence = kwargs.get("first_sequence")
        self.second_sequence = kwargs.get("second_sequence")
        self.label_name = kwargs.get("label_name")
        self.label_enumerate_values = kwargs.get("label_enumerate_values")


class ClassificationRegressionPreprocessor(Preprocessor):
    """ Preprocessor for classification/regression task

    """
    config_class = ClassificationRegressionPreprocessorConfig

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

        if hasattr(self.config, "multi_label") and self.config.multi_label is True:
            self.multi_label = True
            self.max_num_labels = self.config.max_num_labels if hasattr(self.config, "max_num_labels") else 5
        else:
            self.multi_label = False
            self.max_num_labels = None

    def set_feature_schema(self):
        if self.mode.startswith("predict") or self.mode == "preprocess":
            self.output_schema = self.config.output_schema
        self.output_tensor_names = ["input_ids", "input_mask", "segment_ids", "label_id"]
        if self.multi_label:
            self.seq_lens = [self.config.sequence_length] * 3 + [self.max_num_labels]
            self.feature_value_types = [tf.int64] * 3 + [tf.int64]
        else:
            self.seq_lens = [self.config.sequence_length] * 3 + [1]
            if len(self.label_idx_map) >= 2:
                self.feature_value_types = [tf.int64] * 4
            else:
                self.feature_value_types = [tf.int64] * 3 + [tf.float32]

    def convert_example_to_features(self, items):
        """ Convert single example to classifcation/regression features

        Args:
            items (`dict`): inputs from the reader
        Returns:
            features (`tuple`): (input_ids, input_mask, segment_ids, label_id)
        """
        text_a = items[self.input_tensor_names.index(self.config.first_sequence)]
        tokens_a = self.config.tokenizer.tokenize(convert_to_unicode(text_a))
        if self.config.second_sequence in self.input_tensor_names:
            text_b = items[self.input_tensor_names.index(self.config.second_sequence)]
            tokens_b = self.config.tokenizer.tokenize(convert_to_unicode(text_b))
            truncate_seq_pair(tokens_a, tokens_b, self.config.sequence_length - 3)
        else:
            if len(tokens_a) > self.config.sequence_length - 2:
                tokens_a = tokens_a[0:(self.config.sequence_length - 2)]
            tokens_b = None

        tokens = []
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                segment_ids.append(1)
            tokens.append("[SEP]")
            segment_ids.append(1)

        input_ids = self.config.tokenizer.convert_tokens_to_ids(tokens)
        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < self.config.sequence_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == self.config.sequence_length
        assert len(input_mask) == self.config.sequence_length
        assert len(segment_ids) == self.config.sequence_length

        if self.config.label_name is not None:
            label_value = items[self.input_tensor_names.index(self.config.label_name)]
            if isinstance(label_value, str) or isinstance(label_value, bytes):
                label = convert_to_unicode(label_value)
            else:
                label = str(label_value)

            if self.multi_label:
                label_ids = [self.label_idx_map[convert_to_unicode(x)] for x in label.split(",") if x]
                label_ids = label_ids[:self.max_num_labels]
                label_ids = label_ids + [-1 for _ in range(self.max_num_labels - len(label_ids))]
                label_ids = [str(t) for t in label_ids]
                label_id = ' '.join(label_ids)
            elif len(self.label_idx_map) >= 2:
                label_id = str(self.label_idx_map[convert_to_unicode(label)])
            else:
                label_id = label

        else:
            label_id = '0'

        return ' '.join([str(t) for t in input_ids]), \
               ' '.join([str(t) for t in input_mask]), \
               ' '.join([str(t) for t in segment_ids]), label_id


class PairedClassificationRegressionPreprocessor(ClassificationRegressionPreprocessor):
    """ Preprocessor for paired classification/regression task

    """
    config_class = ClassificationRegressionPreprocessorConfig

    def __init__(self, config, **kwargs):
        super(PairedClassificationRegressionPreprocessor, self).__init__(config, **kwargs)

    def set_feature_schema(self):
        if self.mode.startswith("predict") or self.mode == "preprocess":
            self.output_schema = self.config.output_schema
        #self.output_tensor_names = ["input_ids", "input_mask", "segment_ids", "label_id"]
        self.output_tensor_names = ["input_ids_a", "input_mask_a", "segment_ids_a",
                                    "input_ids_b", "input_mask_b", "segment_ids_b",
                                    "label_id"]
        self.seq_lens = [self.config.sequence_length] * 6 + [1]
        if len(self.label_idx_map) >= 2:
            self.feature_value_types = [tf.int64] * 6 + [tf.int64]
        else:
            self.feature_value_types = [tf.int64] * 6 + [tf.float32]

    def convert_example_to_features(self, items):
        """ Convert single example to classifcation/regression features

        Args:
            items (`dict`): inputs from the reader
        Returns:
            features (`tuple`): (input_ids_a, input_mask_a, segment_ids_a,
                                 input_ids_b, input_mask_b, segment_ids_b,
                                 label_id)
        """
        assert self.config.first_sequence in self.input_tensor_names \
               and self.config.second_sequence in self.input_tensor_names
        text_a = items[self.input_tensor_names.index(self.config.first_sequence)]
        tokens_a = self.config.tokenizer.tokenize(convert_to_unicode(text_a))

        text_b = items[self.input_tensor_names.index(self.config.second_sequence)]
        tokens_b = self.config.tokenizer.tokenize(convert_to_unicode(text_b))

        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > self.config.sequence_length - 2:
            tokens_a = tokens_a[0:(self.config.sequence_length - 2)]

        if len(tokens_b) > self.config.sequence_length - 2:
            tokens_b = tokens_b[0:(self.config.sequence_length - 2)]

        tokens = []
        segment_ids_a = []
        tokens.append("[CLS]")
        segment_ids_a.append(0)
        for token in tokens_a:
            tokens.append(token)
            segment_ids_a.append(0)
        tokens.append("[SEP]")
        segment_ids_a.append(0)

        input_ids_a = self.config.tokenizer.convert_tokens_to_ids(tokens)

        tokens = []
        segment_ids_b = []
        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                segment_ids_b.append(1)
            tokens.append("[SEP]")
            segment_ids_b.append(1)

        input_ids_b = self.config.tokenizer.convert_tokens_to_ids(tokens)
        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask_a = [1] * len(input_ids_a)
        input_mask_b = [1] * len(input_ids_b)

        # Zero-pad up to the sequence length.
        while len(input_ids_a) < self.config.sequence_length:
            input_ids_a.append(0)
            input_mask_a.append(0)
            segment_ids_a.append(0)

        # Zero-pad up to the sequence length.
        while len(input_ids_b) < self.config.sequence_length:
            input_ids_b.append(0)
            input_mask_b.append(0)
            segment_ids_b.append(0)

        assert len(input_ids_a) == self.config.sequence_length
        assert len(input_mask_a) == self.config.sequence_length
        assert len(segment_ids_a) == self.config.sequence_length
        assert len(input_ids_b) == self.config.sequence_length
        assert len(input_mask_b) == self.config.sequence_length
        assert len(segment_ids_b) == self.config.sequence_length

        # support single/multi classification and regression
        if self.config.label_name is not None:

            label_value = items[self.input_tensor_names.index(self.config.label_name)]
            if isinstance(label_value, str) or isinstance(label_value, bytes):
                label = convert_to_unicode(label_value)
            else:
                label = str(label_value)

            if len(self.label_idx_map) >= 2:
                label_id = str(self.label_idx_map[convert_to_unicode(label)])
            else:
                label_id = label
        else:
            label_id = '0'
        return ' '.join([str(t) for t in input_ids_a]), \
               ' '.join([str(t) for t in input_mask_a]), \
               ' '.join([str(t) for t in segment_ids_a]), \
               ' '.join([str(t) for t in input_ids_b]), \
               ' '.join([str(t) for t in input_mask_b]), \
               ' '.join([str(t) for t in segment_ids_b]), label_id
