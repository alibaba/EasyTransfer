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
import random
from collections import OrderedDict
import collections
from .tokenization import convert_to_unicode
from .preprocessor import Preprocessor, PreprocessorConfig, truncate_seq_pair

MaskedLmInstance = collections.namedtuple("MaskedLmInstance", ["index", "label"])

def create_chinese_subwords(segment):
    new_segment = []
    for token in segment:
        if u'\u4e00' > token[0] or token[0] > u'\u9fa5':
            new_segment.append(token)
        else:
            if len(token) > 1:
                new_segment.append(token[0])
                for ele in token[1:]:
                    new_segment.append("##"+ele)
            else:
                new_segment.append(token)
    return new_segment

def create_int_feature(values):
    feature = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
    return feature


def create_float_feature(values):
    feature = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
    return feature


def create_masked_lm_predictions(tokens, masked_lm_prob,
                                 max_predictions_per_seq, vocab_words, do_whole_word_mask, rng):
    """Creates the predictions for the masked LM objective."""
    cand_indexes = []
    for (i, token) in enumerate(tokens):
        if token == "[CLS]" or token == "[SEP]":
            continue
        # Whole Word Masking means that if we mask all of the wordpieces
        # corresponding to an original word. When a word has been split into
        # WordPieces, the first token does not have any marker and any subsequence
        # tokens are prefixed with ##. So whenever we see the ## token, we
        # append it to the previous set of word indexes.
        #
        # Note that Whole Word Masking does *not* change the training code
        # at all -- we still predict each WordPiece independently, softmaxed
        # over the entire vocabulary.
        if (do_whole_word_mask and len(cand_indexes) >= 1 and
                token.startswith("##")):
            cand_indexes[-1].append(i)
        else:
            cand_indexes.append([i])

    rng.shuffle(cand_indexes)

    output_tokens = list(tokens)

    num_to_predict = min(max_predictions_per_seq,
                         max(1, int(round(len(tokens) * masked_lm_prob))))

    masked_lms = []
    covered_indexes = set()
    for index_set in cand_indexes:
        if len(masked_lms) >= num_to_predict:
            break
        # If adding a whole-word mask would exceed the maximum number of
        # predictions, then just skip this candidate.
        if len(masked_lms) + len(index_set) > num_to_predict:
            continue
        is_any_index_covered = False
        for index in index_set:
            if index in covered_indexes:
                is_any_index_covered = True
                break
        if is_any_index_covered:
            continue
        for index in index_set:
            covered_indexes.add(index)

            masked_token = None
            # 80% of the time, replace with [MASK]
            if rng.random() < 0.8:
                masked_token = "[MASK]"
            else:
                # 10% of the time, keep original
                if rng.random() < 0.5:
                    masked_token = tokens[index]
                # 10% of the time, replace with random word
                else:
                    masked_token = vocab_words[rng.randint(0, len(vocab_words) - 1)]

            output_tokens[index] = masked_token

            masked_lms.append(MaskedLmInstance(index=index, label=tokens[index]))
    assert len(masked_lms) <= num_to_predict
    masked_lms = sorted(masked_lms, key=lambda x: x.index)

    masked_lm_positions = []
    masked_lm_labels = []
    for p in masked_lms:
        masked_lm_positions.append(p.index)
        masked_lm_labels.append(p.label)

    return (output_tokens, masked_lm_positions, masked_lm_labels)

class TrainingInstance(object):
    """A single training instance (sentence pair)."""

    def __init__(self, tokens, segment_ids, masked_lm_positions, masked_lm_labels,
                 is_random_next):
        self.tokens = tokens
        self.segment_ids = segment_ids
        self.is_random_next = is_random_next
        self.masked_lm_positions = masked_lm_positions
        self.masked_lm_labels = masked_lm_labels

class PretrainPreprocessorConfig(PreprocessorConfig):

    def __init__(self, **kwargs):
        super(PretrainPreprocessorConfig, self).__init__(**kwargs)

        self.input_schema = kwargs.get("input_schema")
        self.sequence_length = kwargs.get("sequence_length")
        self.first_sequence = kwargs.get("first_sequence")
        self.second_sequence = kwargs.get("second_sequence")
        self.label_name = kwargs.get("label_name")
        self.label_enumerate_values = kwargs.get("label_enumerate_values")
        self.max_predictions_per_seq = kwargs.get("max_predictions_per_seq")
        self.masked_lm_prob = kwargs.get("masked_lm_prob", 0.15)
        self.do_whole_word_mask = kwargs.get("do_whole_word_mask", True)

class PretrainPreprocessor(Preprocessor):

    config_class = PretrainPreprocessorConfig

    def __init__(self, config, **kwargs):
        Preprocessor.__init__(self, config, **kwargs)
        self.config = config

        self.input_tensor_names = []
        for schema in config.input_schema.split(","):
            name = schema.split(":")[0]
            self.input_tensor_names.append(name)

        self.vocab_words = list(self.config.tokenizer.vocab.keys())

        self.rng = random.Random(12345)


        self.label_idx_map = OrderedDict()
        if self.config.label_enumerate_values is not None:
            for (i, label) in enumerate(self.config.label_enumerate_values.split(",")):
                self.label_idx_map[convert_to_unicode(label)] = i

        self.feature_type = kwargs.get('feature_type', "pretrain_lm")


    def set_feature_schema(self):
        if self.mode.startswith("predict") or self.mode == "preprocess":
            self.output_schema = self.config.output_schema
        self.output_tensor_names = ["input_ids", "input_mask", "segment_ids",
                                    "masked_lm_positions", "masked_lm_ids", "masked_lm_weights"]

        if self.feature_type == "pretrain_lm":
            self.Tout = [tf.string] * 6
            self.seq_lens = [self.config.sequence_length] * 3 + [self.config.max_predictions_per_seq] * 3
            self.feature_value_types = [tf.int64] * 5 + [tf.float32]
        elif self.feature_type == "pretrain_multimodel":
            self.Tout = [tf.string] * 7
            self.seq_lens = [self.config.sequence_length] * 3 + [self.config.max_predictions_per_seq] * 3 + [4]
            self.feature_value_types = [tf.int64] * 5 + [tf.float32] + [tf.int64]




    def convert_example_to_features(self, items):

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

        tokens, masked_lm_positions, masked_lm_labels = \
            create_masked_lm_predictions(tokens, self.config.masked_lm_prob,
                                     self.config.max_predictions_per_seq, self.vocab_words, self.config.do_whole_word_mask, self.rng)

        input_ids = self.config.tokenizer.convert_tokens_to_ids(tokens)
        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < self.config.sequence_length:
            input_ids.append(0)
            input_mask.append(0)

        assert len(input_ids) == self.config.sequence_length
        assert len(input_mask) == self.config.sequence_length
        assert len(segment_ids) == self.config.sequence_length

        masked_lm_positions = list(masked_lm_positions)
        masked_lm_ids = self.config.tokenizer.convert_tokens_to_ids(masked_lm_labels)

        masked_lm_weights = [1.0] * len(masked_lm_ids)

        if len(masked_lm_positions) >= self.config.max_predictions_per_seq:
            masked_lm_positions = masked_lm_positions[0:self.config.max_predictions_per_seq]
            masked_lm_ids = masked_lm_ids[0:self.config.max_predictions_per_seq]
            masked_lm_weights = masked_lm_weights[0:self.config.max_predictions_per_seq]

        while len(masked_lm_positions) < self.config.max_predictions_per_seq:
            masked_lm_positions.append(0)
            masked_lm_ids.append(0)
            masked_lm_weights.append(0.0)

        if self.feature_type == "pretrain_lm":
            return ' '.join([str(t) for t in input_ids]), \
                   ' '.join([str(t) for t in input_mask]), \
                   ' '.join([str(t) for t in segment_ids]), \
                   ' '.join([str(t) for t in masked_lm_positions]), \
                   ' '.join([str(t) for t in masked_lm_ids]), \
                   ' '.join([str(t) for t in masked_lm_weights])

        elif self.feature_type == "pretrain_multimodel":
            return ' '.join([str(t) for t in input_ids]), \
                   ' '.join([str(t) for t in input_mask]), \
                   ' '.join([str(t) for t in segment_ids]), \
                   ' '.join([str(t) for t in masked_lm_positions]), \
                   ' '.join([str(t) for t in masked_lm_ids]), \
                   ' '.join([str(t) for t in masked_lm_weights]), \
                   ' '.join(sorted(random.sample([str(x) for x in range(10)], 4)))