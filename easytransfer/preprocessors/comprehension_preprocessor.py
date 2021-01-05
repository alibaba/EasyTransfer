# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
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

import collections
from copy import deepcopy
import os
import six
import uuid
import numpy as np
import tensorflow as tf
from .preprocessor import Preprocessor, PreprocessorConfig
from .tokenization import convert_to_unicode, printable_text


def is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
        return True
    return False


def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a piece of text."""
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens


def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer,
                         orig_answer_text):
    """Returns tokenized answer spans that better match the annotated answer."""

    # The SQuAD annotations are character based. We first project them to
    # whitespace-tokenized words. But then after WordPiece tokenization, we can
    # often find a "better match". For example:
    #
    #   Question: What year was John Smith born?
    #   Context: The leader was John Smith (1895-1943).
    #   Answer: 1895
    #
    # The original whitespace-tokenized answer will be "(1895-1943).". However
    # after tokenization, our tokens will be "( 1895 - 1943 ) .". So we can match
    # the exact answer, 1895.
    #
    # However, this is not always possible. Consider the following:
    #
    #   Question: What country is the top exporter of electornics?
    #   Context: The Japanese electronics industry is the lagest in the world.
    #   Answer: Japan
    #
    # In this case, the annotator chose "Japan" as a character sub-span of
    # the word "Japanese". Since our WordPiece tokenizer does not split
    # "Japanese", we just use "Japanese" as the annotation. This is fairly rare
    # in SQuAD, but does happen.
    tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(doc_tokens[new_start:(new_end + 1)])
            if text_span == tok_answer_text:
                return (new_start, new_end)

    return (input_start, input_end)


def _check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""

    # Because of the sliding window approach taken to scoring documents, a single
    # token can appear in multiple documents. E.g.
    #  Doc: the man went to the store and bought a gallon of milk
    #  Span A: the man went to the
    #  Span B: to the store and bought
    #  Span C: and bought a gallon of
    #  ...
    #
    # Now the word 'bought' will have two scores from spans B and C. We only
    # want to consider the score with "maximum context", which we define as
    # the *minimum* of its left and right context (the *sum* of left and
    # right context will always be the same, of course).
    #
    # In the example the maximum context for 'bought' would be span C since
    # it has 1 left context and 3 right context, while span B has 4 left context
    # and 0 right context.
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span.start + doc_span.length - 1
        if position < doc_span.start:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span.start
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index


class ComprehensionPreprocessorConfig(PreprocessorConfig):
    def __init__(self, **kwargs):
        super(ComprehensionPreprocessorConfig, self).__init__(**kwargs)

        self.input_schema = kwargs.get("input_schema")
        self.sequence_length = kwargs.get("sequence_length")
        self.first_sequence = kwargs.get("first_sequence")
        self.second_sequence = kwargs.get("second_sequence")
        self.label_name = kwargs.get("label_name")
        self.label_enumerate_values = kwargs.get("label_enumerate_values")


class Example(object):
    """A single training/test example for simple sequence classification.

       For scripts without an answer, the start and end position are -1.
    """

    def __init__(self,
                 qas_id,
                 question_text,
                 doc_tokens,
                 orig_answer_text=None,
                 start_position=None,
                 end_position=None,
                 is_impossible=False):
        self.qas_id = qas_id
        self.question_text = question_text
        self.doc_tokens = doc_tokens
        self.orig_answer_text = orig_answer_text
        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "qas_id: %s" % (printable_text(self.qas_id))
        s += ", question_text: %s" % (
            printable_text(self.question_text))
        s += ", doc_tokens: [%s]" % (" ".join(self.doc_tokens))
        if self.start_position:
            s += ", start_position: %d" % (self.start_position)
        if self.start_position:
            s += ", end_position: %d" % (self.end_position)
        if self.start_position:
            s += ", is_impossible: %r" % (self.is_impossible)
        return s


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 unique_id,
                 qas_id,
                 example_index,
                 doc_span_index,
                 doc_tokens,
                 tokens,
                 token_to_orig_map,
                 token_is_max_context,
                 input_ids,
                 input_mask,
                 segment_ids,
                 start_position=None,
                 end_position=None,
                 is_impossible=None):
        self.unique_id = unique_id
        self.qas_id = qas_id
        self.example_index = example_index
        self.doc_span_index = doc_span_index
        self.doc_tokens = doc_tokens
        self.tokens = tokens
        self.token_to_orig_map = token_to_orig_map
        self.token_is_max_context = token_is_max_context
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible


class ComprehensionPreprocessor(Preprocessor):
    """ Preprocessor for single-turn text comprehension

    """
    config_class = ComprehensionPreprocessorConfig

    def __init__(self, config, thread_num=1, **kwargs):
        super(ComprehensionPreprocessor, self).__init__(config, thread_num=thread_num, **kwargs)
        self.config = config

        self.max_seq_length = config.sequence_length
        self.context_col_name = config.first_sequence
        self.max_query_length = int(config.max_query_length)
        self.doc_stride = int(config.doc_stride) if hasattr(config, "doc_stride") else 128
        self.query_col_name = config.second_sequence
        self.answer_col_name = config.label_name

        self.input_tensor_names = []
        if "/" in config.pretrain_model_name_or_path:
            dirname = os.path.dirname(config.pretrain_model_name_or_path)
            self.language = dirname.split("-")[-1]
        else:
            self.language = config.pretrain_model_name_or_path.split("-")[-1]

        input_schema = config.input_schema
        self.input_tensor_names = []
        for schema in input_schema.split(","):
            name = schema.split(":")[0]
            self.input_tensor_names.append(name)
        self.example_count = 0

    def convert_example_to_features(self, items):
        """ Convert single example to multiple input features

        Args:
            items (`dict`): inputs from the reader
        Returns:
            features (`list`): list of `InputFeature`
        """
        paragraph_text = convert_to_unicode(items[self.context_col_name])
        question_id_list = convert_to_unicode(items[self.query_col_name]).split("||")
        questions = list(zip(question_id_list[::2], question_id_list[1::2]))
        if self.answer_col_name in self.input_tensor_names:
            answer_starts_list = convert_to_unicode(items[self.answer_col_name]).split("||")
            answers = list(zip(answer_starts_list[::2], [int(t) for t in answer_starts_list[1::2]]))
            is_training = True
        else:
            answers = list()
            is_training = False
        if self.mode.startswith("predict"):
            is_training = False

        doc_tokens = []
        char_to_word_offset = []
        prev_is_whitespace = True
        for c in paragraph_text:
            if is_whitespace(c):
                prev_is_whitespace = True
            else:
                if prev_is_whitespace:
                    doc_tokens.append(c)
                else:
                    doc_tokens[-1] += c
                prev_is_whitespace = False
            char_to_word_offset.append(len(doc_tokens) - 1)

        examples = list()
        for idx, (question_text, qas_id), in enumerate(questions):
            start_position = None
            end_position = None
            orig_answer_text = None
            is_impossible = False
            if is_training:
                orig_answer_text, answer_offset = answers[idx]
                is_impossible = (answer_offset == -1)
                if not is_impossible:
                    answer_length = len(orig_answer_text)
                    start_position = char_to_word_offset[answer_offset]
                    end_position = char_to_word_offset[answer_offset + answer_length - 1]
                    # Only add answers where the text can be exactly recovered from the
                    # document. If this CAN'T happen it's likely due to weird Unicode
                    # stuff so we will just skip the example.
                    #
                    # Note that this means for training mode, every example is NOT
                    # guaranteed to be preserved.
                    actual_text = " ".join(
                        doc_tokens[start_position:(end_position + 1)])
                    cleaned_answer_text = " ".join(
                        whitespace_tokenize(orig_answer_text))
                    if actual_text.find(cleaned_answer_text) == -1:
                        tf.logging.warning("Could not find answer: '%s' vs. '%s'",
                                           actual_text, cleaned_answer_text)
                        continue
                else:
                    start_position = -1
                    end_position = -1
                    orig_answer_text = ""

            example = Example(
                qas_id=qas_id,
                question_text=question_text,
                doc_tokens=doc_tokens,
                orig_answer_text=orig_answer_text,
                start_position=start_position,
                end_position=end_position,
                is_impossible=is_impossible)
            examples.append(example)

        features = list()
        for (example_index, example) in enumerate(examples):
            query_tokens = self.config.tokenizer.tokenize(example.question_text)

            if len(query_tokens) > self.max_query_length:
                query_tokens = query_tokens[0:self.max_query_length]

            tok_to_orig_index = []
            orig_to_tok_index = []
            all_doc_tokens = []
            for (i, token) in enumerate(example.doc_tokens):
                orig_to_tok_index.append(len(all_doc_tokens))
                sub_tokens = self.config.tokenizer.tokenize(token)
                for sub_token in sub_tokens:
                    tok_to_orig_index.append(i)
                    all_doc_tokens.append(sub_token)

            tok_start_position = None
            tok_end_position = None
            if is_training and example.is_impossible:
                tok_start_position = -1
                tok_end_position = -1
            if is_training and not example.is_impossible:
                tok_start_position = orig_to_tok_index[example.start_position]
                if example.end_position < len(example.doc_tokens) - 1:
                    tok_end_position = orig_to_tok_index[example.end_position + 1] - 1
                else:
                    tok_end_position = len(all_doc_tokens) - 1
                (tok_start_position, tok_end_position) = _improve_answer_span(
                    all_doc_tokens, tok_start_position, tok_end_position, self.config.tokenizer,
                    example.orig_answer_text)

            # The -3 accounts for [CLS], [SEP] and [SEP]
            max_tokens_for_doc = self.max_seq_length - len(query_tokens) - 3

            # We can have documents that are longer than the maximum sequence length.
            # To deal with this we do a sliding window approach, where we take chunks
            # of the up to our max length with a stride of `doc_stride`.
            _DocSpan = collections.namedtuple(  # pylint: disable=invalid-name
                "DocSpan", ["start", "length"])
            doc_spans = []
            start_offset = 0
            while start_offset < len(all_doc_tokens):
                length = len(all_doc_tokens) - start_offset
                if length > max_tokens_for_doc:
                    length = max_tokens_for_doc
                doc_spans.append(_DocSpan(start=start_offset, length=length))
                if start_offset + length == len(all_doc_tokens):
                    break
                start_offset += min(length, self.doc_stride)

            for (doc_span_index, doc_span) in enumerate(doc_spans):
                tokens = []
                token_to_orig_map = {}
                token_is_max_context = {}
                segment_ids = []
                tokens.append("[CLS]")
                segment_ids.append(0)
                for token in query_tokens:
                    tokens.append(token)
                    segment_ids.append(0)
                tokens.append("[SEP]")
                segment_ids.append(0)

                for i in range(doc_span.length):
                    split_token_index = doc_span.start + i
                    token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]

                    is_max_context = _check_is_max_context(doc_spans, doc_span_index,
                                                           split_token_index)
                    token_is_max_context[len(tokens)] = is_max_context
                    tokens.append(all_doc_tokens[split_token_index])
                    segment_ids.append(1)
                tokens.append("[SEP]")
                segment_ids.append(1)

                input_ids = self.config.tokenizer.convert_tokens_to_ids(tokens)

                # The mask has 1 for real tokens and 0 for padding tokens. Only real
                # tokens are attended to.
                input_mask = [1] * len(input_ids)

                # Zero-pad up to the sequence length.
                while len(input_ids) < self.max_seq_length:
                    input_ids.append(0)
                    input_mask.append(0)
                    segment_ids.append(0)

                assert len(input_ids) == self.max_seq_length
                assert len(input_mask) == self.max_seq_length
                assert len(segment_ids) == self.max_seq_length

                start_position = None
                end_position = None
                if is_training and not example.is_impossible:
                    # For training, if our document chunk does not contain an annotation
                    # we throw it out, since there is nothing to predict.
                    doc_start = doc_span.start
                    doc_end = doc_span.start + doc_span.length - 1
                    out_of_span = False
                    if not (tok_start_position >= doc_start and
                            tok_end_position <= doc_end):
                        out_of_span = True
                    if out_of_span:
                        start_position = 0
                        end_position = 0
                    else:
                        doc_offset = len(query_tokens) + 2
                        start_position = tok_start_position - doc_start + doc_offset
                        end_position = tok_end_position - doc_start + doc_offset

                if is_training and example.is_impossible:
                    start_position = 0
                    end_position = 0

                unique_id = str(uuid.uuid4())
                if self.example_count < 20:
                    tf.logging.info("*** Example ***")
                    tf.logging.info("unique_id: %s" % (unique_id))
                    tf.logging.info("example_index: %s" % (example_index))
                    tf.logging.info("doc_span_index: %s" % (doc_span_index))
                    tf.logging.info("tokens: %s" % " ".join(
                        [printable_text(x) for x in tokens]))
                    tf.logging.info("token_to_orig_map: %s" % " ".join(
                        ["%d:%d" % (x, y) for (x, y) in six.iteritems(token_to_orig_map)]))
                    tf.logging.info("token_is_max_context: %s" % " ".join([
                        "%d:%s" % (x, y) for (x, y) in six.iteritems(token_is_max_context)
                    ]))
                    tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                    tf.logging.info(
                        "input_mask: %s" % " ".join([str(x) for x in input_mask]))
                    tf.logging.info(
                        "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
                    if is_training and example.is_impossible:
                        tf.logging.info("impossible example")
                    if is_training and not example.is_impossible:
                        answer_text = " ".join(tokens[start_position:(end_position + 1)])
                        tf.logging.info("start_position: %d" % (start_position))
                        tf.logging.info("end_position: %d" % (end_position))
                        tf.logging.info(
                            "answer: %s" % (printable_text(answer_text)))
                    self.example_count += 1

                feature = InputFeatures(
                    unique_id=unique_id,
                    qas_id=example.qas_id,
                    example_index=example_index,
                    doc_span_index=doc_span_index,
                    doc_tokens=doc_tokens,
                    tokens=tokens,
                    token_to_orig_map=token_to_orig_map,
                    token_is_max_context=token_is_max_context,
                    input_ids=input_ids,
                    input_mask=input_mask,
                    segment_ids=segment_ids,
                    start_position=start_position,
                    end_position=end_position,
                    is_impossible=example.is_impossible)
                features.append(feature)
        return features

    def call(self, inputs):
        # HAE does not support on the fly mode, return the inputs
        items = []
        for name in self.input_tensor_names:
            items.append(inputs[name])

        return items

    def process(self, inputs):
        if isinstance(inputs, dict):
            inputs = [inputs]

        all_feature_list = []
        for idx, example in enumerate(inputs):
            feature_list = self.convert_example_to_features(example)
            for feature in feature_list:
                for key, val in example.items():
                    setattr(feature, key, val)
            all_feature_list.extend(feature_list)

        ret = dict()
        for key in all_feature_list[0].__dict__.keys():
            ret[key] = list()
            for features in all_feature_list:
                ret[key].append(getattr(features, key))

        for key, val in ret.items():
            ret[key] = np.array(val)

        return ret


class CQAExample(object):
    """A single training/test example for multi-turn comprehension."""

    def __init__(self,
                 qas_id,
                 question_text,
                 doc_tokens,
                 orig_answer_text=None,
                 start_position=None,
                 end_position=None,
                 history_answer_marker=None,
                 metadata=None):
        self.qas_id = qas_id
        self.question_text = question_text
        self.doc_tokens = doc_tokens
        self.orig_answer_text = orig_answer_text
        self.start_position = start_position
        self.end_position = end_position
        self.history_answer_marker = history_answer_marker
        self.metadata = metadata


class CQAInputFeatures(object):
    """A single set of features of data for multi-turn comprehension"""

    def __init__(self,
                 qas_id,
                 unique_id,
                 example_index,
                 doc_span_index,
                 tokens,
                 doc_tokens,
                 token_to_orig_map,
                 token_is_max_context,
                 input_ids,
                 input_mask,
                 segment_ids,
                 start_position=None,
                 end_position=None,
                 history_answer_marker=None,
                 metadata=None):
        self.qas_id = qas_id
        self.unique_id = unique_id
        self.example_index = example_index
        self.doc_span_index = doc_span_index
        self.tokens = tokens
        self.doc_tokens = doc_tokens
        self.token_to_orig_map = token_to_orig_map
        self.token_is_max_context = token_is_max_context
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.start_position = start_position
        self.end_position = end_position
        self.history_answer_marker = history_answer_marker
        self.metadata = metadata


class MultiTurnComprehensionPreprocessor(Preprocessor):
    """ Preprocessor for multi-turn text comprehension

    """
    config_class = ComprehensionPreprocessorConfig

    def __init__(self, config, **kwargs):
        super(MultiTurnComprehensionPreprocessor, self).__init__(config, **kwargs)
        self.config = config

        self.doc_stride = int(config.doc_stride) if hasattr(config, "doc_stride") else 128
        self.max_seq_length = int(config.sequence_length) if hasattr(config, "sequence_length") else 384
        self.max_query_length = int(config.max_query_length) if hasattr(config, "max_query_length") else 64
        self.max_considered_history_turns = int(config.max_considered_history_turns) \
            if hasattr(config, "max_considered_history_turns") else 11

        self.context_col_name = config.first_sequence
        self.query_col_name = config.second_sequence
        self.answer_col_name = config.label_name

        if "/" in config.pretrain_model_name_or_path:
            dirname = os.path.dirname(config.pretrain_model_name_or_path)
            self.language = dirname.split("-")[-1]
        else:
            self.language = config.pretrain_model_name_or_path.split("-")[-1]

        self.input_tensor_names = []

        input_schema = config.input_schema
        for schema in input_schema.split(","):
            name = schema.split(":")[0]
            self.input_tensor_names.append(name)

    @staticmethod
    def convert_examples_to_example_variations(examples, max_considered_history_turns):
        # an example is "question + passage + markers (M3 + M4) + markers_list (M3, M4)"
        # an example variation is "question + passage + markers (M3)"
        # meaning that we only have one marker for each example variation
        # because we want to make a binary choice for every example variation,
        # and combine all variations to form an example

        new_examples = []
        for example in examples:
            # if the example is the first question in the dialog, it does not contain history answers,
            # so we simply append it.
            if len(example.metadata['tok_history_answer_markers']) == 0:
                example.metadata['history_turns'] = []
                new_examples.append(example)
            else:
                for history_turn, marker, history_turn_text in zip(
                        example.metadata['history_turns'][- max_considered_history_turns:],
                        example.metadata['tok_history_answer_markers'][- max_considered_history_turns:],
                        example.metadata['history_turns_text'][- max_considered_history_turns:]):
                    each_new_example = deepcopy(example)
                    each_new_example.history_answer_marker = marker
                    each_new_example.metadata['history_turns'] = [history_turn]
                    each_new_example.metadata['tok_history_answer_markers'] = [marker]
                    each_new_example.metadata['history_turns_text'] = [history_turn_text]
                    new_examples.append(each_new_example)
        return new_examples

    def convert_example_to_features(self, example):
        """ Convert single example to multiple input features

        Args:
            items (`dict`): inputs from the reader
        Returns:
            features (`list`): list of `CQAInputFeatures`
        """
        paragraph_text = convert_to_unicode(example[self.context_col_name])
        question_id_list = convert_to_unicode(example[self.query_col_name]).split("||")
        questions = list(zip(question_id_list[::2], question_id_list[1::2]))
        answer_starts_list = convert_to_unicode(example[self.answer_col_name]).split("||")
        answers = list(zip(answer_starts_list[::2], [int(t) for t in answer_starts_list[1::2]]))
        if len(answers) != len(questions):
            assert len(questions) == len(answers) + 1, "Need put same number of history " \
                                                       "questions and answer."
            answers.append(("", -1))
            is_training = False
        else:
            is_training = True
        # Build paragraph doc tokens
        doc_tokens = []
        char_to_word_offset = []
        prev_is_whitespace = True
        for c in paragraph_text:
            if is_whitespace(c):
                prev_is_whitespace = True
            else:
                if prev_is_whitespace:
                    doc_tokens.append(c)
                else:
                    doc_tokens[-1] += c
                prev_is_whitespace = False
            char_to_word_offset.append(len(doc_tokens) - 1)

        # Prepare question-answer list
        qas = []
        for i, (question, answer) in enumerate(zip(questions, answers)):
            metadata = {'turn': i + 1, 'history_turns': [], 'tok_history_answer_markers': [],
                        'history_turns_text': []}
            end_index = i
            question_with_histories = ''

            start_index = 0  # we read all the histories no matter we use RL or not. we will make approporiate selections afterwards
            history_answer_marker = []
            for history_turn, (each_answer, each_question) in enumerate(
                    zip(answers[start_index: end_index], questions[start_index: end_index])):
                # [history_answer_start, history_answer_end, history_answer_text]
                each_marker = [each_answer[1], each_answer[1] + len(each_answer[0]), each_answer[0]]
                history_answer_marker.append(each_marker)
                metadata['history_turns'].append(history_turn + start_index + 1)
                metadata['history_turns_text'].append((each_question[0], each_answer[0]))  # [(q1, a1), (q2, a2), ...]

            # add the current question
            question_with_histories += question[0]
            qas.append({'id': question[1], 'question': question_with_histories,
                        'answers': [{'answer_start': answer[1], 'text': answer[0]}],
                        'history_answer_marker': history_answer_marker, 'metadata': metadata})

        examples = list()
        for qa in qas:
            qas_id = qa["id"]
            question_text = qa["question"]

            # if is_training:
            # we read in the groundtruth answer bothing druing training and predicting, because we need to compute acc and f1 at predicting time.
            if len(qa["answers"]) != 1:
                raise ValueError(
                    "For training, each question should have exactly 1 answer.")
            answer = qa["answers"][0]
            orig_answer_text = answer["text"]
            answer_offset = answer["answer_start"]
            answer_length = len(orig_answer_text)
            start_position = char_to_word_offset[answer_offset]
            end_position = char_to_word_offset[answer_offset + answer_length - 1]
            # Only add answers where the text can be exactly recovered from the
            # document. If this CAN'T happen it's likely due to weird Unicode
            # stuff so we will just skip the example.
            #
            # Note that this means for training mode, every example is NOT
            # guaranteed to be preserved.
            actual_text = " ".join(doc_tokens[start_position:(end_position + 1)])
            cleaned_answer_text = " ".join(
                whitespace_tokenize(orig_answer_text))

            if is_training and actual_text.find(cleaned_answer_text) == -1:
                tf.logging.warning("Could not find answer: '%s' vs. '%s'",
                                   actual_text, cleaned_answer_text)
                continue

            # we construct a tok_history_answer_marker to store the aggregated history answer markers for a question.
            # we also construct each_tok_history_answer_marker to store a single history answer marker.
            tok_history_answer_marker = [0] * len(doc_tokens)

            for marker_index, marker in enumerate(qa['history_answer_marker']):
                each_tok_history_answer_marker = [0] * len(doc_tokens)
                history_orig_answer_text = marker[2]
                history_answer_offset = marker[0]
                history_answer_length = len(history_orig_answer_text)
                history_start_position = char_to_word_offset[history_answer_offset]
                history_end_position = char_to_word_offset[history_answer_offset + history_answer_length - 1]
                history_actual_text = " ".join(doc_tokens[history_start_position:(history_end_position + 1)])
                history_cleaned_answer_text = " ".join(whitespace_tokenize(history_orig_answer_text))
                if history_actual_text.find(history_cleaned_answer_text) != -1:
                    tok_history_answer_marker = tok_history_answer_marker[: history_start_position] + \
                                                [1] * (history_end_position - history_start_position + 1) + \
                                                tok_history_answer_marker[history_end_position + 1:]
                    each_tok_history_answer_marker = each_tok_history_answer_marker[: history_start_position] + \
                                                     [1] * (history_end_position - history_start_position + 1) + \
                                                     each_tok_history_answer_marker[history_end_position + 1:]
                    assert len(tok_history_answer_marker) == len(doc_tokens)
                    assert len(each_tok_history_answer_marker) == len(doc_tokens)
                    qa['metadata']['tok_history_answer_markers'].append(each_tok_history_answer_marker)
                else:
                    tf.logging.warning("Could not find history answer: '%s' vs. '%s'", history_actual_text,
                                       history_cleaned_answer_text)
            example = CQAExample(
                qas_id=qas_id,
                question_text=question_text,
                doc_tokens=doc_tokens,
                orig_answer_text=orig_answer_text,
                start_position=start_position,
                end_position=end_position,
                history_answer_marker=tok_history_answer_marker,
                metadata=qa['metadata'])
            examples.append(example)

        features = []
        for (example_index, example) in enumerate(examples):
            variations = self.convert_examples_to_example_variations([example], self.max_considered_history_turns)
            for example in variations:
                metadata = example.metadata
                query_tokens = self.config.tokenizer.tokenize(example.question_text)

                if len(query_tokens) > self.max_query_length:
                    query_tokens = query_tokens[0:self.max_query_length]

                history_answer_marker = example.history_answer_marker
                tok_to_orig_index = []
                orig_to_tok_index = []
                all_doc_tokens = []
                all_history_answer_marker = []
                for (i, token) in enumerate(example.doc_tokens):
                    orig_to_tok_index.append(len(all_doc_tokens))
                    sub_tokens = self.config.tokenizer.tokenize(token)
                    for sub_token in sub_tokens:
                        tok_to_orig_index.append(i)
                        all_doc_tokens.append(sub_token)
                        all_history_answer_marker.append(history_answer_marker[i])

                # we do this for both training and predicting, because we need also start/end position at testing time to compute acc and f1
                tok_start_position = orig_to_tok_index[example.start_position]
                if example.end_position < len(example.doc_tokens) - 1:
                    tok_end_position = orig_to_tok_index[example.end_position + 1] - 1
                else:
                    tok_end_position = len(all_doc_tokens) - 1
                (tok_start_position, tok_end_position) = _improve_answer_span(
                    all_doc_tokens, tok_start_position, tok_end_position, self.config.tokenizer,
                    example.orig_answer_text)

                # The -3 accounts for [CLS], [SEP] and [SEP]
                max_tokens_for_doc = self.max_seq_length - len(query_tokens) - 3

                # We can have documents that are longer than the maximum sequence length.
                # To deal with this we do a sliding window approach, where we take chunks
                # of the up to our max length with a stride of `doc_stride`.
                _DocSpan = collections.namedtuple(  # pylint: disable=invalid-name
                    "DocSpan", ["start", "length"])
                doc_spans = []
                start_offset = 0
                while start_offset < len(all_doc_tokens):
                    length = len(all_doc_tokens) - start_offset
                    if length > max_tokens_for_doc:
                        length = max_tokens_for_doc
                    doc_spans.append(_DocSpan(start=start_offset, length=length))
                    if start_offset + length == len(all_doc_tokens):
                        break
                    start_offset += min(length, self.doc_stride)

                for (doc_span_index, doc_span) in enumerate(doc_spans):
                    marker = []
                    tokens = []
                    token_to_orig_map = {}
                    token_is_max_context = {}
                    segment_ids = []
                    tokens.append("[CLS]")
                    marker.append(0)
                    segment_ids.append(0)
                    for token in query_tokens:
                        tokens.append(token)
                        marker.append(0)
                        segment_ids.append(0)
                    tokens.append("[SEP]")
                    marker.append(0)
                    segment_ids.append(0)

                    for i in range(doc_span.length):
                        split_token_index = doc_span.start + i
                        token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]

                        is_max_context = _check_is_max_context(doc_spans, doc_span_index,
                                                               split_token_index)
                        token_is_max_context[len(tokens)] = is_max_context
                        tokens.append(all_doc_tokens[split_token_index])
                        marker.append(all_history_answer_marker[split_token_index])
                        segment_ids.append(1)
                    tokens.append("[SEP]")
                    marker.append(0)
                    segment_ids.append(1)

                    input_ids = self.config.tokenizer.convert_tokens_to_ids(tokens)

                    # The mask has 1 for real tokens and 0 for padding tokens. Only real
                    # tokens are attended to.
                    input_mask = [1] * len(input_ids)

                    # Zero-pad up to the sequence length.
                    while len(input_ids) < self.max_seq_length:
                        input_ids.append(0)
                        input_mask.append(0)
                        segment_ids.append(0)
                        marker.append(0)

                    assert len(input_ids) == self.max_seq_length
                    assert len(input_mask) == self.max_seq_length
                    assert len(segment_ids) == self.max_seq_length
                    assert len(marker) == self.max_seq_length

                    if is_training:
                        # For training, if our document chunk does not contain an annotation
                        # we throw it out, since there is nothing to predict.
                        doc_start = doc_span.start
                        doc_end = doc_span.start + doc_span.length - 1
                        if (example.start_position < doc_start or
                                example.end_position < doc_start or
                                example.start_position > doc_end or example.end_position > doc_end):
                            continue

                        doc_offset = len(query_tokens) + 2
                        start_position = tok_start_position - doc_start + doc_offset
                        end_position = tok_end_position - doc_start + doc_offset
                    else:
                        start_position = -1
                        end_position = -1

                    features.append(
                        CQAInputFeatures(
                            unique_id=str(uuid.uuid4()),
                            example_index=example_index,
                            doc_span_index=doc_span_index,
                            doc_tokens=doc_tokens,
                            tokens=tokens,
                            token_to_orig_map=token_to_orig_map,
                            token_is_max_context=token_is_max_context,
                            input_ids=input_ids,
                            input_mask=input_mask,
                            segment_ids=segment_ids,
                            start_position=start_position,
                            end_position=end_position,
                            history_answer_marker=marker,
                            metadata=metadata,
                            qas_id=example.qas_id))
        return features

    def call(self, inputs):
        # HAE does not support on the fly mode, return the inputs
        items = []
        for name in self.input_tensor_names:
            items.append(inputs[name])

        return items

    def process(self, inputs):
        if isinstance(inputs, dict):
            inputs = [inputs]

        all_feature_list = []
        for idx, example in enumerate(inputs):
            feature_list = self.convert_example_to_features(example)
            for feature in feature_list:
                for key, val in example.items():
                    setattr(feature, key, val)
            all_feature_list.extend(feature_list)

        ret = dict()
        for key in all_feature_list[0].__dict__.keys():
            ret[key] = list()
            for features in all_feature_list:
                ret[key].append(getattr(features, key))

        total_sample_num = len(ret["input_ids"])
        if hasattr(self.config, "preprocess_batch_size"):
            batch_size = self.config.preprocess_batch_size
        elif hasattr(self.config, "predict_batch_size"):
            batch_size = self.config.predict_batch_size
        else:
            batch_size = 12
        for i in range(total_sample_num // batch_size + 1):
            st = i * batch_size
            end = (i + 1) * batch_size
            if st >= total_sample_num:
                continue
            new_ret = dict()
            for key, val in ret.items():
                new_ret[key] = np.array(val[st:end])
            self.put(new_ret)