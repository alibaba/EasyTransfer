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

import collections
import tensorflow as tf
from easytransfer import FLAGS, Config
from easytransfer.preprocessors.tokenization import convert_to_unicode, load_vocab
from easytransfer.preprocessors.pretrain_preprocessor import truncate_seq_pair, TrainingInstance
from easytransfer.preprocessors.pretrain_preprocessor import create_int_feature, create_float_feature, \
    create_masked_lm_predictions
from easytransfer.preprocessors.pretrain_preprocessor import create_chinese_subwords

_app_flags = tf.app.flags
_app_flags.DEFINE_string("input_dir", default=None, help='')
_app_flags.DEFINE_string("output_dir", default=None, help='')
_app_flags.DEFINE_integer("num_threads", default=None, help='')
_app_flags.DEFINE_string("data_format", default=None, help='')
_app_flags.DEFINE_string("tokenizer", default="wordpiece", help='')
_app_flags.DEFINE_string("spm_model_fp", None, "The model file for sentence piece tokenization.")
_app_flags.DEFINE_string("vocab_fp", None, "The model file for word piece tokenization.")
_app_flags.DEFINE_bool("do_whole_word_mask", True,
                       "Whether to use whole word masking rather than per-WordPiece masking.")
_app_flags.DEFINE_bool("do_chinese_whole_word_mask", False,
                       "Whether to use whole word masking rather than per-WordPiece masking.")
_app_flags.DEFINE_bool("random_next_sentence", False, "")
_app_flags.DEFINE_integer("dupe_factor", 40, "Number of times to duplicate the input data (with different masks).")
_app_flags.DEFINE_integer("max_seq_length", 512, "Maximum sequence length.")
_app_flags.DEFINE_integer("max_predictions_per_seq", 20, "Maximum number of masked LM predictions per sequence.")
_app_flags.DEFINE_float("masked_lm_prob", 0.15, "Masked LM probability.")
_app_flags.DEFINE_float("short_seq_prob", 0.1,
                        "Probability of creating sequences which are shorter than the maximum length.")

_app_flags.DEFINE_string("loss", None, "loss")
_app_flags.DEFINE_string("data_reader", default='tfrecord', help="data_reader")
_app_flags.DEFINE_bool("do_multitaks_pretrain", default=False, help="do_multitaks_pretrain")
_app_flags.DEFINE_string("model_type", None, "model_type")
_app_flags.DEFINE_string("input_schema", default=None, help='input_schema')
_app_flags.DEFINE_string("pretrain_model_name_or_path", default=None, help='pretrain_model_name_or_path')
_app_flags.DEFINE_string("train_input_fp", default=None, help='train_input_fp')
_app_flags.DEFINE_string("eval_input_fp", default=None, help='eval_input_fp')
_app_flags.DEFINE_integer("train_batch_size", default=128, help='train_batch_size')
_app_flags.DEFINE_integer("eval_batch_size", default=128, help='eval_batch_size')
_app_flags.DEFINE_float("num_epochs", default=1, help='num_epochs')
_app_flags.DEFINE_string("model_dir", default='', help='model_dir')
_app_flags.DEFINE_float("learning_rate", 1e-4, "learning_rate")
_app_flags.DEFINE_integer("input_sequence_length", default=None, help='')
_app_flags.DEFINE_integer("hidden_size", default=None, help='')
_app_flags.DEFINE_integer("vocab_size", default=1, help='')
_app_flags.DEFINE_integer("factorized_size", default=None, help='')
_app_flags.DEFINE_integer("intermediate_size", default=None, help='')
_app_flags.DEFINE_integer("num_hidden_layers", default=None, help='')
_app_flags.DEFINE_integer("num_attention_heads", default=None, help='')
_app_flags.DEFINE_integer("num_accumulated_batches", default=1, help='')
_app_flags.DEFINE_integer("num_model_replica", default=1, help='')
_app_flags.DEFINE_string("distribution_strategy", default=None, help='')
_APP_FLAGS = _app_flags.FLAGS

if FLAGS.config is None:
    vocab = load_vocab(_APP_FLAGS.vocab_fp)
    _APP_FLAGS.vocab_size = len(vocab)

tf.logging.info("*********Pretrain loss is {}**********".format(_APP_FLAGS.loss))
tf.logging.info("*********Vocab Size is {}**********".format(_APP_FLAGS.vocab_size))

class PretrainConfig(Config):
    def __init__(self):

        if _APP_FLAGS.pretrain_model_name_or_path is not None:
            pretrain_model_name_or_path = _APP_FLAGS.pretrain_model_name_or_path
        else:
            pretrain_model_name_or_path = _APP_FLAGS.model_dir + "/model.ckpt"

        config_json = {
            "preprocess_config": {
                "input_schema": _APP_FLAGS.input_schema,
            },
            'model_config': {
                'pretrain_model_name_or_path': pretrain_model_name_or_path
            },
            'train_config': {
                "train_input_fp": _APP_FLAGS.train_input_fp,
                "train_batch_size": _APP_FLAGS.train_batch_size,
                "num_epochs": _APP_FLAGS.num_epochs,
                "model_dir": _APP_FLAGS.model_dir,
                "save_steps": None,
                "optimizer_config": {
                    "optimizer": "adam",
                    "weight_decay_ratio": 0,
                    "warmup_ratio": 0.1,
                    "learning_rate": _APP_FLAGS.learning_rate,
                },
                "distribution_config": {
                    "distribution_strategy": _APP_FLAGS.distribution_strategy,
                    "num_accumulated_batches":_APP_FLAGS.num_accumulated_batches,
                    "num_model_replica":_APP_FLAGS.num_model_replica
                }
            },
            "evaluate_config": {
                "eval_input_fp": _APP_FLAGS.eval_input_fp,
                "eval_batch_size":_APP_FLAGS.eval_batch_size,
                "num_eval_steps": 1000
            }
        }
        config_json["worker_hosts"] = FLAGS.worker_hosts
        config_json["task_index"] = FLAGS.task_index
        config_json["job_name"] = FLAGS.job_name
        config_json["num_gpus"] = FLAGS.workerGPU
        config_json["num_workers"] = FLAGS.workerCount
        super(PretrainConfig, self).__init__(mode="train_and_evaluate", config_json=config_json)

def create_training_instances(input_file, tokenizer, max_seq_length,
                              dupe_factor, short_seq_prob, masked_lm_prob,
                              max_predictions_per_seq, do_whole_word_mask, rng):
    all_documents = [[]]
    with tf.gfile.GFile(input_file, "r") as reader:
        while True:
            line = convert_to_unicode(reader.readline())
            if not line:
                break
            line = line.strip()

            # Empty lines are used as document delimiters
            if not line:
                all_documents.append([])
            tokens = tokenizer.tokenize(line)
            if tokens:
                all_documents[-1].append(tokens)


    # Remove empty documents
    all_documents = [x for x in all_documents if x]
    rng.shuffle(all_documents)

    vocab_words = list(tokenizer.vocab.keys())
    instances = []
    for dup_idx in range(dupe_factor):
        tf.logging.info("Dupe_factor {}".format(dup_idx))
        if _APP_FLAGS.data_format == "segment_pair":
            for document_index in range(len(all_documents)):
                    instances.extend(
                        create_instances_from_document_segment_pair(
                            all_documents, document_index, max_seq_length, short_seq_prob,
                            masked_lm_prob, max_predictions_per_seq, vocab_words, do_whole_word_mask, rng))

        elif _APP_FLAGS.data_format == "full_sentences":
            instances.extend(create_instances_from_document_full_sentences(
                    all_documents, max_seq_length,
                    masked_lm_prob, max_predictions_per_seq, vocab_words, do_whole_word_mask, rng))
        else:
            raise ValueError("data_format: segment_pair or full_sentences")

    rng.shuffle(instances)
    return instances


def create_instances_from_document_segment_pair(
        all_documents, document_index, max_seq_length, short_seq_prob,
        masked_lm_prob, max_predictions_per_seq, vocab_words, do_whole_word_mask, rng):
    """Creates `TrainingInstance`s for a single document."""
    document = all_documents[document_index]

    # Account for [CLS], [SEP], [SEP]
    max_num_tokens = max_seq_length - 3

    # We *usually* want to fill up the entire sequence since we are padding
    # to `max_seq_length` anyways, so short sequences are generally wasted
    # computation. However, we *sometimes*
    # (i.e., short_seq_prob == 0.1 == 10% of the time) want to use shorter
    # sequences to minimize the mismatch between pre-training and fine-tuning.
    # The `target_seq_length` is just a rough target however, whereas
    # `max_seq_length` is a hard limit.
    target_seq_length = max_num_tokens
    if rng.random() < short_seq_prob:
        target_seq_length = rng.randint(2, max_num_tokens)

    # We DON'T just concatenate all of the tokens from a document into a long
    # sequence and choose an arbitrary split point because this would make the
    # next sentence prediction task too easy. Instead, we split the input into
    # segments "A" and "B" based on the actual "sentences" provided by the user
    # input.
    instances = []
    current_chunk = []
    current_length = 0
    i = 0
    while i < len(document):
        segment = document[i]
        if _APP_FLAGS.do_chinese_whole_word_mask:
            segment = create_chinese_subwords(segment)

        current_chunk.append(segment)
        current_length += len(segment)
        if i == len(document) - 1 or current_length >= target_seq_length:
            if current_chunk:
                # `a_end` is how many segments from `current_chunk` go into the `A`
                # (first) sentence.
                a_end = 1
                if len(current_chunk) >= 2:
                    a_end = rng.randint(1, len(current_chunk) - 1)

                tokens_a = []
                for j in range(a_end):
                    tokens_a.extend(current_chunk[j])

                tokens_b = []
                # Random next
                is_random_next = False
                if len(current_chunk) == 1 or \
                        (_APP_FLAGS.random_next_sentence and rng.random() < 0.5):
                    is_random_next = True
                    target_b_length = target_seq_length - len(tokens_a)

                    # This should rarely go for more than one iteration for large
                    # corpora. However, just to be careful, we try to make sure that
                    # the random document is not the same as the document
                    # we're processing.
                    for _ in range(10):
                        random_document_index = rng.randint(0, len(all_documents) - 1)
                        if random_document_index != document_index:
                            break

                    random_document = all_documents[random_document_index]
                    random_start = rng.randint(0, len(random_document) - 1)
                    for j in range(random_start, len(random_document)):
                        tokens_b.extend(random_document[j])
                        if len(tokens_b) >= target_b_length:
                            break
                    # We didn't actually use these segments so we "put them back" so
                    # they don't go to waste.
                    num_unused_segments = len(current_chunk) - a_end
                    i -= num_unused_segments
                elif not _APP_FLAGS.random_next_sentence and rng.random() < 0.5:
                    is_random_next = True
                    for j in range(a_end, len(current_chunk)):
                        tokens_b.extend(current_chunk[j])
                    # Note(mingdachen): in this case, we just swap tokens_a and tokens_b
                    tokens_a, tokens_b = tokens_b, tokens_a
                # Actual next
                else:
                    is_random_next = False
                    for j in range(a_end, len(current_chunk)):
                        tokens_b.extend(current_chunk[j])
                truncate_seq_pair(tokens_a, tokens_b, max_num_tokens)

                assert len(tokens_a) >= 1
                assert len(tokens_b) >= 1

                tokens = []
                segment_ids = []
                tokens.append("[CLS]")
                segment_ids.append(0)
                for token in tokens_a:
                    tokens.append(token)
                    segment_ids.append(0)

                tokens.append("[SEP]")
                segment_ids.append(0)

                for token in tokens_b:
                    tokens.append(token)
                    segment_ids.append(1)
                tokens.append("[SEP]")
                segment_ids.append(1)

                (tokens, masked_lm_positions, masked_lm_labels) = create_masked_lm_predictions(
                    tokens, masked_lm_prob, max_predictions_per_seq, vocab_words, do_whole_word_mask, rng)
                instance = TrainingInstance(
                    tokens=tokens,
                    segment_ids=segment_ids,
                    is_random_next=is_random_next,
                    masked_lm_positions=masked_lm_positions,
                    masked_lm_labels=masked_lm_labels)
                instances.append(instance)
            current_chunk = []
            current_length = 0
        i += 1

    return instances


def create_instances_from_document_full_sentences(all_documents, max_seq_length,
        masked_lm_prob, max_predictions_per_seq, vocab_words, do_whole_word_mask, rng):
    """Creates `TrainingInstance`s for a single document."""

    # Account for [CLS], [SEP]
    max_num_tokens = max_seq_length - 2

    instances = []
    current_chunk = []
    current_length = 0

    d = 0
    while d < len(all_documents):
        document = all_documents[d]
        i = 0
        while i < len(document):
            segment = document[i]
            if _APP_FLAGS.do_chinese_whole_word_mask:
                segment = create_chinese_subwords(segment)
            current_chunk.append(segment)
            current_length += len(segment)

            if current_length >= max_num_tokens:
                if current_chunk:
                    tokens_a = []
                    for seg in current_chunk:
                        tokens_a.extend(seg)

                    while True:
                        if len(tokens_a) > max_num_tokens:
                            tokens_a.pop()
                        else:
                            break

                    assert len(tokens_a) == max_num_tokens

                    tokens = []
                    segment_ids = []
                    tokens.append("[CLS]")
                    segment_ids.append(0)
                    for token in tokens_a:
                        tokens.append(token)
                        segment_ids.append(0)

                    tokens.append("[SEP]")
                    segment_ids.append(0)

                    (tokens, masked_lm_positions,
                     masked_lm_labels) = create_masked_lm_predictions(
                        tokens, masked_lm_prob, max_predictions_per_seq, vocab_words, do_whole_word_mask, rng)
                    instance = TrainingInstance(
                        tokens=tokens,
                        segment_ids=segment_ids,
                        is_random_next=False,
                        masked_lm_positions=masked_lm_positions,
                        masked_lm_labels=masked_lm_labels)
                    instances.append(instance)

                current_chunk = []
                current_length = 0
            i += 1
        d +=1

    return instances


def write_instance_to_file(instances, tokenizer, max_seq_length,
                           max_predictions_per_seq, output_file):
    writer = tf.python_io.TFRecordWriter(output_file)
    total_written = 0
    for (inst_index, instance) in enumerate(instances):
        input_ids = tokenizer.convert_tokens_to_ids(instance.tokens)
        input_mask = [1] * len(input_ids)
        segment_ids = list(instance.segment_ids)
        assert len(input_ids) <= max_seq_length

        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        masked_lm_positions = list(instance.masked_lm_positions)
        masked_lm_ids = tokenizer.convert_tokens_to_ids(instance.masked_lm_labels)
        masked_lm_weights = [1.0] * len(masked_lm_ids)

        while len(masked_lm_positions) < max_predictions_per_seq:
            masked_lm_positions.append(0)
            masked_lm_ids.append(0)
            masked_lm_weights.append(0.0)

        sentence_order_label = 1 if instance.is_random_next else 0

        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(input_ids)
        features["input_mask"] = create_int_feature(input_mask)
        features["segment_ids"] = create_int_feature(segment_ids)
        features["masked_lm_positions"] = create_int_feature(masked_lm_positions)
        features["masked_lm_ids"] = create_int_feature(masked_lm_ids)
        features["masked_lm_weights"] = create_float_feature(masked_lm_weights)
        # Note: We keep this feature name `next_sentence_labels` to be compatible
        # with the original data created by lanzhzh@. However, in the ALBERT case
        # it does contain sentence_order_label.
        features["next_sentence_labels"] = create_int_feature(
            [sentence_order_label])

        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())
        total_written += 1

    writer.close()
    tf.logging.info("Wrote %d total instances", total_written)