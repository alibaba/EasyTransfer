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

import functools
import json
import six

import numpy as np
import os
import tensorflow as tf

import easytransfer
from easytransfer.engines.distribution import Process
from easytransfer.engines.model import FLAGS
from .tokenization import FullTokenizer
from easytransfer.model_zoo import get_config_path

sentencepiece_model_name_vocab_path_map = {
    'google-albert-base-zh': "albert/google-albert-base-zh/vocab.txt",
    'google-albert-large-zh': "albert/google-albert-large-zh/vocab.txt",
    'google-albert-xlarge-zh': "albert/google-albert-xlarge-zh/vocab.txt",
    'google-albert-xxlarge-zh': "albert/google-albert-xxlarge-zh/vocab.txt",
    'google-albert-base-en': "albert/google-albert-base-en/30k-clean.model",
    'google-albert-large-en': "albert/google-albert-large-en/30k-clean.model",
    'google-albert-xlarge-en': "albert/google-albert-xlarge-en/30k-clean.model",
    'google-albert-xxlarge-en': "albert/google-albert-xxlarge-en/30k-clean.model",
    'pai-albert-base-zh': "albert/pai-albert-base-zh/vocab.txt",
    'pai-albert-large-zh': "albert/pai-albert-large-zh/vocab.txt",
    'pai-albert-xlarge-zh': "albert/pai-albert-xlarge-zh/vocab.txt",
    'pai-albert-xxlarge-zh': "albert/pai-albert-xxlarge-zh/vocab.txt",
    'pai-albert-base-en': "albert/pai-albert-base-en/30k-clean.model",
    'pai-albert-large-en': "albert/pai-albert-large-en/30k-clean.model",
    'pai-albert-xlarge-en': "albert/pai-albert-xlarge-en/30k-clean.model",
    'pai-albert-xxlarge-en': "albert/pai-albert-xxlarge-en/30k-clean.model",
}

wordpiece_model_name_vocab_path_map = {
    'google-bert-tiny-zh': "bert/google-bert-tiny-zh/vocab.txt",
    'google-bert-tiny-en': "bert/google-bert-tiny-en/vocab.txt",
    'google-bert-small-zh': "bert/google-bert-small-zh/vocab.txt",
    'google-bert-small-en': "bert/google-bert-small-en/vocab.txt",
    'google-bert-base-zh': "bert/google-bert-base-zh/vocab.txt",
    'google-bert-base-en': "bert/google-bert-base-en/vocab.txt",
    'google-bert-large-zh': "bert/google-bert-large-zh/vocab.txt",
    'google-bert-large-en': "bert/google-bert-large-en/vocab.txt",
    'pai-bert-tiny-zh-L2-H768-A12': "bert/pai-bert-tiny-zh-L2-H768-A12/vocab.txt",
    'pai-bert-tiny-zh-L2-H128-A2': "bert/pai-bert-tiny-zh-L2-H128-A2/vocab.txt",
    'pai-bert-tiny-en': "bert/pai-bert-tiny-en/vocab.txt",
    'pai-bert-tiny-zh': "bert/pai-bert-tiny-zh/vocab.txt",
    'pai-bert-small-zh': "bert/pai-bert-small-zh/vocab.txt",
    'pai-bert-small-en': "bert/pai-bert-small-en/vocab.txt",
    'pai-bert-base-zh': "bert/pai-bert-base-zh/vocab.txt",
    'pai-bert-base-en': "bert/pai-bert-base-en/vocab.txt",
    'pai-bert-large-zh': "bert/pai-bert-large-zh/vocab.txt",
    'pai-bert-large-en': "bert/pai-bert-large-en/vocab.txt",
    'hit-roberta-base-zh': "roberta/hit-roberta-base-zh/vocab.txt",
    'hit-roberta-large-zh': "roberta/hit-roberta-large-zh/vocab.txt",
    'pai-imagebert-base-zh': "imagebert/pai-imagebert-base-zh/vocab.txt",
    'pai-videobert-base-zh': "imagebert/pai-videobert-base-zh/vocab.txt",
    'brightmart-roberta-small-zh': "roberta/brightmart-roberta-small-zh/vocab.txt",
    'brightmart-roberta-large-zh': "roberta/brightmart-roberta-large-zh/vocab.txt",
    'icbu-imagebert-small-en': "imagebert/icbu-imagebert-small-en/vocab.txt",
    'pai-transformer-base-zh': "transformer/pai-transformer-base-zh/vocab.txt",
    'pai-linformer-base-en': "linformer/pai-linformer-base-en/vocab.txt",
    'pai-xformer-base-en': "xformer/pai-xformer-base-en/vocab.txt",
    'pai-imagebert-base-en': "imagebert/pai-imagebert-base-en/vocab.txt",
    'pai-synthesizer-base-en': "synthesizer/pai-synthesizer-base-en/vocab.txt",
    'pai-sentimentbert-base-zh': "sentimentbert/pai-sentimentbert-base-zh/vocab.txt"
}

def truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

class PreprocessorConfig(object):

    def __init__(self, **kwargs):

        self.mode = kwargs.get("mode")

        # multitask classification
        self.append_feature_columns = kwargs.get("append_feature_columns")

        # configurate tokenizer
        pretrain_model_name_or_path = kwargs['pretrain_model_name_or_path']

        if "/" not in pretrain_model_name_or_path:
            model_type = pretrain_model_name_or_path.split("-")[1]
            if tf.gfile.Exists(os.path.join(FLAGS.modelZooBasePath, model_type,
                                                pretrain_model_name_or_path, "config.json")):
                # If exists directory, not download
                pass
            else:
                if six.PY2:
                    import errno
                    def mkdir_p(path):
                        try:
                            os.makedirs(path)
                        except OSError as exc:  # Python >2.5 (except OSError, exc: for Python <2.5)
                            if exc.errno == errno.EEXIST and os.path.isdir(path):
                                pass
                            else:
                                raise
                    mkdir_p(os.path.join(FLAGS.modelZooBasePath, model_type))
                else:
                    os.makedirs(os.path.join(FLAGS.modelZooBasePath, model_type), exist_ok=True)

                des_path = os.path.join(os.path.join(FLAGS.modelZooBasePath, model_type),
                                        pretrain_model_name_or_path + ".tgz")
                if not os.path.exists(des_path):
                    tf.logging.info("********** Begin to download to {} **********".format(des_path))
                    os.system(
                        'wget -O ' + des_path + ' https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/eztransfer_modelzoo/' + model_type + '/' + pretrain_model_name_or_path + ".tgz")
                    os.system('tar -zxvf ' + des_path + ' --directory ' + FLAGS.modelZooBasePath + "/" + model_type)

        if "train" in self.mode:
            model_dir = kwargs['model_dir']
            assert model_dir is not None, "model_dir should be set in config"
            if "/" not in pretrain_model_name_or_path:
                model_type = pretrain_model_name_or_path.split("-")[1]
                config_path = get_config_path(model_type, pretrain_model_name_or_path)
                config_path = os.path.join(FLAGS.modelZooBasePath, config_path)
                dir_path = os.path.dirname(config_path)
            else:
                dir_path = os.path.dirname(pretrain_model_name_or_path)

            if not tf.gfile.Exists(model_dir):
                tf.gfile.MakeDirs(model_dir)

            if not tf.gfile.Exists(os.path.join(model_dir, "config.json")):
                tf.gfile.Copy(os.path.join(dir_path, "config.json"),
                              os.path.join(model_dir, "config.json"))
                if tf.gfile.Exists(os.path.join(dir_path, "vocab.txt")):
                    tf.gfile.Copy(os.path.join(dir_path, "vocab.txt"),
                                  os.path.join(model_dir, "vocab.txt"))
                if tf.gfile.Exists(os.path.join(dir_path, "30k-clean.model")):
                    tf.gfile.Copy(os.path.join(dir_path, "30k-clean.model"),
                                  os.path.join(model_dir, "30k-clean.model"))

        albert_language = "zh"
        if "/" not in pretrain_model_name_or_path:
            model_type = pretrain_model_name_or_path.split("-")[1]
            if model_type == "albert":
                vocab_path = os.path.join(FLAGS.modelZooBasePath,
                                          sentencepiece_model_name_vocab_path_map[pretrain_model_name_or_path])
                if "30k-clean.model" in vocab_path:
                    albert_language = "en"
                else:
                    albert_language = "zh"
            else:
                vocab_path = os.path.join(FLAGS.modelZooBasePath,
                                          wordpiece_model_name_vocab_path_map[pretrain_model_name_or_path])

        else:
            with tf.gfile.GFile(os.path.join(os.path.dirname(pretrain_model_name_or_path), "config.json")) as reader:
                text = reader.read()
            json_config = json.loads(text)
            model_type = json_config["model_type"]
            if model_type == "albert":
                if tf.gfile.Exists(os.path.join(os.path.dirname(pretrain_model_name_or_path), "vocab.txt")):
                    albert_language = "zh"
                    vocab_path = os.path.join(os.path.dirname(pretrain_model_name_or_path), "vocab.txt")
                elif tf.gfile.Exists(os.path.join(os.path.dirname(pretrain_model_name_or_path), "30k-clean.model")):
                    albert_language = "en"
                    vocab_path = os.path.join(os.path.dirname(pretrain_model_name_or_path), "30k-clean.model")
            else:
                vocab_path = os.path.join(os.path.dirname(pretrain_model_name_or_path), "vocab.txt")

        assert model_type is not None, "you must specify model_type in pretrain_model_name_or_path"

        if model_type == "albert":
            if albert_language == "en":
                self.tokenizer = FullTokenizer(spm_model_file=vocab_path)
            else:
                self.tokenizer = FullTokenizer(vocab_file=vocab_path)
        else:
            self.tokenizer = FullTokenizer(vocab_file=vocab_path)

        # Additional attributes without default values
        for key, value in kwargs.items():
            try:
                setattr(self, key, value)
            except AttributeError as err:
                tf.logging.error("Can't set {} with value {} for {}".format(key, value, self))
                raise err

    @classmethod
    def from_json_file(cls, **kwargs):
        config = cls(**kwargs)
        return config


class Preprocessor(easytransfer.layers.Layer, Process):

    def __init__(self,
                 config,
                 thread_num=1,
                 input_queue=None,
                 output_queue=None,
                 job_name='DISTPreprocessor',
                 **kwargs):

        kwargs.clear()
        easytransfer.layers.Layer.__init__(self, **kwargs)

        if config.mode.startswith("predict"):
            Process.__init__(
                self, job_name, thread_num, input_queue, output_queue, batch_size=config.predict_batch_size)

        elif config.mode == "preprocess":
            Process.__init__(
                self, job_name, thread_num, input_queue, output_queue, batch_size=config.preprocess_batch_size)

        self.append_tensor_names = []
        if hasattr(config, "append_feature_columns") and config.append_feature_columns is not None:
            for schema in config.append_feature_columns.split(","):
                name = schema.split(":")[0]
                self.append_tensor_names.append(name)

        self.mode = config.mode

    @classmethod
    def get_preprocessor(cls, **kwargs):
        if kwargs.get("user_defined_config", None) is not None:
            config = kwargs["user_defined_config"]
            for key, val in config.__dict__.items():
                kwargs[key] = val
            if kwargs["mode"] == "export":
                kwargs["input_schema"] = config.input_tensors_schema
        else:
            json_file = FLAGS.config
            with tf.gfile.GFile(json_file, mode='r') as reader:
                text = reader.read()

            config_dict = json.loads(text)
            for values in config_dict.values():
                if isinstance(values, str):
                    continue
                for k, v in values.items():
                    if isinstance(v, dict) and k != "label_enumerate_values":
                        continue
                    else:
                        kwargs[k] = v
            kwargs["mode"] = FLAGS.mode
            if FLAGS.mode == "export":
                kwargs["input_schema"] = config_dict['export_config']['input_tensors_schema']

        config = cls.config_class.from_json_file(**kwargs)


        preprocessor = cls(config, **kwargs)
        return preprocessor

    def set_feature_schema(self):
        raise NotImplementedError("must be implemented in descendants")

    def convert_example_to_features(self, items):
        raise NotImplementedError("must be implemented in descendants")

    def _convert(self, convert_example_to_features, *args):

        # mode check
        if not ("on_the_fly" in self.mode or self.mode == "preprocess"):
            raise ValueError("Please using on_the_fly or preprocess mode")

        batch_features = []
        batch_size = len(args[0])
        for i in range(batch_size):
            items = []
            for feat in args:
                if isinstance(feat[i], np.ndarray):
                    assert feat[i][0] is not None, "In on the fly mode where object is ndarray, column has null value"
                    items.append(feat[i][0])
                else:
                    assert feat[i] is not None, "In on the fly mode, column has null value"
                    items.append(feat[i])
            features = convert_example_to_features(items)
            batch_features.append(features)

        stacked_features = np.stack(batch_features, axis=1)
        concat_features = []
        for i in range(stacked_features.shape[0]):
            concat_features.append(np.asarray(" ".join(stacked_features[i])))
        return concat_features

    # Inputs from Reader's map_batch_prefetch method
    def call(self, inputs):
        self.set_feature_schema()

        items = []
        for name in self.input_tensor_names:
            items.append(inputs[name])

        if not ("on_the_fly" in self.mode or self.mode == "preprocess"):
            return items

        self.Tout = [tf.string] * len(self.seq_lens)

        batch_features = tf.py_func(functools.partial(self._convert,
                                                      self.convert_example_to_features),
                                    items, self.Tout)

        ret = []
        for idx, feature in enumerate(batch_features):
            seq_len = self.seq_lens[idx]
            feature_type = self.feature_value_types[idx]
            if feature_type == tf.int64:
                input_tensor = tf.string_to_number(
                    tf.string_split(tf.expand_dims(feature, axis=0), delimiter=" ").values,
                    tf.int64)
            elif feature_type == tf.float32:
                input_tensor = tf.string_to_number(
                    tf.string_split(tf.expand_dims(feature, axis=0), delimiter=" ").values,
                    tf.float32)
            elif feature_type == tf.string:
                input_tensor = feature
            else:
                raise NotImplementedError

            input_tensor = tf.reshape(input_tensor, [-1, seq_len])
            ret.append(input_tensor)

        for name in self.append_tensor_names:
            ret.append(inputs[name])

        return ret

    def process(self, inputs):
        self.set_feature_schema()

        if isinstance(inputs, dict):
            inputs = [inputs]

        batch_features = []
        for input in inputs:
            items = []
            for name in self.input_tensor_names:
                items.append(input[name])
            features = self.convert_example_to_features(items)
            batch_features.append(features)

        stacked_features = np.stack(batch_features, axis=1)
        concat_features = []
        for i in range(stacked_features.shape[0]):
            concat_features.append(np.asarray(" ".join(stacked_features[i])))

        if self.mode.startswith("predict") or self.mode == "preprocess":
            for name in self.output_schema.split(","):
                if name in self.input_tensor_names:
                    self.output_tensor_names.append(name)

        ret = {}
        for idx, name in enumerate(self.output_tensor_names):
            if idx < len(concat_features):
                feature = concat_features[idx]
                seq_len = self.seq_lens[idx]
                feature_type = self.feature_value_types[idx]
                feature = feature.tolist()
                if feature_type == tf.int64:
                    input_tensor = [int(x) for x in feature.split(" ")]
                elif feature_type == tf.float32:
                    input_tensor = [float(x) for x in feature.split(" ")]
                elif feature_type == tf.string:
                    input_tensor = feature
                else:
                    raise NotImplementedError
                input_tensor = np.reshape(input_tensor, [-1, seq_len])
                name = self.output_tensor_names[idx]
                ret[name] = input_tensor
            else:
                left = []
                for ele in inputs:
                    left.append(ele[name])
                left_tensor = np.asarray(left)
                ret[name] = np.reshape(left_tensor, [-1, 1])

        return ret
