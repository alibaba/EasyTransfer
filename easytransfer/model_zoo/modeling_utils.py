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


import json
import re
import os
import tensorflow as tf
from tensorflow.python import pywrap_tensorflow
from tensorflow.python.framework import errors_impl
from tensorflow.python.platform import gfile
from easytransfer.engines.model import FLAGS
from easytransfer import layers

class PretrainedConfig(object):
    def __init__(self, **kwargs):
        # Additional attributes without default values
        for key, value in kwargs.items():
            try:
                setattr(self, key, value)
            except AttributeError as err:
                tf.logging.error("Can't set {} with value {} for {}".format(key, value, self))
                raise err

    @classmethod
    def get(cls, json_file, **kwargs):

        config_dict = cls._dict_from_json_file(json_file)
        return cls.from_dict(config_dict, **kwargs)

    @classmethod
    def from_dict(cls, config_dict, **kwargs):
        config = cls(**config_dict)
        for key, value in kwargs.items():
            setattr(config, key, value)
        return config

    @classmethod
    def _dict_from_json_file(cls, json_file):
        with gfile.GFile(json_file, mode='r') as reader:
            text = reader.read()
        return json.loads(text)

class PreTrainedModel(layers.Layer):
    config_class = None
    pretrained_model_archive_map = {}
    pretrained_config_archive_map = {}

    @classmethod
    def dummy_inputs(self, seq_length):
        """ Dummy inputs to build the network.

        Returns:
            tf.Tensor with dummy inputs
        """
        #input_ids = [[7, 6, 0, 0, 1], [1, 2, 3, 0, 0], [0, 0, 0, 4, 5]]
        input_ids = [[1]*seq_length]
        return tf.constant(input_ids)

    def __init__(self, config, **kwargs):
        kwargs.clear()
        super(PreTrainedModel, self).__init__(**kwargs)
        if not isinstance(config, PretrainedConfig):
            raise ValueError(
                "Parameter config in `{}(config)` should be an instance of class `PretrainedConfig`. "
                "To create a model from a pretrained model use "
                "`model = {}.from_pretrained(PRETRAINED_MODEL_NAME)`".format(
                    self.__class__.__name__, self.__class__.__name__
                )
            )
        # Save config in model
        self.config = config

    @classmethod
    def get(cls, pretrained_model_name_or_path, **kwargs):
        if pretrained_model_name_or_path in cls.pretrained_config_archive_map:
            config_path = cls.pretrained_config_archive_map[pretrained_model_name_or_path]
            config_path = os.path.join(FLAGS.modelZooBasePath, config_path)
        else:
            config_path = os.path.join(os.path.dirname(pretrained_model_name_or_path), "config.json")

        config = cls.config_class.get(
            config_path,
            **kwargs)
        model = cls(config, **kwargs)

        model(model.dummy_inputs(kwargs.get('input_sequence_length', 512)), mode='eval', output_features=False)

        archive_file = None
        if pretrained_model_name_or_path in cls.pretrained_model_archive_map:
            archive_file = cls.pretrained_model_archive_map[pretrained_model_name_or_path]
            archive_file = os.path.join(FLAGS.modelZooBasePath, archive_file)
        elif "/" in pretrained_model_name_or_path:
            archive_file = pretrained_model_name_or_path

        if tf.gfile.Exists(archive_file+".data-00000-of-00001"):
            model._init_from_pretrained_model(archive_file)
        else:
            tf.logging.info("archive file {} does not exists".format(archive_file))
            tf.logging.info("ckpt {} not in model zoo, random initialization".format(pretrained_model_name_or_path))

        return model

    def _init_from_pretrained_model(self, pretrained_model_path):
        tvars = tf.trainable_variables()
        network_name_to_variable = {}
        for var in tvars:
            name = var.name
            m = re.match("^(.*):\\d+$", name)
            if m is not None:
                name = m.group(1)
            network_name_to_variable[name] = var

        try:
            reader = pywrap_tensorflow.NewCheckpointReader(pretrained_model_path)
            var_to_shape_map = reader.get_variable_to_shape_map()
        except errors_impl.DataLossError:
            raise ImportError(
                '`load_weights` requires correct tf ckpts.')

        assignment_map = {}
        for key in var_to_shape_map:
            if "Adam" in key or "beta1_power" in key or "beta2_power" in key:
                continue
            if "global_step" in key:
                continue

            var = None
            if "pre_trained_model" in key:
                root_key = key.replace(key.split("/")[0]+"/","")
            else:
                root_key = key

            for network_key in network_name_to_variable.keys():
                if root_key in network_key:
                    var = network_name_to_variable[network_key]
                    break
            if var is None:
                print("Variable: {} in ckpt not in trainable variable".format(key))
                continue
                #raise ValueError("ckpt var name {} not in trainable variable".format(key))

            assignment_map[key] = var
        tf.logging.info("Load weights from {}".format(pretrained_model_path))
        tf.train.init_from_checkpoint(pretrained_model_path, assignment_map)


def init_from_checkpoint_without_training_ops(pretrained_model_path):
    tvars = tf.trainable_variables()
    network_name_to_variable = {}
    for var in tvars:
        name = var.name
        m = re.match("^(.*):\\d+$", name)
        if m is not None:
            name = m.group(1)
        network_name_to_variable[name] = var

    try:
        reader = pywrap_tensorflow.NewCheckpointReader(pretrained_model_path)
        var_to_shape_map = reader.get_variable_to_shape_map()
    except errors_impl.DataLossError:
        raise ImportError(
            '`load_weights` requires correct tf ckpts.')

    assignment_map = {}
    for key in var_to_shape_map:
        if "Adam" in key or "beta1_power" in key or "beta2_power" in key:
            continue
        if "global_step" in key:
            continue

        var = None
        if "pre_trained_model" in key:
            root_key = key.replace(key.split("/")[0]+"/","")
        else:
            root_key = key

        for network_key in network_name_to_variable.keys():
            if root_key in network_key:
                var = network_name_to_variable[network_key]
                break
        if var is None:
            print("Variable: {} in ckpt not in trainable variable".format(key))
            continue
            #raise ValueError("ckpt var name {} not in trainable variable".format(key))

        assignment_map[key] = var
    tf.logging.info("Load weights from {}".format(pretrained_model_path))
    tf.train.init_from_checkpoint(pretrained_model_path, assignment_map)
