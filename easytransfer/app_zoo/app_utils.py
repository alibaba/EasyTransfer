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
import os
import re
import time
import tensorflow as tf
from easytransfer.datasets import CSVReader, CSVWriter, OdpsTableReader, OdpsTableWriter
from easytransfer import FLAGS
from easytransfer.model_zoo.modeling_bert import BERT_PRETRAINED_MODEL_ARCHIVE_MAP
from easytransfer.preprocessors.tokenization import convert_to_unicode


def copy_pretrain_model_files_to_dir(pretrain_model_name_or_path, output_dir):
    if pretrain_model_name_or_path not in BERT_PRETRAINED_MODEL_ARCHIVE_MAP:
        pretrain_checkpoint_path = pretrain_model_name_or_path
    else:
        pretrain_checkpoint_path = os.path.join(
            FLAGS.modelZooBasePath, BERT_PRETRAINED_MODEL_ARCHIVE_MAP[pretrain_model_name_or_path])
    predict_checkpoint_dir = os.path.dirname(pretrain_checkpoint_path)
    vocab_file = os.path.join(predict_checkpoint_dir, "vocab.txt")
    vocab_out_file = os.path.join(output_dir, "vocab.txt")
    if tf.gfile.Exists(vocab_file) and not tf.gfile.Exists(vocab_out_file):
        tf.gfile.Copy(vocab_file, vocab_out_file)
    config_file = os.path.join(predict_checkpoint_dir, "config.json")
    config_out_file = os.path.join(output_dir, "config.json")
    if tf.gfile.Exists(config_file) and not tf.gfile.Exists(config_out_file):
        tf.gfile.Copy(config_file, config_out_file)


def copy_file_to_new_path(old_dir, new_dir, fname, newfname=None):
    newfname = newfname if newfname else fname
    src_path = os.path.join(old_dir, fname)
    tgt_path = os.path.join(new_dir, newfname)
    if tf.gfile.Exists(src_path) and not tf.gfile.Exists(tgt_path):
        tf.gfile.Copy(src_path, tgt_path)


def get_reader_fn(input_fp=None):
    """ Automatically  get ez_transfer's reader for different env
    """
    if input_fp is None:
        return OdpsTableReader if "PAI" in tf.__version__ else CSVReader

    if "odps://" in input_fp:
        return OdpsTableReader
    else:
        return CSVReader


def get_writer_fn(output_fp=None):
    """ Automatically  get ez_transfer's writer for different env
    """
    if output_fp is None:
        return OdpsTableWriter if "PAI" in tf.__version__ else CSVWriter

    if "odps://" in output_fp:
        return OdpsTableWriter
    else:
        return CSVWriter


def get_all_columns_name(input_glob):
    """ Get all of the column names for ODPSTable

        Args:
            input_glob (`str`): odps path of the input table.
            Eg. odps://pai_exp_dev/tables/ez_transfer_toy_train



        Returns:
            result (`set`): A set of column names
    """
    reader = tf.python_io.TableReader(input_glob,
                                      selected_cols="",
                                      excluded_cols="",
                                      slice_id=0,
                                      slice_count=1,
                                      num_threads=0,
                                      capacity=0)
    schemas = reader.get_schema()
    return set([col_name for col_name, _, _ in schemas])


def get_selected_columns_schema(input_glob, selected_columns):
    """ Get all of the column schema for the selected columns for ODPSTable

        Args:
            input_glob (`str`): odps path of the input table
            selected_columns (`set`): A set of column names of the input table

        Returns:
            result (`str`): A string of easy transfer defined input schema
    """
    reader = tf.python_io.TableReader(input_glob,
                                      selected_cols="",
                                      excluded_cols="",
                                      slice_id=0,
                                      slice_count=1,
                                      num_threads=0,
                                      capacity=0)
    schemas = reader.get_schema()
    colname2schema = dict()
    for col_name, odps_type, _ in schemas:
        if odps_type == u"string":
            colname2schema[str(col_name)] = "str"
        elif odps_type == u"double":
            colname2schema[str(col_name)] = "float"
        elif odps_type == u"bigint":
            colname2schema[str(col_name)] = "int"
        else:
            colname2schema[str(col_name)] = "str"

    col_with_schemas = ["{}:{}:1".format(col_name, colname2schema[col_name])
                        for col_name in selected_columns if col_name]

    rst_schema = ",".join(col_with_schemas)
    return rst_schema


# Parse advanced user defined params
def get_user_defined_prams_dict(user_defined_params_str):
    _user_param_dict = dict()
    for item in re.findall("\w+=[\w/.,\-:\/]+", user_defined_params_str):
        key, val = item.strip().split("=")
        _user_param_dict[key] = val
    return _user_param_dict


def get_label_enumerate_values(label_enumerate_value_or_path):
    if label_enumerate_value_or_path is None:
        return ""
    if tf.gfile.Exists(label_enumerate_value_or_path):
        with tf.gfile.Open(label_enumerate_value_or_path) as f:
            label_enumerate_value = ",".join([convert_to_unicode(line.strip()) for line in f])
    else:
        label_enumerate_value = label_enumerate_value_or_path
    return label_enumerate_value


def get_pretrain_model_name_or_path(pretrain_model_name_or_path):
    contrib_models_path = os.path.join(FLAGS.modelZooBasePath, "contrib_models.json")
    if tf.gfile.Exists(contrib_models_path):
        with tf.gfile.Open(os.path.join(FLAGS.modelZooBasePath, "contrib_models.json")) as f:
            contrib_models = json.load(f)
        if pretrain_model_name_or_path in contrib_models:
            pretrain_model_name_or_path = contrib_models[pretrain_model_name_or_path]
    return pretrain_model_name_or_path


def log_duration_time(func):
    def wrapper(*args, **kw):
        _st_time = time.time()
        func(*args, **kw)
        _end_time = time.time()
        tf.logging.info("Duration time: {:.4f}s".format(_end_time - _st_time))
    return wrapper