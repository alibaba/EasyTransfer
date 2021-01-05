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

import sys
sys.path.append("./")

import json
import os
import tensorflow as tf
try:
    import tensorflow_io as tfio
except:
    pass
from easytransfer import Config, FLAGS
from easytransfer.app_zoo.app_utils import get_all_columns_name, get_selected_columns_schema
from easytransfer.app_zoo.feature_extractor import BertFeatureExtractor


_app_flags = tf.app.flags
_app_flags.DEFINE_string("inputTable", default=None, help='Input table (only for pai cmd)')
_app_flags.DEFINE_string("outputTable", default=None, help='Output table (only for pai cmd)')
_app_flags.DEFINE_string("inputSchema", default=None,
                          help='Only for csv data, the schema of input table')
_app_flags.DEFINE_string("firstSequence", default=None,
                          help='Which column is the first sequence mapping to')
_app_flags.DEFINE_string("secondSequence", default=None,
                          help='Which column is the second sequence mapping to')
_app_flags.DEFINE_string("appendCols", default=None,
                          help='Which columns will be appended on the outputs')
_app_flags.DEFINE_string("outputSchema", default="pool_output,first_token_output,all_hidden_outputs",
                          help='The choices of output features')
_app_flags.DEFINE_integer("sequenceLength", default=128,
                          help='Maximum overall sequence length.')
_app_flags.DEFINE_string("modelName", default='',
                          help='Name of pretrained model')
_app_flags.DEFINE_integer("batchSize", default=32,
                          help='Maximum overall sequence length.')
_APP_FLAGS = _app_flags.FLAGS


class BertFeatConfig(Config):
    def __init__(self):
        """ Configuration adapter for `ez_bert_feat`
            It adapts user command args to configuration protocol of `ez_transfer` engine
        """
        self.mode = "predict_on_the_fly"

        input_table = _APP_FLAGS.inputTable
        output_table = _APP_FLAGS.outputTable

        predict_checkpoint_path = _APP_FLAGS.modelName

        predict_checkpoint_dir = os.path.dirname(predict_checkpoint_path)
        if tf.gfile.Exists(os.path.join(predict_checkpoint_dir, "train_config.json")):
            with tf.gfile.Open(os.path.join(predict_checkpoint_dir, "train_config.json")) as f:
                train_config_json = json.load(f)
            if "model_name" in train_config_json:
                finetune_model_name = train_config_json["model_name"]
            else:
                finetune_model_name = None
            if "_config_json" in train_config_json:
                train_model_config = train_config_json["_config_json"]["model_config"]
            else:
                train_model_config = None
        else:
            finetune_model_name = None
            train_model_config = None

        if "odps://" in FLAGS.inputTable and "PAI" in tf.__version__:
            all_input_col_names = get_all_columns_name(input_table)
        else:
            all_input_col_names = set([t.split(":")[0] for t in _APP_FLAGS.inputSchema.split(",")])
        first_sequence = _APP_FLAGS.firstSequence
        assert first_sequence in all_input_col_names, "The first sequence should be in input schema"
        second_sequence = _APP_FLAGS.secondSequence
        if second_sequence not in all_input_col_names:
            second_sequence = ""
        append_columns = [t for t in _APP_FLAGS.appendCols.split(",") if t and t in all_input_col_names] \
                          if _APP_FLAGS.appendCols else []
        tf.logging.info(input_table)
        if "odps://" in FLAGS.inputTable and "PAI" in tf.__version__:
            selected_cols_set = [first_sequence]
            if second_sequence:
                selected_cols_set.append(second_sequence)
            selected_cols_set.extend(append_columns)
            selected_cols_set = set(selected_cols_set)
            input_schema = get_selected_columns_schema(input_table, selected_cols_set)
        else:
            assert _APP_FLAGS.inputSchema is not None
            input_schema = _APP_FLAGS.inputSchema
        output_schema = _APP_FLAGS.outputSchema
        for column_name in append_columns:
            output_schema += "," + column_name

        config_json = {
            'preprocess_config': {
                'input_schema': input_schema,
                'first_sequence': first_sequence,
                'second_sequence': second_sequence,
                'output_schema': output_schema,
                'sequence_length': _APP_FLAGS.sequenceLength,
                "max_predictions_per_seq": 20
            },
            'model_config': {
                'model_name': 'feat_ext_bert',
                'pretrain_model_name_or_path': _APP_FLAGS.modelName,
                'finetune_model_name': finetune_model_name,
            },
            'predict_config': {
                'predict_checkpoint_path': predict_checkpoint_path,
                'predict_batch_size': _APP_FLAGS.batchSize,
                'predict_input_fp': input_table,
                'predict_output_fp': output_table
            }
        }
        if train_model_config:
            for key, val in train_model_config.items():
                if key not in config_json["model_config"]:
                    config_json["model_config"][str(key)] = val

        config_json["worker_hosts"] = FLAGS.worker_hosts
        config_json["task_index"] = FLAGS.task_index
        config_json["job_name"] = FLAGS.job_name
        config_json["num_gpus"] = FLAGS.workerGPU
        config_json["num_workers"] = FLAGS.workerCount
        tf.logging.info("{}".format(config_json))
        super(BertFeatConfig, self).__init__(mode="predict_on_the_fly", config_json=config_json)

        for key, val in self.__dict__.items():
            tf.logging.info("  {}: {}".format(key, val))


def main():
    # Here is a hack for DSW access OSS
    for argname in ["inputTable", "outputTable", "modelName"]:
        arg = getattr(_APP_FLAGS, argname)
        if arg:
            arg =  arg.replace("\\x01", "\x01").replace("\\x02", "\x02")
            setattr(_APP_FLAGS, argname, arg)
    FLAGS.modelZooBasePath = FLAGS.modelZooBasePath.replace("\\x01", "\x01").replace("\\x02", "\x02")

    # Main function start
    config = BertFeatConfig()
    app = BertFeatureExtractor(user_defined_config=config)
    app.run()

if __name__ == "__main__":
    main()