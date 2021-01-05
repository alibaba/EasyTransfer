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
import tensorflow as tf
from easytransfer import Config, FLAGS
from easytransfer.app_zoo import _name_to_app_model
from easytransfer.app_zoo.app_utils import get_selected_columns_schema, \
    get_user_defined_prams_dict, copy_pretrain_model_files_to_dir, get_label_enumerate_values, \
    get_pretrain_model_name_or_path


class AppConfig(Config):
    def __init__(self, mode, flags):
        if mode == "preprocess":
            config_json = self.build_preprocess_config(flags)
            internal_mode = "preprocess"
        elif mode == "train":
            config_json = self.build_train_config(flags)
            internal_mode = "train_and_evaluate_on_the_fly"
        elif mode == "evaluate":
            config_json = self.build_evaluate_config(flags)
            internal_mode = "evaluate_on_the_fly"
        elif mode == "predict":
            config_json = self.build_predict_config(flags)
            internal_mode = "predict_on_the_fly"
        elif mode == "export":
            config_json = self.build_export_config(flags)
            internal_mode = "export"
        else:
            raise RuntimeError("Invalid mode {}".format(mode))

        # Parse general FLAGS
        config_json["worker_hosts"] = FLAGS.worker_hosts
        config_json["task_index"] = FLAGS.task_index
        config_json["job_name"] = FLAGS.job_name
        config_json["num_gpus"] = FLAGS.workerGPU
        config_json["num_workers"] = FLAGS.workerCount
        super(AppConfig, self).__init__(mode=internal_mode, config_json=config_json)

        if mode == "train" and FLAGS.task_index == 0:
            self.build_train_resource_files(flags)

    def build_train_resource_files(self, flags):
        if not tf.gfile.Exists(flags.checkpointDir):
            tf.gfile.MakeDirs(flags.checkpointDir)
        with tf.gfile.GFile(os.path.join(flags.checkpointDir, "train_config.json"), mode='w') as f:
            json.dump(self.__dict__, f)
        if hasattr(self, "pretrain_model_name_or_path"):
            copy_pretrain_model_files_to_dir(self.pretrain_model_name_or_path, flags.checkpointDir)
        if self.label_enumerate_values and "," in self.label_enumerate_values:
            label_dict = {label: idx for idx, label in enumerate(self.label_enumerate_values.split(","))}
            with tf.gfile.GFile(os.path.join(flags.checkpointDir, "label_mapping.json"), mode='w') as f:
                json.dump(label_dict, f)

    def build_preprocess_config(self, flags):
        first_sequence, second_sequence, label_name = \
            flags.firstSequence, flags.secondSequence, flags.labelName
        input_table = flags.inputTable
        output_table = flags.outputTable
        append_columns = flags.appendCols.split(",") if flags.appendCols else []

        user_param_dict = get_user_defined_prams_dict(flags.advancedParameters)

        if "odps://" in input_table and "PAI" in tf.__version__:
            selected_columns = [first_sequence, second_sequence, label_name] + append_columns
            for key, val in user_param_dict.items():
                if key.endswith("column_name"):
                    selected_columns.append(val)
            selected_columns = set(selected_columns)
            input_schema = get_selected_columns_schema(
                input_table, selected_columns)
        else:
            input_schema = flags.inputSchema
        output_schema = flags.outputSchema
        for column_name in append_columns:
            output_schema += "," + column_name

        if flags.modelName in _name_to_app_model:
            tokenizer_name_or_path = user_param_dict.get("tokenizer_name_or_path", "google-bert-base-zh")
            setattr(self, "model_name", "serialization")
            setattr(self, "app_model_name", flags.modelName)
        else:
            tokenizer_name_or_path = flags.modelName
            setattr(self, "model_name", "serialization")
            setattr(self, "app_model_name", "text_classify_bert")

        config_json = {
            "preprocess_config": {
                "preprocess_input_fp": input_table,
                "preprocess_output_fp": output_table,
                "preprocess_batch_size": flags.batchSize,
                "sequence_length": flags.sequenceLength,
                "tokenizer_name_or_path": tokenizer_name_or_path,
                "input_schema": input_schema,
                "first_sequence": flags.firstSequence,
                "second_sequence": flags.secondSequence,
                "label_name": flags.labelName,
                "label_enumerate_values": get_label_enumerate_values(flags.labelEnumerateValues),
                "output_schema": output_schema
            }
        }
        for key, val in user_param_dict.items():
            setattr(self, key, val)
        return config_json

    def build_train_config(self, flags):
        # Parse input table/csv schema
        first_sequence, second_sequence, label_name = \
            flags.firstSequence, flags.secondSequence, flags.labelName
        label_enumerate_values = get_label_enumerate_values(flags.labelEnumerateValues)

        user_param_dict = get_user_defined_prams_dict(flags.advancedParameters)
        tf.logging.info(user_param_dict)

        train_input_fp, eval_input_fp = FLAGS.inputTable.split(",")
        train_input_fp, eval_input_fp = train_input_fp.strip(), eval_input_fp.strip()

        if "odps://" in FLAGS.inputTable and "PAI" in tf.__version__:
            if first_sequence is None:
                assert flags.sequenceLength is not None
                input_schema = _name_to_app_model[flags.modelName].get_input_tensor_schema(
                    sequence_length=flags.sequenceLength)
            else:
                selected_columns = [first_sequence, second_sequence, label_name]
                for key, val in user_param_dict.items():
                    if key.endswith("column_name"):
                        selected_columns.append(val)
                selected_columns = set(selected_columns)
                input_schema = get_selected_columns_schema(
                    train_input_fp, selected_columns)
        else:
            input_schema = flags.inputSchema

        # Parse args from APP's FLAGS
        config_json = {
            "preprocess_config": {
                "input_schema": input_schema,
                "first_sequence": first_sequence,
                "second_sequence": second_sequence,
                "sequence_length": flags.sequenceLength,
                "label_name": label_name,
                "label_enumerate_values": label_enumerate_values
            },
            "model_config": {
                "model_name": flags.modelName,
            },
            "train_config": {
                "train_input_fp": train_input_fp,
                "num_epochs": flags.numEpochs,
                "save_steps": flags.saveCheckpointSteps,
                "train_batch_size": flags.batchSize,
                "model_dir": flags.checkpointDir,
                "optimizer_config": {
                    "optimizer": flags.optimizerType,
                    "learning_rate": flags.learningRate
                },
                "distribution_config": {
                    "distribution_strategy": flags.distributionStrategy,
                }
            },
            "evaluate_config": {
                "eval_input_fp": eval_input_fp,
                "eval_batch_size": 32,
                "num_eval_steps": None
            }
        }

        if flags.modelName in _name_to_app_model:
            default_model_params = _name_to_app_model[flags.modelName].default_model_params()
        else:
            raise NotImplementedError("model %s not implemented" % flags.modelName)
        for key, _ in default_model_params.items():
            default_val = default_model_params[key]
            if key in user_param_dict:
                if isinstance(default_val, bool):
                    tmp_val = (user_param_dict[key].lower() == "true")
                else:
                    tmp_val = type(default_val)(user_param_dict[key])
                config_json["model_config"][key] = tmp_val
            else:
                config_json["model_config"][key] = default_val

        config_json["model_config"]["num_labels"] = len(label_enumerate_values.split(","))
        if "pretrain_model_name_or_path" in config_json["model_config"]:
            pretrain_model_name_or_path = get_pretrain_model_name_or_path(
                config_json["model_config"]["pretrain_model_name_or_path"])

            config_json["model_config"]["pretrain_model_name_or_path"] = pretrain_model_name_or_path
            config_json["preprocess_config"]["tokenizer_name_or_path"] = pretrain_model_name_or_path
        else:
            config_json["preprocess_config"]["tokenizer_name_or_path"] = ""

        if "num_accumulated_batches" in user_param_dict:
            config_json["train_config"]["distribution_config"]["num_accumulated_batches"] = \
                user_param_dict["num_accumulated_batches"]

        if "pull_evaluation_in_multiworkers_training" in user_param_dict:
            config_json["train_config"]["distribution_config"]["pull_evaluation_in_multiworkers_training"] = \
                (user_param_dict["pull_evaluation_in_multiworkers_training"].lower() == "true")

        other_param_keys = {
            "train_config": ["throttle_secs", "keep_checkpoint_max", "log_step_count_steps"],
            "optimizer_config": ["weight_decay_ratio", "lr_decay", "warmup_ratio", "gradient_clip", "clip_norm_value"],
            "evaluate_config": ["eval_batch_size", "num_eval_steps"],
        }
        for first_key, second_key_list in other_param_keys.items():
            for second_key in second_key_list:
                if second_key in user_param_dict:
                    obj = config_json["train_config"][first_key] if first_key == "optimizer_config" \
                        else config_json[first_key]
                    obj[second_key] = user_param_dict[second_key]

        if "shuffle_buffer_size" in user_param_dict:
            setattr(self, "shuffle_buffer_size", int(user_param_dict["shuffle_buffer_size"]))
        else:
            setattr(self, "shuffle_buffer_size", None)

        if "init_checkpoint_path" in user_param_dict:
            setattr(self, "init_checkpoint_path", user_param_dict["init_checkpoint_path"])

        if "export_best_checkpoint" in user_param_dict:
            assert user_param_dict["export_best_checkpoint"].lower() in ["true", "false"]
            if user_param_dict["export_best_checkpoint"].lower() == "true":
                setattr(self, "export_best_checkpoint", True)
            else:
                setattr(self, "export_best_checkpoint", False)

        if "export_best_checkpoint_metric" in user_param_dict:
            setattr(self, "export_best_checkpoint_metric", user_param_dict["export_best_checkpoint_metric"])
        else:
            if flags.modelName.startswith("text_classify"):
                setattr(self, "export_best_checkpoint_metric", "py_accuracy")
            elif flags.modelName.startswith("text_match") and label_enumerate_values is None:
                setattr(self, "export_best_checkpoint_metric", "mse")
            else:
                setattr(self, "export_best_checkpoint_metric", "accuracy")

        return config_json

    def build_evaluate_config(self, flags):
        input_table = flags.inputTable

        if flags.modelName.startswith("modelhub"):
            checkpoint_path = os.path.join(
                FLAGS.modelZooBasePath, "modelhub",
                flags.modelName.split(":")[-1], "model.ckpt")
        else:
            checkpoint_path = flags.checkpointPath

        ckp_dir = os.path.dirname(checkpoint_path)
        train_config_path = os.path.join(ckp_dir, "train_config.json")
        if tf.gfile.Exists(train_config_path):
            predict_checkpoint_path = checkpoint_path
        else:
            raise RuntimeError("Checkpoint in {} not found".format(ckp_dir))

        with tf.gfile.Open(train_config_path, "r") as f:
            tf.logging.info("config file is {}".format(train_config_path))
            train_config_dict = json.load(f)

        config_json = {
            'preprocess_config': train_config_dict['_config_json']['preprocess_config'],
            'model_config': train_config_dict['_config_json']['model_config'],
            'evaluate_config': {
                'eval_checkpoint_path': predict_checkpoint_path,
                'eval_input_fp': input_table,
                'eval_batch_size': flags.batchSize,
                'num_eval_steps': None
            }
        }

        if "pretrain_model_name_or_path" in config_json['model_config']:
            config_json['model_config']['pretrain_model_name_or_path'] = get_pretrain_model_name_or_path(
                config_json['model_config']['pretrain_model_name_or_path'])

        if not config_json['model_config']['model_name'].startswith("text_match_bert"):
            config_json['model_config']['vocab_path'] =  os.path.join(
                os.path.dirname(predict_checkpoint_path), "train_vocab.txt")
        return config_json

    def build_predict_config(self, flags):
        input_table = flags.inputTable
        output_table = flags.outputTable

        if flags.modelName.startswith("modelhub"):
            checkpoint_path = os.path.join(FLAGS.modelZooBasePath, "modelhub", flags.modelName.split(":")[-1])
        else:
            checkpoint_path = flags.checkpointPath

        ckp_dir =  checkpoint_path if tf.gfile.IsDirectory(checkpoint_path) \
            else os.path.dirname(checkpoint_path)
        train_config_path = os.path.join(ckp_dir, "train_config.json")
        if tf.gfile.Exists(train_config_path):
            predict_checkpoint_path = checkpoint_path
        else:
            raise RuntimeError("Checkpoint in {} not found".format(ckp_dir))

        with tf.gfile.Open(train_config_path, "r") as f:
            tf.logging.info("config file is {}".format(train_config_path))
            train_config_dict = json.load(f)

        first_sequence = flags.firstSequence
        second_sequence = flags.secondSequence
        append_columns = flags.appendCols.split(",") if flags.appendCols else []
        if flags.inputSchema:
            input_schema = flags.inputSchema
        else:
            if "odps://" in input_table and "PAI" in tf.__version__:
                selected_columns = [first_sequence, second_sequence] + append_columns
                for key, val in train_config_dict['_config_json']["model_config"].items():
                    if key.endswith("column_name"):
                        selected_columns.append(val)
                selected_columns = set(selected_columns)
                input_schema = get_selected_columns_schema(
                    input_table, selected_columns)
            else:
                input_schema = flags.inputSchema
        output_schema = flags.outputSchema
        for column_name in append_columns:
            output_schema += "," + column_name

        config_json = {
            'preprocess_config': {
                'input_schema': input_schema,
                'first_sequence': flags.firstSequence,
                'second_sequence': flags.secondSequence,
                'output_schema': output_schema,
                'sequence_length': train_config_dict["sequence_length"],
                'label_name': flags.labelName,
                'label_enumerate_values': flags.labelEnumerateValues if flags.labelEnumerateValues \
                        else train_config_dict["label_enumerate_values"]
            },
            'model_config': train_config_dict['_config_json']['model_config'],
            'predict_config': {
                'predict_checkpoint_path': predict_checkpoint_path,
                'predict_input_fp': input_table,
                'predict_output_fp': output_table,
                'predict_batch_size': flags.batchSize
            }
        }
        if not 'bert' in config_json['model_config']['model_name']:
            config_json['model_config']['vocab_path'] =  os.path.join(
                os.path.dirname(predict_checkpoint_path), "train_vocab.txt")
        user_param_dict = get_user_defined_prams_dict(flags.advancedParameters)
        for key, val in user_param_dict.items():
            setattr(self, key, val)
        return config_json

    def build_export_config(self, flags):
        export_type = flags.exportType
        self.export_type = export_type
        checkpoint_path = flags.checkpointPath
        export_dir_base = flags.exportDirBase
        if export_type == "ez_bert_feat":
            checkpoint_dir = os.path.dirname(checkpoint_path)
            train_config_path = os.path.join(checkpoint_dir, "train_config.json")
            if tf.gfile.Exists(train_config_path):
                with tf.gfile.Open(train_config_path) as f:
                    train_config_json = json.load(f)
                if "model_name" in train_config_json:
                    finetune_model_name = train_config_json["model_name"]
                else:
                    finetune_model_name = None
                pretrain_model_name_or_path = train_config_json["pretrain_model_name_or_path"]
                if "_config_json" in train_config_json:
                    train_model_config = train_config_json["_config_json"]["model_config"]
                else:
                    train_model_config = None
            else:
                pretrain_model_name_or_path = checkpoint_path
                finetune_model_name = None
                train_model_config = None

            config_json = {
                "model_config": {
                    "model_name": "feat_ext_bert",
                    "pretrain_model_name_or_path": pretrain_model_name_or_path,
                    "finetune_model_name": finetune_model_name
                },
                "export_config": {
                    "input_tensors_schema": "input_ids:int:64,input_mask:int:64,segment_ids:int:64,label_ids:int:1",
                    "receiver_tensors_schema": "input_ids:int:64,input_mask:int:64,segment_ids:int:64",
                    "checkpoint_path": checkpoint_path,
                    "export_dir_base": export_dir_base
                }
            }
            if train_model_config:
                for key, val in train_model_config.items():
                    if str(key) not in config_json["model_config"]:
                        config_json["model_config"][str(key)] = val
        elif export_type.startswith("convert"):
            config_json = {
                "model_config": {
                    "model_name": "conversion"
                },
                "export_config": {
                    "checkpoint_path": checkpoint_path,
                    "export_dir_base": export_dir_base,
                    "input_tensors_schema": "",
                    "receiver_tensors_schema": "",
                }
            }
            self.model_zoo_base_path = FLAGS.modelZooBasePath
        else:
            checkpoint_dir = os.path.dirname(checkpoint_path)
            train_config_path = os.path.join(checkpoint_dir, "train_config.json")
            with tf.gfile.Open(train_config_path) as f:
                train_config_json = json.load(f)
            model_config = train_config_json["_config_json"]["model_config"]

            model_name = model_config["model_name"]

            config_json = {
                "model_config": model_config,
                "export_config": {
                    "input_tensors_schema": _name_to_app_model[model_name].get_input_tensor_schema(),
                    "receiver_tensors_schema": _name_to_app_model[model_name].get_received_tensor_schema(),
                    "checkpoint_path": checkpoint_path,
                    "export_dir_base": export_dir_base
                }
            }
            vocab_path = os.path.join(checkpoint_dir, "train_vocab.txt")
            config_json["model_config"]["vocab_path"] = vocab_path
            self.input_schema = config_json["export_config"]["input_tensors_schema"]
            self.sequence_length = train_config_json["sequence_length"]
            self.label_enumerate_values =  train_config_json.get("label_enumerate_values", None)

        return config_json