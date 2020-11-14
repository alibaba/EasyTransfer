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

import copy
import json
import random
import numpy as np
import os
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.INFO)
tf.logging.info("*********** tf.__version__ is {} ******".format(tf.__version__))
SEED = 123123
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.reset_default_graph()
tf.set_random_seed(SEED)

flags = tf.app.flags
flags.DEFINE_string("config", default=None, help='')
flags.DEFINE_string("tables", default=None, help='')
flags.DEFINE_string("outputs", default=None, help='')
flags.DEFINE_integer('task_index', 0, 'task_index')
flags.DEFINE_string('worker_hosts', 'localhost:5001', 'worker_hosts')
flags.DEFINE_string('job_name', 'worker', 'job_name')
flags.DEFINE_string("mode", default=None, help="Which mode")
flags.DEFINE_string("modelZooBasePath", default=os.path.join(os.getenv("HOME"), ".eztransfer_modelzoo"), help="eztransfer_modelzoo")
flags.DEFINE_integer("workerCount", default=1, help="num_workers")
flags.DEFINE_integer("workerGPU", default=1, help="num_gpus")
flags.DEFINE_integer("workerCPU", default=1, help="num_cpus")
flags.DEFINE_string('f', '', 'kernel')
FLAGS = flags.FLAGS

from easytransfer.utils.hooks import avgloss_logger_hook
from easytransfer.optimizers import get_train_op


class Config(object):
    def __init__(self, mode, config_json):
        self._config_json = copy.deepcopy(config_json)
        self.mode = mode

        self.worker_hosts = str(config_json["worker_hosts"])
        self.task_index = int(config_json["task_index"])
        self.job_name = str(config_json["job_name"])
        self.num_gpus = int(config_json["num_gpus"])
        self.num_workers = int(config_json["num_workers"])
        if "oss://" not in FLAGS.modelZooBasePath:
            FLAGS.modelZooBasePath = config_json.get("modelZooBasePath", os.path.join(os.getenv("HOME"), ".eztransfer_modelzoo"))
        tf.logging.info("***************** modelZooBasePath {} ***************".format(FLAGS.modelZooBasePath))
        if self.mode == 'train' or self.mode == "train_and_evaluate" \
                or self.mode == "train_and_evaluate_on_the_fly" or self.mode == "train_on_the_fly":

            self.enable_xla = bool(config_json["train_config"].get('distribution_config', {}).get(
                'enable_xla', False))

            self.distribution_strategy = str(
                config_json["train_config"].get('distribution_config', {}).get("distribution_strategy", None))
            self.pull_evaluation_in_multiworkers_training = bool(config_json["train_config"].get('distribution_config', {}).get(
                'pull_evaluation_in_multiworkers_training', False))

            self.num_accumulated_batches = int(config_json["train_config"].get('distribution_config', {}).get(
                'num_accumulated_batches', 1))

            self.num_model_replica = int(config_json["train_config"].get('distribution_config', {}).get(
                'num_model_replica', 1))

            self.num_communicators = int(config_json["train_config"].get('distribution_config', {}).get(
                'num_communicators', 1))

            self.num_splits = int(config_json["train_config"].get('distribution_config', {}).get(
                'num_splits', 1))

            # optimizer
            self.optimizer = str(config_json["train_config"]['optimizer_config'].get('optimizer', "adam"))
            self.learning_rate = float(config_json['train_config']['optimizer_config'].get('learning_rate', 0.001))

            self.weight_decay_ratio = float(
                config_json['train_config']['optimizer_config'].get('weight_decay_ratio', 0))
            self.lr_decay = config_json['train_config']['optimizer_config'].get('lr_decay', "polynomial")
            self.warmup_ratio = float(config_json['train_config']['optimizer_config'].get('warmup_ratio', 0.1))
            self.clip_norm_value = float(config_json['train_config']['optimizer_config'].get('clip_norm_value', 1.0))
            self.gradient_clip = bool(config_json['train_config']['optimizer_config'].get('gradient_clip', True))
            self.num_freezed_layers = int(config_json['train_config']['optimizer_config'].get('num_freezed_layers', 0))

            # misc
            self.num_epochs = float(config_json['train_config'].get('num_epochs', 1))
            try:
                self.model_dir = str(config_json['train_config'].get('model_dir', None))
            except:
                raise ValueError("input model dir")

            self.throttle_secs = int(config_json['train_config'].get('throttle_secs', 100))
            self.keep_checkpoint_max = int(config_json['train_config'].get('keep_checkpoint_max', 10))

            if 'save_steps' not in config_json['train_config']:
                self.save_steps = None
            else:
                self.save_steps = int(config_json['train_config']['save_steps']) \
                    if config_json['train_config']['save_steps'] else \
                    config_json['train_config']['save_steps']

            self.log_step_count_steps = int(config_json['train_config'].get('log_step_count_steps', 100))

            # model
            for key, val in config_json['model_config'].items():
                setattr(self, key, val)

            # data
            self.input_schema = str(config_json['preprocess_config'].get('input_schema', None))

            if self.mode == 'train_and_evaluate_on_the_fly' or self.mode == 'train_on_the_fly':
                self.sequence_length = int(config_json['preprocess_config']['sequence_length'])
                self.first_sequence = str(config_json['preprocess_config']['first_sequence'])
                self.second_sequence = str(config_json['preprocess_config'].get('second_sequence', None))
                self.label_name = str(config_json['preprocess_config']['label_name'])
                self.label_enumerate_values = config_json['preprocess_config'].get('label_enumerate_values', None)
                self.append_feature_columns = config_json['preprocess_config'].get('append_feature_columns', None)

            if self.mode == 'train_and_evaluate' or self.mode == 'train_and_evaluate_on_the_fly':
                self.eval_batch_size = int(config_json['evaluate_config']['eval_batch_size'])

                if 'num_eval_steps' not in config_json['evaluate_config']:
                    self.num_eval_steps = None
                else:
                    self.num_eval_steps = int(config_json['evaluate_config']['num_eval_steps']) \
                        if config_json['evaluate_config']['num_eval_steps'] else \
                        config_json['evaluate_config']['num_eval_steps']

                self.eval_input_fp = str(config_json['evaluate_config']['eval_input_fp'])

            self.train_input_fp = str(config_json['train_config']['train_input_fp'])
            self.train_batch_size = int(config_json['train_config']['train_batch_size'])

        elif self.mode == "evaluate" or self.mode == "evaluate_on_the_fly":
            self.eval_ckpt_path = config_json['evaluate_config']['eval_checkpoint_path']

            self.input_schema = config_json['preprocess_config']['input_schema']
            if self.mode == "evaluate_on_the_fly":
                self.sequence_length = int(config_json['preprocess_config']['sequence_length'])
                self.first_sequence = str(config_json['preprocess_config']['first_sequence'])
                self.second_sequence = str(config_json['preprocess_config']['second_sequence'])
                self.label_name = str(config_json['preprocess_config']['label_name'])
                self.label_enumerate_values = config_json['preprocess_config'].get('label_enumerate_values', None)

            for key, val in config_json['model_config'].items():
                setattr(self, key, val)

            if "train_config" in config_json:
                self.model_dir = str(config_json['train_config'].get('model_dir', None))
                self.distribution_strategy = str(
                    config_json["train_config"].get('distribution_config', {}).get("distribution_strategy", None))
            self.eval_batch_size = config_json['evaluate_config']['eval_batch_size']
            self.num_eval_steps = config_json['evaluate_config'].get('num_eval_steps', None)
            self.eval_input_fp = config_json['evaluate_config']['eval_input_fp']

        elif self.mode == 'predict' or self.mode == 'predict_on_the_fly':
            self.predict_checkpoint_path = config_json['predict_config'].get('predict_checkpoint_path', None)
            self.input_schema = config_json['preprocess_config']['input_schema']
            self.label_name = config_json['preprocess_config'].get('label_name', None)
            self.label_enumerate_values = config_json['preprocess_config'].get('label_enumerate_values', None)
            self.append_feature_columns = config_json['preprocess_config'].get('append_feature_columns', None)
            self.model_dir = config_json.get('train_config', {}).get('model_dir', None)
            self.output_schema = config_json['preprocess_config'].get('output_schema', None)

            if self.mode == 'predict_on_the_fly':
                self.first_sequence = config_json['preprocess_config']['first_sequence']
                self.second_sequence = config_json['preprocess_config'].get('second_sequence', None)
                self.sequence_length = config_json['preprocess_config']['sequence_length']
                self.max_predictions_per_seq = config_json['preprocess_config'].get('max_predictions_per_seq', None)

            self.predict_batch_size = config_json['predict_config']['predict_batch_size']

            if config_json['preprocess_config'].get('output_schema', None) == "bert_finetune":
                self.output_schema = "input_ids,input_mask,segment_ids,label_id"

            elif config_json['preprocess_config'].get('output_schema', None) == "bert_pretrain":
                self.output_schema = "input_ids,input_mask,segment_ids,masked_lm_positions,masked_lm_ids,masked_lm_weights"

            elif config_json['preprocess_config'].get('output_schema', None) == "bert_predict":
                self.output_schema = "input_ids,input_mask,segment_ids"
            else:
                self.output_schema = config_json['preprocess_config'].get('output_schema', None)

            self.model_config = config_json['model_config']
            for key, val in config_json['model_config'].items():
                setattr(self, key, val)

            self.predict_input_fp = config_json['predict_config']['predict_input_fp']
            self.predict_output_fp = config_json['predict_config'].get('predict_output_fp', None)

        elif self.mode == 'export':
            self.checkpoint_path = config_json['export_config']['checkpoint_path']
            for key, val in config_json['model_config'].items():
                setattr(self, key, val)
            self.export_dir_base = config_json['export_config']['export_dir_base']
            self.checkpoint_path = config_json['export_config']['checkpoint_path']
            self.receiver_tensors_schema = config_json['export_config']['receiver_tensors_schema']
            self.input_tensors_schema = config_json['export_config']['input_tensors_schema']

        elif self.mode == 'preprocess':
            self.input_schema = config_json['preprocess_config']['input_schema']
            self.first_sequence = config_json['preprocess_config']['first_sequence']
            self.second_sequence = config_json['preprocess_config'].get('second_sequence', None)
            self.label_name = config_json['preprocess_config'].get('label_name', None)
            self.label_enumerate_values = config_json['preprocess_config'].get('label_enumerate_values', None)
            self.sequence_length = config_json['preprocess_config']['sequence_length']
            self.max_predictions_per_seq = config_json['preprocess_config'].get('max_predictions_per_seq', None)

            if config_json['preprocess_config']['output_schema'] == "bert_finetune":
                self.output_schema = "input_ids,input_mask,segment_ids,label_id"

            elif config_json['preprocess_config']['output_schema'] == "bert_pretrain":
                self.output_schema = "input_ids,input_mask,segment_ids,masked_lm_positions,masked_lm_ids,masked_lm_weights"

            elif config_json['preprocess_config']['output_schema'] == "bert_predict":
                self.output_schema = "input_ids,input_mask,segment_ids"
            else:
                self.output_schema = config_json['preprocess_config']['output_schema']

            self.preprocess_input_fp = config_json['preprocess_config']['preprocess_input_fp']
            self.preprocess_output_fp = config_json['preprocess_config']['preprocess_output_fp']
            self.preprocess_batch_size = config_json['preprocess_config']['preprocess_batch_size']
            self.tokenizer_name_or_path = config_json['preprocess_config']['tokenizer_name_or_path']

    def __str__(self):
        return json.dumps(self.__dict__, sort_keys=False, indent=4)


class EzTransEstimator(object):
    def __init__(self, **kwargs):

        if self.config.mode == 'train' or self.config.mode == "train_and_evaluate" or \
                self.config.mode == "train_and_evaluate_on_the_fly" or self.config.mode == "train_on_the_fly":

            tf.logging.info("***********Running in {} mode***********".format(self.config.mode))

            if self.config.enable_xla is True:
                tf.logging.info("***********Enable Tao***********")
                os.environ['BRIDGE_ENABLE_TAO'] = 'True'
                os.environ["TAO_ENABLE_CHECK"] = "false"
                os.environ["TAO_COMPILATION_MODE_ASYNC"] = "false"
                os.environ["DISABLE_DEADNESS_ANALYSIS"] = "true"
            else:
                tf.logging.info("***********Disable Tao***********")

            NCCL_MAX_NRINGS = "4"
            NCCL_MIN_NRINGS = "2"
            NCCL_IB_DISABLE = "0"
            NCCL_P2P_DISABLE = "0"
            NCCL_SHM_DISABLE = "0"
            NCCL_LAUNCH_MODE = "PARALLEL"

            TF_JIT_PROFILING = 'False'
            PAI_ENABLE_HLO_DUMPER = 'False'

            os.environ['PAI_ENABLE_HLO_DUMPER'] = PAI_ENABLE_HLO_DUMPER
            os.environ['TF_JIT_PROFILING'] = TF_JIT_PROFILING

            os.environ["NCCL_MAX_NRINGS"] = NCCL_MAX_NRINGS
            os.environ["NCCL_MIN_NRINGS"] = NCCL_MIN_NRINGS
            os.environ["NCCL_IB_DISABLE"] = NCCL_IB_DISABLE
            os.environ["NCCL_P2P_DISABLE"] = NCCL_P2P_DISABLE
            os.environ["NCCL_SHM_DISABLE"] = NCCL_SHM_DISABLE
            os.environ["NCCL_LAUNCH_MODE"] = NCCL_LAUNCH_MODE

            tf.logging.info("***********NCCL_IB_DISABLE {}***********".format(NCCL_IB_DISABLE))
            tf.logging.info("***********NCCL_P2P_DISABLE {}***********".format(NCCL_P2P_DISABLE))
            tf.logging.info("***********NCCL_SHM_DISABLE {}***********".format(NCCL_SHM_DISABLE))
            tf.logging.info("***********NCCL_MAX_NRINGS {}***********".format(NCCL_MAX_NRINGS))
            tf.logging.info("***********NCCL_MIN_NRINGS {}***********".format(NCCL_MIN_NRINGS))
            tf.logging.info("***********NCCL_LAUNCH_MODE {}***********".format(NCCL_LAUNCH_MODE))
            tf.logging.info("***********TF_JIT_PROFILING {}***********".format(TF_JIT_PROFILING))
            tf.logging.info("***********PAI_ENABLE_HLO_DUMPER {}***********".format(PAI_ENABLE_HLO_DUMPER))

            self.strategy = None
            if self.config.num_gpus >= 1 and self.config.num_workers >= 1 and \
                    (self.config.distribution_strategy == "ExascaleStrategy" or
                     self.config.distribution_strategy == "CollectiveAllReduceStrategy"):

                if "PAI" in tf.__version__:
                    import pai
                    worker_hosts = self.config.worker_hosts.split(',')
                    tf.logging.info("***********Job Name is {}***********".format(self.config.job_name))
                    tf.logging.info("***********Task Index is {}***********".format(self.config.task_index))
                    tf.logging.info("***********Worker Hosts is {}***********".format(self.config.worker_hosts))
                    pai.distribute.set_tf_config(self.config.job_name,
                                                 self.config.task_index,
                                                 worker_hosts,
                                                 has_evaluator=self.config.pull_evaluation_in_multiworkers_training)

                if self.config.distribution_strategy == "ExascaleStrategy":
                    tf.logging.info("*****************Using ExascaleStrategy*********************")
                    if "PAI" in tf.__version__:
                        tf.logging.info("***********ESS_COMMUNICATION_NUM_COMMUNICATORS {}***********".format(
                            self.config.num_communicators))
                        tf.logging.info(
                            "***********ESS_COMMUNICATION_NUM_SPLITS {}***********".format(self.config.num_splits))
                        import pai
                        self.strategy = pai.distribute.ExascaleStrategy(num_gpus=self.config.num_gpus,
                                                                        num_micro_batches=self.config.num_accumulated_batches,
                                                                        num_communicators=self.config.num_communicators,
                                                                        max_splits=self.config.num_splits)
                    else:
                        raise ValueError("Please run ExascaleStrategy in DLC")

                elif self.config.distribution_strategy == "CollectiveAllReduceStrategy":
                    tf.logging.info("*****************Using CollectiveAllReduceStrategy*********************")
                    if "PAI" in tf.__version__:
                        cross_tower_ops_type = "horovod"
                        tf.logging.info("*****************cross_tower_ops_type is {}*********************".format(
                            cross_tower_ops_type))
                        self.strategy = tf.contrib.distribute.CollectiveAllReduceStrategy(
                            num_gpus_per_worker=self.config.num_gpus,
                            cross_tower_ops_type=cross_tower_ops_type,
                            all_dense=True,
                            iter_size=self.config.num_accumulated_batches)
                    else:
                        self.strategy = tf.contrib.distribute.CollectiveAllReduceStrategy(
                            num_gpus_per_worker=self.config.num_gpus)

                if self.config.pull_evaluation_in_multiworkers_training is True:
                    real_num_workers = self.config.num_workers - 1
                else:
                    real_num_workers = self.config.num_workers

                global_batch_size = self.config.train_batch_size * self.config.num_gpus * real_num_workers * self.config.num_accumulated_batches


            elif self.config.num_gpus > 1 and self.config.num_workers == 1 and \
                    self.config.distribution_strategy == "MirroredStrategy":
                tf.logging.info("*****************Using MirroredStrategy*********************")
                if "PAI" in tf.__version__:
                    tf.logging.info("*****************Using PAI MirroredStrategy*********************")
                    if 'TF_CONFIG' in os.environ:
                        del os.environ['TF_CONFIG']
                    from tensorflow.contrib.distribute.python import cross_tower_ops as cross_tower_ops_lib
                    cross_tower_ops = cross_tower_ops_lib.AllReduceCrossTowerOps('nccl')
                    self.strategy = tf.contrib.distribute.MirroredStrategy(
                        num_gpus=self.config.num_gpus,
                        cross_tower_ops=cross_tower_ops,
                        all_dense=True,
                        iter_size=self.config.num_accumulated_batches)
                else:
                    self.strategy = tf.contrib.distribute.MirroredStrategy(num_gpus=self.config.num_gpus)

                global_batch_size = self.config.train_batch_size * self.config.num_gpus * self.config.num_accumulated_batches

            elif self.config.num_gpus >= 1 and self.config.num_workers >= 1 and \
                    self.config.distribution_strategy == "WhaleStrategy":

                if "PAI" in tf.__version__:
                    tf.logging.info("***********Job Name is {}***********".format(self.config.job_name))
                    tf.logging.info("***********Task Index is {}***********".format(self.config.task_index))
                    tf.logging.info("***********Worker Hosts is {}***********".format(self.config.worker_hosts))

                tf.logging.info("*****************Using WhaleStrategy*********************")
                WHALE_COMMUNICATION_SPARSE_AS_DENSE = "True"
                WHALE_UNBALANCED_IO_SLICING = "True"
                os.environ["WHALE_COMMUNICATION_SPARSE_AS_DENSE"] = WHALE_COMMUNICATION_SPARSE_AS_DENSE
                os.environ["WHALE_COMMUNICATION_NUM_COMMUNICATORS"] = str(self.config.num_communicators)
                os.environ["WHALE_COMMUNICATION_NUM_SPLITS"] = str(self.config.num_splits)
                os.environ["WHALE_UNBALANCED_IO_SLICING"] = WHALE_UNBALANCED_IO_SLICING
                tf.logging.info("***********WHALE_COMMUNICATION_SPARSE_AS_DENSE {}***********".format(WHALE_COMMUNICATION_SPARSE_AS_DENSE))
                tf.logging.info("***********WHALE_COMMUNICATION_NUM_COMMUNICATORS {}***********".format(self.config.num_communicators))
                tf.logging.info("***********WHALE_COMMUNICATION_NUM_SPLITS {}***********".format(self.config.num_splits))
                tf.logging.info("***********WHALE_UNBALANCED_IO_SLICING {}***********".format(WHALE_UNBALANCED_IO_SLICING))
                global_batch_size = self.config.train_batch_size * self.config.num_accumulated_batches * self.config.num_model_replica

            elif self.config.num_gpus == 1 and self.config.num_workers == 1:
                if 'TF_CONFIG' in os.environ:
                    del os.environ['TF_CONFIG']
                global_batch_size = self.config.train_batch_size * self.config.num_accumulated_batches
                tf.logging.info("***********Single worker, Single gpu, Don't use distribution strategy***********")

            elif self.config.num_gpus == 0 and self.config.num_workers == 1:
                global_batch_size = self.config.train_batch_size * self.config.num_accumulated_batches
                tf.logging.info("***********Single worker, Running on CPU***********")

            else:
                raise ValueError(
                    "In train model, Please set correct num_workers, num_gpus and distribution_strategy, \n"
                    "num_workers>=1, num_gpus>=1, distribution_strategy=WhaleStrategy|ExascaleStrategy|CollectiveAllReduceStrategy \n"
                    "num_workers>1, num_gpus==1, distribution_strategy=MirroredStrategy \n"
                    "num_workers=1, num_gpus=1, distribution_strategy=None")

            # Validate optional keyword arguments.
            if "num_train_examples" not in kwargs:
                raise ValueError('Please pass num_train_examples')

            self.num_train_examples = kwargs['num_train_examples']

            # if save steps is None, save per epoch
            if self.config.save_steps is None:
                self.save_steps = int(self.num_train_examples / global_batch_size)
            else:
                self.save_steps = self.config.save_steps

            self.train_steps = int(self.num_train_examples *
                                   self.config.num_epochs / global_batch_size) + 1

            self.throttle_secs = self.config.throttle_secs
            self.model_dir = self.config.model_dir
            tf.logging.info("model_dir: {}".format(self.config.model_dir))
            tf.logging.info("num workers: {}".format(self.config.num_workers))
            tf.logging.info("num gpus: {}".format(self.config.num_gpus))
            tf.logging.info("learning rate: {}".format(self.config.learning_rate))
            tf.logging.info("train batch size: {}".format(self.config.train_batch_size))
            tf.logging.info("global batch size: {}".format(global_batch_size))
            tf.logging.info("num accumulated batches: {}".format(self.config.num_accumulated_batches))
            tf.logging.info("num model replica: {}".format(self.config.num_model_replica))
            tf.logging.info("num train examples per epoch: {}".format(self.num_train_examples))
            tf.logging.info("num epochs: {}".format(self.config.num_epochs))
            tf.logging.info("train steps: {}".format(self.train_steps))
            tf.logging.info("save steps: {}".format(self.save_steps))
            tf.logging.info("throttle secs: {}".format(self.throttle_secs))
            tf.logging.info("keep checkpoint max: {}".format(self.config.keep_checkpoint_max))
            tf.logging.info("warmup ratio: {}".format(self.config.warmup_ratio))
            tf.logging.info("gradient clip: {}".format(self.config.gradient_clip))
            tf.logging.info("clip norm value: {}".format(self.config.clip_norm_value))
            tf.logging.info("log step count steps: {}".format(self.config.log_step_count_steps))

            if self.config.distribution_strategy != "WhaleStrategy":
                self.estimator = tf.estimator.Estimator(
                    model_fn=self._build_model_fn(),
                    model_dir=self.config.model_dir,
                    config=self._get_run_train_config(config=self.config))
            else:
                tf.logging.info("***********Using Whale Estimator***********")
                try:
                    from easytransfer.engines.whale_estimator import WhaleEstimator
                    import whale as wh
                    wh.init()
                    self.estimator = WhaleEstimator(
                        model_fn=self._build_model_fn(),
                        model_dir=self.config.model_dir,
                        num_model_replica=self.config.num_model_replica,
                        num_accumulated_batches=self.config.num_accumulated_batches,
                        keep_checkpoint_max=self.config.keep_checkpoint_max,
                        save_checkpoints_steps=self.config.save_steps,
                        task_index=self.config.task_index)
                except:
                    raise NotImplementedError("WhaleStrategy doesn't work well")


            if self.config.mode == 'train_and_evaluate' or self.config.mode == 'train_and_evaluate_on_the_fly':
                self.num_eval_steps = self.config.num_eval_steps
                tf.logging.info("num eval steps: {}".format(self.num_eval_steps))

        elif self.config.mode == 'evaluate' or self.config.mode == 'evaluate_on_the_fly':
            self.num_eval_steps = self.config.num_eval_steps
            tf.logging.info("num eval steps: {}".format(self.num_eval_steps))
            tf.logging.info("***********Running in {} mode***********".format(self.config.mode))
            self.estimator = tf.estimator.Estimator(
                model_fn=self._build_model_fn(),
                config=self._get_run_predict_config())

        elif self.config.mode == 'predict' or self.config.mode == 'predict_on_the_fly':
            tf.logging.info("***********Running in {} mode***********".format(self.config.mode))
            self.estimator = tf.estimator.Estimator(
                model_fn=self._build_model_fn(),
                config=self._get_run_predict_config())

        elif self.config.mode == 'export':
            tf.logging.info("***********Running in {} mode***********".format(self.config.mode))
            self.estimator = tf.estimator.Estimator(
                model_fn=self._build_model_fn(),
                config=self._get_run_predict_config())

        elif self.config.mode == 'preprocess':
            tf.logging.info("***********Running in {} mode***********".format(self.config.mode))
            self.estimator = tf.estimator.Estimator(
                model_fn=self._build_model_fn(),
                config=tf.estimator.RunConfig())

            self.first_sequence = self.config.first_sequence
            self.second_sequence = self.config.second_sequence
            self.label_enumerate_values = self.config.label_enumerate_values
            self.label_name = self.config.label_name

    def _get_run_train_config(self, config):
        session_config = tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False,
            intra_op_parallelism_threads=1024,
            inter_op_parallelism_threads=1024,
            gpu_options=tf.GPUOptions(allow_growth=True,
                                      force_gpu_compatible=True,
                                      per_process_gpu_memory_fraction=1.0))

        run_config = tf.estimator.RunConfig(session_config=session_config,
                                            model_dir=config.model_dir,
                                            tf_random_seed=123123,
                                            train_distribute=self.strategy,
                                            log_step_count_steps=100,
                                            save_checkpoints_steps=self.save_steps,
                                            keep_checkpoint_max=config.keep_checkpoint_max
                                            )
        return run_config

    def _get_run_predict_config(self):

        session_config = tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False,
            intra_op_parallelism_threads=1024,
            inter_op_parallelism_threads=1024,
            gpu_options=tf.GPUOptions(allow_growth=True,
                                      force_gpu_compatible=True,
                                      per_process_gpu_memory_fraction=1.0))

        run_config = tf.estimator.RunConfig(session_config=session_config)
        return run_config

    def _build_model_fn(self):
        def model_fn(features, labels, mode, params):
            if mode == tf.estimator.ModeKeys.TRAIN:
                logits, labels = self.build_logits(features, mode=mode)
                total_loss = self.build_loss(logits, labels)
                num_towers = self.config.num_workers * self.config.num_gpus
                train_op = get_train_op(learning_rate=self.config.learning_rate,
                                        weight_decay_ratio=self.config.weight_decay_ratio,
                                        loss=total_loss,
                                        num_towers=num_towers,
                                        lr_decay=self.config.lr_decay,
                                        warmup_ratio=self.config.warmup_ratio,
                                        optimizer_name=self.config.optimizer,
                                        tvars=self.tvars if hasattr(self, "tvars") else None,
                                        train_steps=self.train_steps,
                                        clip_norm=self.config.gradient_clip,
                                        clip_norm_value=self.config.clip_norm_value,
                                        num_freezed_layers=self.config.num_freezed_layers
                                        )

                if self.config.distribution_strategy == "WhaleStrategy":
                    return total_loss, train_op

                avgloss_hook = avgloss_logger_hook(self.train_steps,
                                                   total_loss,
                                                   self.model_dir,
                                                   self.config.log_step_count_steps,
                                                   self.config.task_index)

                summary_hook = tf.train.SummarySaverHook(save_steps=100, summary_op=tf.summary.merge_all())
                return tf.estimator.EstimatorSpec(
                    mode=mode, loss=total_loss, train_op=train_op,
                    training_hooks=[summary_hook, avgloss_hook])

            elif mode == tf.estimator.ModeKeys.EVAL:
                logits, labels = self.build_logits(features, mode=mode)
                eval_loss = self.build_loss(logits, labels)
                tf.summary.scalar("eval_loss", eval_loss)
                metrics = self.build_eval_metrics(logits, labels)
                summary_hook = tf.train.SummarySaverHook(save_steps=100,
                                                         summary_op=tf.summary.merge_all())
                return tf.estimator.EstimatorSpec(mode, loss=eval_loss,
                                                  eval_metric_ops=metrics,
                                                  evaluation_hooks=[summary_hook])

            elif mode == tf.estimator.ModeKeys.PREDICT:
                if self.config.mode == 'predict' or self.config.mode == 'export':
                    output = self.build_logits(features, mode=mode)
                    predictions = self.build_predictions(output)

                elif self.config.mode == 'predict_on_the_fly':
                    output = self.build_logits(features, mode=mode)
                    predictions = self.build_predictions(output)

                elif self.config.mode == 'preprocess':
                    output = self.build_logits(features, mode=mode)
                    predictions = self.build_predictions(output)
                else:
                    predictions = features

                output = {'serving_default': tf.estimator.export.PredictOutput(predictions)}
                predictions.update(features)

                return tf.estimator.EstimatorSpec(
                    mode=mode,
                    predictions=predictions,
                    export_outputs=output)

        return model_fn

    def run_train_and_evaluate(self, train_reader, eval_reader):
        train_spec = tf.estimator.TrainSpec(input_fn=train_reader.get_input_fn(),
                                            max_steps=self.train_steps)

        eval_spec = tf.estimator.EvalSpec(input_fn=eval_reader.get_input_fn(),
                                          steps=self.num_eval_steps,
                                          throttle_secs=self.throttle_secs)

        tf.logging.info("*********Calling tf.estimator.train_and_evaluate *********")
        tf.estimator.train_and_evaluate(self.estimator,
                                        train_spec=train_spec,
                                        eval_spec=eval_spec)

    def run_train(self, reader):
        self.estimator.train(input_fn=reader.get_input_fn(),
                             max_steps=self.train_steps)

    def run_evaluate(self, reader, checkpoint_path=None):
        return self.estimator.evaluate(input_fn=reader.get_input_fn(),
                                       steps=self.num_eval_steps,
                                       checkpoint_path=checkpoint_path)

    def run_predict(self, reader, writer=None, checkpoint_path=None, yield_single_examples=False):

        if writer is None:
            return self.estimator.predict(
                input_fn=reader.get_input_fn(),
                yield_single_examples=yield_single_examples,
                checkpoint_path=checkpoint_path)

        for batch_idx, outputs in enumerate(self.estimator.predict(input_fn=reader.get_input_fn(),
                                                                   yield_single_examples=yield_single_examples,
                                                                   checkpoint_path=checkpoint_path)):

            if batch_idx % 100 == 0:
                tf.logging.info("Processing %d batches" % (batch_idx))
            writer.process(outputs)

        writer.close()

    def run_preprocess(self, reader, writer):
        for batch_idx, outputs in enumerate(self.estimator.predict(input_fn=reader.get_input_fn(),
                                                                   yield_single_examples=False,
                                                                   checkpoint_path=None)):
            if batch_idx % 100 == 0:
                tf.logging.info("Processing %d batches" % (batch_idx))
            writer.process(outputs)

        writer.close()

    def export_model(self):

        export_dir_base = self.config.export_dir_base
        checkpoint_path = self.config.checkpoint_path

        def serving_input_receiver_fn():
            export_features, receiver_tensors = self.get_export_features()
            return tf.estimator.export.ServingInputReceiver(
                features=export_features, receiver_tensors=receiver_tensors, receiver_tensors_alternatives={})

        return self.estimator.export_savedmodel(export_dir_base, serving_input_receiver_fn,
                                                checkpoint_path=checkpoint_path)

class base_model(EzTransEstimator):
    def __init__(self, **kwargs):
        user_defined_config = kwargs.get("user_defined_config", None)
        if user_defined_config is None:
            assert FLAGS.mode is not None
            with tf.gfile.Open(FLAGS.config, "r") as f:
                tf.logging.info("config file is {}".format(FLAGS.config))
                config_json = json.load(f)
            # enhance config_json
            config_json["worker_hosts"] = FLAGS.worker_hosts
            config_json["task_index"] = FLAGS.task_index
            config_json["job_name"] = FLAGS.job_name
            config_json["num_gpus"] = FLAGS.workerGPU
            config_json["num_workers"] = FLAGS.workerCount

            if FLAGS.tables is not None:
                if FLAGS.mode.startswith("train_and_evaluate"):
                    config_json['train_config']['train_input_fp'] = FLAGS.tables.split(",")[0]
                    config_json['evaluate_config']['eval_input_fp'] = FLAGS.tables.split(",")[1]
                elif FLAGS.mode.startswith("train"):
                    config_json['train_config']['train_input_fp'] = FLAGS.tables.split(",")[0]
                elif FLAGS.mode.startswith("evaluate"):
                    config_json['evaluate_config']['eval_input_fp'] = FLAGS.tables.split(",")[0]
                elif FLAGS.mode.startswith("predict"):
                    config_json['predict_config']['predict_input_fp'] = FLAGS.tables.split(",")[0]
                elif FLAGS.mode.startswith("preprocess"):
                    config_json['preprocess_config']['preprocess_input_fp'] = FLAGS.tables.split(",")[0]
                else:
                    raise RuntimeError

            if FLAGS.outputs is not None:
                if FLAGS.mode.startswith("predict"):
                    config_json['predict_config']['predict_output_fp'] = FLAGS.outputs.split(",")[0]
                elif FLAGS.mode.startswith("preprocess"):
                    config_json['preprocess_config']['preprocess_output_fp'] = FLAGS.outputs.split(",")[0]
                else:
                    raise RuntimeError

            if "predict" in FLAGS.mode:
                if config_json['predict_config'].get('predict_checkpoint_path', None) is not None:
                    model_ckpt = config_json['predict_config']['predict_checkpoint_path'].split("/")[-1]
                    config_fp = config_json['predict_config']['predict_checkpoint_path'].replace(model_ckpt,
                                                                                                 "train_config.json")
                    if tf.gfile.Exists(config_fp):
                        with tf.gfile.Open(config_fp, "r") as f:
                            saved_config = json.load(f)
                            model_config = saved_config.get("model_config", None)
                            config_json["model_config"] = model_config

            self.config = Config(mode=FLAGS.mode, config_json=config_json)

            if "train" in FLAGS.mode:
                assert self.config.model_dir is not None
                if not tf.gfile.Exists(self.config.model_dir):
                    tf.gfile.MakeDirs(self.config.model_dir)

                if not tf.gfile.Exists(self.config.model_dir + "/train_config.json"):
                    with tf.gfile.GFile(self.config.model_dir + "/train_config.json", mode='w') as f:
                        json.dump(config_json, f)

        else:
            self.config = user_defined_config

        for key, val in self.config.__dict__.items():
            setattr(self, key, val)

        num_train_examples = 0
        num_predict_examples = 0
        if "train" in self.config.mode:
            if "odps://" in self.config.train_input_fp:
                reader = tf.python_io.TableReader(self.config.train_input_fp,
                                                  selected_cols="",
                                                  excluded_cols="",
                                                  slice_id=0,
                                                  slice_count=1,
                                                  num_threads=0,
                                                  capacity=0)
                num_train_examples = reader.get_row_count()
            elif ".tfrecord" in self.config.train_input_fp:
                for record in tf.python_io.tf_record_iterator(self.config.train_input_fp):
                    num_train_examples += 1

            elif ".list_tfrecord" in self.config.train_input_fp:
                with tf.gfile.Open(self.config.train_input_fp, 'r') as f:
                    for i, line in enumerate(f):
                        if i == 0 and line.strip().isdigit():
                            num_train_examples = int(line.strip())
                            tf.logging.info("Reading {} training examples from list_tfrecord".format(str(num_train_examples)))
                            break
                        if i%10 ==0:
                            tf.logging.info("Reading {} files".format(i))
                        fp = line.strip()
                        for record in tf.python_io.tf_record_iterator(fp):
                            num_train_examples += 1
            elif ".list_csv" in self.config.train_input_fp:
                with tf.gfile.Open(self.config.train_input_fp, 'r') as f:
                    for i, line in enumerate(f):
                        if i == 0 and line.strip().isdigit():
                            num_train_examples = int(line.strip())
                            tf.logging.info("Reading {} training examples from list_csv".format(str(num_train_examples)))
                            break
                        if i%10 ==0:
                            tf.logging.info("Reading {} files".format(i))
                        fp = line.strip()
                        with tf.gfile.Open(fp, 'r') as f:
                            for record in f:
                                num_train_examples += 1

            else:
                with tf.gfile.Open(self.config.train_input_fp, 'r') as f:
                    for record in f:
                        num_train_examples += 1

            assert num_train_examples > 0
            tf.logging.info("total number of training examples {}".format(num_train_examples))
        elif "predict" in self.config.mode:
            if "odps" in self.config.predict_input_fp:
                reader = tf.python_io.TableReader(self.config.predict_input_fp,
                                                  selected_cols="",
                                                  excluded_cols="",
                                                  slice_id=0,
                                                  slice_count=1,
                                                  num_threads=0,
                                                  capacity=0)
                num_predict_examples = reader.get_row_count()

            elif ".tfrecord" in self.config.predict_input_fp:
                for record in tf.python_io.tf_record_iterator(self.config.predict_input_fp):
                    num_predict_examples += 1

            elif ".list_csv" in self.config.predict_input_fp:
                with tf.gfile.Open(self.config.predict_input_fp, 'r') as f:
                    for i, line in enumerate(f):
                        if i == 0 and line.strip().isdigit():
                            num_predict_examples = int(line.strip())
                            tf.logging.info("Use preset num training examples")
                            break
                        if i%10 ==0:
                            tf.logging.info("Reading {} files".format(i))
                        fp = line.strip()
                        with tf.gfile.Open(fp, 'r') as f:
                            for record in f:
                                num_predict_examples += 1

            else:
                with tf.gfile.Open(self.config.predict_input_fp, 'r') as f:
                    for record in f:
                        num_predict_examples += 1

            assert num_predict_examples > 0
            tf.logging.info("total number of predicting examples {}".format(num_predict_examples))

        super(base_model, self).__init__(num_train_examples=num_train_examples)

    def get_export_features(self):
        export_features = {}

        for feat in self.config.input_tensors_schema.split(","):
            feat_name = feat.split(":")[0]
            feat_type = feat.split(":")[1]
            seq_len = int(feat.split(":")[2])
            feat = {}
            feat['name'] = feat_name
            feat['type'] = feat_type
            if feat_type == "int":
                dtype = tf.int32
            elif feat_type == "float":
                dtype = tf.float32
            if seq_len == 1:
                ph = tf.placeholder(dtype=dtype, shape=[None], name=feat_name)
            else:
                ph = tf.placeholder(dtype=dtype, shape=[None, None], name=feat_name)

            export_features[feat_name] = ph

        receiver_tensors = {}
        feat_names = []
        for feat in self.config.receiver_tensors_schema.split(","):
            feat_names.append(feat.split(":")[0])
        for feat_name in feat_names:
            receiver_tensors[feat_name] = export_features[feat_name]
        return export_features, receiver_tensors

    def build_logits(self, features, mode):
        """ Given features, this method take care of building graph for train/eval/predict

        Args:

            features : either raw text features or numerical features such as input_ids, input_mask ...
            mode : tf.estimator.ModeKeys.TRAIN | tf.estimator.ModeKeys.EVAL | tf.estimator.ModeKeys.PREDICT

        Returns:

            logits, labels

        Examples::

            def build_logits(self, features, mode=None):
                preprocessor = preprocessors.get_preprocessor(self.pretrain_model_name_or_path)
                model = model_zoo.get_pretrained_model(self.pretrain_model_name_or_path)

                dense = layers.Dense(self.num_labels,
                                     kernel_initializer=layers.get_initializer(0.02),
                                     name='dense')

                input_ids, input_mask, segment_ids, label_ids = preprocessor(features)
                outputs = model([input_ids, input_mask, segment_ids], mode=mode)
                pooled_output = outputs[1]

                logits = dense(pooled_output)
                return logits, label_ids

        """
        raise NotImplementedError("must be implemented in descendants")

    def build_loss(self, logits, labels):
        """Build loss function

        Args:

            logits : logits returned from build_logits
            labels : labels returned from build_logits

        Returns:

            loss

        Examples::

            def build_loss(self, logits, labels):
                return softmax_cross_entropy(labels, depth=self.config.num_labels, logits=logits)

        """

        raise NotImplementedError("must be implemented in descendants")

    def build_eval_metrics(self, logits, labels):
        """Build evaluation metrics

        Args:

            logits : logits returned from build_logits
            labels : labels returned from build_logits

        Returns:

            metric_dict

        Examples::

            def build_eval_metrics(self, logits, labels):
                predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
                info_dict = {
                    "predictions": predictions,
                    "labels": labels,
                }
                evaluator = PyEvaluator()
                labels = [i for i in range(self.num_labels)]
                metric_dict = evaluator.get_metric_ops(info_dict, labels)
                ret_metrics = evaluator.evaluate(labels)
                tf.summary.scalar("eval accuracy", ret_metrics['py_accuracy'])
                tf.summary.scalar("eval F1 micro score", ret_metrics['py_micro_f1'])
                tf.summary.scalar("eval F1 macro score", ret_metrics['py_macro_f1'])
                return metric_dict

        """

        raise NotImplementedError("must be implemented in descendants")

    def build_predictions(self, logits):
        """Build predictions

        Args:

            logits : logits returned from build_logits


        Returns:

            predictions

        Examples::

            def build_predictions(self, output):
                logits, _ = output
                predictions = dict()
                predictions["predictions"] = tf.argmax(logits, axis=-1, output_type=tf.int32)
                return predictions

        """
        raise NotImplementedError("must be implemented in descendants")
