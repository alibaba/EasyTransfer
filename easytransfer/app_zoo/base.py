# coding=utf-8
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

import os
import tensorflow as tf
from easytransfer.engines.model import base_model
from easytransfer.model_zoo.modeling_utils import init_from_checkpoint_without_training_ops
from easytransfer.app_zoo.app_utils import get_reader_fn, get_writer_fn, copy_file_to_new_path, log_duration_time
from easytransfer.app_zoo.predictors import run_app_predictor


class ApplicationModel(base_model):
    def __init__(self, **kwargs):
        """ Basic application class """
        super(ApplicationModel, self).__init__(**kwargs)
        self.on_the_fly = False
        self.pre_build_vocab = False

    @staticmethod
    def get_input_tensor_schema():
        raise NotImplementedError

    @staticmethod
    def get_received_tensor_schema():
        return NotImplementedError

    @log_duration_time
    def run(self):
        if self.pre_build_vocab:
            self._build_vocab()
        getattr(self, self.config.mode.replace("_on_the_fly", ""))()

    def train(self):
        train_reader = get_reader_fn(self.config.train_input_fp)(
            input_glob=self.config.train_input_fp,
            is_training=True,
            input_schema=self.config.input_schema,
            batch_size=self.config.train_batch_size,
            distribution_strategy=self.config.distribution_strategy)
        self.run_train(reader=train_reader)

    def train_and_evaluate(self):
        shuffle_buffer_size = self.config.shuffle_buffer_size if \
            hasattr(self.config, "shuffle_buffer_size") else None
        train_reader = get_reader_fn(self.config.train_input_fp)(input_glob=self.config.train_input_fp,
                                                                 is_training=True,
                                                                 input_schema=self.config.input_schema,
                                                                 batch_size=self.config.train_batch_size,
                                                                 distribution_strategy=self.config.distribution_strategy,
                                                                 shuffle_buffer_size=shuffle_buffer_size)

        eval_reader = get_reader_fn(self.config.eval_input_fp)(input_glob=self.config.eval_input_fp,
                                                               is_training=False,
                                                               input_schema=self.config.input_schema,
                                                               batch_size=self.config.eval_batch_size)

        if hasattr(self.config, "export_best_checkpoint") and self.config.export_best_checkpoint:
            tf.logging.info("First train, then search for best checkpoint...")
            self.run_train(reader=train_reader)

            ckpts = set()
            with tf.gfile.GFile(os.path.join(self.config.model_dir, "checkpoint"), mode='r') as reader:
                for line in reader:
                    line = line.strip()
                    line = line.replace("oss://", "")
                    ckpts.add(
                        int(line.split(":")[1].strip().replace("\"", "").split("/")[-1].replace("model.ckpt-", "")))

            best_score = float('-inf')
            best_ckpt = None
            best_eval_results = None
            best_metric_name = self.config.export_best_checkpoint_metric
            eval_results_list = list()
            for ckpt in sorted(ckpts):
                checkpoint_path = os.path.join(self.config.model_dir, "model.ckpt-" + str(ckpt))
                eval_results = self.run_evaluate(reader=eval_reader, checkpoint_path=checkpoint_path)
                eval_results.pop('global_step')
                eval_results.pop('loss')
                score = eval_results[best_metric_name]
                _score = -1 * score if self.config.export_best_checkpoint_metric == "mse" else score
                if _score > best_score:
                    best_ckpt = ckpt
                    best_score = _score
                    best_eval_results = eval_results
                tf.logging.info("Ckpt {} 's {}: {:.4f}; Best ckpt {} 's {}: {:.4f}".format(
                    ckpt, best_metric_name, score, best_ckpt, best_metric_name, best_score))
                eval_results_list.append((ckpt, eval_results))
            for ckpt, eval_results in eval_results_list:
                tf.logging.info("Checkpoint-%d: " % ckpt)
                for metric_name, score in eval_results.items():
                    tf.logging.info("\t{}: {:.4f}".format(metric_name, score))
            tf.logging.info("Best checkpoint: {}".format(best_ckpt))
            for metric_name, score in best_eval_results.items():
                tf.logging.info("\t{}: {:.4f}".format(metric_name, score))

            # Export best checkpoints to saved_model
            checkpoint_path = os.path.join(self.config.model_dir, "model.ckpt-" + str(best_ckpt))

        else:
            self.run_train_and_evaluate(train_reader=train_reader, eval_reader=eval_reader)

            # Export the last checkpoints to saved_model
            checkpoint_path = tf.train.latest_checkpoint(self.config.model_dir, latest_filename=None)

        try:
            tf.logging.info("Export checkpoint {}".format(checkpoint_path))
            self.config.export_dir_base = self.config.model_dir
            self.config.checkpoint_path = checkpoint_path
            self.config.input_tensors_schema = self.get_input_tensor_schema()
            self.config.input_schema = self.config.input_tensors_schema
            self.config.receiver_tensors_schema = self.get_received_tensor_schema()
            self.config.mode = "export"
            tf.reset_default_graph()
            self.estimator = tf.estimator.Estimator(
                model_fn=self._build_model_fn(),
                config=self._get_run_predict_config())
            self.export()
        except Exception as e:
            tf.logging.info(str(e))

    def evaluate(self):
        eval_reader = get_reader_fn(self.config.eval_input_fp)(input_glob=self.config.eval_input_fp,
                                                               input_schema=self.config.input_schema,
                                                               is_training=False,
                                                               batch_size=self.config.eval_batch_size)

        self.run_evaluate(reader=eval_reader, checkpoint_path=self.config.eval_ckpt_path)

    def predict(self):
        if ".ckpt" in self.config.predict_checkpoint_path:
            predict_reader = get_reader_fn(self.config.predict_input_fp)(input_glob=self.config.predict_input_fp,
                                                                         batch_size=self.config.predict_batch_size,
                                                                         is_training=False,
                                                                         input_schema=self.config.input_schema)

            predict_writer = get_writer_fn(self.config.predict_output_fp)(output_glob=self.config.predict_output_fp,
                                                                          output_schema=self.config.output_schema,
                                                                          slice_id=0,
                                                                          input_queue=None)

            self.run_predict(predict_reader,
                             predict_writer,
                             checkpoint_path=self.config.predict_checkpoint_path)
        else:
            self.config.mode = "predict_on_the_fly"
            self.mode = "predict_on_the_fly"
            run_app_predictor(self.config)

    def export(self):
        export_dir = self.export_model()
        if not isinstance(export_dir, str):
            export_dir = export_dir.decode("utf-8")
        export_dir_base = self.config.export_dir_base
        tf.gfile.Rename(os.path.join(export_dir, "saved_model.pb"),
                        os.path.join(export_dir_base, "saved_model.pb"), overwrite=True)
        if not tf.gfile.Exists(os.path.join(export_dir_base, "variables")):
            tf.gfile.MkDir(os.path.join(export_dir_base, "variables"))
        for fname in tf.gfile.ListDirectory(os.path.join(export_dir, "variables")):
            src_path = os.path.join(export_dir, "variables", fname)
            tgt_path = os.path.join(export_dir_base, "variables", fname)
            tf.gfile.Rename(src_path, tgt_path, overwrite=True)
        checkpoint_dir = os.path.dirname(self.config.checkpoint_path)
        copied_file_candidates = ["vocab.txt", "config.json", "train_vocab.txt", "train_config.json",
                                  "30k-clean.model", "30k-clean.vocab"]
        for fname in copied_file_candidates:
            copy_file_to_new_path(checkpoint_dir, export_dir_base, fname)
        if tf.gfile.Exists(export_dir):
            tf.gfile.DeleteRecursively(export_dir)
        tmp_dir = os.path.join(export_dir_base, "temp-" + export_dir.split("/")[-1])
        if tf.gfile.Exists(tmp_dir):
            tf.gfile.DeleteRecursively(tmp_dir)

    def check_and_init_from_checkpoint(self, mode):
        if hasattr(self.config, "init_checkpoint_path") and self.config.init_checkpoint_path \
                and mode == tf.estimator.ModeKeys.TRAIN:
            init_from_checkpoint_without_training_ops(self.config.init_checkpoint_path)

    def _build_vocab(self):
        if hasattr(self.config, "vocab_path") and tf.gfile.Exists(self.config.vocab_path):
            return
        else:
            import os
            from easytransfer.preprocessors.deeptext_preprocessor import DeepTextVocab

            vocab = DeepTextVocab()
            reader = get_reader_fn(self.config.train_input_fp)(input_glob=self.config.train_input_fp,
                                                               is_training=False,
                                                               input_schema=self.config.input_schema,
                                                               batch_size=self.config.train_batch_size,
                                                               distribution_strategy=self.config.distribution_strategy)
            for batch_idx, outputs in enumerate(self.estimator.predict(input_fn=reader.get_input_fn(),
                                                                       yield_single_examples=False,
                                                                       checkpoint_path=None)):
                if self.config.first_sequence in outputs:
                    for line in outputs[self.config.first_sequence]:
                        vocab.add_line(line)
                if self.config.second_sequence in outputs:
                    for line in outputs[self.config.second_sequence]:
                        vocab.add_line(line)
            vocab.filter_vocab_to_fix_length(self.config.max_vocab_size)
            self.config.vocab_path = os.path.join(self.config.model_dir, "train_vocab.txt")
            vocab.export_to_file(self.config.vocab_path)