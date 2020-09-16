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

from collections import Counter
import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score

from easytransfer.datasets import CSVReader
from easytransfer.model_zoo import AdaBERTStudent

from utils import get_assignment_map_from_checkpoint, load_npy, load_arch, SearchResultsSaver

flags = tf.app.flags
flags.DEFINE_string('open_ess', None, 'open_ess')
flags.DEFINE_string('distribution_strategy', "MirroredStrategy", "ds")
# Model
flags.DEFINE_integer(
    "embed_size", default=128, help="word/position embedding dimensions")
flags.DEFINE_integer(
    "num_token", default=30522, help="Number of distinct tokens")
flags.DEFINE_integer(
    "is_pair_task", default=0, help="single sentence or paired sentences.")
flags.DEFINE_integer(
    "num_classes", default=2, help="Number of categories to be discriminated")
flags.DEFINE_integer("seq_length", default=128, help="sequence length")
# Training
flags.DEFINE_integer(
    "temp_decay_steps",
    default=18000,
    help="Number of steps for annealing temperature.")
flags.DEFINE_float(
    "model_opt_lr",
    default=5e-4,
    help="learning rate for updating model parameters")
flags.DEFINE_float(
    "arch_opt_lr",
    default=1e-4,
    help="learning rate for updating arch parameters")
flags.DEFINE_float(
    "model_l2_reg",
    default=3e-4,
    help="coefficient for the l2regularization of model parameters")
flags.DEFINE_float(
    "arch_l2_reg",
    default=1e-3,
    help="coefficient for the l2regularization of arch parameters")
flags.DEFINE_float("loss_gamma", default=0.8, help="loss weight gamma")
flags.DEFINE_float("loss_beta", default=4.0, help="loss weight beta")
flags.DEFINE_string(
    "emb_pathes", default=None, help="given embeddings")
flags.DEFINE_string(
    "arch_path", default=None, help="given architectures")
flags.DEFINE_string(
    "model_dir",
    default="./model_dir",
    help="Directory for saving the finetuned model.")
flags.DEFINE_string(
    "searched_model", default=None, help="searched_model_ckpt")
flags.DEFINE_string("train_file", default="", help="train file.")
# mirror ds actual bs = num_core_per_host * train_batch_size
flags.DEFINE_integer(
    "num_core_per_host", default=1, help="the number of GPUs used.")
flags.DEFINE_integer(
    "train_batch_size", default=32, help="batch size for training")
flags.DEFINE_integer(
    "train_steps", default=20000, help="Number of training steps")
flags.DEFINE_integer(
    "save_steps", default=2000, help="If None, not to save any model.")
flags.DEFINE_integer(
    "max_save", default=1, help="Max number of checkpoints to save. ")
# these parameters are reserved for PAI
flags.DEFINE_boolean("is_training", default=True, help="training or not.")
flags.DEFINE_string("checkpointDir", default='', help="checkpoint Dir")
FLAGS = flags.FLAGS


def get_run_config():
    session_config = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=False,
        intra_op_parallelism_threads=64,
        inter_op_parallelism_threads=64,
        gpu_options=tf.GPUOptions(
            allow_growth=True,
            force_gpu_compatible=True,
            per_process_gpu_memory_fraction=1.0))
    #session_config.graph_options.optimizer_options.opt_level = -1
    from tensorflow.core.protobuf import rewriter_config_pb2
    session_config.graph_options.rewrite_options.constant_folding = (
        rewriter_config_pb2.RewriterConfig.OFF)

    if FLAGS.distribution_strategy == "ExascaleStrategy":
        tf.logging.info(
            "*****************Using ExascaleStrategy*********************")
        import pai
        worker_hosts = FLAGS.worker_hosts.split(',')
        if len(worker_hosts) > 1:
            pai.distribute.set_tf_config(FLAGS.job_name, FLAGS.task_index,
                                         worker_hosts)
        strategy = pai.distribute.ExascaleStrategy(
            optimize_clip_by_global_norm=True)
    elif FLAGS.distribution_strategy == "MirroredStrategy":
        tf.logging.info(
            "*****************Using MirroredStrategy*********************")
        from tensorflow.contrib.distribute.python import cross_tower_ops as cross_tower_ops_lib
        cross_tower_ops = cross_tower_ops_lib.AllReduceCrossTowerOps('nccl')
        strategy = tf.contrib.distribute.MirroredStrategy(
            num_gpus=FLAGS.num_core_per_host)
    elif FLAGS.distribution_strategy == "None":
        strategy = None
    else:
        raise ValueError(
            "Set correct distribution strategy, ExascaleStrategy | MirroredStrategy | None"
        )

    #model_dir set in tf.estimator.Estimator
    run_config = tf.estimator.RunConfig(
        model_dir=FLAGS.model_dir,
        session_config=session_config,
        keep_checkpoint_max=FLAGS.max_save,
        save_checkpoints_secs=None,
        save_checkpoints_steps=FLAGS.save_steps,
        #log_step_count_steps=50,
        train_distribute=strategy,
    )
    return run_config


def get_model_fn():
    def model_fn(features, labels, mode, params):
        inputs = []
        if FLAGS.is_training:
            for feature_name in ["ids", "mask", "seg_ids", "prob_logits", "labels"]:
                inputs.append(features[feature_name])
        else:
            for feature_name in ["ids", "mask", "seg_ids", "labels"]:
                inputs.append(features[feature_name])

        if FLAGS.emb_pathes and FLAGS.is_training:
            pathes = FLAGS.emb_pathes.split(',')
            pretrained_word_embeddings = load_npy(pathes[0])
            pretrained_pos_embeddings = load_npy(pathes[1])
        else:
            pretrained_word_embeddings, pretrained_pos_embeddings = None, None

        Kmax = 8
        given_arch = None
        if FLAGS.arch_path:
            Kmax, given_arch = load_arch(FLAGS.arch_path)

        model = AdaBERTStudent(
            inputs, (mode == tf.estimator.ModeKeys.TRAIN),
            vocab_size=FLAGS.num_token,
            is_pair_task=bool(FLAGS.is_pair_task),
            num_classes=FLAGS.num_classes,
            Kmax=Kmax,
            emb_size=FLAGS.embed_size,
            seq_len=FLAGS.seq_length,
            keep_prob=0.9 if mode == tf.estimator.ModeKeys.TRAIN else 1.0,
            temp_decay_steps=FLAGS.temp_decay_steps,
            model_opt_lr=FLAGS.model_opt_lr,
            arch_opt_lr=FLAGS.arch_opt_lr,
            model_l2_reg=FLAGS.model_l2_reg,
            arch_l2_reg=FLAGS.arch_l2_reg,
            loss_gamma=FLAGS.loss_gamma,
            loss_beta=FLAGS.loss_beta,
            pretrained_word_embeddings=pretrained_word_embeddings,
            pretrained_pos_embeddings=pretrained_pos_embeddings,
            given_arch=given_arch)

        if mode == tf.estimator.ModeKeys.TRAIN:
            logging_tensors = dict(
                [(var.name, var) for var in model.arch_params])
            logging_tensors['step'] = model.global_step
            logging_tensors['loss'] = model.loss
            logging_hook = tf.train.LoggingTensorHook(
                logging_tensors, every_n_iter=50)
            chief_only_hooks = [logging_hook]
            if given_arch is None:
                search_result_hook = SearchResultsSaver(
                    model.global_step, model.arch_params, model.ld_embs, FLAGS.model_dir, FLAGS.save_steps)
                chief_only_hooks.append(search_result_hook)
            # handle the save/restore related issues
            if FLAGS.searched_model:
                # has pretrained
                tvars = tf.trainable_variables()
                initialized_variable_names = {}
                init_checkpoint = os.path.join(FLAGS.searched_model)
                tf.logging.info("Init from %s" % init_checkpoint)
                (assignment_map, initialized_variable_names
                 ) = get_assignment_map_from_checkpoint(tvars, init_checkpoint, ["wemb", "pemb"])
                tf.train.init_from_checkpoint(init_checkpoint,
                                              assignment_map)
                tf.logging.info("**** Trainable Variables ****")
                for var in tvars:
                    init_string = ""
                    if var.name in initialized_variable_names:
                        init_string = ", *INIT_FROM_CKPT*"
                    tf.logging.info("  name = %s, shape = %s%s", var.name,
                                    var.shape, init_string)
            return tf.estimator.EstimatorSpec(
                mode=mode,
                loss=model.loss,
                train_op=model.update,
                training_chief_hooks=chief_only_hooks)
        elif mode == tf.estimator.ModeKeys.EVAL:
            # Define the metrics:
            metrics_dict = {
                'Acc': model.acc,
                #'AUC': tf.metrics.auc(train_labels, probabilities, num_thresholds=2000)
            }
            return tf.estimator.EstimatorSpec(
                mode, loss=model.loss, eval_metric_ops=metrics_dict)
        else:
            if FLAGS.is_training:
                predictions = dict()
                predictions["predicted"] = model.predictions
                predictions["labels"] = features["labels"]
            else:
                predictions = features.copy()
                predictions["logits"] = model.logits
                predictions["predicted"] = model.predictions
            return tf.estimator.EstimatorSpec(
                mode=mode, predictions=predictions)

    return model_fn


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)

    print("\nParameters:")
    for attr, _ in sorted(FLAGS.__flags.items()):
        print('  {}={}'.format(attr, FLAGS[attr].value))
    print("")

    feature_schema = "labels:int:1,ids:int:{},mask:int:{},seg_ids:int:{},prob_logits:float:26".format(
        FLAGS.seq_length, FLAGS.seq_length, FLAGS.seq_length)
    estimator = tf.estimator.Estimator(
        model_fn=get_model_fn(), config=get_run_config(), params=None)
    reader_fn = CSVReader
    reader = reader_fn(
        input_glob=FLAGS.train_file.split(',')[0],
        input_schema=feature_schema,
        is_training=FLAGS.is_training,
        batch_size=FLAGS.train_batch_size,
        num_parallel_batches=8,
        shuffle_buffer_size=1024,
        prefatch_buffer_size=1024)

    if FLAGS.is_training:
        valid_reader = reader_fn(
            input_glob=FLAGS.train_file.split(',')[1],
            input_schema=feature_schema,
            is_training=False,
            batch_size=FLAGS.train_batch_size,
            num_parallel_batches=8,
            shuffle_buffer_size=1024,
            prefatch_buffer_size=1024)

        train_spec = tf.estimator.TrainSpec(
            input_fn=reader.get_input_fn(), max_steps=FLAGS.train_steps)
        eval_spec = tf.estimator.EvalSpec(
            input_fn=valid_reader.get_input_fn(), steps=None, throttle_secs=0)
        tf.estimator.train_and_evaluate(
            estimator, train_spec=train_spec, eval_spec=eval_spec)
    else:
        # assume there is just one given output table
        # fout = tf.python_io.TableWriter(FLAGS.outputs)
        fout = open(FLAGS.outputs, "w")
        num_samples, num_correct = 0, 0
        all_predictions = list()
        all_labels = list()
        for batch_idx, result in enumerate(estimator.predict(
                input_fn=reader.get_input_fn(), yield_single_examples=False)):
            predicted = result["predicted"]
            labels = result["labels"].squeeze(1)
            all_predictions.extend(predicted)
            all_labels.extend(labels)
            num_samples += len(labels)
            num_correct += np.sum(
                np.asarray(labels == predicted, dtype=np.int32))
            to_be_wrote = list()
            for i in range(labels.shape[0]):
                row = list()
                row.append(
                    ','.join([str(v) for v in result["ids"][i]]))
                row.append(
                    ','.join([str(v) for v in result["mask"][i]]))
                row.append(
                    ','.join([str(v) for v in result["seg_ids"][i]]))
                row.append(labels[i])
                row.append(
                    ','.join([str(v) for v in result["logits"][i]]))
                row.append(int(result["predicted"][i]))
                to_be_wrote.append(tuple(row))
            # fout.write(to_be_wrote, (0, 1, 2, 3, 4, 5))
            for items in to_be_wrote:
                fout.write("\t".join([str(t) for t in items]) + "\n")
            if batch_idx % 50 == 0:
                print("======> predicted for {} instances".format(num_samples))

        print(Counter(all_predictions))
        print("Accuracy={} / {} = {}".format(num_correct, num_samples, float(num_correct)/float(num_samples)))
        print("f1={}".format(f1_score(all_labels, all_predictions)))

if __name__ == "__main__":
    tf.app.run()