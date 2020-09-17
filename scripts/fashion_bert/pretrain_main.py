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

import tensorflow as tf
import numpy as np
import time
import os
from easytransfer import base_model, FLAGS
from easytransfer import model_zoo
from easytransfer import preprocessors
from easytransfer.datasets import BundleCSVReader
from easytransfer.losses import masked_language_model_loss, next_sentence_prediction_loss, image_reconstruction_kld_loss
from easytransfer.evaluators import masked_language_model_eval_metrics, next_sentence_prediction_eval_metrics
from fashionbert_utils import prediction_analysis, PretrainConfig, append_to_file, delete_exists_file

_app_flags = tf.app.flags
_app_flags.DEFINE_string("type", default=None, help='')
_app_flags.DEFINE_integer("input_sequence_length", default=None, help='')
_app_flags.DEFINE_integer("vocab_size", default=30522, help='')
_app_flags.DEFINE_integer("image_feature_size", default=None, help='')
_APP_FLAGS = _app_flags.FLAGS


class ImageBertPretrain(base_model):

    def __init__(self, **kwargs):
        super(ImageBertPretrain, self).__init__(**kwargs)
        self.user_defined_config = kwargs["user_defined_config"]

    def build_logits(self, features, mode=None):

        preprocessor = preprocessors.get_preprocessor(self.pretrain_model_name_or_path,
                                                           app_model_name="pretrain_language_model",
                                                           feature_type="pretrain_multimodel",
                                                           user_defined_config=self.user_defined_config)

        self.model = model_zoo.get_pretrained_model(self.pretrain_model_name_or_path,
                                                    input_sequence_length=_APP_FLAGS.input_sequence_length)

        if mode == tf.estimator.ModeKeys.PREDICT:
            image_feature, image_mask, input_ids, input_mask, segment_ids,\
            nx_sent_labels, prod_desc, text_prod_id, image_prod_id, prod_img_id = preprocessor(features)
            # TODO: DONOT Need these features in predict. BUT to compatible the data format
            masked_patch_positions = tf.constant(np.random.randint(0, self.config.predict_batch_size, (self.model.config.masked_image_token_num,)))
            masked_lm_positions = tf.constant(np.random.randint(0, self.config.predict_batch_size, (self.model.config.masked_text_token_num,)))   
            masked_lm_ids = tf.constant(np.random.randint(0, self.config.predict_batch_size, (self.model.config.masked_text_token_num, 1,)))
            masked_lm_weights = tf.ones(self.config.predict_batch_size, self.model.config.masked_text_token_num)
        else:
            image_feature, image_mask, masked_patch_positions, input_ids, input_mask, segment_ids,\
            masked_lm_positions, masked_lm_ids, masked_lm_weights, nx_sent_labels = preprocessor(features)

        mlm_logits, nsp_logits, mpm_logits, target_raw_patch_features, pooled_output = \
            self.model(input_ids,
                  input_mask=input_mask,
                  segment_ids=segment_ids,
                  masked_lm_positions=masked_lm_positions,
                  image_feature=image_feature,
                  image_mask=image_mask,
                  masked_patch_positions=masked_patch_positions,
                  output_features=False,
                  mode=mode,
                  image_feature_size=_APP_FLAGS.image_feature_size)

        logits = (mlm_logits, nsp_logits, mpm_logits)
        labels = (masked_lm_ids, masked_lm_weights, nx_sent_labels, target_raw_patch_features)

        return logits, labels

    def build_loss(self, logits, labels):
        mlm_logits, nsp_logits, mpm_logits = logits
        masked_lm_ids, masked_lm_weights, nx_sent_labels, target_raw_patch_features = labels

        masked_lm_loss = masked_language_model_loss(mlm_logits, masked_lm_ids, masked_lm_weights,
                                                    _APP_FLAGS.vocab_size)
        next_sentence_loss = next_sentence_prediction_loss(nsp_logits, nx_sent_labels)

        image_loss = image_reconstruction_kld_loss(mpm_logits, target_raw_patch_features,
                                      self.model.config.masked_image_token_num,
                                                   self.model.config.patch_feature_size)

        G = tf.reshape(tf.stack([masked_lm_loss, next_sentence_loss, image_loss]), shape=[3])
        w0 = 1.0
        w1 = 1.0
        w2 = 1.0
        isAdaptive = True
        if isAdaptive:
            nG = tf.math.square(tf.nn.softmax(G))
            alpha = 1.0
            K = 3.0
            denominator = (alpha * K - nG[0]) * (alpha * K - nG[1]) + \
                          (alpha * K - nG[1]) * (alpha * K - nG[2]) + \
                          (alpha * K - nG[2]) * (alpha * K - nG[0])
            w0 = (alpha * K - nG[1]) * (alpha * K - nG[2]) / denominator
            w1 = (alpha * K - nG[2]) * (alpha * K - nG[0]) / denominator
            w2 = (alpha * K - nG[0]) * (alpha * K - nG[1]) / denominator

        adaptive_loss = w0 * masked_lm_loss + w1 * next_sentence_loss + w2 * image_loss
        return adaptive_loss

    def build_eval_metrics(self, logits, labels):
        mlm_logits, nsp_logits, _ = logits
        masked_lm_ids, masked_lm_weights, next_sentence_labels, _ = labels

        mlm_metrics = masked_language_model_eval_metrics(mlm_logits, masked_lm_ids, masked_lm_weights,
                                                         self.model.config.vocab_size)
        nsp_metrics = next_sentence_prediction_eval_metrics(nsp_logits, next_sentence_labels)
        return mlm_metrics.update(nsp_metrics)

    def build_predictions(self, output):
        logits, _ = output
        mlm_logits, nsp_logits, mpm_logits = logits
        return {"nsp_logits": nsp_logits}

def main():

    config = PretrainConfig()
    app = ImageBertPretrain(user_defined_config=config)

    if FLAGS.mode == "train_and_evaluate":
        train_reader = BundleCSVReader(input_glob=app.train_input_fp,
                                       is_training=True,
                                       shuffle_buffer_size=4096,
                                       input_schema=app.input_schema,
                                       batch_size=app.train_batch_size,
                                       worker_hosts=app.config.worker_hosts,
                                       task_index=app.config.task_index
                                       )
        eval_reader = BundleCSVReader(input_glob=app.eval_input_fp,
                                      input_schema=app.input_schema,
                                      is_training=False,
                                      batch_size=app.eval_batch_size,
                                      worker_hosts=app.config.worker_hosts,
                                      task_index=app.config.task_index)

        app.run_train_and_evaluate(train_reader=train_reader, eval_reader=eval_reader)

    elif FLAGS.mode == "predict":

        predict_reader = BundleCSVReader(input_glob=app.predict_input_fp,
                                      input_schema=app.input_schema,
                                      batch_size=app.predict_batch_size,
                                      worker_hosts=app.config.worker_hosts,
                                      task_index=app.config.task_index)


        localtime = time.strftime("%Y%m%d-%H%M-", time.localtime())

        if _APP_FLAGS.type == "img2txt":
           print("************ predict task img2txt ********")
           result_filename = "eval_img2txt_results.txt"
           analysis_type = "img2txt"
        else:
           print("************ predict task txt2img ********")
           result_filename = "eval_txt2img_results.txt"
           analysis_type = "txt2img"

        if not tf.gfile.Exists(_APP_FLAGS.output_dir):
            tf.gfile.MkDir(_APP_FLAGS.output_dir)

        result_fp_path = os.path.join(_APP_FLAGS.output_dir, str(localtime) +  result_filename)
        print("result_fp_path: ", result_fp_path)
        delete_exists_file(result_fp_path)
        for result in app.run_predict(reader=predict_reader,
                                      checkpoint_path=app.config.predict_checkpoint_path,
                                      yield_single_examples=False):
            nsp_logits = result["nsp_logits"]
            labels = result["nx_sent_labels"]
            text_prod_id = result["text_prod_id"]
            image_prod_id = result["image_prod_id"]
            prod_img_id = result["prod_img_id"]

            batch_pred_result = str(text_prod_id.tolist()) + "\001" \
                                + str(image_prod_id.tolist()) + "\001" \
                                + str(prod_img_id.tolist()) + "\001" \
                                + str(labels.tolist()) + "\001" \
                                + str(np.reshape(nsp_logits, [-1]).tolist()) + "\n"

            append_to_file(result_fp_path, batch_pred_result)
        prediction_analysis(result_fp_path, type=analysis_type)

if __name__ == "__main__":
    main()


