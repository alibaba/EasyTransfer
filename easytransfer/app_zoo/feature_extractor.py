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
from easytransfer import preprocessors, model_zoo
from easytransfer.app_zoo.base import ApplicationModel


class BertFeatureExtractor(ApplicationModel):
    """ Bert Feature Extraction Model (Only for predicting)"""
    def __init__(self, **kwargs):
        super(BertFeatureExtractor, self).__init__(**kwargs)
        self.finetune_model_name = self.config.finetune_model_name if \
            hasattr(self.config, "finetune_model_name") else None

    def build_logits(self, features, mode):
        """ Building BERT feature extraction graph

        Args:
            features (`OrderedDict`): A dict mapping raw input to tensors
            mode (`bool`): tell the model whether it is under training
        Returns:
            pooled_output (`Tensor`): The output after pooling. Shape of [None, 768]
            all_hidden_outputs (`Tensor`): The last hidden outputs of all sequence.
             Shape of [None, seq_len, hidden_size]
        """
        bert_preprocessor = preprocessors.get_preprocessor(self.config.pretrain_model_name_or_path,
                                                           user_defined_config=self.config)

        input_ids, input_mask, segment_ids = bert_preprocessor(features)[:3]


        if self.finetune_model_name == "text_match_bert_two_tower":
            with tf.variable_scope('text_match_bert_two_tower', reuse=tf.AUTO_REUSE):
                bert_backbone = model_zoo.get_pretrained_model(self.config.pretrain_model_name_or_path)
                sequence_output, pooled_output = bert_backbone(
                    [input_ids, input_mask, segment_ids], output_features=True, mode=mode)

                if hasattr(self.config, "projection_dim") and self.config.projection_dim != -1:
                    first_token_output_a = sequence_output[:, 0, :]

                    pooled_output = tf.layers.dense(inputs=first_token_output_a, units=self.config.projection_dim,
                                                    activation=None, name='output_dense_layer')
        else:
            bert_backbone = model_zoo.get_pretrained_model(self.config.pretrain_model_name_or_path)
            sequence_output, pooled_output = bert_backbone(
                [input_ids, input_mask, segment_ids], output_features=True, mode=mode)
        return sequence_output, pooled_output

    def build_predictions(self, predict_output):
        """ Building BERT feature extraction prediction dict.

        Args:
            predict_output (`tuple`): (sequence_output, pooled_output)
        Returns:
            ret_dict (`dict`): A dict with (`pool_output`, `first_token_output`,
            `all_hidden_outputs`)
        """
        all_hidden_outputs, pool_output = predict_output
        first_token_output = all_hidden_outputs[:, 0, :]
        ret_dict = {
            "pool_output": pool_output,
            "first_token_output": first_token_output,
            "all_hidden_outputs": all_hidden_outputs
        }
        return ret_dict
