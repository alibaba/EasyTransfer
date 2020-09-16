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
from easytransfer import layers
from .modeling_utils import PreTrainedModel
from .modeling_bert import BertConfig, BertBackbone

ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP = {
    'hit-roberta-base-zh': "roberta/hit-roberta-base-zh/model.ckpt",
    'hit-roberta-large-zh': "roberta/hit-roberta-large-zh/model.ckpt",
    'brightmart-roberta-small-zh':"roberta/brightmart-roberta-small-zh/model.ckpt",
    'brightmart-roberta-large-zh':"roberta/brightmart-roberta-large-zh/model.ckpt",
}

ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    'hit-roberta-base-zh': "roberta/hit-roberta-base-zh/config.json",
    'hit-roberta-large-zh': "roberta/hit-roberta-large-zh/config.json",
    'brightmart-roberta-small-zh':"roberta/brightmart-roberta-small-zh/config.json",
    'brightmart-roberta-large-zh':"roberta/brightmart-roberta-large-zh/config.json",
}

class RobertaPreTrainedModel(PreTrainedModel):
    config_class = BertConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    pretrained_config_archive_map = ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP

    def __init__(self, config, **kwargs):
        super(RobertaPreTrainedModel, self).__init__(config, **kwargs)

        self.bert = BertBackbone(config, name="bert")
        self.mlm = layers.MLMHead(config, self.bert.embeddings, name="cls/predictions")
        self.nsp = layers.NSPHead(config, name="cls/seq_relationship")

    def call(self, inputs,
             masked_lm_positions=None,
             **kwargs):

        """

        Args:

            inputs : [input_ids, input_mask, segment_ids]
            masked_lm_positions: masked_lm_positions

        Returns:

            sequence_output, pooled_output

        Examples::

            hit-roberta-base-zh

            hit-roberta-large-zh

            pai-roberta-base-zh

            pai-roberta-large-zh

            model = model_zoo.get_pretrained_model('hit-roberta-base-zh')
            outputs = model([input_ids, input_mask, segment_ids], mode=mode)

        """

        training = kwargs['mode'] == tf.estimator.ModeKeys.TRAIN

        if kwargs.get("output_features", True) == True:
            outputs = self.bert(inputs, training=training)
            sequence_output = outputs[0]
            pooled_output = outputs[1]
            return sequence_output, pooled_output
        else:
            outputs = self.bert(inputs, training=training)
            sequence_output = outputs[0]
            pooled_output = outputs[1]
            input_shape = layers.get_shape_list(sequence_output)
            batch_size = input_shape[0]
            seq_length = input_shape[1]
            if masked_lm_positions is None:
                masked_lm_positions = tf.ones(shape=[batch_size, seq_length], dtype=tf.int64)

            mlm_logits = self.mlm(sequence_output, masked_lm_positions)
            nsp_logits = self.nsp(pooled_output)

            return mlm_logits, nsp_logits, pooled_output





