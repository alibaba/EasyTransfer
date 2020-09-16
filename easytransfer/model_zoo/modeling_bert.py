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
from .modeling_utils import PretrainedConfig, PreTrainedModel

BERT_PRETRAINED_MODEL_ARCHIVE_MAP = {
    'google-bert-tiny-en': "bert/google-bert-tiny-en/model.ckpt",
    'google-bert-small-en': "bert/google-bert-small-en/model.ckpt",
    'google-bert-base-zh': "bert/google-bert-base-zh/model.ckpt",
    'google-bert-base-en': "bert/google-bert-base-en/model.ckpt",
    'google-bert-large-en': "bert/google-bert-large-en/model.ckpt",
    'pai-bert-tiny-zh-L2-H768-A12': "bert/pai-bert-tiny-zh-L2-H768-A12/model.ckpt",
    'pai-bert-tiny-zh': "bert/pai-bert-tiny-zh/model.ckpt",
    'pai-bert-small-zh': "bert/pai-bert-small-zh/model.ckpt",
    'pai-bert-base-zh': "bert/pai-bert-base-zh/model.ckpt",
    'pai-bert-large-zh': "bert/pai-bert-large-zh/model.ckpt",
}

BERT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    'google-bert-tiny-en': "bert/google-bert-tiny-en/config.json",
    'google-bert-small-en': "bert/google-bert-small-en/config.json",
    'google-bert-base-zh': "bert/google-bert-base-zh/config.json",
    'google-bert-base-en': "bert/google-bert-base-en/config.json",
    'google-bert-large-en': "bert/google-bert-large-en/config.json",
    'pai-bert-tiny-zh-L2-H768-A12': "bert/pai-bert-tiny-zh-L2-H768-A12/config.json",
    'pai-bert-tiny-zh': "bert/pai-bert-tiny-zh/config.json",
    'pai-bert-small-zh': "bert/pai-bert-small-zh/config.json",
    'pai-bert-base-zh': "bert/pai-bert-base-zh/config.json",
    'pai-bert-large-zh': "bert/pai-bert-large-zh/config.json",

}


class BertConfig(PretrainedConfig):
    """Configuration for `Bert`.

    Args:

      vocab_size: Vocabulary size of `inputs_ids` in `BertModel`.
      hidden_size: Size of the encoder layers and the pooler layer.
      num_hidden_layers: Number of hidden layers in the Transformer encoder.
      num_attention_heads: Number of attention heads for each attention layer in
        the Transformer encoder.
      intermediate_size: The size of the "intermediate" (i.e., feed-forward)
        layer in the Transformer encoder.
      hidden_dropout_prob: The dropout probability for all fully connected
        layers in the embeddings, encoder, and pooler.
      attention_probs_dropout_prob: The dropout ratio for the attention
        probabilities.
      max_position_embeddings: The maximum sequence length that this model might
        ever be used with. Typically set this to something large just in case
        (e.g., 512 or 1024 or 2048).
      type_vocab_size: The vocabulary size of the `token_type_ids` passed into
        `BertModel`.
      initializer_range: The stdev of the truncated_normal_initializer for
        initializing all weight matrices.

    """

    def __init__(self,
                 vocab_size,
                 hidden_size,
                 intermediate_size,
                 num_hidden_layers,
                 num_attention_heads,
                 max_position_embeddings,
                 type_vocab_size,
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 initializer_range=0.02,
                 **kwargs):
        super(BertConfig, self).__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range


class BertBackbone(layers.Layer):
    def __init__(self, config, **kwargs):

        self.embeddings = layers.BertEmbeddings(config, name="embeddings")
        if not kwargs.pop('enable_whale', False):
            self.encoder = layers.Encoder(config, name="encoder")
        else:
            self.encoder = layers.Encoder_whale(config, name="encoder")

        self.pooler = layers.Dense(
            units=config.hidden_size,
            activation='tanh',
            kernel_initializer=layers.get_initializer(config.initializer_range),
            name="pooler/dense")

        super(BertBackbone, self).__init__(config, **kwargs)

    def call(self, inputs,
             input_mask=None,
             segment_ids=None,
             training=False):

        if isinstance(inputs, (tuple, list)):
            input_ids = inputs[0]
            input_mask = inputs[1] if len(inputs) > 1 else input_mask
            segment_ids = inputs[2] if len(inputs) > 2 else segment_ids
        else:
            input_ids = inputs

        input_shape = layers.get_shape_list(input_ids)
        batch_size = input_shape[0]
        seq_length = input_shape[1]

        if input_mask is None:
            input_mask = tf.ones(shape=[batch_size, seq_length], dtype=tf.int32)

        if segment_ids is None:
            segment_ids = tf.zeros(shape=[batch_size, seq_length], dtype=tf.int32)

        embedding_output = self.embeddings([input_ids, segment_ids], training=training)
        attention_mask = layers.get_attn_mask_bert(input_ids, input_mask)
        encoder_outputs = self.encoder([embedding_output, attention_mask], training=training)
        pooled_output = self.pooler(encoder_outputs[0][-1][:, 0])
        outputs = (encoder_outputs[0][-1], pooled_output)
        return outputs


class BertPreTrainedModel(PreTrainedModel):
    config_class = BertConfig
    pretrained_model_archive_map = BERT_PRETRAINED_MODEL_ARCHIVE_MAP
    pretrained_config_archive_map = BERT_PRETRAINED_CONFIG_ARCHIVE_MAP

    def __init__(self, config, **kwargs):
        super(BertPreTrainedModel, self).__init__(config, **kwargs)
        self.bert = BertBackbone(config, name="bert", enable_whale=kwargs.get("enable_whale", False))
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


            google-bert-tiny-zh

            google-bert-tiny-en

            google-bert-small-zh

            google-bert-small-en

            google-bert-base-zh

            google-bert-base-en

            google-bert-large-zh

            google-bert-large-en

            pai-bert-tiny-zh

            pai-bert-tiny-en

            pai-bert-small-zh

            pai-bert-small-en

            pai-bert-base-zh

            pai-bert-base-en

            pai-bert-large-zh

            pai-bert-large-en

            model = model_zoo.get_pretrained_model('google-bert-base-zh')
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
