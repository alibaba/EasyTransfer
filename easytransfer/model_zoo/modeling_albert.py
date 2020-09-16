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

ALBERT_PRETRAINED_MODEL_ARCHIVE_MAP = {
    'google-albert-base-zh': "albert/google-albert-base-zh/model.ckpt",
    'google-albert-base-en': "albert/google-albert-base-en/model.ckpt",
    'google-albert-large-zh': "albert/google-albert-large-zh/model.ckpt",
    'google-albert-large-en': "albert/google-albert-large-en/model.ckpt",
    'google-albert-xlarge-zh': "albert/google-albert-xlarge-zh/model.ckpt",
    'google-albert-xlarge-en': "albert/google-albert-xlarge-en/model.ckpt",
    'google-albert-xxlarge-zh': "albert/google-albert-xxlarge-zh/model.ckpt",
    'google-albert-xxlarge-en': "albert/google-albert-xxlarge-en/model.ckpt",
    'pai-albert-base-zh': "albert/pai-albert-base-zh/model.ckpt",
    'pai-albert-base-en': "albert/pai-albert-base-en/model.ckpt",
    'pai-albert-large-zh': "albertpai-albert-large-zh/model.ckpt",
    'pai-albert-large-en': "albert/pai-albert-large-en/model.ckpt",
    'pai-albert-xlarge-zh': "albert/pai-albert-xlarge-zh/model.ckpt",
    'pai-albert-xlarge-en': "albert/pai-albert-xlarge-en/model.ckpt",
    'pai-albert-xxlarge-zh': "albert/pai-albert-xxlarge-zh/model.ckpt",
    'pai-albert-xxlarge-en': "albert/pai-albert-xxlarge-en/model.ckpt",
}

ALBERT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    'google-albert-base-zh': "albert/google-albert-base-zh/config.json",
    'google-albert-base-en': "albert/google-albert-base-en/config.json",
    'google-albert-large-zh': "albert/google-albert-large-zh/config.json",
    'google-albert-large-en': "albert/google-albert-large-en/config.json",
    'google-albert-xlarge-zh': "albert/google-albert-xlarge-zh/config.json",
    'google-albert-xlarge-en': "albert/google-albert-xlarge-en/config.json",
    'google-albert-xxlarge-zh': "albert/google-albert-xxlarge-zh/config.json",
    'google-albert-xxlarge-en': "albert/google-albert-xxlarge-en/config.json",
    'pai-albert-base-zh': "albert/pai-albert-base-zh/config.json",
    'pai-albert-base-en': "albert/pai-albert-base-en/config.json",
    'pai-albert-large-zh': "albertpai-albert-large-zh/config.json",
    'pai-albert-large-en': "albert/pai-albert-large-en/config.json",
    'pai-albert-xlarge-zh': "albert/pai-albert-xlarge-zh/config.json",
    'pai-albert-xlarge-en': "albert/pai-albert-xlarge-en/config.json",
    'pai-albert-xxlarge-zh': "albert/pai-albert-xxlarge-zh/config.json",
    'pai-albert-xxlarge-en': "albert/pai-albert-xxlarge-en/config.json",
}

#In this albert V2 version, we apply 'no dropout'
# hidden_dropout_prob, and attention_probs_dropout_prob are set to 0
class AlbertConfig(PretrainedConfig):
    """Configuration for `Albert`.

    Args:

      vocab_size: Vocabulary size of `inputs_ids` in `AlbertModel`.
      embedding_size: size of voc embeddings.
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
        `AlbertModel`.
      initializer_range: The stdev of the truncated_normal_initializer for
        initializing all weight matrices.

    """
    def __init__(self,
                 vocab_size,
                 embedding_size,
                 hidden_size,
                 intermediate_size,
                 num_hidden_layers,
                 num_attention_heads,
                 max_position_embeddings,
                 type_vocab_size,
                 hidden_dropout_prob=0,
                 attention_probs_dropout_prob=0,
                 initializer_range=0.02,
                 **kwargs):

        super(AlbertConfig, self).__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.hidden_dropout_prob=hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range=initializer_range


class AlbertSelfOutput(layers.Layer):
    def __init__(self, config, **kwargs):
        super(AlbertSelfOutput, self).__init__(**kwargs)
        self.dense = layers.Dense(
            config.hidden_size,
            kernel_initializer=layers.get_initializer(config.initializer_range),
            name="dense"
        )
        self.dropout = layers.Dropout(config.hidden_dropout_prob)

    def call(self, inputs, training=False):
        hidden_states, input_tensor = inputs
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states, training=training)
        return hidden_states

class AlbertAttention(layers.Layer):
    def __init__(self, config, **kwargs):
        super(AlbertAttention, self).__init__(**kwargs)
        self.self_attention = layers.SelfAttention(config, name="self")
        self.dense_output = AlbertSelfOutput(config, name="output")

    def call(self, inputs, training=False):
        input_tensor, attention_mask = inputs
        self_outputs = self.self_attention(input_tensor, attention_mask, training=training)
        attention_output = self.dense_output([self_outputs, input_tensor], training=training)
        return attention_output

class AlbertOutput(layers.Layer):
    def __init__(self, config, **kwargs):
        super(AlbertOutput, self).__init__(**kwargs)
        self.dense = layers.Dense(
            config.hidden_size, kernel_initializer=layers.get_initializer(config.initializer_range), name="dense"
        )
    def call(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        return hidden_states

class AlbertIntermediate(layers.Layer):
    def __init__(self, config, **kwargs):
        super(AlbertIntermediate, self).__init__(**kwargs)
        self.dense = layers.Dense(
            config.intermediate_size,
            activation=layers.gelu_new,
            kernel_initializer=layers.get_initializer(config.initializer_range),
            name="dense")

        self.dense_output = AlbertOutput(config, name="output")
        self.dropout = layers.Dropout(config.hidden_dropout_prob)

    def call(self, hidden_states, training):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dense_output(hidden_states)
        hidden_states = self.dropout(hidden_states, training=training)
        return hidden_states

class AlbertFFN(layers.Layer):
    def __init__(self, config, **kwargs):
        super(AlbertFFN, self).__init__(**kwargs)
        self.intermediate = AlbertIntermediate(config, name="intermediate")

    def call(self, attention_output, training=False):
        ffn_output = self.intermediate(attention_output, training=training)
        return ffn_output

class AlbertAttentionFFN(layers.Layer):
    def __init__(self, config, **kwargs):
        super(AlbertAttentionFFN, self).__init__(**kwargs)
        self.attention = AlbertAttention(config, name="attention_1")
        self.ffn = AlbertFFN(config, name="ffn_1")
        self.LayerNorm = layers.LayerNormalization
        self.LayerNorm_1 = layers.LayerNormalization

    def call(self, inputs, training=False):
        hidden_states, attention_mask = inputs
        attention_output = self.attention([hidden_states, attention_mask], training=training)
        attention_output = self.LayerNorm(attention_output + hidden_states)
        ffn_output  = self.ffn(attention_output, training=training)
        ffn_output = self.LayerNorm_1(ffn_output + attention_output)
        return ffn_output, attention_output

class AlbertEncoder(layers.Layer):
    def __init__(self, config, **kwargs):
        super(AlbertEncoder, self).__init__(**kwargs)

        self.num_hidden_layers = config.num_hidden_layers
        self.inner_group = AlbertAttentionFFN(config, name="group_0/inner_group_0")

    def call(self, inputs, training=False):
        hidden_states, attention_mask = inputs

        all_hidden_states = ()
        all_att_outputs = ()
        for i in range(self.num_hidden_layers):
            layer_output, att_output = self.inner_group([hidden_states, attention_mask], training=training)
            hidden_states = layer_output
            all_hidden_states = all_hidden_states + (hidden_states,)
            all_att_outputs = all_att_outputs + (att_output, )

        final_outputs = []
        for hidden_states in all_hidden_states:
            final_outputs.append(hidden_states)

        return final_outputs, all_att_outputs

class AlbertBackbone(layers.Layer):
    def __init__(self, config, **kwargs):
        super(AlbertBackbone, self).__init__(**kwargs)

        self.embeddings = layers.AlbertEmbeddings(config, name="embeddings")
        self.embedding_hidden_mapping_in = layers.Dense(
            config.hidden_size,
            kernel_initializer=layers.get_initializer(config.initializer_range),
            name="encoder/embedding_hidden_mapping_in",
        )
        self.encoder = AlbertEncoder(config, name="encoder/transformer")
        self.pooler = layers.Dense(
            units=config.hidden_size,
            activation='tanh',
            kernel_initializer=layers.get_initializer(config.initializer_range),
            name="pooler/dense")

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
        embedding_output = self.embedding_hidden_mapping_in(embedding_output)
        attention_mask = layers.get_attn_mask_bert(input_ids, input_mask)
        encoder_outputs = self.encoder([embedding_output, attention_mask], training=training)
        pooled_output = self.pooler(encoder_outputs[0][-1][:, 0])
        outputs = (encoder_outputs[0][-1], pooled_output)
        return outputs

class AlbertPreTrainedModel(PreTrainedModel):
    config_class = AlbertConfig
    pretrained_model_archive_map = ALBERT_PRETRAINED_MODEL_ARCHIVE_MAP
    pretrained_config_archive_map = ALBERT_PRETRAINED_CONFIG_ARCHIVE_MAP

    def __init__(self, config, **kwargs):
        super(AlbertPreTrainedModel, self).__init__(config, **kwargs)

        self.bert = AlbertBackbone(config, name="bert")
        self.mlm = layers.AlbertMLMHead(config, self.bert.embeddings, name="cls/predictions")
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


            google-albert-base-zh

            google-albert-base-en

            google-albert-large-zh

            google-albert-large-en

            google-albert-xlarge-zh

            google-albert-xlarge-en

            google-albert-xxlarge-zh

            google-albert-xxlarge-en

            pai-albert-base-zh

            pai-albert-base-en

            pai-albert-large-zh

            pai-albert-large-en

            pai-albert-xlarge-zh

            pai-albert-xlarge-en

            pai-albert-xxlarge-zh

            pai-albert-xxlarge-en

            model = model_zoo.get_pretrained_model('google-albert-base-zh')
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





