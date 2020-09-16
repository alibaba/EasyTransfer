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

IMAGEBERT_PRETRAINED_MODEL_ARCHIVE_MAP = {
    'icbu-imagebert-small-en': "imagebert/icbu-imagebert-small-en/model.ckpt",
    'pai-imagebert-base-zh': "imagebert/pai-imagebert-base-zh/model.ckpt",
    'pai-imagebert-base-en': "imagebert/pai-imagebert-base-en/model.ckpt"
}

IMAGEBERT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    'icbu-imagebert-small-en': "imagebert/icbu-imagebert-small-en/config.json",
    'pai-imagebert-base-zh': "imagebert/pai-imagebert-base-zh/config.json",
    'pai-imagebert-base-en': "imagebert/pai-imagebert-base-en/config.json"
}


class ImageBertConfig(PretrainedConfig):
    """Configuration for `ImageBert`.

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
      patch_feature_size: patch feature size
      max_patch_position_embeddings: max_patch_position_embeddings

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
                 patch_type_vocab_size=2,
                 patch_feature_size=2048,
                 max_patch_position_embeddings=64,
                 **kwargs):
        super(ImageBertConfig, self).__init__(**kwargs)
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
        self.patch_type_vocab_size = patch_type_vocab_size
        self.patch_feature_size = patch_feature_size
        self.max_patch_position_embeddings = max_patch_position_embeddings


class ImageEmbeddings(layers.Layer):
    """Construct the embeddings from word, position and token_type embeddings.
    """

    def __init__(self, config, **kwargs):
        super(ImageEmbeddings, self).__init__(**kwargs)

        self.patch_feature_size = config.patch_feature_size
        self.hidden_size = config.hidden_size
        self.initializer_range = config.initializer_range
        self.patch_type_vocab_size = config.patch_type_vocab_size
        self.max_patch_position_embeddings = config.max_patch_position_embeddings

        self.LayerNorm = layers.LayerNormalization
        self.dropout_input = layers.Dropout(config.hidden_dropout_prob)
        self.dropout_output = layers.Dropout(config.hidden_dropout_prob)
        self.initializer = layers.get_initializer(self.initializer_range)

    def build(self, input_shape):
        """Build shared word embedding layer """
        self.image_projector = self.add_weight(
            "image_projector",
            dtype=tf.float32,
            shape=[self.patch_feature_size, self.hidden_size],
            initializer=self.initializer,
        )

        self.patch_position_embeddings = self.add_weight(
            "patch_position_embeddings",
            dtype=tf.float32,
            shape=[self.max_patch_position_embeddings, self.hidden_size],
            initializer=self.initializer,
        )

        self.patch_type_embeddings = self.add_weight(
            "patch_type_embeddings",
            dtype=tf.float32,
            shape=[self.patch_type_vocab_size, self.hidden_size],
            initializer=self.initializer,
        )
        super(ImageEmbeddings, self).build(input_shape)

    def call(self, inputs, training=False):
        input_image_feature, patch_type_ids = inputs

        input_image_feature = self.dropout_input(input_image_feature, training=training)
        patch_embeddings = tf.einsum("abc,cd->abd",
                                     input_image_feature, self.image_projector)

        input_shape = layers.get_shape_list(patch_embeddings)
        batch_size = input_shape[0]
        seq_length = input_shape[1]
        width = input_shape[2]

        # This vocab will be small so we always do one-hot here, since it is always
        # faster for a small vocabulary.
        flat_token_type_ids = tf.reshape(patch_type_ids, [-1])
        one_hot_ids = tf.one_hot(flat_token_type_ids, depth=self.patch_type_vocab_size)
        type_embeddings = tf.matmul(one_hot_ids, self.patch_type_embeddings)
        type_embeddings = tf.reshape(type_embeddings,
                                    [batch_size, seq_length, width])

        position_embeddings = tf.gather(self.patch_position_embeddings, tf.range(0, seq_length))
        position_embeddings = tf.expand_dims(position_embeddings, 0)

        embeddings = patch_embeddings + type_embeddings + position_embeddings
        embeddings = self.LayerNorm(embeddings, name="ImageEmbLayerNorm")
        embeddings = self.dropout_output(embeddings, training=training)
        return embeddings


# MPM: Masked Patch Modeling
class ImageBertMPMHead(layers.Layer):
    def __init__(self, config, **kwargs):
        super(ImageBertMPMHead, self).__init__(**kwargs)
        self.hidden_size = config.hidden_size
        self.patch_feature_size = config.patch_feature_size
        self.initializer_range = config.initializer_range

    def build(self, input_shape):
        self.output_weights = self.add_weight(shape=[self.hidden_size, self.patch_feature_size],
                                              initializer=layers.get_initializer(self.initializer_range),
                                              trainable=True, name="output_weights")

        super(ImageBertMPMHead, self).build(input_shape)

    def call(self, image_seq_output, masked_patch_positions):
        pred_patch_features = layers.gather_indexes(image_seq_output, masked_patch_positions)
        logits = tf.matmul(pred_patch_features, self.output_weights)
        return logits


class ImageBertBackbone(layers.Layer):

    def __init__(self, config, **kwargs):
        super(ImageBertBackbone, self).__init__(**kwargs)
        self.num_hidden_layers = config.num_hidden_layers
        self.embeddings = layers.BertEmbeddings(config, name="embeddings")
        self.image_embeddings = ImageEmbeddings(config, name="image_embeddings")
        self.encoder = layers.Encoder(config, name="encoder")
        self.pooler = layers.Dense(
            units=config.hidden_size,
            activation='tanh',
            kernel_initializer=layers.get_initializer(config.initializer_range),
            name="pooler/dense")

    def call(self, inputs, training=False):
        input_ids, input_mask, segment_ids, masked_image_feature, image_mask = inputs

        token_embedding_output = self.embeddings([input_ids, segment_ids], training=training)

        from_shape = layers.get_shape_list(masked_image_feature)
        batch_size = from_shape[0]
        image_seq_length = from_shape[1]
        input_patch_type_ids = tf.ones(shape=[batch_size, image_seq_length], dtype=tf.int32)

        image_embedding_output = self.image_embeddings([masked_image_feature,
                                                        input_patch_type_ids], training=training)

        embedding_output = tf.concat([token_embedding_output, image_embedding_output],
                                     axis=1)

        attention_mask = layers.get_attn_mask_imagebert(input_ids, input_mask,
                                                        masked_image_feature, image_mask)

        encoder_outputs = self.encoder([embedding_output, attention_mask], training=training)
        pooled_output = self.pooler(encoder_outputs[0][-1][:, 0])
        outputs = (encoder_outputs[0][-1], pooled_output)
        return outputs


class ImageBertPreTrainedModel(PreTrainedModel):
    config_class = ImageBertConfig
    pretrained_model_archive_map = IMAGEBERT_PRETRAINED_MODEL_ARCHIVE_MAP
    pretrained_config_archive_map = IMAGEBERT_PRETRAINED_CONFIG_ARCHIVE_MAP

    def __init__(self, config, **kwargs):
        super(ImageBertPreTrainedModel, self).__init__(config, **kwargs)

        self.bert = ImageBertBackbone(config, name="bert")
        self.mlm = layers.MLMHead(config, self.bert.embeddings, name="cls/predictions")
        self.nsp = layers.NSPHead(config, name="cls/seq_relationship")
        self.mpm = ImageBertMPMHead(config, name="cls/img_predictions")

    def mask_patch_features(self, patch_features, masked_patch_positions):
        onehot_image_mask = tf.reduce_sum(tf.one_hot(masked_patch_positions, self.config.max_patch_position_embeddings, dtype=tf.float32),
                                          axis=1)
        reverse_onehot_image_mask = 1 - (onehot_image_mask[:, :, tf.newaxis])
        masked_patch = tf.multiply(patch_features, reverse_onehot_image_mask)
        return masked_patch

    def call(self, input_ids,
             input_mask=None,
             segment_ids=None,
             masked_lm_positions=None,
             image_feature=None,
             image_mask=None,
             masked_patch_positions=None,
             **kwargs):

        """
        Examples::

            model = model_zoo.get_pretrained_model('icbu-imagebert-small-en')

            mlm_logits, nsp_logits, mpm_logits, target_raw_patch_features = \
                model(input_ids,
                      input_mask=input_mask,
                      segment_ids=token_type_ids,
                      image_feature=image_feature,
                      image_mask=image_mask,
                      masked_lm_positions=lm_positions,
                      masked_patch_positions=masked_patch_positions,
                      output_features=False,
                      mode=mode)

        """

        training = kwargs['mode'] == tf.estimator.ModeKeys.TRAIN

        input_shape = layers.get_shape_list(input_ids)
        batch_size = input_shape[0]
        seq_length = input_shape[1]

        if input_mask is None:
            input_mask = tf.ones(shape=[batch_size, seq_length], dtype=tf.int32)

        if segment_ids is None:
            segment_ids = tf.zeros(shape=[batch_size, seq_length], dtype=tf.int32)

        if image_mask is None:
            image_mask = tf.ones(shape=[batch_size, self.config.max_patch_position_embeddings], dtype=tf.int32)

        if masked_lm_positions is None:
            masked_lm_positions = tf.ones(shape=[batch_size, seq_length], dtype=tf.int64)

        if masked_patch_positions is None:
            masked_patch_positions = tf.ones(shape=[batch_size, self.config.masked_image_token_num], dtype=tf.int64)

        if image_feature is None:
            image_feature = tf.constant([[1.0] * kwargs.get("image_feature_size", 131072)], dtype=tf.float32)

        image_feature = tf.reshape(image_feature, [-1, self.config.max_patch_position_embeddings, self.config.patch_feature_size])
        if kwargs['mode'] == tf.estimator.ModeKeys.PREDICT:
            masked_image_feature = image_feature
        else:
            masked_image_feature = self.mask_patch_features(image_feature, masked_patch_positions)

        inputs = [input_ids, input_mask, segment_ids, masked_image_feature, image_mask]

        if kwargs.get("output_features", True) == True:
            outputs = self.bert(inputs, training=training)
            sequence_output = outputs[0]
            pooled_output = outputs[1]
            return sequence_output, pooled_output
        else:
            outputs = self.bert(inputs, training=training)
            sequence_output = outputs[0]
            text_sequence_output = sequence_output[:, :seq_length, :]
            image_sequence_output = sequence_output[:, seq_length:, :]
            pooled_output = outputs[1]
            mlm_logits = self.mlm(text_sequence_output, masked_lm_positions)
            nsp_logits = self.nsp(pooled_output)
            mpm_logits = self.mpm(image_sequence_output, masked_patch_positions)

            target_raw_patch_features = layers.gather_indexes(image_feature, masked_patch_positions)

            return mlm_logits, nsp_logits, mpm_logits, target_raw_patch_features, pooled_output

