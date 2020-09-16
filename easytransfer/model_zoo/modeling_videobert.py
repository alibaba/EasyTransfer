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
VIDEOBERT_PRETRAINED_MODEL_ARCHIVE_MAP = {
    'pai-videobert-base-zh': "videobert/pai-videobert-base-zh/model.ckpt"
}

VIDEOBERT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    'pai-videobert-base-zh': "videobert/pai-videobert-base-zh/config.json"
}


class VideoBertConfig(PretrainedConfig):
    """Configuration for `VideoBert`.

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
      clip_feature_size: clip feature size
      max_clip_position_embeddings: max_clip_position_embeddings

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
                 clip_size=10,
                 clip_feature_size=1536,
                 max_clip_position_embeddings=10,
                 **kwargs):
        super(VideoBertConfig, self).__init__(**kwargs)
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
        self.clip_size = clip_size
        self.clip_feature_size = clip_feature_size
        self.max_clip_position_embeddings = max_clip_position_embeddings


class VideoBertEmbeddings(layers.Layer):
    """Construct the embeddings from word, position and token_type embeddings.
    """

    def __init__(self, config, **kwargs):
        super(VideoBertEmbeddings, self).__init__(**kwargs)

        self.clip_feature_size = config.clip_feature_size
        self.clip_size = config.clip_size
        self.hidden_size = config.hidden_size
        self.initializer_range = config.initializer_range
        self.max_clip_position_embeddings = config.max_clip_position_embeddings

        self.LayerNorm = layers.LayerNormalization
        self.dropout = layers.Dropout(config.hidden_dropout_prob)
        self.initializer = layers.get_initializer(self.initializer_range)

    def build(self, input_shape):
        """Build shared word embedding layer """
        self.video_embeddings = self.add_weight(
            "video_embeddings",
            dtype=tf.float32,
            shape=[self.clip_feature_size, self.hidden_size],
            initializer=self.initializer,
        )

        self.clip_position_embeddings = self.add_weight(
            "clip_position_embeddings",
            dtype=tf.float32,
            shape=[self.max_clip_position_embeddings, self.hidden_size],
            initializer=self.initializer,
        )

        self.clip_type_embeddings = self.add_weight(
            "clip_type_embeddings",
            dtype=tf.float32,
            shape=[self.clip_size, self.hidden_size],
            initializer=self.initializer,
        )
        super(VideoBertEmbeddings, self).build(input_shape)

    def call(self, inputs, training=False):
        input_video_feature, video_token_type_ids = inputs
        clip_embeddings = tf.einsum("abc,cd->abd",
                                     input_video_feature, self.video_embeddings)

        input_shape = layers.get_shape_list(clip_embeddings)
        batch_size = input_shape[0]
        seq_length = input_shape[1]
        width = input_shape[2]

        # This vocab will be small so we always do one-hot here, since it is always
        # faster for a small vocabulary.
        flat_token_type_ids = tf.reshape(video_token_type_ids, [-1])
        one_hot_ids = tf.one_hot(flat_token_type_ids, depth=self.clip_size)
        clip_type_embeddings = tf.matmul(one_hot_ids, self.clip_type_embeddings)
        clip_type_embeddings = tf.reshape(clip_type_embeddings,
                                           [batch_size, seq_length, width])
        clip_embeddings += clip_type_embeddings

        position_embeddings = tf.gather(self.clip_position_embeddings, tf.range(0, seq_length))
        position_embeddings = tf.expand_dims(position_embeddings, 0)

        clip_embeddings += position_embeddings

        output = self.LayerNorm(clip_embeddings, name="LayerNorm")
        output = self.dropout(output, training=training)
        return output


# MVC: Mask Video Clip
class VideoBertMVCHead(layers.Layer):
    def __init__(self, config, **kwargs):
        super(VideoBertMVCHead, self).__init__(**kwargs)
        self.hidden_size = config.hidden_size
        self.clip_feature_size = config.clip_feature_size
        self.initializer_range = config.initializer_range

    def build(self, input_shape):
        self.output_weights = self.add_weight(shape=[self.hidden_size, self.clip_feature_size],
                                              initializer=layers.get_initializer(self.initializer_range),
                                              trainable=True, name="output_weights")

        super(VideoBertMVCHead, self).build(input_shape)

    def call(self, sequence_output, masked_clip_positions, text_seq_length):
        video_seq_output = sequence_output[:, text_seq_length:, :]
        pred_clip_features = layers.gather_indexes(video_seq_output, masked_clip_positions)
        logits = tf.matmul(pred_clip_features, self.output_weights)
        return logits


class VideoBertBackbone(layers.Layer):

    def __init__(self, config, **kwargs):
        super(VideoBertBackbone, self).__init__(**kwargs)
        self.num_hidden_layers = config.num_hidden_layers
        self.embeddings = layers.BertEmbeddings(config, name="embeddings")
        self.video_embeddings = VideoBertEmbeddings(config, name="video_embeddings")
        self.encoder = layers.Encoder(config, name="encoder")
        self.pooler = layers.Dense(
            units=config.hidden_size,
            activation='tanh',
            kernel_initializer=layers.get_initializer(config.initializer_range),
            name="pooler/dense")

    def call(self, inputs, training=False):
        input_ids, input_mask, segment_ids, masked_video_feature, video_mask = inputs

        token_embedding_output = self.embeddings([input_ids, segment_ids], training=training)

        from_shape = layers.get_shape_list(masked_video_feature)
        batch_size = from_shape[0]
        video_clip_length = from_shape[1]
        input_video_type_ids = tf.ones(shape=[batch_size, video_clip_length], dtype=tf.int32)
        video_embedding_output = self.video_embeddings([masked_video_feature,
                                                        input_video_type_ids], training=training)
        embedding_output = tf.concat([token_embedding_output, video_embedding_output],
                                     axis=1)

        attention_mask = layers.get_attn_mask_videobert(input_ids, input_mask,
                                                        masked_video_feature, video_mask)

        encoder_outputs = self.encoder([embedding_output, attention_mask], training=training)
        pooled_output = self.pooler(encoder_outputs[0][-1][:, 0])
        outputs = (encoder_outputs[0][-1], pooled_output)
        return outputs


class VideoBertPreTrainedModel(PreTrainedModel):
    config_class = VideoBertConfig
    pretrained_model_archive_map = VIDEOBERT_PRETRAINED_MODEL_ARCHIVE_MAP
    pretrained_config_archive_map = VIDEOBERT_PRETRAINED_CONFIG_ARCHIVE_MAP

    def __init__(self, config, **kwargs):
        super(VideoBertPreTrainedModel, self).__init__(config, **kwargs)

        self.bert = VideoBertBackbone(config, name="bert")
        self.mlm = layers.MLMHead(config, self.bert.embeddings, name="cls/predictions")
        self.nsp = layers.NSPHead(config, name="cls/seq_relationship")
        self.mvc = VideoBertMVCHead(config, name="cls/video_predictions")

    def mask_clip_features(self, patch_features, masked_patch_ids):
        onehot_image_mask = tf.reduce_sum(tf.one_hot(masked_patch_ids, 10, dtype=tf.float32),
                                          axis=1)
        reverse_onehot_image_mask = 1 - (onehot_image_mask[:, :, tf.newaxis])
        masked_patch = tf.multiply(patch_features, reverse_onehot_image_mask)
        return masked_patch

    def call(self, input_ids,
             input_mask=None,
             segment_ids=None,
             masked_lm_positions=None,
             video_feature=None,
             video_mask=None,
             masked_clip_positions=None,
             **kwargs):

        training = kwargs['mode'] == tf.estimator.ModeKeys.TRAIN

        input_shape = layers.get_shape_list(input_ids)
        batch_size = input_shape[0]
        seq_length = input_shape[1]

        if input_mask is None:
            input_mask = tf.ones(shape=[batch_size, seq_length], dtype=tf.int32)

        if segment_ids is None:
            segment_ids = tf.zeros(shape=[batch_size, seq_length], dtype=tf.int32)

        if video_mask is None:
            #video_mask = tf.constant([[0] * 10, [0] * 10, [0] * 10], dtype=tf.int32)
            video_mask = tf.ones(shape=[batch_size, 10], dtype=tf.int32)

        if masked_lm_positions is None:
            masked_lm_positions = tf.ones(shape=[batch_size, seq_length], dtype=tf.int64)

        if masked_clip_positions is None:
            masked_clip_positions = tf.ones(shape=[batch_size, 4], dtype=tf.int64)

        if video_feature is None:
            video_feature = tf.constant([[1.0] * 15360, [1.0] * 15360, [1.0] * 15360], dtype=tf.float32)

        video_feature = tf.reshape(video_feature, [-1, 10, 1536])
        masked_video_feature = self.mask_clip_features(video_feature, masked_clip_positions)
        inputs = [input_ids, input_mask, segment_ids, masked_video_feature, video_mask]

        if kwargs.get("output_features", True) == True:
            outputs = self.bert(inputs, training=training)
            sequence_output = outputs[0]
            pooled_output = outputs[1]
            return sequence_output, pooled_output
        else:
            outputs = self.bert(inputs, training=training)
            sequence_output = outputs[0]
            pooled_output = outputs[1]
            mlm_logits = self.mlm(sequence_output, masked_lm_positions)
            nsp_logits = self.nsp(pooled_output)
            mvc_logits = self.mvc(sequence_output, masked_clip_positions, seq_length)

            target_raw_clip_features = layers.gather_indexes(video_feature, masked_clip_positions)

            return mlm_logits, nsp_logits, mvc_logits, target_raw_clip_features, pooled_output
