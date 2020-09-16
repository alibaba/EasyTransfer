# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
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
from .utils import get_shape_list, get_shape_list_imagebert

def get_attn_mask_xlnet(inputs):
    input_mask = inputs
    batch_size = tf.shape(input_mask)[0]
    target_len = tf.shape(input_mask)[1]

    input_mask_trans = tf.transpose(input_mask)

    data_mask = input_mask_trans[None]
    mems_mask = tf.zeros([tf.shape(data_mask)[0], 0, batch_size], dtype=tf.float32)
    data_mask = tf.concat([mems_mask, data_mask], 1)
    attn_mask = data_mask[:, :, :, None]
    attn_mask = tf.cast(attn_mask > 0, dtype=tf.float32)
    non_tgt_mask = -tf.eye(target_len, dtype=tf.float32)
    non_tgt_mask = tf.concat([tf.zeros([target_len, 0], dtype=tf.float32), non_tgt_mask], axis=-1)
    attn_mask = tf.cast((attn_mask + non_tgt_mask[:, :, None, None]) > 0, dtype=tf.float32)
    return attn_mask


def get_attn_mask_bert(from_tensor, to_mask):
    """Create 3D attention mask from a 2D tensor mask.
    Args:
      from_tensor: 2D or 3D Tensor of shape [batch_size, from_seq_length, ...].
      to_mask: int32 Tensor of shape [batch_size, to_seq_length].
    Returns:
      float Tensor of shape [batch_size, from_seq_length, to_seq_length].
    """
    from_shape = get_shape_list(from_tensor)
    batch_size = from_shape[0]
    from_seq_length = from_shape[1]

    to_shape = get_shape_list(to_mask)
    to_seq_length = to_shape[1]

    to_mask = tf.cast(
        tf.reshape(to_mask, [batch_size, 1, to_seq_length]), tf.float32)

    # We don't assume that `from_tensor` is a mask (although it could be). We
    # don't actually care if we attend *from* padding tokens (only *to* padding)
    # tokens so we create a tensor of all ones.
    #
    # `broadcast_ones` = [batch_size, from_seq_length, 1]
    broadcast_ones = tf.ones(
        shape=[batch_size, from_seq_length, 1], dtype=tf.float32)

    # Here we broadcast along two dimensions to create the mask.
    mask = broadcast_ones * to_mask

    return mask


def get_attn_mask_imagebert(from_text_ids,
                            to_text_mask, from_image_feature, to_image_mask):
    from_text_shape = get_shape_list_imagebert(from_text_ids, expected_rank=[2, 3])
    batch_size = from_text_shape[0]
    from_text_seq_length = from_text_shape[1]
    # print("Create FB mask - from_text_shape: ", from_text_shape)

    from_image_shape = get_shape_list_imagebert(from_image_feature, expected_rank=[2, 3])
    from_image_seq_length = from_image_shape[1]
    # print("Create FB mask - from_image_shape: ", from_image_shape)

    to_text_shape = get_shape_list_imagebert(to_text_mask, expected_rank=2)
    to_text_seq_length = to_text_shape[1]
    # print("Create FB mask - to_text_shape: ", to_text_shape)

    to_image_shape = get_shape_list_imagebert(to_image_mask, expected_rank=2)
    to_image_seq_length = to_image_shape[1]
    # print("Create FB mask - to_image_shape: ", to_image_shape)


    to_image_mask = tf.cast(to_image_mask, tf.int32)
    to_text_mask = tf.cast(to_text_mask, tf.int32)
    to_mask = tf.concat([to_text_mask, to_image_mask], axis=1)
    to_seq_length = to_text_seq_length + to_image_seq_length
    from_seq_length = from_text_seq_length + from_image_seq_length
    to_mask = tf.cast(
        tf.reshape(to_mask, [batch_size, 1, to_seq_length]), tf.float32)
    # print("Create FB mask - to_mask_shape: ", to_mask.shape)

    broadcast_ones = tf.ones(
        shape=[batch_size, from_seq_length, 1], dtype=tf.float32)

    mask = broadcast_ones * to_mask
    # print("Create FB mask - mask_shape: ", mask.shape)
    return mask

def get_attn_mask_videobert(from_text_ids,
                            to_text_mask, from_image_feature, to_image_mask):
    from_text_shape = get_shape_list_imagebert(from_text_ids, expected_rank=[2, 3])
    batch_size = from_text_shape[0]
    from_text_seq_length = from_text_shape[1]
    # print("Create FB mask - from_text_shape: ", from_text_shape)

    from_image_shape = get_shape_list_imagebert(from_image_feature, expected_rank=[2, 3])
    from_image_seq_length = from_image_shape[1]
    # print("Create FB mask - from_image_shape: ", from_image_shape)

    to_text_shape = get_shape_list_imagebert(to_text_mask, expected_rank=2)
    to_text_seq_length = to_text_shape[1]
    # print("Create FB mask - to_text_shape: ", to_text_shape)

    to_image_shape = get_shape_list_imagebert(to_image_mask, expected_rank=2)
    to_image_seq_length = to_image_shape[1]
    # print("Create FB mask - to_image_shape: ", to_image_shape)

    to_image_mask = tf.cast(to_image_mask, tf.int32)
    to_text_mask = tf.cast(to_text_mask, tf.int32)
    to_mask = tf.concat([to_text_mask, to_image_mask], axis=1)
    to_seq_length = to_text_seq_length + to_image_seq_length
    from_seq_length = from_text_seq_length + from_image_seq_length
    to_mask = tf.cast(
        tf.reshape(to_mask, [batch_size, 1, to_seq_length]), tf.float32)
    # print("Create FB mask - to_mask_shape: ", to_mask.shape)

    broadcast_ones = tf.ones(
        shape=[batch_size, from_seq_length, 1], dtype=tf.float32)

    mask = broadcast_ones * to_mask
    # print("Create FB mask - mask_shape: ", mask.shape)
    return mask


def create_look_ahead_mask(from_tensor):
    from_shape = get_shape_list(from_tensor, expected_rank=[2, 3])
    batch_size = from_shape[0]
    from_seq_length = from_shape[1]
    mask = tf.linalg.band_part(tf.ones((from_seq_length, from_seq_length)), -1, 0)

    mask = tf.cast(
        tf.reshape(mask, [1, from_seq_length, from_seq_length]), tf.float32)

    broadcast_ones = tf.ones(
        shape=[batch_size, from_seq_length, 1], dtype=tf.float32)
    mask = broadcast_ones * mask
    return mask


def create_padding_mask(from_tensor):
    from_shape = get_shape_list(from_tensor, expected_rank=[2, 3])
    batch_size = from_shape[0]
    from_seq_length = from_shape[1]
    mask = 1 - tf.cast(tf.math.equal(from_tensor, 0), tf.float32)
    mask = tf.cast(
        tf.reshape(mask, [batch_size, 1, from_seq_length]), tf.float32)
    broadcast_ones = tf.ones(
        shape=[batch_size, from_seq_length, 1], dtype=tf.float32)
    mask = broadcast_ones * mask
    return mask
