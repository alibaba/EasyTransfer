#!/usr/bin/env python
# -*- encoding:utf-8 -*-
# Copyright (c) 2020 Alibaba PAI team.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import abc
import logging
import cv2
import tensorflow as tf
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import constants
import numpy as np
from PIL import Image
import six
import json
import math
from tensorflow.core.protobuf import meta_graph_pb2
import argparse


class PredictorInterface(six.with_metaclass(abc.ABCMeta)):
  version = 1

  def __init__(self, model_path, model_config=None):
    """
    init tensorflow session and load tf model

    Args:
      model_path: init model from this directory
      model_config: config string for model to init, in json format
    """
    pass

  @abc.abstractmethod
  def predict(self, input_data, batch_size):
    """
    using session run predict a number of samples using batch_size

    Args:
      input_data:  a list of numpy array, each array is a sample to be predicted
      batch_size: batch_size passed by the caller, you can also ignore this param and 
        use a fixed number if you do not want to adjust batch_size in runtime
    Return:
      result: a list of dict, each dict is the prediction result of one sample
        eg, {"output1": value1, "output2": value2}, the value type can be 
        python int str float, and numpy array
    """
    pass

  def get_output_type(self):
    """
    in this function user should return a type dict, which indicates
    which type of data should the output of predictor be converted to
    * type json, data will be serialized to json str

    * type image, data will be converted to encode image binary and write to oss file,
      whose name is output_dir/${key}/${input_filename}_${idx}.jpg, where input_filename
      is extracted from url, key corresponds to the key in the dict of output_type,
      if the type of data indexed by key is a list, idx is the index of element in list, otherwhile ${idx} will be empty

    * type video, data will be converted to encode video binary and write to oss file,

    eg:  return  {
      'image': 'image',
      'feature': 'json'
    }
    indicating that the image data in the output dict will be save to image
    file and feature in output dict will be converted to json

    """
    return {}


class PredictorImpl(object):
  def __init__(self, model_path, profiling_file=None, decode=True):
    """ Impl class for predictor

    Args:
      model_path:  saved_model directory or frozenpb file path
      profiling_file:  profiling result file, default None.
        if not None, predict function will use Timeline to profiling
        prediction time, and the result json will be saved to profiling_file
    """
    self._inputs_map = {}
    self._outputs_map = {}
    self._is_saved_model = False
    self._decode = decode
    self._profiling_file = profiling_file
    self._model_path = model_path
    self._build_model()
    if self._decode:
      self._load_resource()

  def __del__(self):
    """
    destroy predictor resources
    """
    self._session.close()

  def search_pb(self, directory):
    """
    search pb file recursively, if multiple pb files exist, exception will be
    raised

    Returns:
      directory contain pb file
    """
    dir_list = []
    for root, dirs, files in tf.gfile.Walk(directory):
      for f in files:
        _, ext = os.path.splitext(f)
        if ext == '.pb':
          dir_list.append(root)
    if len(dir_list) == 0:
      raise ValueError('savedmodel is not found in directory %s' % directory)
    elif len(dir_list) > 1:
      raise ValueError('multiple saved model found in directory %s' % directory)

    return dir_list[0]

  def _build_model(self):
    """
    load graph from model_path and create session for this graph
    """

    model_path = self._model_path
    self._graph = tf.Graph()
    gpu_options = tf.GPUOptions(allow_growth=True)
    session_config = tf.ConfigProto(
        gpu_options=gpu_options,
        allow_soft_placement=True,
        log_device_placement=False)
    self._session = tf.Session(config=session_config, graph=self._graph)

    with self._graph.as_default():
      with self._session.as_default():
        # load model
        _, ext = os.path.splitext(model_path)
        tf.logging.info('loading model from %s' % model_path)
        if tf.gfile.IsDirectory(model_path):
          model_path = self.search_pb(model_path)
          logging.info('model find in %s' % model_path)
          assert tf.saved_model.loader.maybe_saved_model_directory(model_path), \
            'saved model does not exists in %s' % model_path
          self._is_saved_model = True
          meta_graph_def = tf.saved_model.loader.load(
              self._session, [tf.saved_model.tag_constants.SERVING], model_path)
          # parse signature
          signature_def = meta_graph_def.signature_def[
            signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
          inputs = signature_def.inputs
          for name, tensor in inputs.items():
            logging.info('Load input binding: %s -> %s' % (name, tensor.name))
            self._inputs_map[name] = self._graph.get_tensor_by_name(tensor.name)
          outputs = signature_def.outputs
          for name, tensor in outputs.items():
            logging.info('Load output binding: %s -> %s' % (name, tensor.name))
            self._outputs_map[name] = self._graph.get_tensor_by_name(
                tensor.name)

          # get assets
          self._assets = {}
          asset_files = tf.get_collection(constants.ASSETS_KEY)
          for any_proto in asset_files:
            asset_file = meta_graph_pb2.AssetFileDef()
            any_proto.Unpack(asset_file)
            type_name = asset_file.tensor_info.name.split(':')[0]
            asset_path = os.path.join(model_path, constants.ASSETS_DIRECTORY,
                                      asset_file.filename)
            assert tf.gfile.Exists(
                asset_path), '%s is missing in saved model' % asset_path
            self._assets[type_name] = asset_path
          logging.info(self._assets)

          # get export config
          self._export_config = {}
          self._use_bgr = False
          export_config_collection = tf.get_collection('EV_EXPORT_CONFIG')
          if len(export_config_collection) > 0:
            self._export_config = json.loads(export_config_collection[0])
            logging.info('load export config info %s' % export_config_collection[0])
            self._use_bgr = self._export_config.get('color_format', 'rgb').lower() == 'bgr'
            if self._use_bgr:
              logging.info('prediction will use image in bgr order')
            else:
              logging.info('prediction will use image in rgb order')

        else:
          raise ValueError('saved model is not found in %s' % self._model_path)

  def _load_resource(self):
    pass

  def predict(self, input_data_dict, output_names=None):
    """

    Args:
      input_data_dict: a dict containing all input data, key is the input name,
        value is the corresponding value
      output_names:  if not None, will fetch certain outputs, if set None, will
        return all the output info according to the output info in model signature
    Return:
      a dict of outputs, key is the output name, value is the corresponding value
    """
    feed_dict = {}
    for input_name, tensor in six.iteritems(self._inputs_map):
      assert input_name in input_data_dict, \
        'input data %s is missing' % input_name
      tensor_shape = tensor.get_shape().as_list()
      input_shape = input_data_dict[input_name].shape
      assert tensor_shape[0] is None or (tensor_shape[0] == input_shape[0]), \
        'input %s  batchsize %d is not the same as the exported batch_size %d' % (
        input_name, input_shape[0], tensor_shape[0])
      feed_dict[tensor] = input_data_dict[input_name]
    fetch_dict = {}
    if output_names is not None:
      for output_name in output_names:
        assert output_name in self._outputs_map, \
          'invalid output name %s' % output_name
        fetch_dict[output_name] = self._outputs_map[output_name]
    else:
      fetch_dict = self._outputs_map

    with self._graph.as_default():
      with self._session.as_default():
        if self._profiling_file is None:
          return self._session.run(fetch_dict, feed_dict)
        else:
          run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
          run_metadata = tf.RunMetadata()
          results = self._session.run(
              fetch_dict,
              feed_dict,
              options=run_options,
              run_metadata=run_metadata)
          # Create the Timeline object, and write it to a json
          from tensorflow.python.client import timeline
          tl = timeline.Timeline(run_metadata.step_stats)
          ctf = tl.generate_chrome_trace_format()
          with tf.gfile.GFile(self._profiling_file, 'w') as f:
            f.write(ctf)
          return results


def batch_images(images, use_bgr):
  assert isinstance(images[0], np.ndarray), 'image type is not np.ndarray but %s' % (
      type(images[0]))
  if len(images) == 1:
    return {'image': np.expand_dims(images[0], 0).astype(np.float32),
            'true_image_shape': np.array([list(images[0].shape)])}
  maxw = 0
  maxh = 0
  image_shapes = np.zeros((len(images), 3), np.int32)
  for idx, img in enumerate(images):
    shape = img.shape
    assert len(shape) == 3 and shape[2] == 3, \
      'image shape should be [height, width, 3]'
    image_shapes[idx, :] = shape
    maxh = max(maxh, shape[0])
    maxw = max(maxw, shape[1])

  batched_images = np.zeros((len(images), maxh, maxw, 3), np.float32)
  for idx, img in enumerate(images):
    h, w, _ = img.shape
    if use_bgr:
      img = img[:, :, ::-1]
    batched_images[idx, :h, :w] = img

  input_data = {}
  input_data['image'] = batched_images
  input_data['true_image_shape'] = image_shapes
  return input_data


class Predictor(PredictorInterface):
  """ Predictor which support all models exported from EasyVision
  """
  def __init__(self, model_path, profiling_file=None, decode=True):
    """
    Args:
      model_path:  saved_model directory or frozenpb file path
      profiling_file:  profiling result file, default None.
        if not None, predict function will use Timeline to profiling
        prediction time, and the result json will be saved to profiling_file
    """
    self._predictor_impl = PredictorImpl(model_path, profiling_file, decode)
    self._inputs_map = self._predictor_impl._inputs_map
    self._outputs_map = self._predictor_impl._outputs_map
    self._profiling_file = profiling_file
    self._export_config = self._predictor_impl._export_config
    self._use_bgr = self._export_config.get('color_format', 'rgb').lower() == 'bgr'
    if hasattr(self._predictor_impl, '_label_map'):
      self._label_map = self._predictor_impl._label_map

    if hasattr(self._predictor_impl, '_char_dict'):
      self._char_dict = self._predictor_impl._char_dict

  @property
  def get_input_names(self):
    """
    Return:
      a list, which conaining the name of input nodes available in model
    """
    return self._inputs_map.keys()

  def get_output_names(self):
    """
    Return:
      a list, which conaining the name of outputs nodes available in model
    """
    return self._outputs_map.keys()

  def predict(self, data_list, output_names=None, batch_size=1):
    """
    Args:
      data_list: list of numpy array with type uint8
      output_names:  if not None, will fetch certain outputs, if set None, will
      batch_size: batch_size used to predict, -1 indicates to use the real batch_size

    Return:
      a list of dict, each dict contain a key-value pair for output_name, output_value
    """
    num_image = len(data_list)
    assert len(data_list) > 0, 'input images should not be an empty list'
    if batch_size > 0:
      num_batches = int(math.ceil(float(num_image) / batch_size))
      num_padded = num_batches * batch_size - num_image
      image_list = data_list + [data_list[-1] for i in range(num_padded)]
    else:
      num_batches = 1
      batch_size = len(data_list)
      image_list = data_list

    outputs_list = []
    for batch_idx in range(num_batches):
      batch_image_list = image_list[batch_idx * batch_size:(batch_idx + 1) *
                                                           batch_size]
      input_data = self.batch(batch_image_list)
      outputs = self._predictor_impl.predict(input_data, output_names)
      for idx in range(batch_size):
        single_result = {}
        for key, batch_value in six.iteritems(outputs):
          single_result[key] = batch_value[idx]
        outputs_list.append(single_result)

    outputs_list = outputs_list[:num_image]
    return outputs_list

  def batch(self, images):
    """
    Pack a list of rgb order images to batch

    Args:
      images:  a list of numpy array
    Return:
      a dict of input feed dict, containing keys:
        image for image data
        true_image_shape for image shapes
    """
    assert isinstance(images, list), 'input data should be a list of np.array'
    assert len(images) > 0, 'number of input data list is zero'
    assert isinstance(images[0], np.ndarray), 'image type should be np.ndarray'
    return batch_images(images, self._use_bgr)

  def get_class_name(self, cls_id):
    if cls_id in self._label_map:
      return self._label_map[cls_id]['name']
    else:
      tf.logging.error('invalid class id %s' % cls_id)
      return ''

class FeatureExtractor(Predictor):

  def __init__(self, model_path, output_feature, profiling_file=None):
    """
    Args:
      model_path:  saved model path or frozenpb path
      output_feature:  output node name 
    """
    super(FeatureExtractor, self).__init__(
        model_path, profiling_file=profiling_file, decode=False)
    assert output_feature in self._outputs_map, \
      'invalid output_name %s, not in model signature' % output_feature
    self._outputs = [output_feature]

  def predict(self, images, batch_size=1):
    """
    Args:
      images:  a list of numpy uint8 array
       batch_size: batch_size used to predict
    Return:
      the output feature, a list of float32 numpy array
    """
    output_list = super(FeatureExtractor, self).predict(images, self._outputs,
                                                        batch_size)
    for idx in range(len(output_list)):
      out = output_list[idx]
      out = out[self._outputs[0]]
      output_list[idx] = {'feature': np.reshape(out, (-1))}
    return output_list

class PatchFeatureExtractor(PredictorInterface):
  def __init__(self, model_path):
    self._predictor = Predictor(model_path)
    self._feature_name = 'AvgPool_1a' # feature output from resnet

    if self._feature_name not in self._predictor.get_output_names():
      tf.logging.error('invalid feature %s, not in model output info' % self._feature_name)

  def extract_patch(self, image, num_patch):
    """
    extract patch from image

    Args:
      image: numpy array
      num_patch: tuple of int(num_height, nun_width),
        number of patch along height and width dimension

    Return:
      a list of numpy array, list of patches
    """
    num_height, num_width = num_patch
    patch_h = image.shape[0] // num_height
    patch_w = image.shape[1] // num_width

    patches = []
    for i in range(num_height):
      for j in range(num_width):
        patch = image[i*patch_h:(i+1)*patch_h, j*patch_w:(j+1)*patch_w, :]
        patches.append(patch)

    return patches

  def resize_image(self, image, max_shape=5000):
    large_shape = max(image.shape)
    if large_shape >= max_shape:
      scale = large_shape / max_shape
      new_width = int(image.shape[1]/scale)
      new_height = int(image.shape[0]/scale)
      new_image = cv2.resize(image, (new_width, new_height))
      print("large image with size {} is resized to {}".format(image.shape, new_image.shape))
      return new_image
    else:
      return image

  def predict(self, input_data, batch_size):
    """
    using session run predict a number of samples using batch_size
    Args:
      input_data:  a list of numpy array, each array is a sample to be predicted
      batch_size: batch_size passed by the caller, you can also ignore this param and
        use a fixed number if you do not want to adjust batch_size in runtime
    Return:
      result: a list of dict, each dict is the prediction result of one sample
        eg, {"output1": value1, "output2": value2}, the value type can be
        python int str float, and numpy array
    """
    num_patches = (4, 4)
    num_patch_per_image = num_patches[0] * num_patches[1]

    total_patches = []
    for image in input_data:
      patches = self.extract_patch(self.resize_image(image), num_patches)
      total_patches.extend(patches)

    features = self._predictor.predict(total_patches,
                            output_names=[self._feature_name], batch_size=num_patch_per_image*batch_size)

    results = []
    for idx in range(len(input_data)):
      features_per_image = []
      start = idx * num_patch_per_image
      for j in range(num_patch_per_image):
        feature = features[start+j][self._feature_name]
        f = np.reshape(feature, -1)
        features_per_image.extend(f)
      features_per_image = np.reshape(features_per_image, -1)
      results.append({'feature': features_per_image})

    return results


def main(args):
  output_feature = ''
  img = cv2.imread(args.img_path)
  # convert to rgb order
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  feature_extractor = PatchFeatureExtractor(args.model_path)
  results_batch = feature_extractor.predict([img], batch_size=1)
  _, fname = os.path.split(args.img_path)
  np.save(fname+'.npy', results_batch[0]['feature'])

if __name__ == '__main__':
  parser = argparse.ArgumentParser('image feature extract tool')
  parser.add_argument('model_path', help='directory of saved model')
  parser.add_argument('img_path', help='img file path')

  args = parser.parse_args()
  main(args)
