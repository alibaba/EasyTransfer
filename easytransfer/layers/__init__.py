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

from .dam import DAMEncoder
from .cnn import BiCNNEncoder, HybridCNNEncoder, TextCNNEncoder
from .core import Dense, Embedding, Dropout, LayerNormalization, dense_dropoutput_layernorm
from .activations import gelu_new
from .kd import HiddenLayerProbes
from .mask import get_attn_mask_bert, get_attn_mask_imagebert, get_attn_mask_videobert
from .utils import get_initializer, get_shape_list, gather_indexes
from .embedding import BertEmbeddings, AlbertEmbeddings
from .encoder_decoder import Encoder, Decoder
try:
    from .encoder_decoder_whale import Encoder as Encoder_whale
except:
    pass
from .header import MLMHead, NSPHead, AlbertMLMHead, FactorizedBertMLMHead
from .attention import SelfAttention
from tensorflow.python.layers.base import Layer







