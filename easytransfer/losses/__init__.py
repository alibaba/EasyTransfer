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

from .classification_regression_loss import softmax_cross_entropy
from .classification_regression_loss import mean_square_error
from .classification_regression_loss import multi_label_sigmoid_cross_entropy
from .labeling_loss import sequence_labeling_loss
from .comprehension_loss import comprehension_loss
from .kd_loss import build_kd_loss, build_kd_probes_loss
from .matching_loss import matching_embedding_margin_loss
from .pretrain_loss import masked_language_model_loss
from .pretrain_loss import next_sentence_prediction_loss
from .pretrain_loss import image_reconstruction_mse_loss
from .pretrain_loss import image_reconstruction_kld_loss

