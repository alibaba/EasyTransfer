# coding=utf-8
#
# Copyright (c) 2019 Alibaba PAI team.
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

from .classification_regression_evaluator import classification_eval_metrics, multi_label_eval_metrics, regression_eval_metrics
from .classification_regression_evaluator import matthew_corr_metrics
from .comprehension_evaluator import comprehension_eval_metrics
from .kd_evaluator import teacher_probes_eval_metrics
from .labeling_evaluator import sequence_labeling_eval_metrics
from .match_evaluator import match_eval_metrics
from .pretrain_evaluator import masked_language_model_eval_metrics, next_sentence_prediction_eval_metrics

