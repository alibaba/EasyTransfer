# coding=utf-8
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

from .classification_postprocessors import ClassificationPostprocessor
from .comprehension_postprocessors import ComprehensionPostprocessor
from .labeling_postprocessors import LabelingPostprocessor


def get_postprocessors(app_model_name=None, **kwargs):
    if app_model_name in ["text_comprehension_bert", "text_comprehension_bert_hae"]:
        return ComprehensionPostprocessor(**kwargs)
    elif app_model_name in ["sequence_labeling_bert"]:
        return LabelingPostprocessor(**kwargs)
    else:
        return ClassificationPostprocessor(**kwargs)
