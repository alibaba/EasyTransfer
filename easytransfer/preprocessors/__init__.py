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


from .classification_regression_preprocessor import ClassificationRegressionPreprocessor, \
    PairedClassificationRegressionPreprocessor
from .comprehension_preprocessor import ComprehensionPreprocessor, MultiTurnComprehensionPreprocessor
from .pretrain_preprocessor import PretrainPreprocessor
from .labeling_preprocessor import SequenceLabelingPreprocessor


def get_preprocessor(pretrain_model_name_or_path,
                     app_model_name="text_classify_bert",
                     is_paired=False,
                     **kwargs):
    if app_model_name == "text_comprehension_bert":
        return ComprehensionPreprocessor.get_preprocessor(
            pretrain_model_name_or_path=pretrain_model_name_or_path, **kwargs)

    elif app_model_name == "text_comprehension_bert_hae":
        return MultiTurnComprehensionPreprocessor.get_preprocessor(
            pretrain_model_name_or_path=pretrain_model_name_or_path, **kwargs)

    elif app_model_name == "sequence_labeling_bert":
        return SequenceLabelingPreprocessor.get_preprocessor(
            pretrain_model_name_or_path=pretrain_model_name_or_path, **kwargs)

    elif app_model_name in ["text_classify_bert", "text_match_bert"]:
        if is_paired:
            return PairedClassificationRegressionPreprocessor.get_preprocessor(
                pretrain_model_name_or_path=pretrain_model_name_or_path, **kwargs)
        else:
            return ClassificationRegressionPreprocessor.get_preprocessor(
                pretrain_model_name_or_path=pretrain_model_name_or_path, **kwargs)

    elif app_model_name == "pretrain_language_model":
        return PretrainPreprocessor.get_preprocessor(
            pretrain_model_name_or_path=pretrain_model_name_or_path, **kwargs)
    else:
        raise NotImplementedError("APP model {} not implemented".format(app_model_name))
