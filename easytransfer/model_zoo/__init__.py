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

import os
import json
from tensorflow.python.platform import gfile
from .modeling_adabert import AdaBERTStudent

def get_pretrained_model(pretrain_model_name_or_path, **kwargs):
    if "/" not in pretrain_model_name_or_path:
        model_type = pretrain_model_name_or_path.split("-")[1]
        if model_type == 'bert':
            from .modeling_bert import BertPreTrainedModel
            return BertPreTrainedModel.get(pretrain_model_name_or_path, **kwargs)
        elif model_type == "roberta":
            from .modeling_roberta import RobertaPreTrainedModel
            return RobertaPreTrainedModel.get(pretrain_model_name_or_path, **kwargs)
        elif model_type == 'albert':
            from .modeling_albert import AlbertPreTrainedModel
            return AlbertPreTrainedModel.get(pretrain_model_name_or_path, **kwargs)
        elif model_type == 'imagebert':
            from .modeling_imagebert import ImageBertPreTrainedModel
            return ImageBertPreTrainedModel.get(pretrain_model_name_or_path, **kwargs)
        elif model_type == 'videobert':
            from .modeling_videobert import VideoBertPreTrainedModel
            return VideoBertPreTrainedModel.get(pretrain_model_name_or_path, **kwargs)
        else:
            raise NotImplementedError
    else:
        config_path = os.path.join(os.path.dirname(pretrain_model_name_or_path), "config.json")
        if "oss" in config_path:
            with gfile.GFile(config_path, mode='r') as reader:
                text = reader.read()
        else:
            with open(config_path, "r") as reader:
                text = reader.read()
        json_config = json.loads(text)
        model_type = json_config["model_type"]
        assert model_type is not None, "you must specify model_type in config.json when pass pretrained_model_path"
        if model_type == 'bert':
            from .modeling_bert import BertPreTrainedModel
            return BertPreTrainedModel.get(pretrain_model_name_or_path, **kwargs)
        elif model_type == "roberta":
            from .modeling_roberta import RobertaPreTrainedModel
            return RobertaPreTrainedModel.get(pretrain_model_name_or_path, **kwargs)
        elif model_type == 'albert':
            from .modeling_albert import AlbertPreTrainedModel
            return AlbertPreTrainedModel.get(pretrain_model_name_or_path, **kwargs)
        elif model_type == 'imagebert':
            from .modeling_imagebert import ImageBertPreTrainedModel
            return ImageBertPreTrainedModel.get(pretrain_model_name_or_path, **kwargs)
        elif model_type == 'videobert':
            from .modeling_videobert import VideoBertPreTrainedModel
            return VideoBertPreTrainedModel.get(pretrain_model_name_or_path, **kwargs)
        else:
            raise ValueError("model_type should be in bert, roberta, albert, imagebert, videobert")

def get_config_path(model_type, pretrain_model_name_or_path):
    if model_type == 'bert':
        from .modeling_bert import BertPreTrainedModel
        config_path = BertPreTrainedModel.pretrained_config_archive_map[
            pretrain_model_name_or_path]
    elif model_type == "roberta":
        from .modeling_roberta import RobertaPreTrainedModel
        config_path = RobertaPreTrainedModel.pretrained_config_archive_map[
            pretrain_model_name_or_path]
    elif model_type == 'albert':
        from .modeling_albert import AlbertPreTrainedModel
        config_path = AlbertPreTrainedModel.pretrained_config_archive_map[
            pretrain_model_name_or_path]
    elif model_type == 'imagebert':
        from .modeling_imagebert import ImageBertPreTrainedModel
        config_path = ImageBertPreTrainedModel.pretrained_config_archive_map[
            pretrain_model_name_or_path]
    elif model_type == 'videobert':
        from .modeling_videobert import VideoBertPreTrainedModel
        config_path = VideoBertPreTrainedModel.pretrained_config_archive_map[
            pretrain_model_name_or_path]
    else:
        raise ValueError("model_type should be in bert, roberta, albert, imagebert, videobert")

    return config_path