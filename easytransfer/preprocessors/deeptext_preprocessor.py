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



from collections import OrderedDict
import json
import numpy as np
import tensorflow as tf

from .tokenization import convert_to_unicode
from .preprocessor import Preprocessor


PAD_WORD = '[PAD]'
UNK_WORD = '[UNK]'
BOS_WORD = '[BOS]'
EOS_WORD = '[EOS]'
MASK_WORD = '[MASK]'
SEP_WORD = '[SEP]'
ALL_SPECIAL_TOKENS = [PAD_WORD, UNK_WORD, BOS_WORD, EOS_WORD, MASK_WORD, SEP_WORD]


def get_pretrained_embedding(stoi, pretrained_w2v_path, init="random"):
    tf.logging.info("Start loading pretrained word embeddings from {}".format(pretrained_w2v_path))
    mu, sigma = 0, 0.01
    hit = 0
    with tf.gfile.GFile(pretrained_w2v_path) as f:
        line = f.readline()
        word_num, vec_dim = line.split(" ")
        vec_dim = int(vec_dim)
        if init == "random":
            res_embed_matrix = np.array([np.random.normal(mu, sigma, vec_dim).tolist()
                                         for _ in range(len(stoi))])
        elif init == "zero":
            res_embed_matrix = np.zeros((len(stoi), vec_dim))
        else:
            raise NotImplementedError
        tf.logging.info("Total pretrained word num: {}".format(word_num))
        for i, line in enumerate(f):
            word = convert_to_unicode(line.split(" ")[0])
            vec = [float(t) for t in line.strip().split(" ")[1:]]
            if len(vec) != vec_dim:
                continue
            if word in stoi:
                hit += 1
                res_embed_matrix[stoi[word]] = vec
        tf.logging.info("Hit: {}/{}".format(hit, len(stoi)))
    tf.logging.info("Loading pretrained Done")
    return res_embed_matrix


class DeepTextVocab(object):
    def __init__(self):
        self.stoi = {}
        self.itos = {}
        self.stof = {}
        self.size = 0
        for token in ALL_SPECIAL_TOKENS:
            self.add_word(token)
            self.stof[token] = float('inf')

    def __str__(self):
        s = ""
        for word, idx in self.stoi.items():
            s += word + "\t" + str(idx) + "\n"
        return s

    def __repr__(self):
        return self.__str__()

    def __len__(self):
        return self.size

    def has(self, word):
        return word in self.stoi

    def add_word(self, word):
        if not self.has(word):
            ind = self.size
            self.stoi[word] = ind
            self.itos[ind] = word
            self.stof[word] = 1
            self.size += 1
        else:
            self.stof[word] += 1

    def add_line(self, line):
        line = convert_to_unicode(line)
        for word in line.lower().split(" "):
            self.add_word(word)

    def to_idx(self, word):
        if self.has(word):
            return self.stoi[word]
        else:
            return self.stoi[UNK_WORD]

    def to_word(self, ind):
        if ind >= self.size:
            return 0
        return self.itos[ind]

    def filter_vocab_to_fix_length(self, max_vocab_size=50000):
        tmp_stof = self.stof
        self.stoi = {}
        self.stof = {}
        sorted_list = list(sorted(tmp_stof.items(), key=lambda x: x[1], reverse=True))
        for i, (word, freq) in enumerate(sorted_list[:max_vocab_size]):
            self.stoi[word] = i
            self.stof[word] = freq
        self.itos = {val: key for key, val in self.stoi.items()}
        self.size = len(self.stoi)

    @classmethod
    def build_from_file(cls, file_path):
        with tf.gfile.GFile(file_path) as f:
            stoi = json.load(f)
        obj = cls()
        obj.stoi = {convert_to_unicode(key): val for key, val in stoi.items()}
        obj.itos = {val: key for key, val in obj.stoi.items()}
        obj.size = len(obj.stoi)
        return obj

    def export_to_file(self, file_path):
        with tf.gfile.GFile(file_path, mode="w") as f:
            json.dump(self.stoi, f)


class DeepTextPreprocessor(Preprocessor):
    """ Preprocessor for deep text models such as CNN, DAM, HCNN, etc.

    """
    def __init__(self, config, **kwargs):
        super(DeepTextPreprocessor, self).__init__(config, **kwargs)
        self.config = config
        self.vocab = DeepTextVocab.build_from_file(self.config.vocab_path)
        if config.mode.startswith("train") and \
            hasattr(self.config, "pretrain_word_embedding_name_or_path") and \
            self.config.pretrain_word_embedding_name_or_path:
            emb_path = self.config.pretrain_word_embedding_name_or_path
            assert tf.gfile.Exists(emb_path)
            self.pretrained_word_embeddings = get_pretrained_embedding(
                self.vocab.stoi, emb_path)
        else:
            self.pretrained_word_embeddings = None

        self.input_tensor_names = []
        for schema in config.input_schema.split(","):
            name = schema.split(":")[0]
            self.input_tensor_names.append(name)

        if not hasattr(config, "first_sequence_length"):
            setattr(self.config, "first_sequence_length", self.config.sequence_length)
        if not hasattr(config, "second_sequence") or \
                not self.config.second_sequence in self.input_tensor_names:
            setattr(self.config, "second_sequence_length", 1)

        self.label_idx_map = OrderedDict()
        if hasattr(self.config, "label_enumerate_values") and self.config.label_enumerate_values is not None:
            for (i, label) in enumerate(self.config.label_enumerate_values.split(",")):
                self.label_idx_map[convert_to_unicode(label)] = i

        self.label_name = config.label_name if hasattr(config, "label_name") else ""

        if hasattr(self.config, "multi_label") and self.config.multi_label is True:
            self.multi_label = True
            self.max_num_labels = self.config.max_num_labels if hasattr(self.config, "max_num_labels") else 5
        else:
            self.multi_label = False
            self.max_num_labels = None

    def set_feature_schema(self):
        if self.mode.startswith("predict") or self.mode == "preprocess":
            self.output_schema = self.config.output_schema
        self.output_tensor_names = ["input_ids_a", "input_mask_a", "input_ids_b", "input_mask_b", "label_id"]
        if self.multi_label:
            self.seq_lens = [self.config.first_sequence_length] * 2 + \
                            [self.config.second_sequence_length] * 2 + [1] + [self.max_num_labels]
            self.feature_value_types = [tf.int64] * 4 + [tf.int64]
        else:
            self.seq_lens = [self.config.first_sequence_length] * 2 + \
                            [self.config.second_sequence_length] * 2 + [1]
            if len(self.label_idx_map) >= 2:
                self.feature_value_types = [tf.int64] * 4 + [tf.int64]
            else:
                self.feature_value_types = [tf.int64] * 4 + [tf.float32]

    def convert_example_to_features(self, items):
        """ Convert single example to classifcation/regression features

        Args:
            items (`dict`): inputs from the reader
        Returns:
            features (`tuple`): (input_ids, input_mask, segment_ids, label_id)
        """
        first_seq_len, second_seq_len = self.config.first_sequence_length, \
                                        self.config.second_sequence_length
        text_a = items[self.input_tensor_names.index(self.config.first_sequence)]
        tokens_a = [t for t in convert_to_unicode(text_a).lower().split(" ")][:first_seq_len]
        indices_a = [self.vocab.to_idx(token) for token in tokens_a]
        masks_a = [1 for _ in tokens_a]
        while len(indices_a) < first_seq_len:
            indices_a.append(self.vocab.to_idx(PAD_WORD))
            masks_a.append(0)

        if self.config.second_sequence in self.input_tensor_names:
            text_b = items[self.input_tensor_names.index(self.config.second_sequence)]
            tokens_b = [t for t in convert_to_unicode(text_b).lower().split(" ")][:second_seq_len]
            indices_b = [self.vocab.to_idx(token) for token in tokens_b]
            masks_b = [1 for _ in tokens_b]
            while len(indices_b) < second_seq_len:
                indices_b.append(self.vocab.to_idx(PAD_WORD))
                masks_b.append(0)
        else:
            indices_b = [0]
            masks_b = [0]

        # support classification and regression
        if self.config.label_name is not None:

            label_value = items[self.input_tensor_names.index(self.config.label_name)]
            if isinstance(label_value, str) or isinstance(label_value, bytes):
                label = convert_to_unicode(label_value)
            else:
                label = str(label_value)

            if self.multi_label:
                label_ids = [self.label_idx_map[convert_to_unicode(x)] for x in label.split(",") if x]
                label_ids = label_ids[:self.max_num_labels]
                label_ids = label_ids + [-1 for _ in range(self.max_num_labels - len(label_ids))]
                label_ids = [str(t) for t in label_ids]
                label_id = ' '.join(label_ids)
            elif len(self.label_idx_map) >= 2:
                label_id = str(self.label_idx_map[convert_to_unicode(label)])
            else:
                label_id = label

        else:
            label_id = '0'

        return ' '.join([str(t) for t in indices_a]), \
               ' '.join([str(t) for t in masks_a]), \
               ' '.join([str(t) for t in indices_b]), \
               ' '.join([str(t) for t in masks_b]), label_id