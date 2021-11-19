from __future__ import absolute_import, division, print_function
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
import json
import logging
import numpy as np
import torch
from tqdm import tqdm, trange
import sys
import shutil
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset, Dataset)
from torch.utils.data.distributed import DistributedSampler
from random import random, shuffle, choice, sample, randint
from random import seed as rd_seed
# from pytorch_pretrained_bert.file_utils import WEIGHTS_NAME, CONFIG_NAME, MAX_TARGET, MAX_NUM_PAIR, MAX_LONG_WORD_USE, \
# MAX_SHORT_WORD_USE, MAX_SEG_USE
from transformers import RobertaConfig  # Note that we use the sop, rather than nsp task.
from transformers import RobertaForMaskedLM
from transformers import RobertaTokenizer
from torch.optim import AdamW
from transformers.optimization import get_linear_schedule_with_warmup
import argparse
import pickle
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
# from apex import amp

def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    rd_seed(seed)
set_seed(42)

data_path = './roberta_pretrain_data/'

long_tail_entity_frq = {}
with open('99%_training_data_long_tail_entity.txt', 'r') as file:
    long_tail_entitys = file.readlines()
    for long_tail_entity in long_tail_entitys:
        entityQname, entity_frq = long_tail_entity.strip().split('\t')
        if entityQname not in long_tail_entity_frq.keys():
            long_tail_entity_frq[entityQname] = entity_frq
        else:
            raise ValueError('existing long tail entity')

with open('/apsarapangu/disk3/zhangtaolin.ztl/aaai22_v3/roberta_pretrain_data/roberta_entity2wikidata5m_20.pickle', 'rb') as file:
    # {'Qxxx': {'entity':xxx, 'relation':xxx, 'knowledge_label': +-1}}
    entityQ2textID = pickle.load(file)
# with open('/apsarapangu/disk3/zhangtaolin.ztl/ernie/kg_embed/entity2id.txt', 'r') as file:
#     entity2id = file.readlines()
# from tqdm import tqdm
# id2entity_dict = {}
# for item in tqdm(entity2id[1:]):
#     entityName_Q, entityId = item.strip().split('\t')
#     if entityId not in id2entity_dict.keys():
#         id2entity_dict[entityId] = entityName_Q

# ent_file_name = './wiki_embedding/Wikidata/embeddings/dimension_100/transe/entity2vec.bin'
# ent_vec = np.memmap(ent_file_name, dtype='float32', mode='r')
#
# rel_file_name = './wiki_embedding/Wikidata/embeddings/dimension_100/transe/relation2vec.bin'
# rel_vec = np.memmap(rel_file_name, dtype='float32', mode='r')
#
# ent_embed = torch.from_numpy(ent_vec).view(-1, 100)
# # print(ent_embed.size())
# ent_embed = torch.cat((ent_embed, torch.mean(ent_embed, dim=0, keepdim=True)), dim=0)
# # print(ent_embed.size())
# ent_embed = torch.nn.Embedding.from_pretrained(ent_embed)
# ent_pad_id = ent_embed.weight.size(0) - 1
#
# rel_embed = torch.FloatTensor(rel_vec).view(-1, 100)
# rel_embed = torch.cat((rel_embed, torch.mean(rel_embed, dim=0, keepdim=True)), dim=0)
# rel_embed = torch.nn.Embedding.from_pretrained(rel_embed)
# rel_pad_id = rel_embed.weight.size(0) - 1
#
# del rel_vec, ent_vec
#
# qid2pos = {}
# pid2pos = {}
# qid2frq = {}
# # idx2neighbor = {}
# idx2name = {}
# qid2alias = {}
#
# with open('wiki_embedding/Wikidata/knowledge graphs/triple2id.pkl','rb') as tf:
#     idx2neighbor = pickle.load(tf)
# with open('wiki_embedding/Wikidata/knowledge graphs/relation2id.pkl', 'rb') as tf:
#     pid2pos = pickle.load(tf)
# with open('entity2id.pkl', 'rb') as tf:
#     qid2pos = pickle.load(tf)
with open('qid2frq.pkl', 'rb') as tf:
    qid2frq = pickle.load(tf)
# with open('qid2alias.pkl','rb') as tf:
#     qid2alias = pickle.load(tf)

CONFIG_NAME = "config.json"
WEIGHTS_NAME = "pytorch_model.bin"

MAX_TARGET = 20
MAX_NEGBHOR_USE = 5  # rel_num .
SAMPLE_REVESERD_RATE = 0.05
GRADIENT_CLIP = 25
GRADIENT_SKIP = 30

# torch.autograd.set_detect_anomaly(True)
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

class OurENRIEDataset(Dataset):
    def __init__(self, args, data_path, max_seq_length, masked_lm_prob,
                 max_predictions_per_seq, tokenizer,
                 data_type='train', min_len=100):
        self.args = args
        # self.longer_128_offset = longer_128_offset
        self.data_path = data_path
        self.max_seq_length = max_seq_length
        self.masked_lm_prob = masked_lm_prob
        self.max_predictions_per_seq = max_predictions_per_seq
        self.tokenizer = tokenizer
        self.min_len = min_len
        self.max_num_tokens = max_seq_length - 2
        self.examples = []
        # print(tokenizer.vocab)
        self.vocab = tokenizer.get_vocab()
        self.data_type = data_type
        self.__read_data__()
        self.data_sources = os.path.join(self.data_path, '{}_pretrain_data.txt'.format('total'))
        self.args = args

    def __getitem__(self, index):
        offset = self.examples[index]
        sources = open(self.data_sources, 'r', encoding='utf-8')
        sources.seek(offset)
        example = json.loads(sources.readline())
        sources.close()
        # example = self.examples[index]
        masked_example = self.__get_example__(example)
        feature = self.__get_feature__(masked_example)
        tensor_tuple = self.__feature2tensor__(feature)
        return tensor_tuple

    def __get_example__(self, example):
        token_ids, masked_lm_positions, masked_label_ids, entity_qid, entity_pos = create_wwm_lm_predictions(self.args,
                                                                                                             example,
                                        self.masked_lm_prob, self.max_predictions_per_seq, self.vocab, self.tokenizer,
                                                                                                             self.data_type)
        segment_ids = [0] * len(token_ids)  # We Do NOT use sequence-level task.
        example = {
            "token_ids": token_ids,
            "segment_ids": segment_ids,
            "masked_lm_positions": masked_lm_positions,
            "masked_label_ids": masked_label_ids,
            'entity_qid': entity_qid,
            'entity_pos': entity_pos
        }
        return example

    def __get_feature__(self, example):
        max_seq_length = self.max_seq_length
        input_ids = example["token_ids"]
        segment_ids = example["segment_ids"]
        masked_lm_positions = example["masked_lm_positions"]
        masked_label_ids = example["masked_label_ids"]
        assert len(input_ids) == len(segment_ids) <= max_seq_length  # The preprocessed data should be already truncated

        input_array = np.full(max_seq_length, dtype=int, fill_value=self.tokenizer.convert_tokens_to_ids(['<pad>'])[0])
        input_array[:len(input_ids)] = input_ids

        mask_array = np.zeros(max_seq_length, dtype=int)
        mask_array[:len(input_ids)] = 1

        segment_array = np.zeros(max_seq_length, dtype=int)
        segment_array[:len(segment_ids)] = segment_ids

        lm_label_array = np.full(max_seq_length, dtype=int, fill_value=-100)
        lm_label_array[masked_lm_positions] = masked_label_ids

        knowledge_input_ids = np.zeros((max_seq_length, self.args.knowledge_text_len), dtype=int)
        knowledge_input_masks = np.zeros((max_seq_length, self.args.knowledge_text_len), dtype=int)
        knowledge_segment_ids = np.zeros((max_seq_length, self.args.knowledge_text_len), dtype=int)
        knowledge_entity_start = np.zeros((max_seq_length, 1), dtype=int)
        knowledge_entity_end = np.zeros((max_seq_length, 1), dtype=int)
        knowledge_relation_start = np.zeros((max_seq_length, 1), dtype=int)
        knowledge_relation_end = np.zeros((max_seq_length, 1), dtype=int)
        knowledge_add_sub_label = np.zeros((max_seq_length, 1), dtype=int)
        knowledge_label_mask = np.zeros((max_seq_length, 1), dtype=int)
        knowledge_pseudo_mask = np.ones((max_seq_length, 1), dtype=int)

        entity_qid = example['entity_qid']
        entity_pos = example['entity_pos']
        if len(entity_qid) != 0:
            for index, qid in enumerate(entity_qid):
                entity_start_in_context, entity_end_in_context = entity_pos[index]
                if qid in entityQ2textID.keys():
                    entity_input_ids = entityQ2textID[qid]['knowledge_input_ids']
                    # 由于Roberta 模型训练的时候只有一种segment的输入，所以token_type embedding只有一个，就是0。后面就靠input_mask来区分
                    # entity_segment_ids = entityQ2textID[qid]['knowledge_segment_ids']
                    entity_input_mask = entityQ2textID[qid]['knowledge_input_mask']
                    knowledge_label = entityQ2textID[qid]['knowledge_label']
                    start_ent = entityQ2textID[qid]['start_ent']
                    end_ent = entityQ2textID[qid]['end_ent']
                    start_rel = entityQ2textID[qid]['start_rel']
                    end_rel = entityQ2textID[qid]['end_rel']

                    # 实体对应的token位置全部注入
                    for insert_index in range(entity_start_in_context, entity_end_in_context+1):
                        knowledge_input_ids[insert_index] = entity_input_ids
                        # knowledge_segment_ids[insert_index] = entity_segment_ids
                        knowledge_input_masks[insert_index] = entity_input_mask
                        knowledge_add_sub_label[insert_index][0] = knowledge_label
                        knowledge_entity_start[insert_index][0] = start_ent
                        knowledge_entity_end[insert_index][0] = end_ent
                        knowledge_relation_start[insert_index][0] = start_rel
                        knowledge_relation_end[insert_index][0] = end_rel
                        knowledge_label_mask[insert_index][0] = 1
                        knowledge_pseudo_mask[insert_index][0] = 0
                else:
                    continue
        feature = InputFeatures(input_ids=input_array,
                                input_mask=mask_array,
                                segment_ids=segment_array,
                                label_id=lm_label_array,
                                knowledge_input_ids=knowledge_input_ids,
                                knowledge_segment_ids=knowledge_segment_ids,
                                knowledge_input_masks=knowledge_input_masks,
                                knowledge_entity_start=knowledge_entity_start,
                                knowledge_entity_end=knowledge_entity_end,
                                knowledge_relation_start=knowledge_relation_start,
                                knowledge_relation_end=knowledge_relation_end,
                                knowledge_add_sub_label=knowledge_add_sub_label,
                                knowledge_label_mask=knowledge_label_mask,
                                knowledge_pseudo_mask=knowledge_pseudo_mask)
        return feature

    def __feature2tensor__(self, feature):
        f = feature
        all_input_ids = torch.tensor(f.input_ids, dtype=torch.long)
        all_input_mask = torch.tensor(f.input_mask, dtype=torch.long)
        all_segment_ids = torch.tensor(f.segment_ids, dtype=torch.long)
        all_label_ids = torch.tensor(f.label_id, dtype=torch.long)
        knowledge_input_ids = torch.tensor(f.knowledge_input_ids, dtype=torch.long)
        knowledge_segment_ids = torch.tensor(f.knowledge_segment_ids, dtype=torch.long)
        knowledge_input_masks = torch.tensor(f.knowledge_input_masks, dtype=torch.long)
        knowledge_entity_start = torch.tensor(f.knowledge_entity_start, dtype=torch.long)
        knowledge_entity_end = torch.tensor(f.knowledge_entity_end, dtype=torch.long)
        knowledge_relation_start = torch.tensor(f.knowledge_relation_start, dtype=torch.long)
        knowledge_relation_end = torch.tensor(f.knowledge_relation_end, dtype=torch.long)
        knowledge_add_sub_label = torch.tensor(f.knowledge_add_sub_label, dtype=torch.long)
        knowledge_label_mask = torch.tensor(f.knowledge_label_mask, dtype=torch.long)
        knowledge_pseudo_mask = torch.tensor(f.knowledge_pseudo_mask, dtype=torch.long)

        return all_input_ids, all_input_mask, all_segment_ids, all_label_ids, \
               knowledge_input_ids, knowledge_segment_ids, knowledge_input_masks, \
               knowledge_entity_start, knowledge_entity_end, knowledge_relation_start, \
               knowledge_relation_end, knowledge_add_sub_label, knowledge_label_mask, \
               knowledge_pseudo_mask

    def __len__(self):
        return len(self.examples)

    def __read_data__(self):
        if self.data_type == 'train':
            with open('/apsarapangu/disk3/zhangtaolin.ztl/aaai22_v3/roberta_pretrain_data/train_sample_data_99%.txt', 'r') as file:
                self.examples = tuple([int(item.strip()) for item in file.readlines()])
                logger.info('training samples loaded!')
            # with open('/apsarapangu/disk3/zhangtaolin.ztl/aaai22_v7/all_openwebQA_roberta_ids.txt', 'r') as file:
            #     data = file.readlines()[:7000000]
            # for item in data:
            #     example_id = [int(token_id) for token_id in item.strip().split(' ')]
            #     self.examples.append(example_id)
        else:
            with open('/apsarapangu/disk3/zhangtaolin.ztl/aaai22_v3/roberta_pretrain_data/dev_sample_data_1%.txt', 'r') as file:
                self.examples = tuple([int(item.strip()) for item in file.readlines()])
                logger.info('dev samples loaded!')
            # with open('/apsarapangu/disk3/zhangtaolin.ztl/aaai22_v7/all_openwebQA_roberta_ids.txt', 'r') as file:
            #     data = file.readlines()[7000000:]
            # for item in data:
            #     example_id = [int(token_id) for token_id in item.strip().split(' ')]
            #     self.examples.append(example_id)
        # with open(os.path.join(self.data_path, 'offset_ids.pkl'), 'rb') as tf:
        #     self.examples = list(pickle.load(tf))

        # if self.data_type == 'train':
        #     global train_offset
        #     # 定义全局变量，在train的时候加入已经选中的数据offset，删除大于128的数据
        #     for example_offset in tqdm(self.examples, desc='remove longer than 128 in train mode'):
        #         if example_offset in self.longer_128_offset:
        #             self.examples.remove(example_offset)
        # if self.data_type == 'dev':
        #     for example_offset in tqdm(self.examples, desc='remove longer than 128 and used samples in dev mode'):
        #         if example_offset in self.longer_128_offset or example_offset in train_offset:
        #             self.examples.remove(example_offset)

        # shuffle(self.examples)
        # total_num = len(self.examples)
        # sample_num = int(total_num * SAMPLE_REVESERD_RATE * (.05 if self.data_type != 'train' else 1))

        # if self.data_type == 'train':
        #     # 这个变量可以全局调用，注意！！！
        #     train_offset = self.examples[:sample_num]

        # self.examples = tuple(self.examples[:sample_num])
        # print('{} samples loaded, {} used.'.format(total_num, sample_num))

# logging.basicConfig(filename='logger.log', level=logging.INFO)
def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x / warmup
    return 1.0 - x


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id,
                                    knowledge_input_ids,
                                    knowledge_segment_ids,
                                    knowledge_input_masks,
                                    knowledge_entity_start,
                                    knowledge_entity_end,
                                    knowledge_relation_start,
                                    knowledge_relation_end,
                                    knowledge_add_sub_label,
                                    knowledge_label_mask,
                                    knowledge_pseudo_mask):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.knowledge_input_ids = knowledge_input_ids
        self.knowledge_segment_ids = knowledge_segment_ids
        self.knowledge_input_masks = knowledge_input_masks
        self.knowledge_entity_start = knowledge_entity_start
        self.knowledge_entity_end = knowledge_entity_end
        self.knowledge_relation_start = knowledge_relation_start
        self.knowledge_relation_end = knowledge_relation_end
        self.knowledge_add_sub_label = knowledge_add_sub_label
        self.knowledge_label_mask = knowledge_label_mask
        self.knowledge_pseudo_mask = knowledge_pseudo_mask


def create_masked_lm_predictions(tokens, masked_lm_prob, max_predictions_per_seq, vocab):
    """Creates the predictions for the masked LM objective. This is mostly copied from the Google BERT repo, but
    with several refactors to clean it up and remove a lot of unnecessary variables."""
    cand_indices = []
    for (i, token) in enumerate(tokens):
        if token == "[CLS]" or token == "[SEP]" or not token.isalnum():
            continue
        cand_indices.append(i)

    num_to_mask = min(max_predictions_per_seq,
                      max(1, int(round(len(tokens) * masked_lm_prob))))
    # print(num_to_mask)
    # print("tokens", len(tokens))
    # print("cand", len(cand_indices))
    shuffle(cand_indices)
    mask_indices = sorted(sample(cand_indices, num_to_mask))
    masked_token_labels = []
    for index in mask_indices:
        # 80% of the time, replace with [MASK]
        if random() < 0.8:
            masked_token = "[MASK]"
        else:
            # 10% of the time, keep original
            if random() < 0.5:
                masked_token = tokens[index]
            # 10% of the time, replace with random word
            else:
                masked_token = choice(vocab)
        masked_token_labels.append(tokens[index])
        # Once we've saved the true label for that token, we can overwrite it with the masked version
        tokens[index] = masked_token

    return tokens, mask_indices, masked_token_labels


def isSkipToken(token):
    return token == "[CLS]" or token == "[SEP]" or (not token.isalnum() and len(token) == 1)


def get_reduce_dict(d, n):
    tmp = list(d.items())
    shuffle(tmp)
    tmp = tmp[:n]
    return dict(tmp)

def create_roberta_lm_predictions(args, example, masked_lm_prob, max_predictions_per_seq, vocab, tokenizer, data_type):
    """Creates the predictions for the masked LM objective. This is mostly copied from the Google BERT repo, but
    with several refactors to clean it up and remove a lot of unnecessary variables."""
    token_ids = example['token_ids']
    if len(token_ids) > args.max_seq_length-2:
        token_ids = token_ids[:args.max_seq_length-2]
        token_ids = [0] + token_ids + [2]
    else:
        token_ids = [0] + token_ids + [2]
    total_num_to_mask = min(max_predictions_per_seq, max(1, int(round(len(token_ids) * masked_lm_prob))))
    # token_ids = [0] + token_ids + [2]

    token_masked_lm_positions = []

    tmp = list(i for i in range(1, len(token_ids) - 1))
    shuffle(tmp)
    for t in tmp:
        if (total_num_to_mask <= 0): break
        if (t not in token_masked_lm_positions):
            token_masked_lm_positions.append(t)
            total_num_to_mask -= 1

    masked_label_ids = []
    for pos in token_masked_lm_positions:
        masked_label_ids.append(token_ids[pos])
        masked_token = 50264
        if (random() < .2):
            masked_token = token_ids[pos]
            if (random() < .5):
                masked_token = randint(3, len(vocab) - 100)
        token_ids[pos] = masked_token
    entity_qid, entity_pos = [], []
    return token_ids, token_masked_lm_positions, masked_label_ids, entity_qid, entity_pos

def create_wwm_backup_lm_predictions(args, example, masked_lm_prob, max_predictions_per_seq, vocab, tokenizer, data_type):
    """Creates the predictions for the masked LM objective. This is mostly copied from the Google BERT repo, but
    with several refactors to clean it up and remove a lot of unnecessary variables."""
    """Creates the predictions for the masked LM objective. This is mostly copied from the Google BERT repo, but
        with several refactors to clean it up and remove a lot of unnecessary variables."""
    token_ids = example
    if len(token_ids) > args.max_seq_length - 2:
        token_ids = token_ids[:args.max_seq_length - 2]
    token_ids = [0] + token_ids + [2]
    total_num_to_mask = min(max_predictions_per_seq, max(1, int(round(len(token_ids) * masked_lm_prob))))

    token_masked_lm_positions = []
    # 长尾实体mask
    # 长尾实体多跳
    tmp = list(i for i in range(1, len(token_ids) - 1))
    shuffle(tmp)
    for t in tmp:
        if (total_num_to_mask <= 0): break
        if (t not in token_masked_lm_positions):
            token_masked_lm_positions.append(t)
            total_num_to_mask -= 1

    masked_label_ids = []
    for pos in token_masked_lm_positions:
        masked_label_ids.append(token_ids[pos])
        masked_token = 50264
        if (random() < .2):
            masked_token = token_ids[pos]
            if (random() < .5):
                masked_token = randint(3, len(vocab) - 100)
        token_ids[pos] = masked_token
    entity_qid, entity_pos = [], []
    return token_ids, token_masked_lm_positions, masked_label_ids

def create_wwm_lm_predictions(args, example, masked_lm_prob, max_predictions_per_seq, vocab, tokenizer, data_type):
    """Creates the predictions for the masked LM objective. This is mostly copied from the Google BERT repo, but
    with several refactors to clean it up and remove a lot of unnecessary variables."""
    token_ids = example['token_ids']
    token_ids = [0] + token_ids + [2]

    new_entity_pos = []
    entity_qid = example['entity_qid']
    entity_pos = example['entity_pos']  # 不包括最后一个位置的token
    temp_tmp = list(zip(entity_pos, entity_qid))
    tmp = []
    if len(temp_tmp) == 0:
        pass
    elif len(temp_tmp) == 1:
        tmp.append(temp_tmp[0])
    else:
        tmp.append(temp_tmp[0])
        count_pos = temp_tmp[0][0][1]
        for i in range(1, len(temp_tmp), 1):
            item = temp_tmp[i]
            entityPos, entityQname = item[0], item[1]
            start_ent, end_ent = entityPos[0], entityPos[1]
            if start_ent - count_pos >= 5:
                tmp.append(item)
                count_pos = end_ent
            else:
                continue
    # 长尾实体mask
    # for item in temp_tmp:
    #     entityQname = item[1]
    #     if entityQname in long_tail_entity_frq.keys():
    #         entity_frq = long_tail_entity_frq[entityQname]
    #         if int(entity_frq) < args.long_tail_entity_frq_ths:
    #             tmp.append(item)
    #     else:
    #         tmp.append(item)
    # 低频实体多跳取概率

    # 如果有实体，直接mask实体，不mask token
    if len(tmp) > 0:
        total_num_to_mask = min(max_predictions_per_seq, max(1, int(round(len(token_ids) * masked_lm_prob))))
        entity_masked_lm_positions = []
        for select in range(len(entity_pos)):
            if len(entity_masked_lm_positions) > total_num_to_mask:
                break
            else:
                entity_masked_lm_positions.extend(j for j in range(entity_pos[select][0], entity_pos[select][1]))

        masked_label_ids = []
        for pos in entity_masked_lm_positions:
            masked_label_ids.append(token_ids[pos])
            masked_token = 50264
            if (random() < .2):
                masked_token = token_ids[pos]
                if (random() < .5):
                    masked_token = randint(3, len(vocab) - 100)
            token_ids[pos] = masked_token
        return token_ids, entity_masked_lm_positions, masked_label_ids, entity_qid, entity_pos
    else:
        total_num_to_mask = min(max_predictions_per_seq, max(1, int(round(len(token_ids) * masked_lm_prob))))
        token_masked_lm_positions = []
        tmp = list(i for i in range(1, len(token_ids) - 1))
        shuffle(tmp)
        for t in tmp:
            if (total_num_to_mask <= 0): break
            if (t not in token_masked_lm_positions):
                token_masked_lm_positions.append(t)
                total_num_to_mask -= 1

        masked_label_ids = []
        for pos in token_masked_lm_positions:
            masked_label_ids.append(token_ids[pos])
            masked_token = 50264
            if (random() < .2):
                masked_token = token_ids[pos]
                if (random() < .5):
                    masked_token = randint(3, len(vocab) - 100)
            token_ids[pos] = masked_token
        return token_ids, token_masked_lm_positions, masked_label_ids, entity_qid, entity_pos


    # 一半儿给token级别，一半儿给entity 级别。总之mask的token总的数量不能太多，继续15%。
    # 需要进行调整，有可能rare entity没有，那么就要扩大token mask的数量，保持15%不变
    # total_entity_len = 0
    # if len(tmp) > 0:
    #     for item in tmp:
    #         total_entity_len += (item[0][1] - item[0][0])
    # # 如果有实体，那就只mask实体，否则就mask token
    # if total_entity_len > 0:
    #     token_num_to_mask = total_entity_len
    # else:
    #     token_num_to_mask = total_num_to_mask
    #
    # if total_entity_len > 0:
    #     # if (random() < .75):
    #     #     tmp = sorted(tmp, key=lambda x: qid2frq[x[1]])
    #     # else:
    #     shuffle(tmp)
    #     entity_pos, entity_qid = zip(*tmp)
    #
    #     for index, qid in enumerate(entity_qid):
    #         s, e = entity_pos[index]
    #         new_entity_pos.append((s + 1, e + 1))
    #     entity_pos = new_entity_pos
    # else:
    #     entity_pos, entity_qid = [], []
    #
    # # Mask机制处理：（1）普通token的MLM，（2）对于Entity的Mask,（3）拼接上SPO以后的Mask
    # # 验证后，拼接上SPO以后的Mask 这个没啥用！
    # # spo_token_ids, rel_mask_pos = concate_spo(copy.deepcopy(token_ids), entity_qid, args, tokenizer)
    # if args.mask_method == 'ori_mlm':
    #     if total_entity_len > 0:
    #         entity_num_to_mask = total_num_to_mask
    #     else:
    #         entity_num_to_mask = 0
    #     # entity_num_to_mask = total_num_to_mask - token_num_to_mask
    # elif args.mask_method == 'ori_mlm_ent_decoder':
    #     # 将entity全部mask, 不控制数量
    #     entity_num_to_mask = 1000000
    # else:
    #     raise ValueError('no mask method implementation!')
    #
    # if total_entity_len == 0:
    #     token_masked_lm_positions = []
    #
    #     tmp = list(i for i in range(1, len(token_ids) - 1))
    #     shuffle(tmp)
    #     # （1）剔除掉实体内部的token, 让token级别的mask全部是token级别的
    #     for entity_pos_i in entity_pos:
    #         start, end = entity_pos_i[0], entity_pos_i[1]
    #         for index_ent_token in range(start, end, 1):
    #             tmp.remove(index_ent_token)
    #
    #     for t in tmp:
    #         if (token_num_to_mask <= 0): break
    #         if (t not in token_masked_lm_positions):
    #             token_masked_lm_positions.append(t)
    #             token_num_to_mask -= 1
    # else:
    #     token_masked_lm_positions = []
    #
    # # (2) Entity Mask 实体中的所有token全部Mask掉
    # entity_masked_lm_positions = []
    # for select in range(len(entity_pos)):
    #     if len(entity_masked_lm_positions) > entity_num_to_mask:
    #         break
    #     else:
    #         entity_masked_lm_positions.extend(j for j in range(entity_pos[select][0], entity_pos[select][1]))
    #
    # if args.mask_method == 'ori_mlm':
    #     masked_lm_positions = token_masked_lm_positions + entity_masked_lm_positions
    #     # for item in rel_mask_pos:
    #     #     start_rel_mask, end_rel_mask = item
    #     #     for mask_rel_index in range(start_rel_mask, end_rel_mask+1, 1):
    #     #         masked_lm_positions.append(mask_rel_index)
    #     masked_label_ids = []
    #     for pos in masked_lm_positions:
    #         masked_label_ids.append(token_ids[pos])
    #         masked_token = 50264
    #         if(random()<.2):
    #             masked_token = token_ids[pos]
    #             if(random()<.5):
    #                 masked_token = randint(3,len(vocab)-100)
    #         token_ids[pos] = masked_token
    # # return token_ids, masked_lm_positions, masked_label_ids,entity_qid,entity_pos,entity2word_pos,entity2word_ids
    #     return token_ids, masked_lm_positions, masked_label_ids, entity_qid, entity_pos
    # elif args.mask_method == 'ori_mlm_ent_decoder':
    #     token_masked_label_ids = []
    #     for pos in token_masked_lm_positions:
    #         token_masked_label_ids.append(token_ids[pos])
    #         masked_token = 50264
    #         if (random() < .2):
    #             masked_token = token_ids[pos]
    #             if (random() < .5):
    #                 masked_token = randint(3, len(vocab) - 100)
    #         token_ids[pos] = masked_token
    #
    #     relation_masked_label_ids = []
    #     relation_masked_lm_positions = []
    #     # for item in rel_mask_pos:
    #     #     start_rel_mask, end_rel_mask = item
    #     #     for mask_rel_index in range(start_rel_mask, end_rel_mask+1, 1):
    #     #         relation_masked_label_ids.append(spo_token_ids[mask_rel_index + 1])
    #     #         relation_masked_lm_positions.append(mask_rel_index)
    #     #         masked_token = 50264
    #     #         if (random() < .2):
    #     #             masked_token = spo_token_ids[mask_rel_index]
    #     #             if (random() < .5):
    #     #                 masked_token = randint(3, len(vocab) - 100)
    #     #         spo_token_ids[mask_rel_index] = masked_token
    #
    #     entity_masked_label_ids = []
    #     for pos in entity_masked_lm_positions:
    #         entity_masked_label_ids.append(token_ids[pos])
    #         masked_token = 50264
    #         if (random() < .2):
    #             masked_token = token_ids[pos]
    #             if (random() < .5):
    #                 masked_token = randint(3, len(vocab) - 100)
    #         token_ids[pos] = masked_token
    #     return token_ids, token_masked_lm_positions+entity_masked_lm_positions+relation_masked_lm_positions, \
    #            token_masked_label_ids+entity_masked_label_ids+relation_masked_label_ids, entity_qid,entity_pos
    # else:
    #     raise ValueError('No this mask method implementation!')

    # masked_label_ids = []
    # for pos in masked_lm_positions:
    #     masked_label_ids.append(token_ids[pos])
    #     masked_token = 50264
    #     # print(masked_token)
    #     if (random() < .2):
    #         masked_token = token_ids[pos]
    #         if (random() < .5):
    #             masked_token = randint(3, len(vocab) - 100)
    #     token_ids[pos] = masked_token
    # return token_ids, masked_lm_positions, masked_label_ids, entity_qid, entity_pos

# def pos_dict(l,max_pre_line):
#     return [ j+i*max_pre_line for i in range(len(l)) for j in range(len(l[i])) ]

# def convert_examples_to_features(args, examples, max_seq_length, tokenizer, do_gc=False):
#     features = []
#     example_num = len(examples)
#     names_list = []
#     save_pre_step = max(int(.25 * example_num), 1)
#     # print(save_pre_step)
#     # example = {
#     #         "tokens": tokens,
#     #         "segment_ids": segment_ids,
#     #         "masked_lm_positions": masked_lm_positions,
#     #         "masked_lm_labels": masked_lm_labels,
#     #         "entiy_ids": entiy_ids,
#     #         'sop_label':sop_label
#     #         }
#     for f_index in tqdm(range(example_num), desc="Converting Feature"):
#         #    for i, example in enumerate(examples):
#         # print(f_index)
#         example = examples[-1]
#         tokens = example["tokens"]
#         segment_ids = example["segment_ids"]
#         masked_lm_positions = example["masked_lm_positions"]
#         masked_lm_labels = example["masked_lm_labels"]
#         entity_ids_mapping = example["entity_ids_mapping"]
#         entity_ids_mapping_mask = example["entity_ids_mapping_mask"]
#
#         add_default_value = args.max_seq_length - len(entity_ids_mapping)
#         for _ in range(add_default_value):
#             number_hop_list = [-1 for _ in range(args.two_hop_entity_num)]
#             entity_ids_mapping.append(number_hop_list)
#             number_default_list = [0 for _ in range(args.two_hop_entity_num)]
#             entity_ids_mapping_mask.append(number_default_list)
#         assert len(entity_ids_mapping) == args.max_seq_length
#         assert len(entity_ids_mapping_mask) == args.max_seq_length
#
#         entity_ids_mapping = np.array(entity_ids_mapping)
#         entity_ids_mapping_mask = np.array(entity_ids_mapping_mask)
#
#         entiy_ids = example["entiy_ids"]
#         sop_label = example['sop_label']
#         # print(list(zip(tokens,range(len(tokens)))))
#
#         assert len(tokens) == len(segment_ids) <= max_seq_length  # The preprocessed data should be already truncated
#
#         input_ids = tokenizer.convert_tokens_to_ids(tokens)
#
#         masked_label_ids = tokenizer.convert_tokens_to_ids(masked_lm_labels)
#         assert (len(masked_label_ids) == len(masked_lm_positions))
#         input_array = np.zeros(max_seq_length, dtype=np.int)
#         input_array[:len(input_ids)] = input_ids
#
#         mask_array = np.zeros(max_seq_length, dtype=np.int)
#         mask_array[:len(input_ids)] = 1
#
#         segment_array = np.zeros(max_seq_length, dtype=np.int)
#         segment_array[:len(segment_ids)] = segment_ids
#
#         lm_label_array = np.full(max_seq_length, dtype=np.int, fill_value=-1)
#         lm_label_array[masked_lm_positions] = masked_label_ids
#
#         entity_array = np.full(max_seq_length, dtype=np.int, fill_value=-1)
#         entity_array[:len(entiy_ids)] = entiy_ids
#         # seg_label_array = np.full(max_seq_length, dtype=np.int, fill_value=-1)
#         # seg_label_array[seg_positions] = seg_labels
#
#         feature = InputFeatures(input_ids=input_array,
#                                 input_mask=mask_array,
#                                 segment_ids=segment_array,
#                                 label_id=lm_label_array,
#                                 entiy_ids=entity_array,
#                                 entity_mapping=entity_ids_mapping,
#                                 entity_mapping_mask=entity_ids_mapping_mask,
#                                 sop_label=sop_label)
#         features.append(feature)
#         examples.pop()
#         del example
#         if (((f_index + 1) % save_pre_step) == 0 or (f_index + 1) == example_num):
#             print("Do Save There")
#             name = 'run_tmp/{}_f.pklf'.format(f_index)
#             sf = open(name, 'wb+')
#             pickle.dump(features, sf)
#             sf.close()
#             names_list.append(name)
#             features.clear()
#             del name
#     del features
#     features = []
#     examples = []
#     for name in tqdm(names_list, desc='Loading features'):
#         sf = open(name, 'rb')
#         f = pickle.load(sf)
#         sf.close()
#         features.extend(f)
#         del f
#     return features


def reduce_tensor(tensor, ws=2):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.reduce_op.SUM)
    rt /= ws
    return rt


def evaluate(args, model, eval_dataloader, device, epoch, train_loss, best_loss):
    torch.cuda.empty_cache()
    model.eval()
    eval_loss = 0
    nb_eval_steps = 0
    if (args.local_rank >= 0):
        torch.distributed.barrier()
    if (args.local_rank != -1):
        eval_dataloader.sampler.set_epoch(args.seed)
    for batch in tqdm(eval_dataloader, desc='Evaluation') if args.local_rank <= 0 else eval_dataloader:
        batch0 = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            all_input_ids, all_input_mask, all_segment_ids, all_label_ids, \
            knowledge_input_ids, knowledge_segment_ids, knowledge_input_masks, \
            knowledge_entity_start, knowledge_entity_end, knowledge_relation_start, \
            knowledge_relation_end, knowledge_add_sub_label, knowledge_label_mask, knowledge_pseudo_mask = batch0
            outputs = model(
                input_ids=all_input_ids,
                attention_mask=all_input_mask,
                token_type_ids=all_segment_ids,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                encoder_hidden_states=None,
                encoder_attention_mask=None,
                labels=all_label_ids,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=False
                # knowledge_input_ids=knowledge_input_ids,
                # knowledge_segment_ids=knowledge_segment_ids,
                # knowledge_input_masks=knowledge_input_masks,
                # knowledge_entity_start=knowledge_entity_start,
                # knowledge_entity_end=knowledge_entity_end,
                # knowledge_relation_start=knowledge_relation_start,
                # knowledge_relation_end=knowledge_relation_end,
                # knowledge_add_sub_label=knowledge_add_sub_label,
                # knowledge_label_mask=knowledge_label_mask,
                # knowledge_pseudo_masks=knowledge_add_sub_label
            )
            loss = outputs[0]
            if (args.local_rank >= 0):
                loss = reduce_tensor(loss, dist.get_world_size())
        eval_loss += loss.mean().item()
        nb_eval_steps += 1
    eval_loss = eval_loss / nb_eval_steps
    if best_loss > eval_loss:
        best_loss = eval_loss
        # save best model
        if args.fp16:
            if (args.local_rank <= 0):
                logger.info('**************************************************************************')
                model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                output_model_file = os.path.join(args.output_dir, "best_pytorch_model.bin")
                logger.info('Saving best Model into {}'.format(output_model_file))
                torch.save(model_to_save.state_dict(), output_model_file)
                logger.info('Saving best Model Done!')
                logger.info('**************************************************************************')
        else:
            # 保存普通训练的模型
            model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
            output_model_file = os.path.join(args.output_dir, "best_pytorch_model.bin")
            if (args.local_rank <= 0):
                logger.info('**************************************************************************')
                logger.info('Saving best Model into {}'.format(output_model_file))
                torch.save(model_to_save.state_dict(), output_model_file)
                logger.info('Saving best Model Done!')
                logger.info('**************************************************************************')

    if (args.local_rank <= 0):
        logger.info("============================ -epoch %d -train_loss %.4f -eval_loss %.4f -best_loss %.4f\n" % (epoch,
                                                                                                   train_loss,
                                                                                                   eval_loss,
                                                                                                   best_loss))
    torch.cuda.empty_cache()
    return best_loss

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--long_tail_entity_frq_ths", type=int, default=347,
                        help="less than 11 is a long tail entity, 77: 70%-30%, 347: 50%, 11: 90%")
    parser.add_argument("--mask_method", type=str, default='ori_mlm',
                        help="mask method, including original mlm and entity decoder")
    parser.add_argument("--pretrain_train_path", type=str, default=data_path,
                        help="pretrain train path to file")
    parser.add_argument("--pretrain_dev_path", type=str,
                        default=data_path,
                        help="pretrain dev path to file")
    parser.add_argument("--max_seq_length", type=int, default=128, help="max seq length of input sequences")
    parser.add_argument("--knowledge_text_len", type=int, default=20, help="max seq length of input knowledge text")

    parser.add_argument("--do_train", type=bool, default=True, help="If do train")
    parser.add_argument("--do_lower_case", type=bool, default=True, help="If do case lower")
    parser.add_argument("--train_batch_size", type=int, default=25, help="train_batch_size")  # May Need to finetune
    parser.add_argument("--eval_batch_size", type=int, default=96, help="eval_batch_size")
    parser.add_argument("--num_train_epochs", type=int, default=2, help="num_train_epochs")
    parser.add_argument("--learning_rate", type=float, default=4e-5, help="learning rate")  # May Need to finetune
    parser.add_argument("--warmup_proportion", type=float, default=.09,
                        help="warmup_proportion")  # May Need to finetune
    parser.add_argument("--no_cuda", type=bool, default=False, help="prevent use GPU")
    parser.add_argument("--local_rank", type=int, default=-1, help="If we are using cluster for training")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="gradient_accumulation_steps")  # May Need to finetune
    parser.add_argument("--fp16", type=bool, default=False, help="If use apex to train")
    parser.add_argument("--loss_scale", type=int, default=0, help="loss_scale")
    parser.add_argument("--bert_config_json", type=str, default="roberta_base/bert_config.json",
                        help="bert_config_json")
    parser.add_argument("--vocab_file", type=str, default="roberta_base/vocab.txt",
                        help="Path to vocab file")
    parser.add_argument("--output_dir", type=str,
                        default="./pretrain_out_rare_entity_v3",
                        help="output_dir")
    parser.add_argument("--masked_lm_prob", type=float, default=0.15, help="masked_lm_prob")
    parser.add_argument("--max_predictions_per_seq", type=int, default=72, help="max_predictions_per_seq")
    parser.add_argument("--cache_dir", type=str, default='roberta_base', help="cache_dir")
    parser.add_argument("--model_name_or_path", type=str, default="/apsarapangu/disk3/zhangtaolin.ztl/aaai22_v3/roberta_base", help="model_name_or_path")
    parser.add_argument('--eval_pre_step', type=float, default=.196,
                        help="The percent of how many train with one eval run")
    # parser.add_argument('--finetune_proportion', type=float, default=.05,
    #                     help="Detemind the proportion of the first training stage")

    args = parser.parse_args()

    # 获取实体相关信息
    # node2entity, combine_entity_type_dict,entityOutNegbhor,entityInNegbhor = entity_info(args)
    # type2id,type_embedd = entity_type_initialize(combine_entity_type_dict)
    # print('My local rank: {}'.format(args.local_rank))
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        torch.distributed.init_process_group(backend='nccl')
        assert (torch.distributed.get_world_size() >= 2)
    
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)

    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
        raise ValueError('Output dir is not empty!')
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    set_seed(args.seed)
    tokenizer = RobertaTokenizer.from_pretrained(pretrained_model_name_or_path=args.model_name_or_path)
    # set dataset
    if args.do_train:
        train_dataset = OurENRIEDataset(args=args,
                                        data_path=args.pretrain_train_path,
                                        max_seq_length=args.max_seq_length,
                                        masked_lm_prob=args.masked_lm_prob,
                                        max_predictions_per_seq=args.max_predictions_per_seq,
                                        tokenizer=tokenizer, data_type='train')

        num_train_optimization_steps = int(len(train_dataset) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs

        eval_dataset = OurENRIEDataset(args=args,
                                       data_path=args.pretrain_dev_path,
                                       max_seq_length=args.max_seq_length,
                                       masked_lm_prob=args.masked_lm_prob,
                                       max_predictions_per_seq=args.max_predictions_per_seq,
                                       tokenizer=tokenizer, data_type='dev')

        if (args.local_rank != -1):
            train_sampler = DistributedSampler(train_dataset, shuffle=False)
            eval_sampler = DistributedSampler(eval_dataset, shuffle=False)

        else:
            train_sampler = RandomSampler(train_dataset)
            eval_sampler = SequentialSampler(eval_dataset)
            print('single gpu sampler!!!')
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler,
                                      batch_size=args.train_batch_size, num_workers=3, shuffle=False)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, num_workers=3, shuffle=False)
    else:
        raise ValueError('Not Training Model! Please set the do_train=True!')

    # Prepare model
    missing_keys = set()
    model = RobertaForMaskedLM.from_pretrained(pretrained_model_name_or_path=args.model_name_or_path)
    model.to(device)

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

    new_add_param = [(n, p) for n, p in param_optimizer if n in missing_keys]
    pretrain_parm = [(n, p) for n, p in param_optimizer if n not in missing_keys]

    new_optimizer_grouped_parameters = [
        {'params': [p for n, p in new_add_param if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in new_add_param if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    old_optimizer_grouped_parameters = [
        {'params': [p for n, p in pretrain_parm if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in pretrain_parm if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(new_optimizer_grouped_parameters, lr=args.learning_rate)
    n_gpu = max(n_gpu, 1)

    for g in old_optimizer_grouped_parameters:
        optimizer.add_param_group(g)

    if (args.fp16):
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
    if args.local_rank != -1:
        model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    if (args.local_rank >= 0):
        torch.distributed.barrier()

    if args.local_rank != -1:
        num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

    # optimizer = None
    # scheduler = None
    # if args.fp16:
    #     try:
    #         from apex.optimizers import FP16_Optimizer
    #         from apex.optimizers import FusedAdam
    #     except ImportError:
    #         raise ImportError(
    #             "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")
    #     optimizer = FusedAdam(new_optimizer_grouped_parameters,
    #                           lr=args.learning_rate,
    #                           bias_correction=False,
    #                           max_grad_norm=1.0)
    #     if args.loss_scale == 0:
    #         optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
    #     else:
    #         optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)
    # else:

    # for g in old_optimizer_grouped_parameters:
    #     optimizer.add_param_group(g)

    # if (args.local_rank >= 0):
    #     torch.distributed.barrier()

    global_step = 0
    best_loss = 100000
    import time
    if args.do_train:

        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_dataset))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)
        model.train()
        import datetime
        fout = None
        if (args.local_rank <= 0):
            writer = SummaryWriter('TensorBoard_roberta_base_logs/tensorboard_loss_output')
            fout = open(os.path.join(args.output_dir, "model_start_time.{}".format(datetime.datetime.now())), 'w')

        total_train_step = len(train_dataloader) * args.num_train_epochs
        total_eval_step = int(len(eval_dataset) / args.eval_batch_size)
        if args.local_rank <= 0:
            logger.info('In DP/DDP, Total step:{}, Eval step: {}'.format(total_train_step, total_eval_step))

        # DDP和DP优化步骤数目不同
        if args.local_rank != -1:
            scheduler = get_linear_schedule_with_warmup(optimizer,
                                    args.warmup_proportion * num_train_optimization_steps*args.num_train_epochs,
                                                    num_train_optimization_steps*args.num_train_epochs)
        else:
            scheduler = get_linear_schedule_with_warmup(optimizer,
                                                        args.warmup_proportion * total_train_step,
                                                        total_train_step)
        tr_loss = 0
        loss_step = 0
        batch_loss = 0
        total_time = []
        total_size = 0
        for epoch in range(int(args.num_train_epochs)):
            if (args.local_rank != -1):
                train_dataloader.sampler.set_epoch(epoch)
            for step, batch in enumerate(train_dataloader):
                batch0 = tuple(t.to(device) for t in batch)
                all_input_ids, all_input_mask, all_segment_ids, all_label_ids, \
                knowledge_input_ids, knowledge_segment_ids, knowledge_input_masks, \
                knowledge_entity_start, knowledge_entity_end, knowledge_relation_start, \
                knowledge_relation_end, knowledge_add_sub_label, knowledge_label_mask, \
                knowledge_pseudo_mask = batch0
                total_size += all_input_ids.size(0)
                start_time = time.time()
                outputs = model(
                    input_ids=all_input_ids,
                    attention_mask=all_input_mask,
                    # token_type_ids=all_segment_ids,
                    position_ids=None,
                    head_mask=None,
                    inputs_embeds=None,
                    encoder_hidden_states=None,
                    encoder_attention_mask=None,
                    labels=all_label_ids,
                    output_attentions=None,
                    output_hidden_states=None,
                    return_dict=False
                    # knowledge_input_ids=knowledge_input_ids,
                    # knowledge_segment_ids=knowledge_segment_ids,
                    # knowledge_input_masks=knowledge_input_masks,
                    # knowledge_entity_start=knowledge_entity_start,
                    # knowledge_entity_end=knowledge_entity_end,
                    # knowledge_relation_start=knowledge_relation_start,
                    # knowledge_relation_end=knowledge_relation_end,
                    # knowledge_add_sub_label=knowledge_add_sub_label,
                    # knowledge_label_mask=knowledge_label_mask,
                    # knowledge_pseudo_masks=knowledge_pseudo_mask
                )

                loss = outputs[0]
                if n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                batch_loss += loss.mean().item()
                optimizer.zero_grad()
                if args.fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()
                end_time = time.time()
                total_time.append(end_time - start_time)
                tr_loss += float(loss.item() * args.gradient_accumulation_steps)
                if args.local_rank <= 0:
                    logger.info("epoch: {}   Step: {} / {}   loss: {}".format(epoch,
                                                                              step,
                                                                              len(train_dataloader),
                                                                              loss.item()))
                loss_step += 1
                scheduler.step()
                if (step + 1) % args.gradient_accumulation_steps == 0 or (step+1) == len(train_dataloader):
                    continue
                    # if args.fp16:
                    #     # modify learning rate with special warm up BERT uses
                    #     # if args.fp16 is False, BertAdam is used that handles this automatically
                    #     lr_this_step = args.learning_rate * warmup_linear(global_step / num_train_optimization_steps,
                    #                                                       args.warmup_proportion)
                    #     for param_group in optimizer.param_groups:
                    #         param_group['lr'] = lr_this_step
                    grad = torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
                    if args.local_rank <= 0:
                        logger.info("epoch: {}   Step: {} / {}   grad: {}".format(epoch,
                                                                                  step,
                                                                                  len(train_dataloader),
                                                                                  grad))
                    if args.local_rank <= 0:
                        writer.add_scalar('LOSS', batch_loss, global_step=global_step)
                    batch_loss = 0
                    # if grad < GRADIENT_SKIP or global_step < USE_GRAD_CLIP_UNTIL:
                    #     optimizer.step()
                    # else:
                    #     print('**********************klfjdska**********************')
                    optimizer.step()

                    global_step += 1

                    if (step + 1) % 12800 == 0:
                        # save checkpoint
                        if args.fp16:
                            if args.local_rank <= 0:
                                logger.info('**************************************************************************')
                                model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                                output_model_file = os.path.join(args.output_dir, str(global_step)+ "_pytorch_model.bin")
                                logger.info('Saving checkpoint into {}'.format(output_model_file))
                                torch.save(model_to_save.state_dict(), output_model_file)
                                logger.info('Saving checkpoint Done!')
                                logger.info('**************************************************************************')
                        else:
                            # 保存普通训练的模型
                            model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                            output_model_file = os.path.join(args.output_dir, str(global_step) + "_pytorch_model.bin")
                            if (args.local_rank <= 0):
                                logger.info('**************************************************************************')
                                logger.info('Saving checkpoint into {}'.format(output_model_file))
                                torch.save(model_to_save.state_dict(), output_model_file)
                                logger.info('Saving checkpoint Done!')
                                logger.info('**************************************************************************')
                        # evaluation and save current best model
                        best_loss = evaluate(args, model, eval_dataloader, device, epoch, tr_loss / loss_step, best_loss)
                        if (args.local_rank <= 0):
                            logger.info('Eval Done')
                        tr_loss = 0
                        loss_step = 0
        print('total time: {}'.format(sum(total_time)))
        if (args.local_rank <= 0):
            logger.info('Training Done!')
        # fout.close()

if __name__ == "__main__":
    main()

