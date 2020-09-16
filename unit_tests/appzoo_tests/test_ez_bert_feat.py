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
import subprocess
import unittest
from scipy import spatial

EPS = 1e-2

def cosine_similarity(vec1, vec2):
    return 1 - spatial.distance.cosine(vec1, vec2)


def equal_2d_array(str1, str2):
    arr1 = str1.split(";")
    arr2 = str2.split(";")
    min_len = min(len(arr1), len(arr2))
    arr1 = arr1[:min_len]
    arr2 = arr2[:min_len]
    for lst_1, lst_2 in zip(arr1, arr2):
        flag = equal_1d_array(lst_1, lst_2)
        if not flag:
            print(lst_1[:50])
            print(lst_2[:50])
            return False
    return True


def equal_1d_array(str1, str2):
    lst_1 = [float(t) for t in str1.split(",")]
    lst_2 = [float(t) for t in str2.split(",")]
    if len(lst_1) != len(lst_2):
        print("1D array Length not matched")
        return False
    sim = cosine_similarity(lst_1, lst_2)
    if abs(1 - sim) > EPS:
        print("Similarity: {}".format(sim))
        return False
    return True


class TestEzBertFeat(unittest.TestCase):
    def test_one_seq_from_checkpoints_encode(self):
        argvs = ['ez_bert_feat',
                 '--inputTable', '../ut_data/ez_bert_feat/one.seq.txt',
                 '--outputTable', 'ez_bert_feat.one.seq.results.txt',
                 '--inputSchema', 'example_id:int:1,query1:str:1,label:str:1,category:str:1,score:float:1',
                 '--firstSequence', 'query1',
                 '--appendCols', 'example_id,label,category,score,xxx',
                 '--outputSchema', 'pool_output,first_token_output,all_hidden_outputs',
                 '--modelName', './google-bert-base-zh/model.ckpt',
                 '--sequenceLength', '50',
                 '--batchSize', '1']
        cmd = ' '.join(argvs)
        print(cmd)
        res = subprocess.check_output(' '.join(argvs), stderr=subprocess.STDOUT, shell=True)
        print(res)
        with open("../ut_data/ez_bert_feat/one.seq.txt") as f:
            grt_cnt = len(f.readlines())

        with open("../ut_data/ez_bert_feat/one.seq.hf.results.txt") as grt_inp, \
            open("ez_bert_feat.one.seq.results.txt") as pred_inp:
            cnt = 0
            for i, (grt_line, pred_line) in enumerate(zip(grt_inp, pred_inp)):
                g_pool_output, g_first_token_output, g_all_hidden_output, \
                g_example_id, g_label, g_category, g_score = grt_line.strip().split("\t")
                p_pool_output, p_first_token_output, p_all_hidden_output, \
                p_example_id, p_label, p_category, p_score = pred_line.strip().split("\t")
                if not equal_1d_array(g_pool_output, p_pool_output):
                    print("[LINE {}] Pool output not matched".format(i))
                    break
                if not equal_1d_array(g_first_token_output, p_first_token_output):
                    print("[LINE {}] First token output not matched".format(i))
                    break
                if not equal_2d_array(g_all_hidden_output, p_all_hidden_output):
                    print("[LINE {}] All hidden outputs not matched".format(i))
                    break
                if not (g_example_id, g_label, g_category) == (p_example_id, p_label, p_category)\
                        and abs(float(p_score) - float(g_score)) > EPS:
                    print((g_example_id, g_label, g_category, g_score))
                    print((p_example_id, p_label, p_category, p_score))
                    print("[LINE {}] Append columns not matched".format(i))
                    break
                cnt += 1
        self.assertTrue(cnt == grt_cnt)
        os.remove('ez_bert_feat.one.seq.results.txt')

    def test_one_seq_encode(self):
        argvs = ['ez_bert_feat',
                 '--inputTable', '../ut_data/ez_bert_feat/one.seq.txt',
                 '--outputTable', 'ez_bert_feat.one.seq.results.txt',
                 '--inputSchema', 'example_id:int:1,query1:str:1,label:str:1,category:str:1,score:float:1',
                 '--firstSequence', 'query1',
                 '--appendCols', 'example_id,label,category,score,xxx',
                 '--outputSchema', 'pool_output,first_token_output,all_hidden_outputs',
                 '--modelName', 'google-bert-base-zh',
                 '--sequenceLength', '50',
                 '--batchSize', '1']
        cmd = ' '.join(argvs)
        print(cmd)
        res = subprocess.check_output(' '.join(argvs), stderr=subprocess.STDOUT, shell=True)
        print(res)
        with open("../ut_data/ez_bert_feat/one.seq.txt") as f:
            grt_cnt = len(f.readlines())

        grt_dict = {}
        pred_dict = {}
        with open("../ut_data/ez_bert_feat/one.seq.hf.results.txt") as grt_inp, \
            open("ez_bert_feat.one.seq.results.txt") as pred_inp:
            for i, (grt_line, pred_line) in enumerate(zip(grt_inp, pred_inp)):
                g_pool_output, g_first_token_output, g_all_hidden_output, \
                g_example_id, g_label, g_category, g_score = grt_line.strip().split("\t")
                p_pool_output, p_first_token_output, p_all_hidden_output, \
                p_example_id, p_label, p_category, p_score = pred_line.strip().split("\t")
                grt_dict[g_example_id] = (g_pool_output, g_first_token_output, g_all_hidden_output,
                                          g_example_id, g_label, g_category, g_score)
                pred_dict[p_example_id] = (p_pool_output, p_first_token_output, p_all_hidden_output,
                                           p_example_id, p_label, p_category, p_score)

        cnt = 0
        for i, example_id in enumerate(grt_dict):
            g_pool_output, g_first_token_output, g_all_hidden_output, \
            g_example_id, g_label, g_category, g_score = grt_dict[example_id]
            p_pool_output, p_first_token_output, p_all_hidden_output, \
            p_example_id, p_label, p_category, p_score = pred_dict[example_id]
            if not equal_1d_array(g_pool_output, p_pool_output):
                print("[LINE {}] Pool output not matched".format(i))
                break
            if not equal_1d_array(g_first_token_output, p_first_token_output):
                print("[LINE {}] First token output not matched".format(i))
                break
            if not equal_2d_array(g_all_hidden_output, p_all_hidden_output):
                print("[LINE {}] All hidden outputs not matched".format(i))
                break
            if not (g_example_id, g_label, g_category) == (p_example_id, p_label, p_category)\
                    and abs(float(p_score) - float(g_score)) > EPS:
                print((g_example_id, g_label, g_category, g_score))
                print((p_example_id, p_label, p_category, p_score))
                print("[LINE {}] Append columns not matched".format(i))
                break
            cnt += 1
        self.assertTrue(cnt == grt_cnt)
        os.remove('ez_bert_feat.one.seq.results.txt')

    def test_two_seq_encode(self):
        argvs = ['ez_bert_feat',
                 '--inputTable', '../ut_data/ez_bert_feat/two.seq.txt',
                 '--outputTable', 'ez_bert_feat.two.seq.ez_transfer.results.txt',
                 '--inputSchema', 'example_id:int:1,query1:str:1,query2:str:1,category:str:1,score:float:1',
                 '--firstSequence', 'query1',
                 '--secondSequence', 'query2',
                 '--appendCols', 'example_id,category,score',
                 '--outputSchema', 'pool_output,first_token_output,all_hidden_outputs',
                 '--modelName', './google-bert-base-zh/',
                 '--sequenceLength', '50',
                 '--batchSize', '1']
        cmd = ' '.join(argvs)
        print(cmd)
        res = subprocess.check_output(' '.join(argvs), stderr=subprocess.STDOUT, shell=True)
        print(res)
        with open("../ut_data/ez_bert_feat/two.seq.txt") as f:
            grt_cnt = len(f.readlines())

        grt_dict = {}
        pred_dict = {}
        with open("../ut_data/ez_bert_feat/two.seq.hf.results.txt") as grt_inp, \
            open("ez_bert_feat.two.seq.ez_transfer.results.txt") as pred_inp:
            for i, (grt_line, pred_line) in enumerate(zip(grt_inp, pred_inp)):
                g_pool_output, g_first_token_output, g_all_hidden_output, \
                g_example_id, g_category, g_score = grt_line.strip().split("\t")
                p_pool_output, p_first_token_output, p_all_hidden_output, \
                p_example_id, p_category, p_score = pred_line.strip().split("\t")
                grt_dict[g_example_id] = (g_pool_output, g_first_token_output, g_all_hidden_output,
                                          g_example_id, g_category, g_score)
                pred_dict[p_example_id] = (p_pool_output, p_first_token_output, p_all_hidden_output,
                                           p_example_id, p_category, p_score)

        cnt = 0
        for i, example_id in enumerate(grt_dict):
            g_pool_output, g_first_token_output, g_all_hidden_output, \
            g_example_id, g_category, g_score = grt_dict[example_id]
            p_pool_output, p_first_token_output, p_all_hidden_output, \
            p_example_id, p_category, p_score = pred_dict[example_id]
            if not equal_1d_array(g_pool_output, p_pool_output):
                print("[LINE {}] Pool output not matched".format(i))
                break
            if not equal_1d_array(g_first_token_output, p_first_token_output):
                print("[LINE {}] First token output not matched".format(i))
                break
            if not equal_2d_array(g_all_hidden_output, p_all_hidden_output):
                print("[LINE {}] All hidden outputs not matched".format(i))
                break
            if not (g_example_id, g_category) == (p_example_id, p_category) \
                    and abs(float(p_score) - float(g_score)) > EPS:
                print((g_example_id, g_category, g_score))
                print((p_example_id, p_category, p_score))
                print("[LINE {}] Append columns not matched".format(i))
            cnt += 1

        self.assertTrue(cnt == grt_cnt)
        os.remove('ez_bert_feat.two.seq.ez_transfer.results.txt')

if __name__ == '__main__':
    unittest.main()