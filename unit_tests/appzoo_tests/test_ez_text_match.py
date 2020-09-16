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
import shutil
import unittest

class TestEzTextMatch(unittest.TestCase):
    def test_1_train_bert(self):
        argvs = ['easy_transfer_app',
                 '--mode', 'train',
                 '--inputTable', '../ut_data/ez_text_match/train.csv,../ut_data/ez_text_match/dev.csv',
                 '--inputSchema', "example_id:int:1,query1:str:1,query2:str:1,is_same_question:str:1,category:str:1,score:float:1",
                 '--firstSequence', 'query1',
                 '--secondSequence', 'query2',
                 '--labelName', 'is_same_question',
                 '--labelEnumerateValues', '0,1',
                 '--checkpointDir', 'ez_text_match_models',
                 '--numEpochs', '1',
                 '--batchSize', '5',
                 '--saveCheckpointSteps', '10000',
                 '--optimizerType', 'adam',
                 '--learningRate', '2e-5',
                 '--modelName', 'text_match_bert',
                 '--distributionStrategy', 'MirroredStrategy',
                 '--workerCount', '1',
                 '--workerGPU', '1',
                 '--workerCPU', '1',
                 '--advancedParameters', """'
                            pretrain_model_name_or_path=google-bert-base-zh # For BERT
                            save_steps=300
                            log_step_count_steps=100
                            throttle_secs=100
                            keep_checkpoint_max=10
                            lr_decay=polynomial
                            warmup_ratio=0.1
                            weight_decay_ratio=0
                            gradient_clip=true
                            num_accumulated_batches=1
                            eval_batch_size=256
                            '
                    """
                 ]
        print(' '.join(argvs))
        try:
            res = subprocess.check_output(' '.join(argvs), stderr=subprocess.STDOUT, shell=True)
            print(res)
        except subprocess.CalledProcessError as e:
            print(e.output)
            raise RuntimeError

    def test_2_predict_mode(self):
        argvs = ['easy_transfer_app',
                 '--mode', 'predict',
                 '--inputTable', '../ut_data/ez_text_match/test.csv',
                 '--outputTable', 'ez_text_match.pred.csv',
                 '--inputSchema', 'example_id:int:1,query1:str:1,query2:str:1,category:str:1,score:float:1',
                 '--firstSequence', 'query1',
                 '--secondSequence', 'query2',
                 '--appendCols', 'example_id,category,score,query1',
                 '--outputSchema', 'predictions,probabilities,logits',
                 '--checkpointPath', 'ez_text_match_models',
                 '--batchSize', '100',
                 '--workerCount', '1',
                 '--workerGPU', '1',
                 '--workerCPU', '1',
                 ]
        print(' '.join(argvs))
        try:
            res = subprocess.check_output(' '.join(argvs), stderr=subprocess.STDOUT, shell=True)
            print(res)
        except subprocess.CalledProcessError as e:
            print(e.output)
            raise RuntimeError
        self.assertTrue(os.path.exists('ez_text_match.pred.csv'))
        os.remove('ez_text_match.pred.csv')

    def test_3_evaluate_mode(self):
        argvs = ['easy_transfer_app',
                 '--mode', 'evaluate',
                 '--inputTable', '../ut_data/ez_text_match/dev.csv',
                 '--checkpointPath', 'ez_text_match_models/model.ckpt-0',
                 '--batchSize', '100'
                 ]
        print(' '.join(argvs))
        try:
            res = subprocess.check_output(' '.join(argvs), stderr=subprocess.STDOUT, shell=True)
            print(res)
        except subprocess.CalledProcessError as e:
            print(e.output)
            raise RuntimeError

    def test_4_export_predict_mode(self):
        argvs = ['easy_transfer_app',
                 '--mode', 'export',
                 '--checkpointPath', 'ez_text_match_models/model.ckpt-0',
                 '--exportType', 'app_model',
                 '--exportDirBase', 'ez_text_match_models/saved_app_model',
                 ]
        print(' '.join(argvs))
        try:
            res = subprocess.check_output(' '.join(argvs), stderr=subprocess.STDOUT, shell=True)
            print(res)
        except subprocess.CalledProcessError as e:
            print(e.output)
            raise RuntimeError

        argvs = ['easy_transfer_app',
                 '--mode', 'predict',
                 '--inputTable', '../ut_data/ez_text_match/test.csv',
                 '--outputTable', 'ez_text_match.pred.csv',
                 '--inputSchema', 'example_id:int:1,query1:str:1,query2:str:1,category:str:1,score:float:1',
                 '--firstSequence', 'query1',
                 '--secondSequence', 'query2',
                 '--appendCols', 'example_id,category,score,query1',
                 '--outputSchema', 'predictions,probabilities,logits',
                 '--checkpointPath', 'ez_text_match_models/saved_app_model',
                 '--batchSize', '100',
                 '--workerCount', '1',
                 '--workerGPU', '1',
                 '--workerCPU', '1',
                 ]
        print(' '.join(argvs))
        try:
            res = subprocess.check_output(' '.join(argvs), stderr=subprocess.STDOUT, shell=True)
            print(res)
        except subprocess.CalledProcessError as e:
            print(e.output)
            raise RuntimeError
        self.assertTrue(os.path.exists('ez_text_match.pred.csv'))
        os.remove('ez_text_match.pred.csv')

    def test_5_ez_bert_feat(self):
        argvs = ['easy_transfer_app',
                 '--mode', 'export',
                 '--checkpointPath', 'ez_text_match_models/model.ckpt-0',
                 '--exportType', 'ez_bert_feat',
                 '--exportDirBase', 'ez_text_match_models/saved_model',
                 ]
        print(' '.join(argvs))
        try:
            res = subprocess.check_output(' '.join(argvs), stderr=subprocess.STDOUT, shell=True)
            print(res)
        except subprocess.CalledProcessError as e:
            print(e.output)
            raise RuntimeError
        argvs = ['ez_bert_feat',
                 '--inputTable', '../ut_data/ez_bert_feat/one.seq.txt',
                 '--outputTable', 'ez_text_match.feats.txt',
                 '--inputSchema', 'example_id:int:1,query1:str:1,label:str:1,category:str:1,score:float:1',
                 '--firstSequence', 'query1',
                 '--appendCols', 'example_id,label,category,score,xxx',
                 '--outputSchema', 'pool_output,first_token_output,all_hidden_outputs',
                 '--modelName', 'ez_text_match_models/model.ckpt-0',
                 '--sequenceLength', '50',
                 '--batchSize', '1']
        print(' '.join(argvs))
        try:
            res = subprocess.check_output(' '.join(argvs), stderr=subprocess.STDOUT, shell=True)
            print(res)
        except subprocess.CalledProcessError as e:
            print(e.output)
            raise RuntimeError
        self.assertTrue(os.path.exists('ez_text_match.feats.txt'))
        os.remove('ez_text_match.feats.txt')
        shutil.rmtree('ez_text_match_models', ignore_errors=True)

    def test_6_test_bert_two_tower(self):
        argvs = ['easy_transfer_app',
                 '--mode', 'train',
                 '--inputTable', '../ut_data/ez_text_match/train.csv,../ut_data/ez_text_match/dev.csv',
                 '--inputSchema', "example_id:int:1,query1:str:1,query2:str:1,is_same_question:str:1,category:str:1,score:float:1",
                 '--firstSequence', 'query1',
                 '--secondSequence', 'query2',
                 '--labelName', 'is_same_question',
                 '--labelEnumerateValues', '0,1',
                 '--checkpointDir', 'ez_text_match_two_tower_models',
                 '--numEpochs', '1',
                 '--batchSize', '5',
                 '--saveCheckpointSteps', '10000',
                 '--optimizerType', 'adam',
                 '--learningRate', '2e-5',
                 '--modelName', 'text_match_bert_two_tower',
                 '--distributionStrategy', 'MirroredStrategy',
                 '--workerCount', '1',
                 '--workerGPU', '1',
                 '--workerCPU', '1',
                 '--advancedParameters', """'
                            pretrain_model_name_or_path=google-bert-base-zh # For BERT
                            '
                    """
                 ]
        print(' '.join(argvs))
        try:
            res = subprocess.check_output(' '.join(argvs), stderr=subprocess.STDOUT, shell=True)
            print(res)
        except subprocess.CalledProcessError as e:
            print(e.output)
            raise RuntimeError

        argvs = ['easy_transfer_app',
                 '--mode', 'predict',
                 '--inputTable', '../ut_data/ez_text_match/test.csv',
                 '--outputTable', 'ez_text_match.pred.csv',
                 '--inputSchema', 'example_id:int:1,query1:str:1,query2:str:1,category:str:1,score:float:1',
                 '--firstSequence', 'query1',
                 '--secondSequence', 'query2',
                 '--appendCols', 'example_id,category,score,query1',
                 '--outputSchema', 'predictions,probabilities,logits',
                 '--checkpointPath', 'ez_text_match_two_tower_models/model.ckpt-0',
                 '--batchSize', '100',
                 '--workerCount', '1',
                 '--workerGPU', '1',
                 '--workerCPU', '1',
                 ]
        print(' '.join(argvs))
        try:
            res = subprocess.check_output(' '.join(argvs), stderr=subprocess.STDOUT, shell=True)
            print(res)
        except subprocess.CalledProcessError as e:
            print(e.output)
            raise RuntimeError
        self.assertTrue(os.path.exists('ez_text_match.pred.csv'))
        shutil.rmtree('ez_text_match_two_tower_models', ignore_errors=True)
        os.remove('ez_text_match.pred.csv')

    def test_7_train_dam(self):
        argvs = ['easy_transfer_app',
                 '--mode', 'train',
                 '--inputTable', '../ut_data/ez_text_match/train_tokenized.csv,../ut_data/ez_text_match/dev_tokenized.csv',
                 '--inputSchema', "query1:str:1,query2:str:1,query1_words:str:1,query2_words:str:1,"
                                  "query1_chars:str:1,query2_chars:str:1,label:str:1",
                 '--firstSequence', 'query1_words',
                 '--secondSequence', 'query2_words',
                 '--labelName', 'label',
                 '--labelEnumerateValues', '0,1',
                 '--checkpointDir', 'ez_text_match_dam_models',
                 '--numEpochs', '1',
                 '--batchSize', '5',
                 '--saveCheckpointSteps', '10000',
                 '--optimizerType', 'adam',
                 '--learningRate', '1e-3',
                 '--modelName', 'text_match_dam',
                 '--distributionStrategy', 'none',
                 '--workerCount', '1',
                 '--workerGPU', '1',
                 '--workerCPU', '1',
                 '--advancedParameters', """'
                            first_sequence_length=16
                            second_sequence_length=16
                            fix_embedding=false
                            max_vocab_size=100
                            embedding_size=20
                            hidden_size=10

                            lr_decay=polynomial
                            warmup_ratio=0.1
                            weight_decay_ratio=0
                            gradient_clip=true
                            clip_norm_value=1.0
                            throttle_secs=10
                            '
                    """
                 ]
        print(' '.join(argvs))
        try:
            res = subprocess.check_output(' '.join(argvs), stderr=subprocess.STDOUT, shell=True)
            print(res)
        except subprocess.CalledProcessError as e:
            print(e.output)
            raise RuntimeError

        argvs = ['easy_transfer_app',
                 '--mode', 'predict',
                 '--inputTable', '../ut_data/ez_text_match/test.csv',
                 '--outputTable', 'ez_text_match.pred.csv',
                 '--inputSchema', 'example_id:int:1,query1:str:1,query2:str:1,category:str:1,score:float:1',
                 '--firstSequence', 'query1',
                 '--secondSequence', 'query2',
                 '--appendCols', 'example_id,category,score,query1',
                 '--outputSchema', 'predictions,probabilities,logits',
                 '--checkpointPath', 'ez_text_match_dam_models/',
                 '--batchSize', '100',
                 '--workerCount', '1',
                 '--workerGPU', '1',
                 '--workerCPU', '1',
                 ]
        print(' '.join(argvs))
        try:
            res = subprocess.check_output(' '.join(argvs), stderr=subprocess.STDOUT, shell=True)
            print(res)
        except subprocess.CalledProcessError as e:
            print(e.output)
            raise RuntimeError
        self.assertTrue(os.path.exists('ez_text_match.pred.csv'))
        os.remove('ez_text_match.pred.csv')

    def test_8_export_predict_dam(self):
        argvs = ['easy_transfer_app',
                 '--mode', 'export',
                 '--checkpointPath', 'ez_text_match_dam_models/model.ckpt-0',
                 '--exportType', 'app_model',
                 '--exportDirBase', 'ez_text_match_dam_models/saved_app_model',
                 ]
        print(' '.join(argvs))
        try:
            res = subprocess.check_output(' '.join(argvs), stderr=subprocess.STDOUT, shell=True)
            print(res)
        except subprocess.CalledProcessError as e:
            print(e.output)
            raise RuntimeError

        argvs = ['easy_transfer_app',
                 '--mode', 'predict',
                 '--inputTable', '../ut_data/ez_text_match/test.csv',
                 '--outputTable', 'ez_text_match.pred.csv',
                 '--inputSchema', 'example_id:int:1,query1:str:1,query2:str:1,category:str:1,score:float:1',
                 '--firstSequence', 'query1',
                 '--secondSequence', 'query2',
                 '--appendCols', 'example_id,category,score,query1',
                 '--outputSchema', 'predictions,probabilities,logits',
                 '--checkpointPath', 'ez_text_match_dam_models/saved_app_model',
                 '--batchSize', '100',
                 '--workerCount', '1',
                 '--workerGPU', '1',
                 '--workerCPU', '1',
                 ]
        print(' '.join(argvs))
        try:
            res = subprocess.check_output(' '.join(argvs), stderr=subprocess.STDOUT, shell=True)
            print(res)
        except subprocess.CalledProcessError as e:
            print(e.output)
            raise RuntimeError
        self.assertTrue(os.path.exists('ez_text_match.pred.csv'))
        shutil.rmtree('ez_text_match_dam_models', ignore_errors=True)
        os.remove('ez_text_match.pred.csv')

    def test_9_regression_bert(self):
        argvs = ['easy_transfer_app',
                 '--mode', 'train',
                 '--inputTable', '../ut_data/ez_text_match/train.csv,../ut_data/ez_text_match/dev.csv',
                 '--inputSchema', "example_id:int:1,query1:str:1,query2:str:1,is_same_question:str:1,category:str:1,score:float:1",
                 '--firstSequence', 'query1',
                 '--secondSequence', 'query2',
                 '--labelName', 'score',
                 '--checkpointDir', 'ez_text_match_models',
                 '--numEpochs', '1',
                 '--batchSize', '5',
                 '--saveCheckpointSteps', '10000',
                 '--optimizerType', 'adam',
                 '--learningRate', '2e-5',
                 '--modelName', 'text_match_bert',
                 '--distributionStrategy', 'MirroredStrategy',
                 '--workerCount', '1',
                 '--workerGPU', '1',
                 '--workerCPU', '1',
                 '--advancedParameters', """'
                            pretrain_model_name_or_path=google-bert-base-zh # For BERT
                            save_steps=300
                            log_step_count_steps=100
                            throttle_secs=100
                            keep_checkpoint_max=10
                            lr_decay=polynomial
                            warmup_ratio=0.1
                            weight_decay_ratio=0
                            gradient_clip=true
                            num_accumulated_batches=1
                            eval_batch_size=256
                            '
                    """
                 ]
        print(' '.join(argvs))
        try:
            res = subprocess.check_output(' '.join(argvs), stderr=subprocess.STDOUT, shell=True)
            print(res)
        except subprocess.CalledProcessError as e:
            print(e.output)
            raise RuntimeError
        shutil.rmtree('ez_text_match_models', ignore_errors=True)

if __name__ == '__main__':
    unittest.main()