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

class TestEzTextClassify(unittest.TestCase):
    def test_1_train_mode(self):
        argvs = ['easy_transfer_app',
                 '--mode', 'train',
                 '--inputTable', '../ut_data/ez_text_classify/train.csv,../ut_data/ez_text_classify/dev.csv',
                 '--inputSchema', "example_id:int:1,content:str:1,label:str:1,label_str:str:1,keywords:str:1",
                 '--firstSequence', 'content',
                 '--labelName', 'label',
                 '--labelEnumerateValues', '100,101,102,103,104,105,106,107,108,109,110,112,113,114,115,116',
                 '--checkpointDir', 'ez_text_classify_models',
                 '--numEpochs', '1',
                 '--batchSize', '5',
                 '--optimizerType', 'adam',
                 '--learningRate', '2e-5',
                 '--modelName', 'text_classify_bert',
                 '--distributionStrategy', 'none',
                 '--workerCount', '1',
                 '--workerGPU', '1',
                 '--workerCPU', '1',
                 '--advancedParameters', """'
                            pretrain_model_name_or_path=google-bert-base-zh # For BERT
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
                 '--inputTable', '../ut_data/ez_text_classify/dev.csv',
                 '--outputTable', 'ez_text_classify.pred.csv',
                 '--inputSchema', 'example_id:int:1,content:str:1,label:str:1,label_str:str:1,keywords:str:1',
                 '--firstSequence', 'content',
                 '--appendCols', 'example_id,keywords,label_str',
                 '--outputSchema', 'predictions,probabilities,logits',
                 '--checkpointPath', 'ez_text_classify_models/',
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
        self.assertTrue(os.path.exists('ez_text_classify.pred.csv'))
        os.remove('ez_text_classify.pred.csv')

    def test_3_evaluate_mode(self):
        argvs = ['easy_transfer_app',
                 '--mode', 'evaluate',
                 '--inputTable', '../ut_data/ez_text_classify/dev.csv',
                 '--checkpointPath', 'ez_text_classify_models/model.ckpt-0',
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
                 '--checkpointPath', 'ez_text_classify_models/model.ckpt-0',
                 '--exportType', 'app_model',
                 '--exportDirBase', 'ez_text_classify_models/saved_app_model/',
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
                 '--inputTable', '../ut_data/ez_text_classify/dev.csv',
                 '--outputTable', 'ez_text_classify.pred.csv',
                 '--inputSchema', 'example_id:int:1,content:str:1,label:str:1,label_str:str:1,keywords:str:1',
                 '--firstSequence', 'content',
                 '--appendCols', 'example_id,keywords,label_str',
                 '--outputSchema', 'predictions,probabilities,logits',
                 '--checkpointPath', 'ez_text_classify_models/saved_app_model',
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
        self.assertTrue(os.path.exists('ez_text_classify.pred.csv'))
        os.remove('ez_text_classify.pred.csv')

    def test_5_ez_bert_feat_mode(self):
        argvs = ['easy_transfer_app',
                 '--mode', 'export',
                 '--checkpointPath', 'ez_text_classify_models/model.ckpt-0',
                 '--exportType', 'ez_bert_feat',
                 '--exportDirBase', 'ez_text_classify_models/saved_model/',
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
                 '--outputTable', 'ez_text_classify.feats.txt',
                 '--inputSchema', 'example_id:int:1,query1:str:1,label:str:1,category:str:1,score:float:1',
                 '--firstSequence', 'query1',
                 '--appendCols', 'example_id,label,category,score,xxx',
                 '--outputSchema', 'pool_output,first_token_output,all_hidden_outputs',
                 '--modelName', 'ez_text_classify_models/saved_model/',
                 '--sequenceLength', '50',
                 '--batchSize', '1']
        print(' '.join(argvs))
        try:
            res = subprocess.check_output(' '.join(argvs), stderr=subprocess.STDOUT, shell=True)
            print(res)
        except subprocess.CalledProcessError as e:
            print(e.output)
            raise RuntimeError
        os.remove('ez_text_classify.feats.txt')

    def test_6_continue_train_mode(self):
        argvs = ['easy_transfer_app',
                 '--mode', 'train',
                 '--inputTable', '../ut_data/ez_text_classify/train.csv,../ut_data/ez_text_classify/dev.csv',
                 '--inputSchema', "example_id:int:1,content:str:1,label:str:1,label_str:str:1,keywords:str:1",
                 '--firstSequence', 'content',
                 '--labelName', 'label',
                 '--labelEnumerateValues', '100,101,102,103,104,105,106,107,108,109,110,112,113,114,115,116',
                 '--checkpointDir', 'ez_text_classify_continue_models',
                 '--numEpochs', '1',
                 '--batchSize', '5',
                 '--optimizerType', 'adam',
                 '--learningRate', '2e-5',
                 '--modelName', 'text_classify_bert',
                 '--distributionStrategy', 'none',
                 '--workerCount', '1',
                 '--workerGPU', '1',
                 '--workerCPU', '1',
                 '--advancedParameters', """'
                            pretrain_model_name_or_path=google-bert-base-zh # For BERT
                            init_checkpoint_path=./ez_text_classify_models/model.ckpt-0
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
        shutil.rmtree('ez_text_classify_models', ignore_errors=True)
        shutil.rmtree('ez_text_classify_continue_models', ignore_errors=True)

    def test_7_textcnn(self):
        argvs = ['easy_transfer_app',
                 '--mode', 'train',
                 '--inputTable', '../ut_data/ez_text_classify/train.csv,../ut_data/ez_text_classify/dev.csv',
                 '--inputSchema', "example_id:int:1,content:str:1,label:str:1,label_str:str:1,keywords:str:1",
                 '--firstSequence', 'content',
                 '--labelName', 'label',
                 '--labelEnumerateValues', '100,101,102,103,104,105,106,107,108,109,110,112,113,114,115,116',
                 '--checkpointDir', 'ez_text_classify_cnn_models',
                 '--numEpochs', '1',
                 '--batchSize', '5',
                 '--optimizerType', 'adadelta',
                 '--learningRate', '0.1',
                 '--modelName', 'text_classify_cnn',
                 '--distributionStrategy', 'none',
                 '--workerCount', '1',
                 '--workerGPU', '1',
                 '--workerCPU', '1',
                 '--advancedParameters', """'
                                fix_embedding=false
                                max_vocab_size=20000
                                num_filters=100
                                filter_sizes=3,4,5
                                dropout_rate=0.5

                                lr_decay=none
                                warmup_ratio=0.0
                                weight_decay_ratio=0
                                gradient_clip=true
                                clip_norm_value=5.0
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
                 '--inputTable', '../ut_data/ez_text_classify/dev.csv',
                 '--outputTable', 'ez_text_classify.pred.csv',
                 '--inputSchema', 'example_id:int:1,content:str:1,label:str:1,label_str:str:1,keywords:str:1',
                 '--firstSequence', 'content',
                 '--appendCols', 'example_id,keywords,label_str',
                 '--outputSchema', 'predictions,probabilities,logits',
                 '--checkpointPath', 'ez_text_classify_cnn_models/',
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
        self.assertTrue(os.path.exists('ez_text_classify.pred.csv'))
        shutil.rmtree('ez_text_classify_cnn_models', ignore_errors=True)
        os.remove('ez_text_classify.pred.csv')

    def test_8_multi_label(self):
        argvs = ['easy_transfer_app',
                 '--mode', 'train',
                 '--inputTable', '../ut_data/ez_text_classify/train.multi_label.csv,../ut_data/ez_text_classify/dev.multi_label.csv',
                 '--inputSchema', "example_id:int:1,content:str:1,accusations:str:1,articles:str:1",
                 '--firstSequence', 'content',
                 '--labelName', 'accusations',
                 '--labelEnumerateValues', '../ut_data/ez_text_classify/multi_labels.txt',
                 '--checkpointDir', 'ez_text_classify_multi_label_models',
                 '--numEpochs', '1',
                 '--batchSize', '5',
                 '--optimizerType', 'adam',
                 '--learningRate', '2e-5',
                 '--modelName', 'text_classify_bert',
                 '--distributionStrategy', 'none',
                 '--workerCount', '1',
                 '--workerGPU', '1',
                 '--workerCPU', '1',
                 '--advancedParameters', """'
                            multi_label=true
                            max_num_labels=5
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
                 '--inputTable', '../ut_data/ez_text_classify/dev.multi_label.csv',
                 '--outputTable', 'ez_text_classify.pred.csv',
                 '--inputSchema', "example_id:int:1,content:str:1,accusations:str:1,articles:str:1",
                 '--firstSequence', 'content',
                 '--appendCols', 'example_id,accusations,articles',
                 '--outputSchema', 'predictions,probabilities,logits',
                 '--checkpointPath', 'ez_text_classify_multi_label_models/',
                 '--batchSize', '100',
                 '--workerCount', '1',
                 '--workerGPU', '1',
                 '--workerCPU', '1'
                 ]
        print(' '.join(argvs))
        try:
            res = subprocess.check_output(' '.join(argvs), stderr=subprocess.STDOUT, shell=True)
            print(res)
        except subprocess.CalledProcessError as e:
            print(e.output)
            raise RuntimeError
        self.assertTrue(os.path.exists('ez_text_classify.pred.csv'))
        shutil.rmtree('ez_text_classify_multi_label_models', ignore_errors=True)
        os.remove('ez_text_classify.pred.csv')

    def test_9_regression_bert(self):
        argvs = ['easy_transfer_app',
                 '--mode', 'train',
                 '--inputTable', '../ut_data/ez_text_match/train.csv,../ut_data/ez_text_match/dev.csv',
                 '--inputSchema', "example_id:int:1,query1:str:1,query2:str:1,is_same_question:str:1,category:str:1,score:float:1",
                 '--firstSequence', 'query1',
                 '--labelName', 'score',
                 '--checkpointDir', 'ez_text_classify_models',
                 '--numEpochs', '1',
                 '--batchSize', '5',
                 '--saveCheckpointSteps', '10000',
                 '--optimizerType', 'adam',
                 '--learningRate', '2e-5',
                 '--modelName', 'text_classify_bert',
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
        shutil.rmtree('ez_text_classify_models', ignore_errors=True)

if __name__ == '__main__':
    unittest.main()