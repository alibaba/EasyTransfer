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


class TestEzTextComprehension(unittest.TestCase):
    def test_1_train_mode(self):
        argvs = ['easy_transfer_app',
                 '--mode', 'preprocess',
                 '--inputTable', '../ut_data/ez_text_comprehension/train.csv',
                 '--outputTable', 'serialization.train.csv',
                 '--inputSchema', "context:str:1,question:str:1,answer:str:1",
                 '--firstSequence', 'context',
                 '--secondSequence', 'question',
                 '--labelName', 'answer',
                 '--sequenceLength', '384',
                 '--modelName', 'text_comprehension_bert',
                 '--outputSchema', 'input_ids,input_mask,segment_ids,start_position,end_position',
                 '--batchSize', '100',
                 '--advancedParameters', "'tokenizer_name_or_path=google-bert-base-zh"
                                         " max_query_length=64'"
                 ]
        print(' '.join(argvs))
        try:
            res = subprocess.check_output(' '.join(argvs), stderr=subprocess.STDOUT, shell=True)
            print(res)
        except subprocess.CalledProcessError as e:
            print(e.output)
            raise RuntimeError
        self.assertTrue(os.path.exists('serialization.train.csv'))

        argvs = ['easy_transfer_app',
                 '--mode', 'preprocess',
                 '--inputTable', '../ut_data/ez_text_comprehension/dev.csv',
                 '--outputTable', 'serialization.dev.csv',
                 '--inputSchema', "context:str:1,question:str:1,answer:str:1",
                 '--firstSequence', 'context',
                 '--secondSequence', 'question',
                 '--labelName', 'answer',
                 '--sequenceLength', '384',
                 '--modelName', 'text_comprehension_bert',
                 '--outputSchema', 'input_ids,input_mask,segment_ids,start_position,end_position',
                 '--batchSize', '100',
                 '--advancedParameters', "'tokenizer_name_or_path=google-bert-base-zh"
                                         " max_query_length=64'"
                 ]
        print(' '.join(argvs))
        try:
            res = subprocess.check_output(' '.join(argvs), stderr=subprocess.STDOUT, shell=True)
            print(res)
        except subprocess.CalledProcessError as e:
            print(e.output)
            raise RuntimeError
        self.assertTrue(os.path.exists('serialization.dev.csv'))

        argvs = ['easy_transfer_app',
                 '--mode', 'train',
                 '--inputTable', 'serialization.train.csv,serialization.dev.csv',
                 '--inputSchema', 'input_ids:int:384,input_mask:int:384,segment_ids:int:384,start_position:int:1,end_position:int:1',
                 '--sequenceLength', '384',
                 '--checkpointDir', 'ez_text_comprehension_models',
                 '--numEpochs', '1',
                 '--batchSize', '5',
                 '--optimizerType', 'adam',
                 '--learningRate', '2e-5',
                 '--modelName', 'text_comprehension_bert',
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
        os.remove('serialization.train.csv')

    def test_2_predict_mode(self):
        argvs = ['easy_transfer_app',
                 '--mode', 'predict',
                 '--inputTable', '../ut_data/ez_text_comprehension/dev.csv',
                 '--outputTable', 'ez_text_comprehension.pred.csv',
                 '--inputSchema', "context:str:1,question:str:1,answer:str:1",
                 '--firstSequence', 'context',
                 '--secondSequence', 'question',
                 '--outputSchema', 'qas_id,predictions',
                 '--checkpointPath', 'ez_text_comprehension_models',
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
        self.assertTrue(os.path.exists('ez_text_comprehension.pred.csv'))
        os.remove('ez_text_comprehension.pred.csv')

    def test_3_evaluate_mode(self):
        argvs = ['easy_transfer_app',
                 '--mode', 'evaluate',
                 '--inputTable', './serialization.dev.csv',
                 '--checkpointPath', 'ez_text_comprehension_models/model.ckpt-0',
                 '--batchSize', '100'
                 ]
        print(' '.join(argvs))
        try:
            res = subprocess.check_output(' '.join(argvs), stderr=subprocess.STDOUT, shell=True)
            print(res)
        except subprocess.CalledProcessError as e:
            print(e.output)
            raise RuntimeError
        os.remove('serialization.dev.csv')
        shutil.rmtree('ez_text_comprehension_models', ignore_errors=True)

    def test_4_train_bert_hae(self):
        argvs = ['easy_transfer_app',
                 '--mode', 'preprocess',
                 '--inputTable', '../ut_data/ez_text_comprehension/train.multi.csv',
                 '--outputTable', 'serialization.train.csv',
                 '--inputSchema', "context:str:1,question:str:1,answer:str:1",
                 '--firstSequence', 'context',
                 '--secondSequence', 'question',
                 '--labelName', 'answer',
                 '--sequenceLength', '384',
                 '--modelName', 'text_comprehension_bert_hae',
                 '--outputSchema', 'input_ids,input_mask,segment_ids,history_answer_marker,start_position,end_position',
                 '--batchSize', '100',
                 '--advancedParameters', "'tokenizer_name_or_path=google-bert-base-zh"
                                         " max_query_length=64 doc_stride=128'"
                 ]
        print(' '.join(argvs))
        try:
            res = subprocess.check_output(' '.join(argvs), stderr=subprocess.STDOUT, shell=True)
            print(res)
        except subprocess.CalledProcessError as e:
            print(e.output)
            raise RuntimeError
        self.assertTrue(os.path.exists('serialization.train.csv'))

        argvs = ['easy_transfer_app',
                 '--mode', 'preprocess',
                 '--inputTable', '../ut_data/ez_text_comprehension/dev.multi.csv',
                 '--outputTable', 'serialization.dev.csv',
                 '--inputSchema', "context:str:1,question:str:1,answer:str:1",
                 '--firstSequence', 'context',
                 '--secondSequence', 'question',
                 '--labelName', 'answer',
                 '--sequenceLength', '384',
                 '--modelName', 'text_comprehension_bert_hae',
                 '--outputSchema', 'input_ids,input_mask,segment_ids,history_answer_marker,start_position,end_position',
                 '--batchSize', '100',
                 '--advancedParameters', "'tokenizer_name_or_path=google-bert-base-zh"
                                         " max_query_length=64 doc_stride=128'"
                 ]
        print(' '.join(argvs))
        try:
            res = subprocess.check_output(' '.join(argvs), stderr=subprocess.STDOUT, shell=True)
            print(res)
        except subprocess.CalledProcessError as e:
            print(e.output)
            raise RuntimeError

        argvs = ['easy_transfer_app',
                 '--mode', 'train',
                 '--inputTable', 'serialization.train.csv,serialization.dev.csv',
                 '--inputSchema', 'input_ids:int:384,input_mask:int:384,segment_ids:int:384,history_answer_marker:int:384,'
                                  'start_position:int:1,end_position:int:1',
                 '--sequenceLength', '384',
                 '--checkpointDir', 'ez_text_comprehension_hae_models',
                 '--numEpochs', '1',
                 '--batchSize', '5',
                 '--optimizerType', 'adam',
                 '--learningRate', '2e-5',
                 '--modelName', 'text_comprehension_bert_hae',
                 '--distributionStrategy', 'none',
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
                 '--inputTable', '../ut_data/ez_text_comprehension/dev.multi.csv',
                 '--outputTable', 'ez_text_comprehension_hae.pred.csv',
                 '--inputSchema', "context:str:1,question:str:1,answer:str:1",
                 '--firstSequence', 'context',
                 '--secondSequence', 'question',
                 '--labelName', 'answer',
                 '--sequenceLength', '384',
                 '--outputSchema', 'qas_id,unique_id,predictions',
                 '--checkpointPath', './ez_text_comprehension_hae_models/',
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
        self.assertTrue(os.path.exists('ez_text_comprehension_hae.pred.csv'))
        os.remove('serialization.train.csv')
        os.remove('serialization.dev.csv')
        os.remove('ez_text_comprehension_hae.pred.csv')
        shutil.rmtree('ez_text_comprehension_hae_models', ignore_errors=True)


if __name__ == '__main__':
    unittest.main()