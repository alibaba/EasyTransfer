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


class TestEzSequenceLabeling(unittest.TestCase):
    def test_1_train_mode(self):
        argvs = ['easy_transfer_app',
                 '--mode', 'train',
                 '--inputTable', '../ut_data/ez_sequence_labeling/train.csv,../ut_data/ez_sequence_labeling/dev.csv',
                 '--inputSchema', 'content:str:1,ner_tags:str:1',
                 '--firstSequence', 'content',
                 '--labelName', 'ner_tags',
                 '--labelEnumerateValues', 'B-LOC,B-ORG,B-PER,I-LOC,I-ORG,I-PER,O',
                 '--checkpointDir', './ez_sequence_labeling_models/',
                 '--numEpochs', '1',
                 '--batchSize', '5',
                 '--optimizerType', 'adam',
                 '--learningRate', '2e-5',
                 '--modelName', 'sequence_labeling_bert',
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

    def test_2_predict_mode(self):
        argvs = ['easy_transfer_app',
                 '--mode', 'predict',
                 '--inputTable', '../ut_data/ez_sequence_labeling/dev.csv',
                 '--outputTable', 'ez_sequence_labeling.pred.csv',
                 '--inputSchema', 'content:str:1,ner_tags:str:1',
                 '--firstSequence', 'content',
                 '--appendCols', 'ner_tags',
                 '--outputSchema', 'predictions',
                 '--checkpointPath', './ez_sequence_labeling_models/',
                 '--batchSize', '5',
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
        self.assertTrue(os.path.exists('ez_sequence_labeling.pred.csv'))
        os.remove('ez_sequence_labeling.pred.csv')

    def test_3_evaluate_mode(self):
        argvs = ['easy_transfer_app',
                 '--mode', 'evaluate',
                 '--inputTable', '../ut_data/ez_sequence_labeling/dev.csv',
                 '--checkpointPath', './ez_sequence_labeling_models/model.ckpt-0',
                 '--batchSize', '5'
                 ]
        print(' '.join(argvs))
        try:
            res = subprocess.check_output(' '.join(argvs), stderr=subprocess.STDOUT, shell=True)
            print(res)
        except subprocess.CalledProcessError as e:
            print(e.output)
            raise RuntimeError
        shutil.rmtree('ez_sequence_labeling_models', ignore_errors=True)


if __name__ == '__main__':
    unittest.main()