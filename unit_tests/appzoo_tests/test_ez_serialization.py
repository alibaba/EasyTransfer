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
import shutil


class TestEzSerialization(unittest.TestCase):
    def test_serialization_finetune(self):
        argvs = ['easy_transfer_app',
                 '--mode', 'preprocess',
                 '--inputTable', '../ut_data/ez_text_classify/dev.csv',
                 '--outputTable', 'serialization.pred.csv',
                 '--inputSchema', 'example_id:int:1,content:str:1,label:str:1,label_str:str:1,keywords:str:1',
                 '--firstSequence', 'content',
                 '--modelName', 'google-bert-base-zh',
                 '--appendCols', 'example_id,keywords,label_str',
                 '--outputSchema', 'input_ids,input_mask,segment_ids,label_id',
                 '--labelName', 'label',
                 '--labelEnumerateValues', '100,101,102,103,104,105,106,107,108,109,110,112,113,114,115,116',
                 '--batchSize', '100',
                 '--workerCount', '1',
                 '--workerGPU', '1',
                 '--workerCPU', '1',
                 ]
        print(' '.join(argvs))
        try:
            res = subprocess.check_output(' '.join(argvs), stderr=subprocess.STDOUT, shell=True)
            print(res)
            self.assertTrue(os.path.exists('serialization.pred.csv'))
            shutil.rmtree('serialization.pred.csv', ignore_errors=True)
        except subprocess.CalledProcessError as e:
            print(e.output)
            raise RuntimeError

    def test_serialization_predict(self):
        argvs = ['easy_transfer_app',
                 '--mode', 'preprocess',
                 '--inputTable', '../ut_data/ez_text_classify/dev.csv',
                 '--outputTable', 'serialization.pred.csv',
                 '--inputSchema', 'example_id:int:1,content:str:1,label:str:1,label_str:str:1,keywords:str:1',
                 '--firstSequence', 'content',
                 '--modelName', 'google-bert-base-zh',
                 '--appendCols', 'example_id,keywords,label_str',
                 '--outputSchema', 'input_ids,input_mask,segment_ids',
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
        self.assertTrue(os.path.exists('serialization.pred.csv'))
        os.remove('serialization.pred.csv')


if __name__ == "__main__":
    unittest.main()