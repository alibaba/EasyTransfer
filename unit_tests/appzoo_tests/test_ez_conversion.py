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

class TestEzConversion(unittest.TestCase):
    def test_conversion(self):
        argvs = ['easy_transfer_app',
                 '--mode', 'export',
                 '--checkpointPath', './pai-bert-base-zh/model.ckpt',
                 '--exportType', 'convert_bert_to_google',
                 '--exportDirBase', 'ez_conversion/',
                 ]
        print(' '.join(argvs))
        try:
            res = subprocess.check_output(' '.join(argvs), stderr=subprocess.STDOUT, shell=True)
            print(res)
        except subprocess.CalledProcessError as e:
            print(e.output)
            raise RuntimeError
        shutil.rmtree('ez_conversion/', ignore_errors=True)


if __name__ == "__main__":
    unittest.main()