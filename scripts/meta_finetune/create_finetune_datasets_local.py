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
import sys
sys.path.append("../..")
import os
import numpy as np
import tensorflow as tf

tf.app.flags.DEFINE_string("input_file", "", "input_file")
tf.app.flags.DEFINE_string("output_file", "", "output_file")

tf.app.flags.DEFINE_string("domain_name", "", "domain_name")
tf.app.flags.DEFINE_integer("domain_column_index", 0, "domain_column_index")
aFLAGS = tf.app.flags.FLAGS

def main(_):
    with open(aFLAGS.input_file, "r") as input_file, open(aFLAGS.output_file, "w+") as output_file: 
        lines = input_file.readlines()
        for line in lines:
            items = line.strip().split("\t")
            cur_domain = items[int(aFLAGS.domain_column_index)]
            if cur_domain == aFLAGS.domain_name:
                output_file.write(line)

if __name__ == '__main__':
    tf.app.run()
    
