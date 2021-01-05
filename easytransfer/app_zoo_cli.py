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
sys.path.append("./")

import tensorflow as tf
try:
    import tensorflow_io as tfio
except:
    pass
from easytransfer import FLAGS
from easytransfer.app_zoo import get_application_model
from easytransfer.app_zoo.app_config import AppConfig


_app_flags = tf.app.flags
_app_flags.DEFINE_string("inputTable", default=None, help='Input table/path,'
                                                          'train_path,eval_path for `train` mode'
                                                          'test_path for `predict` mode')
_app_flags.DEFINE_string("inputSchema", default=None,
                          help='Only for csv data, the schema of input table')
_app_flags.DEFINE_string("outputTable", default=None, help='Output table/path')
_app_flags.DEFINE_string("firstSequence", default=None,
                          help='Which column is the first sequence mapping to')
_app_flags.DEFINE_string("secondSequence", default=None,
                          help='Which column is the second sequence mapping to')
_app_flags.DEFINE_integer("batchSize", default=32,
                          help='Batch size for `train` or `predict`')

# FLAGS only for training
_app_flags.DEFINE_string("labelName", default=None,
                          help='Which column is the label mapping to')
_app_flags.DEFINE_string("labelEnumerateValues", default=None,
                          help='Which column is the label mapping to')
_app_flags.DEFINE_integer("sequenceLength", default=128,
                          help='Maximum overall sequence length.')
_app_flags.DEFINE_string("checkpointDir", default=None,
                          help='Directory of saved checkpoints/configs')
_app_flags.DEFINE_integer("numEpochs", default=1,
                          help='Number of epochs. Checkpoints will be saved after each epoch.')
_app_flags.DEFINE_integer("saveCheckpointSteps", default=None,
                          help='Save checkpoints per ? steps')
_app_flags.DEFINE_string("optimizerType", default='adam',
                          help='Types of optimizers, choices from ["adam", "lamb"]')
_app_flags.DEFINE_float("learningRate", default=2e-5,
                          help='Initial learning rate')
_app_flags.DEFINE_string("modelName", default="text_match_bert",
                          help='Which model you want to use, choices from ['
                               '"text_match_bert", "text_match_bert_two_tower",'
                               '"text_match_bicnn", "text_match_dam", "text_match_damplus"'
                               ']')
_app_flags.DEFINE_string("distributionStrategy", default="MirroredStrategy",
                          help='Distribution Strategy, choices from ["MirroredStrategy"]')
_app_flags.DEFINE_string("advancedParameters", default="",
                          help='Advanced parameter settings')

# FLAGS only for prediction
_app_flags.DEFINE_string("appendCols", default=None,
                          help='Which columns will be appended on the outputs')
_app_flags.DEFINE_string("outputSchema", default="predictions,probabilities,logits",
                          help='The choices of output features')
_app_flags.DEFINE_string("checkpointPath", default='',
                          help='Path of saved checkpoints')

# FLAGS only for export
_app_flags.DEFINE_string("exportType", default='ez_text_match',
                          help='Which type of export model')
_app_flags.DEFINE_string("exportDirBase", default='',
                          help='Directory of saved model')

_APP_FLAGS = _app_flags.FLAGS


def main():
    # Here is a hack for DSW access OSS
    for argname in ["inputTable", "outputTable", "checkpointDir", "checkpointPath", "exportDirBase"]:
        arg = getattr(_APP_FLAGS, argname)
        if arg:
            arg =  arg.replace("\\x01", "\x01").replace("\\x02", "\x02")
            setattr(_APP_FLAGS, argname, arg)
    FLAGS.modelZooBasePath = FLAGS.modelZooBasePath.replace("\\x01", "\x01").replace("\\x02", "\x02")

    # Main function start
    config = AppConfig(mode=FLAGS.mode, flags=_APP_FLAGS)
    app = get_application_model(config)
    app.run()


if __name__ == "__main__":
    main()