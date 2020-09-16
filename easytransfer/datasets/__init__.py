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

from .tfrecord_reader import TFRecordReader, BundleTFRecordReader
from .tfrecord_writer import TFRecordWriter
from .csv_reader import CSVReader, BundleCSVReader
from .csv_writer import CSVWriter
from .odps_table_reader import OdpsTableReader
from .odps_table_writer import OdpsTableWriter
from .reader import Reader
