# Copyright 2018 Anthony H Thomas and Arun Kumar
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

with open(__file__) as fh: print fh.read()
import os
import sys
import textwrap
import numpy as np
import pandas as pd

ROOT = os.getenv('BENCHMARK_PROJECT_ROOT')
if (ROOT is None):
    msg = 'Please set environment variable BENCHMARK_PROJECT_ROOT'
    raise StandardError(msg)

sys.path.append(os.path.join(ROOT,'lib','python'))
from sql_cxn import SQLCxn
from make_utils import hdfs_put
import np_timing_utils as utils

def main():
    stub = 'azuremlsampleexperiments.blob.core.windows.net/criteo/day_1.gz'
    url = 'http://{}'.format(stub)

    path = '../temp/day_1.gz'
    if not (os.path.exists(path) or os.path.exists(path.replace('.gz',''))):
        rc = os.system('wget {} -O {}'.format(url, path))
        if rc != 0:
            raise StandardError('Could not fetch data')

    os.system('hdfs dfs -mkdir /scratch')
    rc = hdfs_put(path)

if __name__ == '__main__':
    main()
