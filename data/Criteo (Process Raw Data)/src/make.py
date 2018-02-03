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

import os
import sys
import numpy as np
from gslab_make.dir_mod import *
from gslab_make.run_program import *
from gslab_make.make_log import *

# Clean up after previous runs
clear_dirs('../external')
try:
    os.mkdir('../temp')
except OSError:
    pass

# create symlinks to external resources
project_root = os.getenv('BENCHMARK_PROJECT_ROOT')
if (project_root is None):
    msg = 'Pease set environment variable "BENCHMARK_PROJECT_ROOT"'
    raise StandardError(msg)

externals = {'lib' : '/lib',
             'disk_data' : '/tests/SimpleMatrixOps (Disk Data)/output'}
for name in externals:
    os.symlink(project_root + externals[name], '../external/' + name)

sys.path.append('../external/lib/python')
import make_utils as utils
import global_params as params

# start logging
start_make_logging()

# compile
makelog = '../../output/make.log'
utils.run_sbt('./spark', makelog=makelog)

utils.run_python(program='get_data.py')
os.putenv('SAVE_STUB', '_1')
utils.run_spark(program='SparkPreclean',
                sbt_dir='./spark',
                cmd_args='/scratch/day_1.gz true')

# stop logging
end_make_logging()
