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
import argparse

import numpy as np
from gslab_make.dir_mod import *
from gslab_make.run_program import *
from gslab_make.make_log import *

# Clean up after previous runs
clear_dirs('../temp')
clear_dirs('../external')

# create symlinks to external resources
project_root = os.getenv('BENCHMARK_PROJECT_ROOT')
if (project_root is None):
    msg = 'Pease set environment variable "BENCHMARK_PROJECT_ROOT"'
    raise StandardError(msg)

externals = {'lib' : '/lib',
             'disk_data' : '/data/SimpleMatrixOps (Disk Data)/output'}
for name in externals:
    os.symlink(project_root + externals[name], '../external/' + name)

sys.path.append('../external/lib/python')
import make_utils as utils
import global_params as params
import gen_data as data

utils.safe_mkdir('../output/scale_mat_size')
utils.safe_mkdir('../output/scale_nodes')

parser = argparse.ArgumentParser()
parser.add_argument(
    '--test-type', type=str, default='scale_mat',
    help='Test to run - one of (scale_nodes, scale_mat) - default scale_mat')
parser.add_argument(
    '--nodes', type=str,
    help='Number of nodes in current cluster (e.g. 2 for a 2 node cluster)')
parser.add_argument(
    '--matsize', type=str, default='2 4 8 16',
    help='Matrix sizes to test. Space delimited. Default "2 4 8 16"')
parser.add_argument(
    '--systems', type=str, default='MLLIB R MADLIB',
    help='Space delimited list of systems to compare. Must be one of MLLIB R MADLIB')
args = parser.parse_args()

# start logging
start_make_logging()

test_type = args.test_type
nodes = args.nodes
matsize = args.matsize
systems = args.systems

# compile
makelog = '../../output/make.log'
utils.run_sbt('./systemml', makelog = makelog)
utils.run_sbt('./mllib',  makelog = makelog)

if test_type == 'scale_nodes':
    utils.run_python(program='node_scaling_tests.py',
                     cmd_args='{} "{}" "{}"'.format(nodes, matsize, systems))
elif test_type == 'scale_mat':
    utils.run_python(program='msize_scaling_tests.py',
                     cmd_args='{} "{}" "{}"'.format(nodes, matsize, systems))
else:
    raise StandardError('TEST_TYPE must be one of: "scale_nodes", "scale_mat"')

remove_dir('scratch_space')

# stop logging
end_make_logging()
