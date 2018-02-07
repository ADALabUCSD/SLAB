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
             'disk_data' : '/tests/SimpleMatrixOps (Disk Data)/output'}
for name in externals:
    os.symlink(project_root + externals[name], '../external/' + name)

sys.path.append('../external/lib/python')
import make_utils as utils
import global_params as params
import gen_data as data

utils.safe_mkdir('../output/scale_mat_size')
utils.safe_mkdir('../output/scale_nodes')

parser = argparse.ArgumentParser()
parser.add_argument('--test-type',type=str, default='scale_mat',
    help='Test type to run. May be one of "scale_mat" or "scale_nodes". Default: scale_mat')
parser.add_argument('--nodes', type=str,
    help='Number of nodes on which test is being run (e.g. 2)')
parser.add_argument('--sparsity', type=str, default='0001 001 01 1',
    help='Space delimited list of matrix densities on which to perform tests. Default: "0001 001 01 1"')
parser.add_argument('--sparse-gb', type=str, default='100',
    help='Logical size of sparse matrix. Must agree with value selected during data generation. Default: 100')
op_types = 'TRANS MVM NORM TSM ADD GMM'
parser.add_argument('--operators', type=str, default=op_types,
    help='Space delimited list of operators to test. Default: "{}"'.format(op_types))
systems = 'R SYSTEMML MLLIB MADLIB'
parser.add_argument('--systems', type=str, default=systems,
    help='Space delimited list of systems to compare. Default: "{}"'.format(systems))
args = parser.parse_args()

# start logging
start_make_logging()

test_type = args.test_type
nodes = args.nodes
sparsity = args.sparsity
systems = args.systems
op_types = args.operators
sparse_gb = args.sparse_gb

# compile
makelog = '../../output/make.log'
utils.run_sbt('./systemml', makelog = makelog)
utils.run_sbt('./mllib',  makelog = makelog)

if test_type == 'scale_nodes':
    utils.run_python(program='node_scaling_tests.py',
                     cmd_args='{} "{}" "{}" "{}" {}'.format(
        nodes, sparsity, systems, op_types, sparse_gb))
elif test_type == 'scale_mat':
    utils.run_python(program='msize_scaling_tests.py',
                     cmd_args='{} "{}" "{}" "{}" {}'.format(
        nodes, sparsity, systems, op_types, sparse_gb))
else:
    raise StandardError('TEST_TYPE must be one of: "scale_nodes", "scale_mat"')

remove_dir('scratch_space')

# stop logging
end_make_logging()
