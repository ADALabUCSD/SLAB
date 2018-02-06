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

parser = argparse.ArgumentParser()
parser.add_argument('--nodes', type=int,
    help='Number of nodes on which test is being run (e.g. 2)')
parser.add_argument('--test-type', type=str, default='scale',
    help='Test type to run. May be "criteo" or "scale". Default "scale"')
sparsity = '0001 001 01 1'
parser.add_argument(
    '--sparsity', type=str, default=sparsity,
    help='Space delimited list of matrix sparsitites to test for "scale" tests')
parser.add_argument(
    '--sparse-gb', type=str, default='100',
    help='Number of GB in logical sparse matrix. Must agree with value used for data generation')
parser.add_argument('--stub', default='_1', type=str,
    help='Unneeded. For debug purposes only')
algorithms = 'reg logit gnmf robust'
parser.add_argument(
    '--algorithms', type=str, default=algorithms,
    help='Space delimited list of algorithms to run. May be any of "{}"'.format(algorithms))
systems = 'MLLIB SYSTEMML MADLIB'
parser.add_argument('--systems', type=str, default=systems,
    help='Space delimited list of systems to compare. May be any of "{}"'.format(systems))

args = parser.parse_args()

# start logging
start_make_logging()

# compile
makelog = '../../output/make.log'
utils.run_sbt('./systemml', makelog = makelog)
utils.run_sbt('./mllib',  makelog = makelog)

if args.test_type == 'criteo':
    utils.run_python(program='run_criteo_tests.py',
                     cmd_args='{} {} "{}" "{}"'.format(args.stub, 
                                                       args.nodes,
                                                       args.systems,
                                                       args.algorithms))
if args.test_type == 'scale':
    utils.run_python(program='run_scale_tests.py',
                     cmd_args='{} {} "{}" "{}" "{}" {}'.format(args.stub, 
                                                               args.nodes,
                                                               args.algorithms,
                                                               args.systems,
                                                               args.sparsity,
                                                               args.sparse_gb))

remove_dir('scratch_space')

# stop logging
end_make_logging()

