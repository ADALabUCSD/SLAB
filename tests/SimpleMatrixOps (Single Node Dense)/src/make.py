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

from gslab_make.dir_mod import *
from gslab_make.run_program import *
from gslab_make.make_log import *

# Clean up after previous runs
clear_dirs('../temp')
clear_dirs('../external')
#delete_files('../output/*')

# create symlinks to external resources
project_root = os.getenv('BENCHMARK_PROJECT_ROOT')
if (project_root is None):
    msg = 'Pease set environment variable "BENCHMARK_PROJECT_ROOT"'
    raise StandardError(msg)

externals = {'lib' : '/lib'}
for name in externals:
    os.symlink(project_root + externals[name], '../external/' + name)

sys.path.append('../external/lib/python')
import make_utils as utils
import global_params as params

parser = argparse.ArgumentParser()
systems = 'TF MADLIB SYSTEMML MLLIB R NUMPY'
parser.add_argument('--systems', type=str, default=systems, 
    help='Space delimited list of systems to run. Default: ({})'.format(systems))
op_types = 'NORM MVM TRANS ADD TSM GMM'
parser.add_argument('--operators', type=str, default=op_types, 
    help='Space delimited list of operators to run. Default: ({})'.format(op_types))
parser.add_argument('--nrows', type=str, default='',
    help='Number of rows to generate in test matrices')
parser.add_argument('--test-type', type=str, default='matrix-ops',
    help='Test to run. May be "matrix-ops" or "cpu". Default: "matrix-ops"')
args = parser.parse_args()

op_types = args.operators.split(' ')
systems = args.systems.split(' ')
test_type = args.test_type
nrows = args.nrows
if nrows == '' and test_type == 'matrix-ops':
    nrows = "2500000 5000000 10000000 20000000"
if nrows == '' and test_type == 'cpu':
    nrows = '20000000'

# start logging
start_make_logging()

# compile
makelog = '../../output/make.log'
utils.run_sbt('./systemml', makelog=makelog)
utils.run_sbt('./mllib', makelog=makelog)

cmd_args_template = 'opType={} mattype={} nrows="{}"'
mattype = 'tall'
if test_type == 'matrix-ops':
    for op in op_types:
        args = (op, mattype, nrows)
        cmd_args = cmd_args_template.format(*args)
        cmd_args += ' fixedAxis=100 step=10'

        if 'NUMPY' in systems:
            utils.run_python(program='np_matrix_ops.py', cmd_args=cmd_args)
        if 'TF' in systems:
            utils.run_python(program='tf_matrix_ops.py', cmd_args=cmd_args)
        if 'R' in systems:
            utils.run_R(program='R_matrix_ops.R', cmd_args=cmd_args)
        if 'SYSTEMML' in systems:
            utils.run_spark(program='SystemMLMatrixOps', 
                            sbt_dir='./systemml',
                            cmd_args=cmd_args)
        if 'MLLIB' in systems:
            utils.run_spark(program='MLLibMatrixOps', 
                            sbt_dir='./mllib', 
                            cmd_args=cmd_args)
        if 'MADLIB' in systems:
            utils.run_python(program='madlib_matrix_ops.py',
                             cmd_args=cmd_args)
        if 'SCIDB' in systems:
            utils.run_python(program='scidb_matrix_ops.py',
                             cmd_args=cmd_args)

if test_type == 'cpu':
#    ops = ['TSM','ADD'] 
    ops = ['TSM']
    nproc = [1,2,4,8,16]
    for num_proc in nproc:
        for op in ops:
            # This will force the process to execute only on a subset of processors
            utils.set_nproc(num_proc)
            os.putenv('OMP_NUM_THREADS','{}'.format(num_proc))
            args = (op, mattype, nrows)
            cmd_args = cmd_args_template.format(*args)
            cmd_args += ' fixedAxis=100 step=10 nproc={}'.format(num_proc)
    
            if 'NUMPY' in systems:
                utils.run_python(program='np_matrix_ops.py', cmd_args=cmd_args)
            if 'TF' in systems:
                utils.run_python(program='tf_matrix_ops.py', cmd_args=cmd_args)
            if 'R' in systems:
                utils.run_R(program='R_matrix_ops.R', cmd_args=cmd_args)
            if 'SYSTEMML' in systems:
                utils.run_spark(program='SystemMLMatrixOps', 
                                sbt_dir='./systemml',
                                driver_cores=str(num_proc),
                                cmd_args=cmd_args)
            if 'MLLIB' in systems:
                utils.run_spark(program='MLLibMatrixOps', 
                                sbt_dir='./mllib',
                                driver_cores=str(num_proc),
                                cmd_args=cmd_args)
            utils.set_nproc(999)
            if 'MADLIB' in systems:
                utils.run_python(program='madlib_matrix_ops.py',
                                cmd_args=cmd_args)
            if 'SCIDB' in systems:
                utils.run_python(program='scidb_matrix_ops.py',
                                cmd_args=cmd_args)

remove_dir('scratch_space')

# stop logging
end_make_logging()
