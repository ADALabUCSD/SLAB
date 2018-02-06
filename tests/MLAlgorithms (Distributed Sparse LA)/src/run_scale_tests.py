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

sys.path.append('../external/lib/python')
import make_utils as utils
import global_params as params
import gen_data as data

stub = sys.argv[1]
nodes = sys.argv[2]
algorithms = sys.argv[3].split(' ')
systems = sys.argv[4].split(' ')
sparsity = sys.argv[5].split(' ')
sparse_gb = sys.argv[6]

ncols=[100]
args_madlib = ('mattype=tall '
               'xTableName=M_{sr}{sparse_gb}_sparse_tall '
               'yTableName=y{sparse_gb}_sparse '
               'nodes={nodes} opType={op} ncol={ncol} sr={sr}')
args_hdfs = ('mattype=tall '
             'Xpath=/scratch/M_{sr}{sparse_gb}_sparse_tall.mtx '
             'Ypath=/scratch/y{sparse_gb}_sparse.csv '
             'passPath=/scratch/pass.csv '
             'nodes={nodes} opType={op} ncol={sparse_gb} sr={sr}')

for sr in sparsity:
    for alg in algorithms:
        for nc in ncols:
            argv = {'nodes': nodes, 'op': alg, 'sr': sr, 'sparse_gb': sparse_gb, 'ncol': nc}
            cmd_args_hdfs = args_hdfs.format(**argv)
            cmd_args_madlib = args_madlib.format(**argv)

            if 'SYSTEMML' in systems:
                utils.run_spark(program='SystemMLMLAlgorithms',
                                sbt_dir='./systemml',
                                driver_memory='64G',
                                cmd_args=cmd_args_hdfs)
            if 'MLLIB' in systems:
                utils.run_spark(program='SparkMLAlgorithms',
                                sbt_dir='./mllib',
                                driver_memory='64G',
                                cmd_args=cmd_args_hdfs)
            if 'MADLIB' in systems:
                utils.run_python(program='madlib_algs.py', 
                                 cmd_args=cmd_args_madlib)
