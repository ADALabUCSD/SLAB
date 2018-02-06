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
systems = sys.argv[3].split(' ')
algorithms = sys.argv[4].split(' ')

data.gen_data_disk('../temp/pass.csv', 2, 2, 2**12)
utils.hdfs_put('../temp/pass.csv')

args_madlib = ('mattype=adclick '
               'xTableName=adclick_clean_indepvars_long '
               'yTableName=adclick_clean_y '
               'nodes={nodes} opType={op}')
args_hdfs = ('mattype=adclick '
             'Xpath=/scratch/pass.csv '
             'Ypath=/scratch/pass.csv '
             'passPath=/scratch/pass.csv '
             'dataPath=/scratch/adclick_clean{stub}_sparse.parquet '
             'nodes={nodes} opType={op}')

for alg in algorithms:
    argv = {'stub': stub,
            'nodes': nodes,
            'op': alg}

    cmd_args_hdfs = args_hdfs.format(**argv)

    if 'SYSTEMML' in systems:
        utils.run_spark(program='SystemMLMLAlgorithms',
                        sbt_dir='./systemml',
                        driver_memory='80G',
                        cmd_args=cmd_args_hdfs)
    if 'MLLIB' in systems:
        utils.run_spark(program='SparkMLAlgorithms',
                        sbt_dir='./mllib',
                        driver_memory='20G',
                        cmd_args=cmd_args_hdfs)
    if 'MADLIB' in systems:
        print 'MADLib Tests Not Implemented for Sparse Criteo'
    #    utils.run_python(program='madlib_bigmat_algs.py', 
    #                     cmd_args=cmd_args_madlib)
