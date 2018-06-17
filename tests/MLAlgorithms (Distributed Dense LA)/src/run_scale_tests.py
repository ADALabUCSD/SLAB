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
msize = sys.argv[3].split(' ')
algorithms = sys.argv[4].split(' ')
systems = sys.argv[5].split(' ')

#data.gen_data_disk('../temp/pass.csv', 2, 2, 2**12)
#utils.hdfs_put('../temp/pass.csv')

args_R = ('mattype=tall '
          'Xpath=../external/disk_data/M{gb}_tall.csv '
          'Ypath=../external/disk_data/y{gb}_tall.csv '
          'nodes={nodes} opType={op}')
args_madlib = ('mattype=tall '
               'xTableName=M{gb}_tall '
               'yTableName=y{gb}_tall '
               'nodes={nodes} opType={op}')
args_hdfs = ('mattype=tall '
             'Xpath=/scratch/M{gb}_tall.csv '
             'Ypath=/scratch/y{gb}_tall.csv '
             'passPath=/scratch/pass.csv '
             'nodes={nodes} opType={op}')

for gb in msize:
    for alg in algorithms:
        if alg == 'logit':
            ytable_name = 'adclick_y_array{}'
        else:
            ytable_name = 'adclick_y_mat{}'
        argv = {'stub': stub,
                'nodes': nodes,
                'op': alg,
                'gb': gb}
        cmd_args_R = args_R.format(**argv)
        cmd_args_madlib = args_madlib.format(**argv)
        cmd_args_hdfs = args_hdfs.format(**argv)
       
        if 'R' in systems:
            utils.run_pbdR(program='ml_algs.R',
                           cmd_args=cmd_args_R)
        if 'SYSTEMML' in systems:
            utils.run_spark(program='SystemMLMLAlgorithms',
                            sbt_dir='./systemml',
                            driver_memory='32G',
                            cmd_args=cmd_args_hdfs)
        if 'MLLIB' in systems:
            utils.run_spark(program='SparkMLAlgorithms',
                            sbt_dir='./mllib',
                            driver_memory='32G',
                            cmd_args=cmd_args_hdfs)
        if 'MADLIB' in systems:
            utils.run_python(program='madlib_algs.py', 
                             cmd_args=cmd_args_madlib)
        if 'SCIDB' in systems:
            utils.run_python(program='scidb_algs.py', 
                             cmd_args=cmd_args_madlib)
