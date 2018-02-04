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

nodes = sys.argv[1]
matsize = sys.argv[2].split(' ')
systems = sys.argv[3].split(' ')
ops = sys.argv[4].split(' ')

if len(matsize) > 1:
    raise StandardError('matsize must be a scalar')

matsize = matsize[0]

#all_files = os.listdir('../output/scale_nodes')
#for s in systems:
#    for op in ops:
#        relevant_files = filter(
#            lambda x: (s in x) and (op in x) and (nodes in x), all_files)
#        map(lambda x:  os.unlink('../output/scale_nodes/{}'.format(x)), 
#             relevant_files)

cmd_args = ('opType={opType} mattype={mattype}'
            ' Mpath={Mpath} Npath={Npath}'
            ' wPath={wPath} tableStub={tableStub}'
            ' nodes={nodes} passPath=/scratch/pass.csv'
            ' outdir=scale_nodes')

data.gen_data_disk('../temp/pass.csv', 2, 2, 2**12)
utils.hdfs_put('../temp/pass.csv')
for op in ops:
    mattype_m = 'tall' if op != 'GMM' else 'wide'
    mattype_n = 'tall'

    Mpath_disk = '../external/disk_data/M{}_{}.csv'.format(matsize,mattype_m)
    wPath_disk = '../external/disk_data/w{}_{}.csv'.format(matsize,mattype_m)
    Npath_disk = '../external/disk_data/N{}_{}.csv'.format(matsize,mattype_n)
    if op == 'GMM':
        NPath_disk = '../external/disk_data/M{}_tall.csv'.format(matsize)

    Mpath_hdfs = Mpath_disk.replace('../external/disk_data', '/scratch')
    wPath_hdfs = wPath_disk.replace('../external/disk_data', '/scratch')
    Npath_hdfs = Npath_disk.replace('../external/disk_data', '/scratch')

    cmd_params_disk = {'mattype' : mattype_m,
               'Mpath'   : Mpath_disk,
               'wPath'   : wPath_disk,
               'Npath'   : Npath_disk,
               'nodes'   : nodes,
               'tableStub' : '{}_{}'.format(matsize, mattype_m)}
    cmd_params_hdfs = {'mattype' : mattype_m,
               'Mpath'   : Mpath_hdfs,
               'wPath'   : wPath_hdfs,
               'Npath'   : Npath_hdfs,
               'nodes'   : nodes,
               'tableStub' : '{}_{}'.format(matsize, mattype_m)}

    cmd_params_disk['opType'] = op
    cmd_params_hdfs['opType'] = op
    args_disk = cmd_args.format(**cmd_params_disk)
    args_hdfs = cmd_args.format(**cmd_params_hdfs)
    
    os.system('rm -rf /tmp/systemml')
    if 'SYSTEMML' in systems:
        utils.run_spark(program='SystemMLMatrixOps',
                        sbt_dir='./systemml',
                        cmd_args = args_hdfs)
    if 'MLLIB' in systems:
        utils.run_spark(program = 'SparkMatrixOps',
                    sbt_dir = './mllib',
                    cmd_args = args_hdfs)
    if 'MADLIB' in systems:
        utils.run_python(program = 'madlib_matrix_ops.py',
                           cmd_args = args_disk)
    if 'R' in systems:
        utils.run_pbdR(program = 'R_matrix_ops.R',
                      cmd_args = args_disk)
