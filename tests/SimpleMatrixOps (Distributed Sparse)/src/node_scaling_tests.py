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
sparsity = sys.argv[2].split(' ')
systems = sys.argv[3].split(' ')
op_types = sys.argv[4].split(' ')
sparse_gb = sys.argv[5]

cmd_args = ('opType={opType} mattype={mattype}'
            ' Mpath={Mpath} Npath={Npath}'
            ' wPath={wPath} tableStub={tableStub}'
            ' nodes={nodes} passPath=/scratch/pass.csv'
            ' savestub={savestub} sr={sr} '
            ' outdir=../output/scale_nodes')

data.gen_data_disk('../temp/pass.csv', 2, 2, 2**12)
utils.hdfs_put('../temp/pass.csv')

gb = sparse_gb
for op in op_types:
    for sr in sparsity:
        mattype_m = 'tall' if op != 'GMM' else 'wide'
        mattype_n = 'tall'
        fmt = (sr, gb, mattype_m)

        Mpath_disk = '../external/disk_data/M_{}{}_sparse_{}.mtx'.format(*fmt)
        wPath_disk = '../external/disk_data/w_{}{}_sparse_{}.mtx'.format(*fmt)
        Npath_disk = '../external/disk_data/N_{}{}_sparse_{}.mtx'.format(*fmt)
        if op == 'GMM':
            Npath_disk = Mpath_disk.replace('wide','tall')

        Mpath_hdfs = Mpath_disk.replace('../external/disk_data', '/scratch')
        wPath_hdfs = wPath_disk.replace('../external/disk_data', '/scratch')
        Npath_hdfs = Npath_disk.replace('../external/disk_data', '/scratch')

        cmd_params_disk = {'mattype' : mattype_m,
                           'Mpath'   : Mpath_disk,
                           'wPath'   : wPath_disk,
                           'Npath'   : Npath_disk,
                           'nodes'   : nodes,
                           'savestub': gb,
                           'sr'      : sr,
                           'tableStub' : '_{}{}_sparse_{}'.format(*fmt)}
        cmd_params_hdfs = {'mattype' : mattype_m,
                           'Mpath'   : Mpath_hdfs,
                           'wPath'   : wPath_hdfs,
                           'Npath'   : Npath_hdfs,
                           'nodes'   : nodes,
                           'savestub': gb,
                           'sr'      : sr,
                           'tableStub' : '_{}{}_sparse_{}'.format(*fmt)}

        cmd_params_disk['opType'] = op
        cmd_params_hdfs['opType'] = op
        args_disk = cmd_args.format(**cmd_params_disk)
        args_hdfs = cmd_args.format(**cmd_params_hdfs)

        if 'SYSTEMML' in systems:
            utils.run_spark(program = 'SystemMLMatrixOps',
                            sbt_dir = './systemml',
                            cmd_args = args_hdfs)
        if 'MLLIB' in systems:
            utils.run_spark(program = 'SparkMatrixOps',
                            sbt_dir = './mllib',
                            cmd_args = args_hdfs)
        if ('MADLIB' in systems) and (op != 'MVM'):
            utils.run_python(program = 'madlib_matrix_ops.py',
                             cmd_args = args_disk)

# stop logging
end_make_logging()
