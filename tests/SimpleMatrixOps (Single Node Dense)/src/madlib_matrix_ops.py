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

with open(__file__) as fh: print fh.read()
import sys
import os
import atexit
import numpy as np
import pandas as pd
import numpy.linalg as alg

ROOT = os.getenv('BENCHMARK_PROJECT_ROOT')
if (ROOT is None):
    msg = 'Please set environment variable BENCHMARK_PROJECT_ROOT'
    raise StandardError(msg)

sys.path.append(os.path.join(ROOT,'lib','python'))
from sql_cxn import SQLCxn
import np_timing_utils as utils

GPDB_PORT_MAP = {'1': 5481, '2': 6431, '4': 6431, '8': 6431, '16': 6431}
def doMatrixOp(kwargs):
    opType = kwargs.get('opType')
    mattype = kwargs.get('mattype')
    fixedAxis = int(kwargs.get('fixedAxis'))
    nrow_scale = map(lambda x: int(x), kwargs['nrows'].split(' '))
    nproc = kwargs.get('nproc')

    port = GPDB_PORT_MAP[nproc] if nproc is not None else None
    
    if nproc is not None:
        cxn = start_gpdb(port, nproc)
        cxn.execute('DROP TABLE IF EXISTS M16_tall')
        atexit.register(stop_gpdb, nproc, cxn)
    else:
        cxn = SQLCxn(username='ubuntu', db='ubuntu', timeout=10000)
    
    colnames = ['rows','time1','time2','time3','time4','time5']
    runTimes = pd.DataFrame(np.zeros((1,len(colnames))))
    runTimes.columns = colnames

    if nproc is None:
        path = os.path.join('..','output','madlib_{}_{}.txt'.format(mattype, opType))
    else:
        path = os.path.join('..','output','madlib_cpu_{}_scale.txt'.format(opType))
    for nr in nrow_scale:
        nrow = fixedAxis if opType == 'GMM' else nr
        ncol = nr if opType == 'GMM' else fixedAxis
        print nrow
        print ncol
        Mname = 'M{}{}'.format(nrow,ncol)
        if not cxn.table_exists('M{}{}'.format(nrow,ncol)):
            cxn.randomMatrix(nrow, ncol, 'M{}{}'.format(nrow, ncol))
        if (opType == 'GMM'):
            if not cxn.table_exists('N{}{}'.format(ncol, nrow)):
                cxn.randomMatrix(ncol, nrow, 'N{}{}'.format(ncol, nrow))
            Nname = 'N{}{}'.format(ncol, nrow)
        elif (opType == 'ADD'):
            if not cxn.table_exists('N{}{}'.format(nrow, ncol)):
                cxn.randomMatrix(nrow, ncol, 'N{}{}'.format(nrow, ncol))
            Nname = 'N{}{}'.format(nrow, ncol)

        cleanup = []
        if (opType == 'TRANS'):
            call = "matrix_trans('{}',NULL,'Mt',NULL)".format(Mname)
            cleanup.append('Mt')
        elif (opType == 'NORM'):
            call = "matrix_norm('{}',NULL,'fro')".format(Mname)
        elif (opType == 'GMM'):
            call = "matrix_mult('{}',NULL,'{}',NULL,'MN',NULL)".format(Mname,Nname)
            cleanup.append('MN')
        elif (opType == 'MVM'):
            array_call = 'SELECT array_agg(random()) FROM generate_series(1,{})'.format(
                ncol)
            call = "matrix_vec_mult('{}',NULL,({}))".format(Mname,array_call)
        elif (opType == 'TSM'):
            call = "matrix_mult('{0}','trans=True','{0}',NULL,'MtM',NULL)".format(Mname)
            cleanup.append('MtM')
        elif (opType == 'ADD'):
            call = "matrix_add('{}',NULL,'{}',NULL,'M_N',NULL)".format(Mname, Nname)
            cleanup.append('M_N')
        else:
            raise NotImplementedError('Invalid Operation')

        sql_call = 'SELECT madlib.{}'.format(call)
        runTimes.ix[:,'rows'] = nr if nproc is None else nproc
        runTimes.ix[:,1:] = cxn.time(sql_call, cleanup)
        writeHeader = False if (os.path.exists(path)) else True
        runTimes.to_csv(path, index=False, header = writeHeader, mode = 'a')

def start_gpdb(port, nproc):
    if port is None:
        os.system('yes | gpstart')
        cxn = SQLCxn(username='ubuntu', db='ubuntu', timeout=10000)
    else:
        call = 'yes | gpstart -d /gpsegs/gpdb-{}/master/gpseg-1'.format(nproc)
        os.system(call)
        cxn = SQLCxn(username='ubuntu', db='ubuntu', timeout=10000, port=port)
    return cxn

def stop_gpdb(nproc, cxn):
    cxn._cxn.close()
    if nproc is None:
        os.system('yes | gpstop')
    else:
        call = 'yes | gpstop -d /gpsegs/gpdb-{}/master/gpseg-1'.format(nproc)
        os.system(call)

if __name__=='__main__':
    kwargs = utils.parse_cmd_args(sys.argv[1:])
    doMatrixOp(kwargs)
