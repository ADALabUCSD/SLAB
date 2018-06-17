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
import os
import sys
import numpy as np
import pandas as pd

ROOT = os.getenv('BENCHMARK_PROJECT_ROOT')
if (ROOT is None):
    msg = 'Please set environment variable BENCHMARK_PROJECT_ROOT'
    raise StandardError(msg)

sys.path.append(os.path.join(ROOT,'lib','python'))
from sql_cxn import SQLCxn
import np_timing_utils as utils

def doMatrixOp(kwargs):
    opType  = kwargs.get('opType')
    mattype = kwargs.get('mattype')
    tableStub = kwargs.get('tableStub')
    savestub = kwargs.get('savestub')
    nodes = kwargs.get('nodes')
    outdir = kwargs.get('outdir')
    sr = kwargs.get('sr')

    try:
        tableStub = int(tableStub)
    except ValueError:
        pass

    Mname = 'M{}'.format(tableStub)
    Nname = 'M{}'.format(tableStub)
    wname = 'w{}'.format(tableStub)
    if opType == 'GMM':
        Nname = Mname.replace('wide','tall')

    print 'Evaluating: {}'.format(opType)

    colnames = ['nodes','sr','time1','time2','time3','time4','time5']
    runTimes = pd.DataFrame(np.zeros((1,len(colnames))))
    runTimes.columns = colnames
    runTimes['nodes'] = runTimes['nodes'].astype('O')
    runTimes['sr'] = runTimes['sr'].astype('O')

    cxn = SQLCxn(username='ubuntu', db='ubuntu', timeout=10000)
    
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
        array_call = 'SELECT array_agg(random()) FROM generate_series(1,100)'
        call = "matrix_vec_mult('{}',NULL,({}))".format(Mname,array_call)
        cleanup.append('Mw')
    elif (opType == 'TSM'):
        call = "matrix_mult('{0}','trans=True','{0}',NULL,'MtM',NULL)".format(Mname)
        cleanup.append('MtM')
    elif (opType == 'ADD'):
        call = "matrix_add('{}',NULL,'{}',NULL,'M_N',NULL)".format(Mname, Nname)
        cleanup.append('M_N')
    else:
        raise NotImplementedError('Invalid Operation')

    for obj in cleanup:
        cxn.execute('DROP TABLE IF EXISTS {}'.format(obj))

    sql_call = 'SELECT madlib.{}'.format(call)
    fmt = (outdir, mattype, opType, nodes)
    path = '../output/{}/madlib_{}_{}{}.txt'.format(*fmt)
    res = cxn.time(sql_call, cleanup)
    if (res is None):
        print 'Timed Out'
        return

    runTimes.ix[:,'nodes'] = nodes
    runTimes.ix[:,'sr'] = sr
    runTimes.ix[:,2:] = res
    writeHeader = False if (os.path.exists(path)) else True
    runTimes.to_csv(path, index=False, header = writeHeader, mode = 'a')

if __name__=='__main__':
    kwargs = utils.parse_cmd_args(sys.argv[1:])
    doMatrixOp(kwargs)
