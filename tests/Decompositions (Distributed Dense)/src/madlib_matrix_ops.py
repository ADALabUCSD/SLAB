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
    savestub  = kwargs.get('savestub')
    nodes  = kwargs.get('nodes')
    outdir = kwargs.get('outdir')

    savestub = '' if (savestub is None) else savestub
    try:
        tableStub = int(tableStub)
    except ValueError:
        pass

    Mname = 'M{}'.format(tableStub)
    Nname = 'N{}'.format(tableStub)
    wname = 'w{}'.format(tableStub)

    print 'Evaluating: {}'.format(opType)

    colnames = ['nodes','rows','cols','time1','time2','time3','time4','time5']
    runTimes = pd.DataFrame(np.zeros((1,len(colnames))))
    runTimes.columns = colnames

    cxn = SQLCxn(username='ubuntu', db='ubuntu', timeout = 2000)
    shape = cxn.get_shape_dense('M{}'.format(tableStub))

    cleanup = []
    if (opType == 'SVD'):
        call = "svd('{}','svd','row_num',10, 10,'svd_summary')".format(
            Mname, shape[1])
        cleanup.append('svd_s')
        cleanup.append('svd_u')
        cleanup.append('svd_v')
        cleanup.append('svd_summary')
    else:
        raise NotImplementedError('Invalid Operation')

    for obj in cleanup:
        cxn.execute('DROP TABLE IF EXISTS {}'.format(obj))

    sql_call = 'SELECT madlib.{}'.format(call)
    rows = shape[0]
    cols = shape[1]
    path = '../output/{}/madlib_{}_{}{}.txt'.format(
        outdir, mattype, opType, int(nodes))
    runTimes.ix[:,['nodes','rows','cols']] = (nodes, rows, cols)
    madlib_timeout = ('../temp/madlib_punked_out.json', opType)
    res = cxn.time(sql_call, cleanup, madlib_timeout)
    if (res is None):
        print 'Timed Out'
        return
    runTimes.ix[:,3:] = res
    writeHeader = False if (os.path.exists(path)) else True
    runTimes.to_csv(path, index=False, header = writeHeader, mode = 'a')

if __name__=='__main__':
    kwargs = utils.parse_cmd_args(sys.argv[1:])
    doMatrixOp(kwargs)
