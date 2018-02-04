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
import numpy as np
import pandas as pd
import numpy.linalg as alg

ROOT = os.getenv('BENCHMARK_PROJECT_ROOT')
if (ROOT is None):
    msg = 'Please set environment variable BENCHMARK_PROJECT_ROOT'
    raise StandardError(msg)

sys.path.append(os.path.join(ROOT,'lib','python'))
import np_timing_utils as utils

print 'Using Numpy: ' + np.__version__

def doMatrixOp(kwargs):
    opType = kwargs.get('opType')
    mattype = kwargs.get('mattype')
    fixedAxis = int(kwargs.get('fixedAxis'))
    nrow_scale = map(lambda x: int(x), kwargs['nrows'].split(' '))
    nproc = kwargs.get('nproc')

    colnames = ['rows','time1','time2','time3','time4','time5']
    runTimes = pd.DataFrame(np.zeros((1,len(colnames))))
    runTimes.columns = colnames

    if (opType == 'TRANS'):
        call = 'M.T'
    elif (opType == 'NORM'):
        call = 'alg.norm(M)'
    elif (opType == 'GMM'):
        call = 'M.dot(N)'
    elif (opType == 'MVM'):
        call = 'M.dot(w)'
    elif (opType == 'TSM'):
        call = 'M.T.dot(M)'
    elif (opType == 'ADD'):
        call = 'M+X'
    elif (opType == 'SM'):
        call = 'np.multiply(10,M)'
    else:
        raise NotImplementedError('Invalid Operation')

    if nproc is None:
        path = os.path.join('..','output','np_{}_{}.txt'.format(mattype, opType))
    else:
        path = os.path.join('..','output','np_cpu_{}_scale.txt'.format(opType))

    for nr in nrow_scale:
        nrow = fixedAxis if opType == 'GMM' else nr
        ncol = nr if opType == 'GMM' else fixedAxis

        env = {'alg' : alg, 'np' : np}
        RNG = np.random
        env['M'] = utils.allocMatrix(nrow, ncol, RNG)

        if (opType == 'GMM'):
            env['N'] = utils.allocMatrix(ncol, nrow, RNG)
        elif (opType == 'MVM'):
            env['w'] = utils.allocMatrix(ncol, 1, RNG)
        elif (opType == 'ADD'):
            env['X'] = utils.allocMatrix(nrow, ncol, RNG)

        runTimes.ix[:,'rows'] = nr if nproc is None else nproc
        runTimes.ix[:,1:] = utils.timeOp(call, env)
        writeHeader = False if (os.path.exists(path)) else True
        runTimes.to_csv(path, index=False, header = writeHeader, mode = 'a')

if __name__=='__main__':
    kwargs = utils.parse_cmd_args(sys.argv[1:])
    doMatrixOp(kwargs)
