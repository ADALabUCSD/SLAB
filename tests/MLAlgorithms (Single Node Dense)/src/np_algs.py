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
import numpy.linalg as alg
import pandas as pd

ROOT = os.getenv('BENCHMARK_PROJECT_ROOT')
if (ROOT is None):
    msg = 'Please set environment variable BENCHMARK_PROJECT_ROOT'
    raise StandardError(msg)

sys.path.append(os.path.join(ROOT,'lib','python'))
from sql_cxn import SQLCxn
import np_timing_utils as utils

def main(kwargs):
    mattype = kwargs['mattype']
    opType = kwargs['opType']
    nrow = int(kwargs['nrow'])
    ncol = int(kwargs['ncol'])
    nproc = int(kwargs['nproc'])

    path = '../output/np_{}.txt'.format(opType)
    colnames = ['nproc','time1','time2','time3','time4','time5']
    runTimes = pd.DataFrame(np.zeros((1,len(colnames))))
    runTimes.columns = colnames

    env = {
        'logit_reg': logit_reg,
        'reg': reg,
        'gnmf': gnmf,
        'robust_se': robust_se
    }
    X = np.random.rand(nrow, ncol)
    y = np.random.rand(nrow,1).ravel() if opType != 'gnmf' else None

    if opType == 'logit':
        call = 'logit_reg(X,y)'
    elif opType == 'reg':
        call = 'reg(X,y)'
    elif opType == 'gnmf':
        call = 'gnmf(X, 10)'
    elif opType == 'robust':
        b = reg(X, y)
        y_hat = X.dot(b)
        env['eps'] = np.power(y_hat, 2)
        call = 'robust_se(X, eps)'
    else:
        raise StandardError('Invalid Operation')

    env['X'] = X
    env['y'] = y
    runTimes.ix[:,'nproc'] = nproc
    runTimes.ix[:,1:] = utils.timeOp(call, env)
    writeHeader = not os.path.exists(path)
    runTimes.to_csv(path, index=False, header = writeHeader, mode = 'a')

def logit_reg(X, y, iterations=3):
    N,K = X.shape
    w = np.random.rand(K,1).ravel()

    iteration = 0
    step_size = 0.001

    while iteration < iterations:
        xb = X.dot(w)
        delta = y - (1/1+np.exp(-xb))
        step_size /= 2
        w = w + step_size*(X.T.dot(delta)/float(N))
        iteration += 1

    return w

def gnmf(X, r, iterations=3):
    N,K = X.shape
    W = np.random.rand(N, r)
    H = np.random.rand(r, K)

    iteration = 0
    while iteration < iterations:
        W = W*((X.dot(H.T))/(W.dot(H.dot(H.T))))
        H = H*((W.T.dot(X))/((W.T.dot(W).dot(H))))
        iteration += 1

    return W,H

def reg(X,y):
    return alg.solve(X.T.dot(X), X.T.dot(y))

def robust_se(X, r2):
    S = X.T*r2
    XTX_INV = alg.inv(X.T.dot(X))
    return XTX_INV.dot(S.dot(X)).dot(XTX_INV)

if __name__=='__main__':
    args = utils.parse_cmd_args(sys.argv[1:])
    main(args)
