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
import time
import sys
import os
import numpy as np
import pandas as pd
import tensorflow as tf

ROOT = os.getenv('BENCHMARK_PROJECT_ROOT')
if (ROOT is None):
    msg = 'Please set environment variable BENCHMARK_PROJECT_ROOT'
    raise StandardError(msg)

sys.path.append(os.path.join(ROOT,'lib','python'))
import tf_timing_utils as utils
from np_timing_utils import parse_cmd_args

print 'Using TensorFlow: ' + tf.__version__

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
        opStr = 'tf.transpose(M)'
    elif (opType == 'NORM'):
        opStr = 'tf.sqrt(tf.reduce_sum(tf.square(M)))'
    elif (opType == 'GMM'):
        opStr = 'tf.matmul(M,N)'
    elif (opType == 'MVM'):
        opStr = 'tf.matmul(M,w)'
    elif (opType == 'TSM'):
        opStr = 'tf.matmul(M,M,transpose_a=True)'
    elif (opType == 'ADD'):
        opStr = 'tf.add(M,X)'
    else:
        raise NotImplementedError('Invalid Operation')

    if nproc is None:
        path = os.path.join('..','output','tf_{}_{}.txt'.format(mattype, opType))
    else:
        path = os.path.join('..','output','tf_cpu_{}_scale.txt'.format(opType))
    for nr in nrow_scale:
        nrow = fixedAxis if opType == 'GMM' else nr
        ncol = nr if opType == 'GMM' else fixedAxis

        G = tf.Graph()
        with G.as_default():
            M = tf.Variable(tf.random_uniform(shape=(nrow,ncol)))
            if (opType == 'GMM'):
                N = tf.Variable(tf.random_uniform(shape=(ncol,nrow)))
            elif (opType == 'MVM'):
                w = tf.Variable(tf.random_uniform(shape=(ncol,1)))
            elif (opType == 'ADD'):
                X = tf.Variable(tf.random_uniform(shape=(nrow,ncol)))
            
            op = eval(opStr)
            init_op = tf.global_variables_initializer()

            with tf.Session() as sess:
                sess.run(init_op)
                runTimes.ix[:,'rows'] = nr if nproc is None else nproc
                result = utils.timeOp(op, sess)
                runTimes.ix[:,1:] = result
                if (result is None):
                    return

        writeHeader = False if (os.path.exists(path)) else True
        runTimes.to_csv(path, 
                        index=False, 
                        header = writeHeader, 
                        mode = 'a')

if __name__=='__main__':
    kwargs = parse_cmd_args(sys.argv[1:])
    doMatrixOp(kwargs)
