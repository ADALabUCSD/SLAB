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
import tensorflow as tf

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

    path = '../output/tf_{}.txt'.format(opType)
    colnames = ['nproc','time1','time2','time3','time4','time5']
    runTimes = pd.DataFrame(np.zeros((1,len(colnames))))
    runTimes.columns = colnames

    env = {
        'np': np, 'tf': tf,
        'logit_reg': logit_reg,
        'reg': reg,
        'gnmf': gnmf,
        'robust_se': robust_se
    }
    X = np.random.rand(nrow, ncol).astype(np.float32)
    if opType != 'gnmf':
        y = (np.random.rand(nrow,1) >= 0.80).astype(np.int64)
    else:
        y = None
    if opType == 'logit':
        call = 'logit_reg(X,y)'
    elif opType == 'reg':
        call = 'reg(X,y)'
    elif opType == 'gnmf':
        call = 'gnmf(X, 10)'
    elif opType == 'robust':
        b = reg(X, y)
        y_hat = X.dot(b)
        env['eps'] = np.power(y_hat, 2).ravel()
        call = 'robust_se(X, eps)'
    else:
        raise StandardError('Invalid Operation')

    env['X'] = X
    env['y'] = y
    runTimes.ix[:,'nproc'] = nproc
    runTimes.ix[:,1:] = utils.timeOp(call, env)
    writeHeader = not os.path.exists(path)
    runTimes.to_csv(path, index=False, header = writeHeader, mode = 'a')

def logit_reg(Xdata, ydata, iterations=3):
    def doLogitIter(w, stepSize, iteration):
        iteration += 1
        xb = tf.matmul(X,w)
        delta = tf.subtract(1/(1+tf.exp(-xb)),y)
        stepSize /= float(4.0)
        w = w - stepSize*(tf.matmul(Xt, delta)/N)
        return (w, stepSize, iteration)

    G = tf.Graph()
    with G.as_default():
        N = Xdata.shape[0]
        X = tf.placeholder(tf.float32, shape=Xdata.shape)
        y = tf.placeholder(tf.float32, shape=ydata.shape)

        Xt = tf.transpose(X)
        w = tf.Variable(tf.zeros(shape=(Xdata.shape[1],1)))
        stepSize = 10.0
        iteration = tf.constant(0)

        cond = lambda x,y,z: tf.less(z, iterations)
        loop = tf.while_loop(cond, doLogitIter, (w, stepSize, iteration))

        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            res = sess.run(loop, feed_dict={X: Xdata, y: ydata})

    return res[0]

def gnmf(Xdata, r, iterations=3):
    def doGNMFIter(W, H, iteration):
        W = tf.multiply(W, tf.div(tf.matmul(X, H, transpose_b=True),
                        tf.matmul(W, tf.matmul(H, H, transpose_b=True))))
        H = tf.multiply(H, tf.div(tf.matmul(W, X, transpose_a=True),
                        tf.matmul(tf.matmul(W, W, transpose_a=True), H)))
        iteration += 1
        return (W, H, iteration)

    G = tf.Graph()
    with G.as_default():
        X = tf.placeholder(tf.float32, shape=Xdata.shape)

        W = tf.Variable(tf.random_uniform((Xdata.shape[0],r)))
        H = tf.Variable(tf.random_uniform((r, Xdata.shape[1])))
        iteration = tf.constant(0)

        cond = lambda x,y,z: tf.less(z, iterations)
        loop = tf.while_loop(cond, doGNMFIter, (W,H,iteration))

        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            res = sess.run(loop, feed_dict={X : Xdata})

    return (res[0], res[1])

def reg(Xdata, ydata):
    G = tf.Graph()
    with G.as_default():
        X = tf.placeholder(tf.float32, shape=Xdata.shape)
        y = tf.placeholder(tf.float32, shape=ydata.shape)

        b = tf.matrix_solve(
                tf.matmul(X, X, transpose_a=True),
                tf.matmul(X, y, transpose_a=True)
            )

        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            res = sess.run(b, feed_dict={X: Xdata, y: ydata})

    return res

def robust_se(Xdata, epsdata):
    G = tf.Graph()
    with G.as_default():
        X = tf.placeholder(tf.float32, shape=Xdata.shape)
        eps = tf.placeholder(tf.float32, shape=epsdata.shape)
        S = tf.transpose( X )*eps
        XTX_INV = tf.matrix_inverse(tf.matmul(X,X,transpose_a=True))
        VAR = tf.matmul(
            tf.matmul(XTX_INV, tf.matmul(S,X)), XTX_INV)
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            res = sess.run(VAR, feed_dict={X: Xdata, eps: epsdata})

    return res

if __name__=='__main__':
    args = utils.parse_cmd_args(sys.argv[1:])
    main(args)
