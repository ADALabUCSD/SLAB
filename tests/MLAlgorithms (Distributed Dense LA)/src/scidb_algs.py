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
import time
import json
import scidbpy
import subprocess

import numpy as np
import pandas as pd

ROOT = os.getenv('BENCHMARK_PROJECT_ROOT')
if (ROOT is None):
    msg = 'Please set environment variable BENCHMARK_PROJECT_ROOT'
    raise StandardError(msg)

sys.path.append(os.path.join(ROOT,'lib','python'))
import np_timing_utils as utils

def main(kwargs):
    op_type  = kwargs.get('opType')
    nodes = kwargs.get('nodes')
    x_table_name = kwargs.get('xTableName')

    nrow, ncol = get_dims(x_table_name)

    path = '../output/scidb_{}{}.txt'.format(op_type, nodes)
    colnames = ['nodes','rows','cols','time1','time2','time3','time4','time5']
    run_times = pd.DataFrame(np.zeros((1,len(colnames))))
    run_times.columns = colnames

    cxn = scidbpy.connect()
    cxn.iquery("load_library('dense_linear_algebra')")

    env = {
        'cxn': cxn, 
        'reg': reg,
        'logit': logit,
        'gnmf': gnmf,
        'robust_se': robust_se
    }

    alloc_matrix(nrow, ncol, 'X{}{}'.format(nrow, ncol), cxn, overwrite=False)
    alloc_matrix(nrow, 0, 'y{}{}'.format(nrow, ncol), cxn,
        overwrite=False, binary=True)
    if op_type == 'reg':
        call = "reg('X{0}{1}', 'y{0}{1}', cxn)".format(nrow, ncol)
    elif op_type == 'logit':
        alloc_matrix(nrow, 0, 'y{}{}b'.format(nrow, ncol), 
            cxn, overwrite=False, binary=True)
        call = "logit('X{0}{1}', 'y{0}{1}b', cxn)".format(nrow, ncol)
    elif op_type == 'gnmf':
        call = "gnmf('X{0}{1}', 10, cxn)".format(nrow, ncol)
    elif op_type == 'robust':
        alloc_matrix(nrow, 0, 'r2{}{}'.format(nrow, ncol), 
            cxn, overwrite=True, val_name='residuals')
        call = "robust_se('X{0}{1}', 'r2{0}{1}', cxn)".format(nrow, ncol)

    run_times.loc[:,['nodes','rows','cols']] = (nodes, nrow, ncol)
    run_times.loc[:,3:] = utils.timeOp(call, env)
    write_header = not os.path.exists(path)
    run_times.to_csv(path, index=False, header=write_header, mode='a')

def get_dims(name):
    with open('../external/disk_data/{}.csv.mtd'.format(name), 'rb') as fh:
        res = json.load(fh)
    return (res['rows'], res['cols'])

def reg(X_table_name, y_table_name, cxn):
    # We need to explicitly form the inverse using SVD since SciDB does not
    # provide an inversion operator nor does it provide an operator to solve
    # a linear system
    
    temp_tables = ['XTX','XTX_INV']
    for t in temp_tables:
        safe_drop(t, cxn)
    
    N,K = get_shape(X_table_name, cxn)
    zname = 'Z{0}{0}'.format(K)
    zeros(K, K, zname, cxn)

    # materialize the inner produce matrix
    cxn.iquery('store(gemm({X}, {X}, {Z}, transa:true), XTX)'.format(
        X=X_table_name, Z=zname))
    
    # compute the inverse of XTX
    stmt = """
        store(gemm(project(apply(cross_join(transpose(gesvd(XTX, 'VT')) as V, 
            project(apply(
                    gesvd(XTX, 'S'), sigma_inv, POW(sigma,-1)), sigma_inv) 
                AS SINV, V.i, SINV.i), vsinv, v*sigma_inv), vsinv), 
            transpose(gesvd(XTX, 'U')), {Z}), XTX_INV)        
    """.format(Z=zname)
    print stmt
    cxn.iquery(stmt)

    # now do the thing
    zname = 'Z{}1'.format(K)
    zeros(K, 0, zname, cxn)
    stmt = """
        gemm(XTX_INV, gemm({X}, {y}, {Z}, transa:true), {Z})
    """.format(X=X_table_name, y=y_table_name, Z=zname)
    print stmt
    cxn.iquery(stmt)

def logit(X_table_name, y_table_name, cxn, iterations=3):
    N,K = get_shape(X_table_name, cxn)
    alloc_matrix(K, 0, 'w', cxn, overwrite=True)
    alloc_matrix(K, 0, 'ZK', cxn, method=0, overwrite=True)
    alloc_matrix(N, 0, 'ZN', cxn, method=0, overwrite=True)

    logit_iter = """
        store(
            project(
            apply(join(w, project(
            apply(
            gemm({X},
            project(apply(
                join(gemm({X}, w, ZN), {y}), delta, val - pow(1+EXP(-gemm),-1)), 
            delta), ZK, transa:true), norm, 0.001*gemm/{N}), norm)
        ), update, val+norm), update), w_new)
    """.format(X=X_table_name, y=y_table_name, N=N)

    iteration = 0
    while iteration < iterations:
        print 'Iteration: {}'.format(iteration)
        cxn.iquery(logit_iter)
        cxn.iquery('remove(w)')
        cxn.iquery('store(project(apply(w_new, val, update), val), w)')
        iteration += 1

def robust_se(X_table_name, r2_table_name, cxn):
    temp_tables = ['XTX','XTX_INV']
    for t in temp_tables:
        safe_drop(t, cxn)
    
    N,K = get_shape(X_table_name, cxn)
    zkkname = 'Z{0}{0}'.format(K)
    zeros(K, K, zkkname, cxn)

    znkname = 'Z{}{}'.format(K,N)
    zeros(N, K, znkname, cxn)

    zknname = 'Z{}{}'.format(N,N)
    zeros(K, N, zknname, cxn)

    # materialize the inner produce matrix
    print 'store(gemm({X}, {X}, {Z}, transa:true), XTX)'.format(
        X=X_table_name, Z=zkkname)
    cxn.iquery('store(gemm({X}, {X}, {Z}, transa:true), XTX)'.format(
        X=X_table_name, Z=zkkname))

    # compute the inverse of XTX
    stmt = """
        store(gemm(project(apply(cross_join(transpose(gesvd(XTX, 'VT')) as V, 
            project(apply(
                    gesvd(XTX, 'S'), sigma_inv, POW(sigma,-1)), sigma_inv) 
                AS SINV, V.i, SINV.i), vsinv, v*sigma_inv), vsinv), 
            transpose(gesvd(XTX, 'U')), {Z}), XTX_INV)        
    """.format(Z=zkkname)
    print stmt
    cxn.iquery(stmt)

    # now compute the sandwich se = (XTX)^{1} x XT x diag(r2) x X x (XTX^{-1})
    stmt = """
        gemm(XTX_INV, gemm(transpose(redimension(project(apply(
                cross_join({X} as X, {r2} as r2, X.row, r2.row),
         XR2, residuals*val), XR2), <XR2:double>[row=0:{N}:0:1000;col=0:{K}:0:1000]))
            , gemm({X}, XTX_INV, {ZNK}), {ZKK}), {ZKK})
    """.format(X=X_table_name, r2=r2_table_name,
               ZKK=zkkname, ZNK=znkname, N=N, K=K)
    print stmt
    cxn.iquery(stmt)

def gnmf(X_table_name, r, cxn, iterations=3):
    N,K = get_shape(X_table_name, cxn)
    
    alloc_matrix(N, r, 'W_old', cxn, overwrite=True)
    alloc_matrix(r, K, 'H_old', cxn, overwrite=True)

    # no actual data is allocated for zeros creation
    zeros(N, r, 'ZXHT', cxn, overwrite=True)
    zeros(r, r, 'ZHHT', cxn, overwrite=True)
    zeros(N, r, 'ZWHHT', cxn, overwrite=True)
    zeros(r, K, 'ZWTX', cxn, overwrite=True)
    zeros(r, r, 'ZWTW', cxn, overwrite=True)
    zeros(r, K, 'ZWTWH', cxn, overwrite=True)

    build_new_W = """
        store(project(apply(project(apply(join(W_old,
            project(apply(join(
                project(apply(gemm({X}, H_old, ZXHT, transb: true), prod, gemm), prod),
                gemm(W_old, gemm(H_old, H_old, ZHHT, transb: true), ZWHHT)),
            div, prod/gemm), div)),
        val_new, val*div), val_new), val, val_new), val), W_new)
    """.format(X=X_table_name)

    build_new_H = """
        store(project(apply(project(apply(join(H_old,
            project(apply(join(
                project(apply(gemm(W_old, {X}, ZWTX, transa: true), prod, gemm), prod),
                gemm(gemm(W_old, W_old, ZWTW, transa: true), H_old, ZWTWH)),
            div, prod/gemm), div)),
        val_new, val*div), val_new), val, val_new), val), H_new)
    """.format(X=X_table_name)

    iteration = 0
    while iteration < iterations:
        print 'Iteration: {}'.format(iteration)
        cxn.iquery(build_new_W)
        cxn.iquery('remove(W_old)')
        cxn.iquery('rename(W_new, W_old)')
        cxn.iquery(build_new_H)
        cxn.iquery('remove(H_old)')
        cxn.iquery('rename(H_new, H_old)')
        iteration += 1

def safe_drop(name, cxn):
    if name in dir(cxn):
        cxn.iquery('remove({})'.format(name))

def get_shape(name, cxn):
    dims = cxn.iquery('dimensions({})'.format(name), fetch=True)
    N = dims.loc[0,'high']
    K = dims.loc[1,'high']
    return (N,K)

def alloc_matrix(rows, cols, name, cxn, 
        method='(RANDOM() % 100) / 10.0',
        overwrite=False, val_name='val',
        binary=False, chunksize=1000):
    if (name in dir(cxn.arrays)) and (overwrite is False):
        return
    elif name in dir(cxn.arrays):
        cxn.iquery('remove({})'.format(name))
    
    if binary is True:
        method = 'iif(RANDOM() % 100 > 80, 1, 0)'
    stmt = """
        store(build(<{vn}:double>[row=0:{r}:0:{cz};col=0:{c}:0:{cz}], {m}), {n})
    """.format(
        vn=val_name, r=rows, c=cols, m=method, n=name, cz=chunksize)
    print stmt
    cxn.iquery(stmt)

def zeros(rows, cols, name, cxn, chunksize=1000, overwrite=True):
    if overwrite and (name in dir(cxn.arrays)):
        cxn.iquery('remove({})'.format(name))
    stmt = """
        CREATE ARRAY {n}<val:double>[row=0:{r}:0:{cz};col=0:{c}:0:{cz}]
    """.format(r=rows, n=name, c=cols, cz=chunksize)
    print stmt
    cxn.iquery(stmt)

if __name__=='__main__':
    argv = utils.parse_cmd_args(sys.argv[1:])
    main(argv)
