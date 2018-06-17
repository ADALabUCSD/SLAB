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
from np_timing_utils import parse_cmd_args

def do_matrix_op(kwargs):
    op_type  = kwargs.get('opType')
    mattype = kwargs.get('mattype')
    tableStub = kwargs.get('tableStub')
    savestub  = kwargs.get('savestub')
    nodes  = kwargs.get('nodes')

    nrow, ncol = get_dims('M{}'.format(tableStub))
    path = '../output/scidb_{}_{}{}.txt'.format(mattype, op_type, nodes)

    cxn = scidbpy.connect()
    print cxn.iquery("list('instances')", fetch=True)
    colnames = ['nodes','rows','time1','time2','time3','time4','time5']
    run_times = pd.DataFrame(np.zeros((1,len(colnames))))
    run_times.columns = colnames

    M_name = 'M{}{}'.format(nrow, ncol)
    if not M_name in dir(cxn.arrays):
        alloc_matrix(nrow, ncol, M_name, cxn)
    if op_type == 'GMM':
        if not 'N{}{}'.format(ncol, nrow) in dir(cxn.arrays):
            alloc_matrix(ncol, nrow, 'N{}{}'.format(ncol, nrow), cxn)
        N_name = 'N{}{}'.format(ncol, nrow)
        zv_name = 'ZEROS{}{}'.format(nrow, nrow)
        zeros(nrow, nrow, zv_name, cxn)
    if op_type == 'TSM':
        zv_name = 'ZEROS{}{}'.format(ncol, ncol)
        zeros(ncol, ncol, zv_name, cxn)
    if op_type == 'ADD':
        if not 'N{}{}'.format(nrow, ncol) in dir(cxn.arrays):
            alloc_matrix(nrow, ncol, 'N{}{}'.format(nrow, ncol), cxn)
        N_name = 'N{}{}'.format(nrow, ncol)
    if op_type == 'MVM':
        v_name = 'v{}'.format(ncol)
        if not v_name in dir(cxn.arrays):
            alloc_vector(ncol, v_name, cxn)
        zv_name = 'ZEROS{}'.format(nrow)
        zeros(nrow, 0, zv_name, cxn)

    cxn.iquery("load_library('dense_linear_algebra')")
    if op_type == 'TRANS':
        call = 'consume(transpose({}))'.format(M_name)
    elif op_type == 'NORM':
        call = 'aggregate(apply({}, val2, pow(val,2.0)), sum(val2))'.format(
            M_name)
    elif op_type == 'GMM':
        call = 'gemm({},{},{})'.format(M_name, N_name, zv_name)
    elif op_type == 'MVM':
        call = 'gemm({},{},{})'.format(M_name, v_name, zv_name)
    elif op_type == 'TSM':
        call = 'gemm({},{},{}, transa:true)'.format(M_name, M_name, zv_name)
    elif op_type == 'ADD':
        call = 'consume(apply(join({0},{1}), sum, {0}.val+{1}.val))'.format(
            M_name, N_name)
    else:
        raise StandardError('Invalid operator type')
    
    main_axis = ncol if op_type == 'GMM' else nrow
    run_times.ix[:,:2] = (nodes, main_axis)
    run_times.ix[:,2:] = time_stmt(call, cxn)
    write_header = False if (os.path.exists(path)) else True
    run_times.to_csv(path, index=False, header=write_header, mode='a')

def get_dims(name):
    with open('../external/disk_data/{}.csv.mtd'.format(name), 'rb') as fh:
        res = json.load(fh)
    return (res['rows'], res['cols'])

def zeros(rows, cols, name, cxn, chunksize=1000, overwrite=True):
    if overwrite and (name in dir(cxn.arrays)):
        cxn.iquery('remove({})'.format(name))
    stmt = """
        CREATE ARRAY {n}<val:double>[row=0:{r}:0:{cz};col=0:{c}:0:{cz}]
    """.format(r=rows, n=name, c=cols, cz=chunksize)
    cxn.iquery(stmt)

def alloc_matrix(rows, cols, name, 
        cxn, method='(RANDOM() % 100) / 10.0',
        chunksize=1000):
    stmt = """
        store(build(<val:double>[row=0:{r}:0:{cz};col=0:{c}:0:{cz}], {m}), {n})
    """.format(r=rows, c=cols, cz=chunksize, m=method, n=name)
    print stmt
    cxn.iquery(stmt)

def alloc_vector(length, name, cxn, method='(RANDOM() % 100) / 10.0'):
    stmt = """
        store(build(<val:double>[row=0:{};col=0:0], {}), {})
    """.format(length, method, name)
    print stmt
    cxn.iquery(stmt)

def time_stmt(stmt, cxn, cleanup=None, n_reps=5):
    times = []
    cleanup = cleanup if cleanup is not None else []
    for ix in range(n_reps):
        if ix == 0:
            print stmt
        start = time.time()
        cxn.iquery(stmt)
        stop = time.time()
        print 'Test {} => {}'.format(ix, stop-start)
        if len(cleanup) > 0:
            for obj in cleanup:
                cxn.iquery('remove({})'.format(obj))
        times.append(stop-start)
    return times

if __name__=='__main__':
    argv = parse_cmd_args(sys.argv[1:])
    do_matrix_op(argv)
