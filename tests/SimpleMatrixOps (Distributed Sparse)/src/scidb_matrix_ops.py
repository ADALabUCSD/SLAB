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
    savestub = kwargs.get('savestub')
    nodes = kwargs.get('nodes')
    outdir = kwargs.get('outdir')
    sr = kwargs.get('sr')
    sr_val = np.float64('0.{}'.format(sr))

    nrow, ncol = 125000000, 100
    path = '../output/scidb_{}_{}{}.txt'.format(mattype, op_type, nodes)

    cxn = scidbpy.connect()
    print cxn.iquery("list('instances')", fetch=True)
    colnames = ['nodes','sr','time1','time2','time3','time4','time5']
    run_times = pd.DataFrame(np.zeros((1,len(colnames))))
    run_times.columns = colnames

    M_name = 'M{}'.format(sr)
    if not M_name in dir(cxn.arrays):
        alloc_matrix(nrow, ncol, M_name, cxn, density=sr_val)
    if op_type == 'GMM':
        if not 'M{}W' in dir(cxn.arrays):
            alloc_matrix(
                ncol, nrow, 'M{}W'.format(sr), cxn, density=sr_val)
        if not 'N{}'.format(sr) in dir(cxn.arrays):
            alloc_matrix(
                nrow, ncol, 'N{}'.format(sr), cxn, density=sr_val)
        N_name = 'N{}'.format(sr)
        M_name = 'M{}W'.format(sr)
    if op_type == 'ADD':
        if not 'N{}'.format(sr) in dir(cxn.arrays):
            alloc_matrix(
                nrow, ncol, 'N{}'.format(sr), cxn, density=sr_val)
        N_name = 'N{}'.format(sr)
    if op_type == 'MVM':
        v_name = 'v{}'.format(ncol)
        if not v_name in dir(cxn.arrays):
            alloc_vector(ncol, v_name, cxn)

    cxn.iquery("load_library('linear_algebra')")
    if op_type == 'TRANS':
        call = 'consume(transpose({}))'.format(M_name)
    elif op_type == 'NORM':
        call = 'aggregate(apply({}, val2, pow(val,2.0)), sum(val2))'.format(
            M_name)
    elif op_type == 'GMM':
        call = 'spgemm({},{})'.format(M_name, N_name)
    elif op_type == 'MVM':
        call = 'spgemm({},{})'.format(M_name, v_name)
    elif op_type == 'TSM':
        call = 'spgemm(transpose({}),{})'.format(M_name, M_name)
    elif op_type == 'ADD':
        call = 'consume(apply(join({0},{1}), sum, {0}.val+{1}.val))'.format(
            M_name, N_name)
    else:
        raise StandardError('Invalid operator type')
    
    run_times.ix[:,:2] = (nodes, sr)
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
        CREATE ARRAY {n}<val:double not null>[row=0:{r}:0:{cz};col=0:{c}:0:{cz}]
    """.format(r=rows, n=name, c=cols, cz=chunksize)
    cxn.iquery(stmt)

def alloc_matrix(rows, cols, name, 
        cxn, chunksize=1000, overwrite=False, density=0.0):
    
    if name in dir(cxn) and overwrite is False:
        return
    if name in dir(cxn) and overwrite:
        cxn.iquery()
    if 'tmp' in dir(cxn):
        cxn.iquery('remove(tmp)')

    method = 'iif((RANDOM() % 100000) / 100000.0 >= {}, (RANDOM() % 100) / 10.0, 0)'.format(1-density)
    stmt = """
        store(build(<val:double not null>[row=0:{r}:0:{cz};col=0:{c}:0:{cz}], {m}), tmp)
    """.format(r=rows, c=cols, cz=chunksize, m=method)
    print stmt
    cxn.iquery(stmt)

    print "store(filter(tmp, val <> 0), {n})".format(n=name)
    cxn.iquery("store(filter(tmp, val <> 0), {n})".format(n=name))
    cxn.iquery("remove(tmp)")

def alloc_vector(length, name, cxn, method='(RANDOM() % 100) / 10.0'):
    stmt = """
        store(build(<val:double not null>[row=0:{};col=0:0], {}), {})
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
