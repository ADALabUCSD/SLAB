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

def main(kwargs):
    opType  = kwargs.get('opType')
    savestub = kwargs.get('savestub')
    nodes = kwargs.get('nodes')
    x_table_name = kwargs.get('xTableName')
    y_table_name = kwargs.get('yTableName')

    savestub = '' if (savestub is None) else savestub

    print 'Evaluating: {}'.format(opType)

    cxn = SQLCxn(username='ubuntu', db='ubuntu', timeout=10000)

    colnames = ['nodes','rows','cols','time1','time2','time3','time4','time5']
    runTimes = pd.DataFrame(np.zeros((1,len(colnames))))
    runTimes.columns = colnames

    shape = cxn.get_shape(x_table_name)

    env = {'x_table_name': x_table_name,
           'y_table_name': y_table_name,
           'do_logit': do_logit,
           'do_gnmf': do_gnmf,
           'do_reg': do_reg,
           'do_robust': do_robust,
           'shape': shape,
           'cxn': cxn}
    cleanup = None
    if opType == 'logit':
        call = 'do_logit(x_table_name, y_table_name, shape, cxn)'
    elif opType == 'gnmf':
        call = 'do_gnmf(x_table_name, shape, 10, cxn)'
    elif opType == 'reg':
        call = 'do_reg(x_table_name, y_table_name, cxn)'
        cleanup = ("map(lambda x: cxn.execute("
                   "'DROP TABLE {}'.format(x)), ['XTX','XTY','XTX_INV','B'])")
    elif opType == 'robust':
        #do_reg(x_table_name, y_table_name, cxn)
        #preproc = """
        #    DROP TABLE IF EXISTS Y_HAT;
        #    SELECT madlib.matrix_mult('{X}',NULL,'B',NULL,'Y_HAT');
        #    CREATE TABLE R2 AS (
        #        SELECT {y}.row_num, ARRAY[POW({y}.val[1]-y_hat.val[1],2)] val
        #          FROM {y}
        #         INNER JOIN y_hat ON {y}.row_num = y_hat.row_num
        #    ) DISTRIBUTED BY (row_num)
        #""".format(X=x_table_name, y=y_table_name)

        # We can just generate a vector of residuals on the fly
        # rather than computing them explicitly.
        cxn.execute('DROP TABLE IF EXISTS R2')
        cxn.randomMatrix(shape[0], 1, 'R2')
        call = 'do_robust(x_table_name, cxn)'

    rows = shape[0]
    cols = shape[1]
    path = '../output/madlib_tall_{}{}.txt'.format(opType, int(nodes))
    runTimes.ix[:,['nodes','rows','cols']] = (nodes, rows, cols)
    res = utils.timeOp(call, env, cleanup)
    res
    runTimes.ix[:,3:] = res
    writeHeader = False if (os.path.exists(path)) else True
    runTimes.to_csv(path, index=False, header = writeHeader, mode = 'a')

def do_logit(x_table_name, 
             y_table_name,
             shape, 
             cxn,
             iterations=3):
    w = np.random.rand(shape[1],1).ravel()

    cxn.execute('DROP TABLE IF EXISTS {}T'.format(x_table_name))
    stmt = """
        SELECT madlib.matrix_trans('{0}',NULL,'{0}T',NULL)
    """.format(x_table_name)
    cxn.execute(stmt)

    logit_iter = """
        SELECT madlib.matrix_vec_mult('{X}T',NULL,(
            SELECT ARRAY_AGG(1/1+EXP(-1*LEAST(
                    madlib.array_dot({X}.val,{w}), 100)) - 
                       {y}.val[1])
                  FROM {X}
                 INNER JOIN {y} ON {X}.row_num = {y}.row_num
        ))
    """

    iteration = 0
    while iteration < iterations:
        call = logit_iter.format(X=x_table_name,
                                 y=y_table_name,
                                 w=gen_w_call(w))
        (_, cursor) = cxn.execute(call)
        w = w - (1e-3/shape[0])*np.array(
            map(lambda x: x[0], cursor.fetchall())).ravel()
        print w
        iteration += 1

def gen_w_call(w_array):
    array_str = map(lambda x: str(x), w_array)
    return 'ARRAY[{}]::DOUBLE PRECISION[]'.format(','.join(array_str))

def do_gnmf(x_table_name, shape, rank, cxn, iterations=3):
    gnmf_iter = """
        SELECT madlib.matrix_mult('{X}',NULL,'H','trans=True','XHT');
        SELECT madlib.matrix_mult('H',NULL,'H','trans=True','HHT');
        SELECT madlib.matrix_mult('W',NULL,'HHT',NULL,'WHHT');
        CREATE TABLE WHHT_INV AS (
            SELECT row_num, madlib.array_pow(
                val, -1.0::DOUBLE PRECISION) val FROM WHHT
        );
        SELECT madlib.matrix_elem_mult('XHT',NULL,'WHHT_INV',NULL,'XHT_WHHT_INV');
        SELECT madlib.matrix_elem_mult('W',NULL,'XHT_WHHT_INV',NULL,'W_NEW');
        DROP TABLE W;
        DROP TABLE XHT;
        DROP TABLE HHT;
        DROP TABLE WHHT;
        DROP TABLE WHHT_INV;
        DROP TABLE XHT_WHHT_INV;
        ALTER TABLE W_NEW RENAME TO W;

        SELECT madlib.matrix_mult('W','trans=True','{X}',NULL,'WTX');
        SELECT madlib.matrix_mult('W','trans=True','W',NULL,'WTW');
        SELECT madlib.matrix_mult('WTW',NULL,'H',NULL,'WTWH');
        CREATE TABLE WTWH_INV AS (
            SELECT row_num, madlib.array_pow(
                val, -1.0::DOUBLE PRECISION) val FROM WTWH
        );
        SELECT madlib.matrix_elem_mult('WTX',NULL,'WTWH_INV',NULL,'WTX_WTWH_INV');
        SELECT madlib.matrix_elem_mult('H',NULL,'WTX_WTWH_INV',NULL,'H_NEW');
        DROP TABLE H;
        DROP TABLE WTX;
        DROP TABLE WTW;
        DROP TABLE WTWH;
        DROP TABLE WTWH_INV;
        DROP TABLE WTX_WTWH_INV;
        ALTER TABLE H_NEW RENAME TO H;
    """.format(X=x_table_name)

    setup = """
        DROP TABLE IF EXISTS W;
        DROP TABLE IF EXISTS H;
        SELECT madlib.matrix_random({N},{r},NULL,'normal','W','row=row_num');
        SELECT madlib.matrix_random({r},{K},NULL,'normal','H','row=row_num');
    """.format(N=shape[0], K=shape[1],r=rank)

    cxn.execute(setup)
    iteration = 0
    while iteration < iterations:
        verbose = iteration > 0
        cxn.execute(gnmf_iter, verbose=verbose)
        iteration += 1

def do_reg(x_table_name, y_table_name, cxn):
    cxn.execute('DROP TABLE IF EXISTS XT')
    cxn.execute('DROP TABLE IF EXISTS XTX')
    cxn.execute('DROP TABLE IF EXISTS XTY')
    cxn.execute('DROP TABLE IF EXISTS XTX_INV')
    cxn.execute('DROP TABLE IF EXISTS B')

    stmt = """
        SELECT madlib.matrix_mult('{0}','trans=True','{0}',NULL,'XTX');
        SELECT madlib.matrix_mult('{0}','trans=True','{1}',NULL,'XTY');
        SELECT madlib.matrix_inverse('XTX',NULL,'XTX_INV',NULL);
        SELECT madlib.matrix_mult('XTX_INV',NULL,'XTY',NULL,'B');
    """.format(x_table_name, y_table_name)
    cxn.execute(stmt)

def do_robust(x_table_name, cxn):
    tables = ['R2_DIAG','XT','XTX','XTX_INV',
              'XTR2','XTX_INV_XTR2',
              'XTX_INV_XTR2X','SE']
    for t in tables:
        cxn.execute('DROP TABLE IF EXISTS {}'.format(t))
    cxn.execute('DROP VIEW IF EXISTS XT')
    stmt = """
        SELECT madlib.matrix_diag(
            (SELECT ARRAY_AGG(val[1] ORDER BY row_num) FROM R2),'r2_diag',
             'row=row_num,col=col_num');
        SELECT madlib.matrix_trans('{0}',NULL,'XT',NULL);
        SELECT madlib.matrix_mult('XT',NULL,'{0}',NULL,'XTX');
        SELECT madlib.matrix_inverse('XTX',NULL,'XTX_INV',NULL);
        SELECT madlib.matrix_mult('XT',NULL,'r2_diag',NULL,'XTR2');
        SELECT madlib.matrix_mult('XTX_INV',NULL,'XTR2',NULL,'XTX_INV_XTR2');
        SELECT madlib.matrix_mult('XTX_INV_XTR2',NULL,'{0}',NULL,'XTX_INV_XTR2X');
        SELECT madlib.matrix_mult('XTX_INV_XTR2X',NULL,'XTX_INV',NULL,'SE');        
    """.format(x_table_name)
    cxn.execute(stmt)

def do_pca(x_table_name, k, cxn):
    cxn.execute('DROP TABLE IF EXISTS XS')
    cxn.execute('DROP TABLE IF EXISTS SVD_XS')
    cxn.execute('DROP TABLE IF EXISTS SVD_SUMMARY_TABLE')
    stmt = """
        CREATE TABLE XS AS (
            SELECT row_num, madlib.array_sub(
                    val, (SELECT madlib.matrix_mean('{0}',NULL,1))
                ) val
              FROM {0}
        ) DISTRIBUTED BY (row_num);
        SELECT madlib.svd('XS','svd_xs','row_num','{1}');
        SELECT madlib.matrix_mult('XS',NULL,'svd_xs_v',
                                  'row=row_id,val=row_vec','prj');
    """.format(x_table_name,k)
    cxn.execute(stmt)

if __name__ == '__main__':
    args = utils.parse_cmd_args(sys.argv[1:])
    main(args)
