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
from gen_data import gen_data_sparse

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
           'do_logit':     do_logit,
           'do_reg':       do_reg,
           'do_gnmf':      do_gnmf,
           'do_robust':    do_robust,
           'shape':        shape,
           'cxn':          cxn}
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
        cxn.execute('DROP TABLE IF EXISTS R2 CASCADE')
        cxn.execute(
            "SELECT MADLIB.matrix_random({},1,NULL,'uniform','R2',NULL)".format(
             shape[0]))
        cxn.execute('ALTER TABLE R2 RENAME COLUMN ROW TO ROW_NUM')
        call = 'do_robust(x_table_name, cxn)'

    rows = shape[0]
    cols = shape[1]
    path = '../output/madlib_adclick_{}{}.txt'.format(opType, int(nodes))
    runTimes.ix[:,['nodes','rows','cols']] = (nodes, rows, cols)
    res = utils.timeOp(call, env, cleanup)
    print res
    runTimes.ix[:,3:] = res
    writeHeader = False if (os.path.exists(path)) else True
    runTimes.to_csv(path, index=False, header = writeHeader, mode = 'a')

def do_logit(x_table_name, 
             y_table_name,
             shape, 
             cxn,
             iterations=1):

    w = np.random.rand(shape[1],1).ravel()
    make_xtrans_sparse(x_table_name, cxn)

    logit_iter = """
        DROP TABLE IF EXISTS EPS;
        CREATE TABLE EPS AS (
            SELECT xb.row_num, 1 AS col_num, (xb.val - y.val[1]) AS val
              FROM (
                SELECT X.row_num, 
                       1/1+EXP(-1*LEAST(SUM(X.val*w.val), 100)) val
                  FROM {X} X
                 INNER JOIN w_curr w ON X.col_num = w.col_num
                 GROUP BY (X.row_num)
              ) xb
             INNER JOIN {y} y ON xb.row_num = y.row_num
        ) DISTRIBUTED BY (row_num);
        DROP TABLE IF EXISTS W_UPDATE;
        SELECT madlib.matrix_mult('{XT}',NULL,'EPS',NULL,'W_UPDATE');
    """

    iteration = 0
    while iteration < iterations:
        build_weight_table(w, cxn)
        call = logit_iter.format(X=x_table_name,
                                 y=y_table_name,
                                 w=gen_w_call(w),
                                 XT='{}_trans'.format(x_table_name))
        cxn.execute(call)
        (_,cursor) = cxn.execute('SELECT val FROM W_UPDATE ORDER BY row_num')
        w = w - (1e-3/shape[0])*np.array(
            map(lambda x: x[0], cursor.fetchall())).ravel()
        print w
        iteration += 1

def build_weight_table(w_array, cxn):
    cxn.execute("DROP TABLE IF EXISTS w_curr")
    stmt = """
        CREATE TABLE w_curr AS (
            SELECT GENERATE_SERIES(1,{0}) col_num, UNNEST({1}) val
        )
    """.format(w_array.size, gen_w_call(w_array))
    cxn.execute(stmt)


def gen_w_call(w_array):
    array_str = map(lambda x: str(x), w_array)
    return 'ARRAY[{}]::DOUBLE PRECISION[]'.format(','.join(array_str))

def do_reg(x_table_name, y_table_name, cxn):
    cxn.execute('DROP TABLE IF EXISTS XTX')
    cxn.execute('DROP TABLE IF EXISTS XTY')
    cxn.execute('DROP TABLE IF EXISTS XTX_INV')
    cxn.execute('DROP TABLE IF EXISTS B')
    cxn.execute('DROP TABLE IF EXISTS T')
    cxn.execute('DROP TABLE IF EXISTS aT')
    cxn.execute('DROP TABLE IF EXISTS aTT')
    cxn.execute('DROP TABLE IF EXISTS XTXaTT')
    make_xtrans_sparse(x_table_name, cxn)

    shape = cxn.get_shape(x_table_name)

    stmt = """
        SELECT madlib.matrix_identity('{}','T','row=row_num,col=col_num,val=val');
        SELECT madlib.matrix_scalar_mult('T',NULL,0.001,'aT',NULL);
        SELECT madlib.matrix_mult('aT','trans=True','aT',NULL,'aTT');
    """.format(shape[1])
    cxn.execute(stmt)

    stmt = """
        SELECT madlib.matrix_mult('{2}',NULL,'{0}',NULL,'XTX');
        SELECT madlib.matrix_add('XTX',NULL,'aTT',NULL,'XTXaTT');
        SELECT madlib.matrix_mult('{2}',NULL,'{1}',NULL,'XTY');
        SELECT madlib.matrix_inverse('XTXaTT',NULL,'XTX_INV',NULL);
        SELECT madlib.matrix_mult('XTX_INV',NULL,'XTY',NULL,'B');
    """.format(x_table_name, y_table_name, '{}_trans'.format(x_table_name))
    cxn.execute(stmt)

def make_xtrans_sparse(name, cxn):
    cxn.execute('DROP VIEW IF EXISTS {}_trans'.format(name))
    stmt = """
        CREATE VIEW {0}_trans AS (
            SELECT row_num AS col_num, col_num AS row_num, val
              FROM {0}
        )
    """.format(name)
    cxn.execute(stmt)

def do_gnmf(x_table_name, shape, rank, cxn, iterations=3):
    gnmf_iter = """
        SELECT madlib.matrix_mult('{X}',NULL,'{H}','trans=True','XHT');
        SELECT madlib.matrix_mult('{H}',NULL,'{H}','trans=True','HHT');
        SELECT madlib.matrix_mult('{W}',NULL,'HHT',NULL,'WHHT');
        CREATE TABLE WHHT_INV AS (
            SELECT row_num, madlib.array_pow(
                val, -1.0::DOUBLE PRECISION) val FROM WHHT
        );
        SELECT madlib.matrix_elem_mult('XHT',NULL,'WHHT_INV',NULL,'XHT_WHHT_INV');
        SELECT madlib.matrix_elem_mult('{W}',NULL,'XHT_WHHT_INV',NULL,'W_NEW');
        DROP TABLE W;
        DROP TABLE XHT;
        DROP TABLE HHT;
        DROP TABLE WHHT;
        DROP TABLE WHHT_INV;
        DROP TABLE XHT_WHHT_INV;
        ALTER TABLE W_NEW RENAME TO W;

        SELECT madlib.matrix_mult('{W}','trans=True','{X}',NULL,'WTX');
        SELECT madlib.matrix_mult('{W}','trans=True','{W}',NULL,'WTW');
        SELECT madlib.matrix_mult('WTW',NULL,'{H}',NULL,'WTWH');
        CREATE TABLE WTWH_INV AS (
            SELECT row_num, madlib.array_pow(
                val, -1.0::DOUBLE PRECISION) val FROM WTWH
        );
        SELECT madlib.matrix_elem_mult('WTX',NULL,'WTWH_INV',NULL,'WTX_WTWH_INV');
        SELECT madlib.matrix_elem_mult('{H}',NULL,'WTX_WTWH_INV',NULL,'H_NEW');
        DROP TABLE H;
        DROP TABLE WTX;
        DROP TABLE WTW;
        DROP TABLE WTWH;
        DROP TABLE WTWH_INV;
        DROP TABLE WTX_WTWH_INV;
        ALTER TABLE H_NEW RENAME TO H;
    """.format(X=x_table_name, W='W{}'.format(label), H='H{}'.format(label))

    setup = """
        DROP TABLE IF EXISTS W;
        DROP TABLE IF EXISTS H;
        SELECT madlib.matrix_random({N},{r},NULL,'normal','W','row=row_num');
        SELECT madlib.matrix_random({r},{K},NULL,'normal','H','row=row_num');
    """.format(N=shape[0], K=shape[1],r=rank)

    cxn.execute(setup)
    iteration = 0
    while iteration < iterations:
        print 'GNMF ITERATION {}'.format(iteration)
        verbose = iteration > 0
        cxn.execute(gnmf_iter)
        iteration += 1

def do_robust(x_table_name, cxn):
    tables = ['R2_DIAG','XT','XTX','XTX_INV',
              'XTR2','XTX_INV_XTR2',
              'XTX_INV_XTR2X','SE']
    for t in tables:
        cxn.execute('DROP TABLE IF EXISTS {}'.format(t))
    make_xtrans_sparse(x_table_name, cxn)
    stmt = """
        CREATE VIEW R2_diag AS (
            SELECT row_num, row_num AS col_num, val[1] AS val
              FROM R2
        );
        SELECT madlib.matrix_mult('{0}_trans',NULL,'{0}',NULL,'XTX');
        SELECT madlib.matrix_inverse('XTX',NULL,'XTX_INV',NULL);
        SELECT madlib.matrix_mult('{0}_trans',NULL,'r2_diag',NULL,'XTR2');
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
