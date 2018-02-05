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

    cxn = SQLCxn(username='ubuntu', db='ubuntu', timeout = 2000)

    colnames = ['nodes','rows','cols','time1','time2','time3','time4','time5']
    runTimes = pd.DataFrame(np.zeros((1,len(colnames))))
    runTimes.columns = colnames

    shape = cxn.get_shape(x_table_name)

    env = {'x_table_name': x_table_name,
           'y_table_name': y_table_name,
           'do_logit': do_logit,
           'do_reg': do_reg,
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
        do_reg(x_table_name, y_table_name, cxn)
        preproc = """
            DROP TABLE IF EXISTS Y_HAT;
            SELECT madlib.matrix_mult('{X}',NULL,'B',NULL,'Y_HAT');
            CREATE TABLE R2 AS (
                SELECT {y}.row_num, ARRAY[POW({y}.val[1]-y_hat.val[1],2)] val
                  FROM {y}
                 INNER JOIN y_hat ON {y}.row_num = y_hat.row_num
            ) DISTRIBUTED BY (row_num)
        """.format(X=x_table_name, y=y_table_name)
        cxn.execute(preproc)
        call = 'do_robust(x_table_name, cxn)'
    elif opType == 'pca':
        print 'Not Implemented'
        return

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
             iterations=3):

    w = np.random.rand(shape[1],1).ravel()
    make_xtrans_sparse(cxn)

    logit_iter = """
        DROP TABLE IF EXISTS EPS;
        CREATE TABLE EPS AS (
            SELECT xb.row_num, 1 AS col_num, (xb.val - y.val) AS val
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
                                 XT='adclick_xtrans_long')
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
    make_xtrans_sparse(cxn)

    stmt = """
        SELECT madlib.matrix_mult('{2}',NULL,'{0}',NULL,'XTX');
        SELECT madlib.matrix_mult('{2}',NULL,'{1}',NULL,'XTY');
        SELECT madlib.matrix_inverse('XTX',NULL,'XTX_INV',NULL);
        SELECT madlib.matrix_mult('XTX_INV',NULL,'XTY',NULL,'B');
    """.format(x_table_name, y_table_name, 'adclick_xtrans_long')
    cxn.execute(stmt)

def make_xtrans_sparse(cxn):
    cxn.execute('DROP VIEW IF EXISTS adlick_xtrans_long')
    stmt = """
        CREATE VIEW adclick_xtrans_long AS (
            SELECT row_num AS col_num, col_num AS row_num, val
              FROM adclick_clean_indepvars_long
        )
    """
    cxn.execute(stmt)

if __name__ == '__main__':
    args = utils.parse_cmd_args(sys.argv[1:])
    main(args)
