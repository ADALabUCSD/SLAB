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
    op_type = kwargs['opType']
    nodes = kwargs['nodes']
    stub = kwargs['stub']

    colnames = ['nodes','rows','cols','time1','time2','time3','time4','time5']
    runTimes = pd.DataFrame(np.zeros((1,len(colnames))))
    runTimes.columns = colnames

    cxn = SQLCxn(username='ubuntu', db='ubuntu', timeout=10000)

    shape = cxn.get_shape('adclick_clean_1_sparse')
    if not cxn.table_exists('adclick_clean_1_vectors_sparse'):
        stmt = """
        CREATE TABLE adclick_clean_1_vectors_sparse AS (
            SELECT x.row_num, madlib.svec_cast_positions_float8arr(
               ARRAY_AGG(x.col_num), ARRAY_AGG(x.val), {}, 0.0
               ) AS indep_vars, y.val AS y
             FROM adclick_clean_1_sparse x
            INNER JOIN adclick_clean_y y ON x.row_num = y.row_num
            GROUP BY x.row_num, y.val
        ) DISTRIBUTED BY (row_num)
        """.format(shape[1])        
        cxn.execute(stmt)

    if op_type == 'logit':
        cxn.execute('DROP TABLE IF EXISTS adclick_logit_summary')
        cxn.execute('DROP TABLE IF EXISTS adclick_logit')
        call = """
            SELECT madlib.logregr_train('adclick_clean_1_vectors_sparse',
                                        'adclick_logit',
                                        'y', 'indep_vars', NULL,
                                        3, 'igd', .000001)
        """
        cleanup = ['adclick_logit_summary','adclick_logit']
    elif op_type == 'reg':
        cxn.execute('DROP TABLE IF EXISTS adclick_reg_summary')
        cxn.execute('DROP TABLE IF EXISTS adclick_reg')
        call = """
            SELECT madlib.linregr_train('adclick_clean_1_vectors_sparse',
                                        'adclick_reg', 'y', 'indep_vars')
        """
        cleanup = ['adclick_reg_summary' ,'adclick_reg']
    elif op_type == 'pca':
        cxn.execute('DROP TABLE IF EXISTS result_table')
        cxn.execute('DROP TABLE IF EXISTS result_table_mean')
        cxn.execute('DROP TABLE IF EXISTS residual_table')
        cxn.execute('DROP TABLE IF EXISTS result_summary_table')
        cxn.execute('DROP TABLE IF EXISTS adlick_prj')
        call = """
            SELECT madlib.pca_sparse_train('adclick_clean_1_sparse',
                                           'result_table',
                                           'row_num',
                                           'col_num',
                                           'val',
                                           '{0}',
                                           '{1}',
                                           5);
            SELECT madlib.pca_sparse_project('adclick_clean_1_sparse',
                                      'result_table',
                                      'adclick_prj',
                                      'row_num',
                                      'col_num',
                                      'val',
                                      '{0}',
                                      '{1}',
                                      'residual_table',
                                      'result_summary_table')
        """.format(*shape)
        cleanup = ['result_table',
                   'result_table_mean',
                   'residual_table',
                   'result_summary_table',
                   'adclick_prj']

    runTimes.ix[:,['rows','cols']] = shape

    path = '../output/madlib_{}{}_sparse.txt'.format(op_type, int(nodes))
    runTimes.ix[:,'nodes'] = nodes
    res = cxn.time(call, cleanup)
    runTimes.ix[:,3:] = res
    runTimes.to_csv(path, index=False)


if __name__ == '__main__':
    args = utils.parse_cmd_args(sys.argv[1:])
    main(args)
