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

    cxn = SQLCxn(username='ubuntu', db='ubuntu')
    shape = cxn.get_shape_dense('adclick_clean{}_dense'.format(stub))
    if not cxn.table_exists('adclick_clean_vectors_split'):
        stmt = """
            CREATE TABLE adclick_clean_vectors_split AS (
                SELECT row_num, val[1]::INTEGER y, val[2:{}]::NUMERIC[] indep_vars
                  FROM adclick_clean{}_dense
            ) DISTRIBUTED BY (row_num)
        """.format(shape[1], stub)
        cxn.execute(stmt)

    # need to do a bit of preprocessing
    if op_type == 'logit':
        cxn.execute('DROP TABLE IF EXISTS adclick_logit_summary')
        cxn.execute('DROP TABLE IF EXISTS adclick_logit')
        call = """
            SELECT madlib.logregr_train('adclick_clean_vectors_split',
                                        'adclick_logit',
                                        'y', 'indep_vars', NULL,
                                        3, 'igd', .000001)
        """
        cleanup = ['adclick_logit_summary','adclick_logit']
    elif op_type == 'reg':
        cxn.execute('DROP TABLE IF EXISTS adclick_reg_summary')
        cxn.execute('DROP TABLE IF EXISTS adclick_reg')
        call = """
            SELECT madlib.linregr_train('adclick_clean_vectors_split',
                                        'adclick_reg', 'y', 'indep_vars')
        """
        cleanup = ['adclick_reg_summary' ,'adclick_reg']
    elif op_type == 'pca':
        cxn.execute('DROP TABLE IF EXISTS result_table')
        cxn.execute('DROP TABLE IF EXISTS result_table_mean')
        cxn.execute('DROP TABLE IF EXISTS residual_table')
        cxn.execute('DROP TABLE IF EXISTS result_summary_table')
        cxn.execute('DROP TABLE IF EXISTS adlick_prj')
        stmt = """
            CREATE TABLE adclick_clean_depvars AS (
                SELECT row_num, val[2:{}]::NUMERIC[] val
                  FROM adclick_clean{}_dense
            ) DISTRIBUTED BY (row_num)
        """.format(shape[1], stub)
        if not cxn.table_exists('adclick_clean_depvars'):
            cxn.execute(stmt)
        call = """
            SELECT madlib.pca_train('adclick_clean_depvars',
                                    'result_table',
                                    'row_num',
                                    5);
            SELECT madlib.pca_project('adclick_clean_depvars',
                                      'result_table',
                                      'adclick_prj',
                                      'row_num',
                                      'residual_table',
                                      'result_summary_table')
        """
        cleanup = ['result_table',
                   'result_table_mean',
                   'residual_table',
                   'result_summary_table',
                   'adclick_prj']

    #shape = cxn.get_shape_dense('adclick_clean{}_dense'.format(stub))
    runTimes.ix[:,['rows','cols']] = shape

    path = '../output/madlib_{}{}_dense.txt'.format(op_type, int(nodes))
    runTimes.ix[:,'nodes'] = nodes
    res = cxn.time(call, cleanup)
    runTimes.ix[:,3:] = res
    runTimes.to_csv(path, index=False)


if __name__ == '__main__':
    args = utils.parse_cmd_args(sys.argv[1:])
    main(args)
