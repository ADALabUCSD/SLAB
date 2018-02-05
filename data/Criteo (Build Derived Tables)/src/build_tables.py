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

ROOT = os.getenv('BENCHMARK_PROJECT_ROOT')
if (ROOT is None):
    msg = 'Please set environment variable BENCHMARK_PROJECT_ROOT'
    raise StandardError(msg)

sys.path.append(os.path.join(ROOT,'lib','python'))
import make_utils as utils
from sql_cxn import SQLCxn

def main():
    stub = sys.argv[1]
    cxn = SQLCxn(timeout=None, username='ubuntu', db='ubuntu')

    if not cxn.table_exists('adclick_clean_vectors_split'):
        shape = cxn.get_shape('adclick_clean{}_dense'.format(stub))
        stmt = """
            CREATE TABLE adclick_clean_vectors_split AS (
                SELECT row_num, val[1]::INTEGER y, val[2:{}]::NUMERIC[] indep_vars
                  FROM adclick_clean{}_dense
            ) DISTRIBUTED BY (row_num)
        """.format(shape[1], stub)
        cxn.execute(stmt)

    if not cxn.table_exists('adclick_clean_indepvars_long'):
        stmt = """
            CREATE TABLE adclick_clean_indepvars_long AS (
                SELECT row_num, ix AS col_num, indep_vars[ix] AS val
                  FROM (
                    SELECT *, GENERATE_SUBSCRIPTS(indep_vars, 1) AS ix
                      FROM adclick_clean_vectors_split
                  ) tmp
            ) DISTRIBUTED BY (row_num, col_num)
        """
        cxn.execute(stmt)

    if not cxn.table_exists('adclick_clean_y'):
        stmt = """
            CREATE TABLE adclick_clean_y AS (
                SELECT row_num, 1 AS col_num, y AS val
                  FROM adclick_clean_vectors_split
            ) DISTRIBUTED BY (row_num)
        """
        cxn.execute(stmt)

if __name__=='__main__':
    main()
