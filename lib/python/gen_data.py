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

import os
import numpy as np
import pandas as pd
import scipy as sp
import scipy.io as io
import scipy.sparse as sparse
import sql_cxn

def get_dims(mtype, k, default_rows, default_cols):
    dims = None
    if (mtype == 'square'):
        dims = (k,k)
    elif (mtype == 'tall_and_skinny'):
        dims = (k, default_cols)
    elif (mtype == 'short_and_wide'):
        dims = (default_rows, k)
    else:
        raise StandardError('Invalid Matrix Type')
    return dims

def gen_data_sparse(rows, cols, density, name, path=None, cxn=None, write_to_disk=True):
    if (path is not None) and os.path.exists(path):
        print 'WARNING: File exists. Delete it to re-create'
        return 0

    # for some annoying reason, the built in SciPy generator cannot
    # guarantee there are no duplicate values

    n = np.int64(density * rows * cols)
    if cxn is None:
        cxn = sql_cxn.SQLCxn(username='ubuntu', db='ubuntu')

    if cxn.table_exists(name):
        print 'WAENING: Table exists. Delete it to re-create'
        return 0

    stmt = """
        CREATE TABLE {name} AS (
            SELECT row_num, col_num, SUM(val) val
              FROM (
                  SELECT TRUNC(RANDOM()*{rows})::INT+1 AS row_num,
                         TRUNC(RANDOM()*{cols})::INT+1 AS col_num,
                         RANDOM() AS val
                    FROM GENERATE_SERIES(1,{n})
              ) raw
             GROUP BY row_num, col_num
        ) DISTRIBUTED BY (row_num,col_num)
    """.format(rows=rows, cols=cols, name=name, n=n)
    cxn.execute(stmt)

    (_,res) = cxn.execute(
        'SELECT COUNT(*) FROM {} WHERE row_num=1 AND col_num=1'.format(name))
    if (res.fetchall()[0][0] == 0):
        cxn.execute('INSERT INTO {} VALUES (1,1,0)'.format(name))

    stmt = """
        SELECT COUNT(*) FROM {} WHERE row_num={} AND col_num={}
    """.format(name, rows, cols)
    (_, res) = cxn.execute(stmt)
    if (res.fetchall()[0][0] == 0):
        cxn.execute('INSERT INTO {} VALUES ({},{},0)'.format(name, rows, cols))

    if write_to_disk:
        cxn.execute(
            'CREATE UNIQUE INDEX pk_{0} ON {0} (row_num,col_num)'.format(name))
        (_,res) = cxn.execute('SELECT COUNT(*) FROM {}'.format(name))
        nnz = res.fetchall()[0][0]

        with open(path, 'wb') as fh:
            fh.write('%%MatrixMarket matrix coordinate real general\n%\n')
            fh.write('{} {} {}\n'.format(rows, cols, nnz))

        stmt = """
            COPY (SELECT * FROM {name})
              TO PROGRAM 'cat >> "{path}"'
            WITH DELIMITER AS ' '
        """.format(name=name, path=path)
        cxn.execute(stmt)
        
        return (nnz / np.float64(rows*cols))

def write_sparse_meta(name, path, cxn):
    shape = cxn.get_shape(name)
    (_,res) = cxn.execute('SELECT COUNT(*) FROM {}'.format(name))
    nnz = res.fetchall()[0][0]
    metadata = ('{{\n'
        '    "data_type": "matrix",\n'
        '    "value_type": "double",\n'
        '    "rows": {nrow},\n'
        '    "cols": {ncol},\n'
        '    "nnz": {nnz},\n'
        '    "format": "ijv",\n'
        '    "header": false,\n'
        '    "sep": ","\n'
    '}}').format(nrow=shape[0], ncol=shape[1], nnz=nnz)

    with open(path + '.mtd', 'w') as fh:
        fh.write(metadata)


def gen_data_disk(path, rows, cols, chunk_rows, indicator=False):
    if os.path.exists(path):
        print 'WARNING: File exists. Delete it to recreate.'
        return 0

    chunk_rows = min(chunk_rows, rows)
    nChunks = max(np.floor(rows/float(chunk_rows)), 1)
    for ix in range(int(nChunks)):
        print 'Writing {} of {} chunks'.format(ix, nChunks)
        if indicator:
            chunk = (np.random.rand(chunk_rows, cols) > .83).astype(np.int64)
        else:
            chunk = np.random.rand(chunk_rows, cols)
        K = pd.DataFrame(chunk)
        K.to_csv(path, header = False, index = False, mode = 'a')

    obs_left = int(rows - (chunk_rows*nChunks))
    if (obs_left > 0):
        K = pd.DataFrame(np.random.rand(obs_left, cols))
        K.to_csv(path, header = False, index = False, mode = 'a')

    # write metadata for systemml
    metadata = ('{{\n'
            '    "data_type": "matrix",\n'
            '    "value_type": "double",\n'
            '    "rows": {nrow},\n'
            '    "cols": {ncol},\n'
            '    "nnz": {nnz},\n'
            '    "format": "csv",\n'
            '    "header": false,\n'
            '    "sep": ","\n'
        '}}').format(nrow=rows, ncol=cols, nnz=rows*cols)

    with open(path + '.mtd', 'w') as fh:
        fh.write(metadata)
