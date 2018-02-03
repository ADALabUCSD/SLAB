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

import time
import sys
import Queue
import os
import json
import textwrap
import numpy as np
import pandas as pd
import multiprocessing as mp
import psycopg2 as psql

class SQLCxn:

    def __init__(self, timeout=6000,
                       port=5432,
                       username='ahthomas',
                       db='ahthomas',
                       host='localhost'):
        self._db = db
        self._username = username
        self._host = host
        self._cxn = None
        self._port = port
        self._allocated_objects = []
        self.newConnection()
        self._timeout = timeout
        if not (self._timeout is None):
            self._timeout = self._timeout*1000
            self.execute('SET statement_timeout TO {}'.format(self._timeout))

    def execute(self, statement, verbose=True):
        if (verbose):
            print textwrap.dedent(statement)

        self.newCursor()
        start = time.time()
        try:
            self._cursor.execute(statement)
        except (psql.ProgrammingError,
                psql.extensions.QueryCanceledError) as e:
            self._cxn.rollback()
            print e
            return None

        stop = time.time()
        self._cxn.commit()
        return (stop-start, self._cursor)

    def time(self, statement, cleanup, timeout=None):
        if timeout is not None:
            if self._check_op_timed_out(*timeout):
                return None

        times = []
        for ix in range(5):
            print 'TIMING: ', ix
            verbose = (ix < 1)
            res = self.execute(statement, verbose)
            if res is None:
                if timeout is not None:
                    self._write_timeout_flag(*timeout)
                return None

            times.append(res[0])
            for table in cleanup:
                self.execute('DROP TABLE IF EXISTS {}'.format(table), verbose=False)
        print times
        return times

    @staticmethod
    def _check_op_timed_out(path, key):
        if not os.path.exists(path):
            return False
        with open(path, 'rb') as fh:
            meta = json.load(fh)
        if key in meta:
            print 'WARNING: PREVIOUS OP TIMED OUT'
        return key in meta

    @staticmethod
    def _write_timeout_flag(path, key):
        meta = {}
        if os.path.exists(path):
            with open(path, 'rb') as fh:
                meta = json.load(fh)
        meta[key] = True
        with open(path, 'wb') as fh:
            json.dump(meta, fh)

    def newCursor(self):
        self._cursor.close()
        self._cursor = self._cxn.cursor()

    def newConnection(self):
        if (self._cxn is not None):
            self._cursor.close()
            self._cxn.close()
        self._cxn = psql.connect(
                "dbname='{}' user='{}' host='{}' port='{}'".format(
                    self._db, self._username, self._host, self._port
                )
            )
        self._cursor = self._cxn.cursor()

    def close(self):
        self._cxn.close()

    def randomMatrix(self,rows,cols,name):
        if (name == ""):
            raise StandardError("Must Specify Table Name")
        self.execute("SELECT madlib.matrix_random({},{},NULL,'uniform','{}','row=row_num')".format(
                int(rows), int(cols), name
            ))

    def get_shape(self, name, valname='val'):
        if self.is_sparse_mat(name, valname):
            shape = self.get_shape_sparse(name)
        else:
            shape = self.get_shape_dense(name)
        return shape

    def is_sparse_mat(self, name, valname='val'):
        stmt = """
            SELECT data_type
              FROM (
                SELECT column_name, data_type
                  FROM information_schema.columns
                 WHERE table_name = '{}'
            ) tmp WHERE column_name = '{}'
        """.format(name.lower(), valname.lower())
        (_,res) = self.execute(stmt)
        dtype = res.fetchall()[0][0]
        return dtype != 'ARRAY'

    def get_shape_dense(self, name):
        (_,res) = self.execute('SELECT COUNT(*) FROM {}'.format(name))
        rows = res.fetchall()[0][0]
        (_,res) = self.execute(
            'SELECT ARRAY_LENGTH(val,1) FROM {} WHERE row_num=1'.format(name))
        cols = res.fetchall()[0][0]
        return (rows,cols)

    def get_shape_sparse(self, name):
        (_,res) = self.execute(
            'SELECT MAX(row_num), MAX(col_num) FROM {}'.format(name))
        return res.fetchall()[0]

    def get_nnz(self, name):
        if self.is_sparse_mat(name):
            (_,res) = self.execute('SELECT COUNT(*) FROM {}'.format(name))
            return res.fetchall()[0][0]
        shape = self.get_shape_dense(name)
        return shape[0]*shape[1]

    def table_exists(self, name):
        stmt = "SELECT COUNT(*) FROM pg_tables WHERE tablename = '{}'".format(
            name.lower())
        (_,res) = self.execute(stmt)
        tuples = res.fetchall()
        return tuples[0][0] > 0

    def load_dense_matrix(self, path, table_name,
                          force_reload = False,
                          create_from_view='',
                          id_var_name='row_num',
                          row_vec_name='val'):

        # check if it's necessary to load new data
        stmt = "SELECT COUNT(*) FROM pg_tables WHERE tablename = '{}'".format(
            table_name.lower())
        (_,res) = self.execute(stmt)
        tuples = res.fetchall()
        if ((tuples[0][0] > 0) and (force_reload is False)):
            print 'Warning: using existent table'
            self._allocated_objects.append(table_name)
            return None

        # create a view
        if len(create_from_view) > 0:
            cxn.execute(
                'CREATE OR REPLACE VIEW {} AS (SELECT * FROM {})'.format(
                table_name, create_from_view))

        # figure out the number of columns
        data = pd.read_csv(path, nrows = 2)
        ncols = data.shape[1]

        self.execute('DROP TABLE IF EXISTS LOAD')
        self.execute('CREATE TABLE LOAD (row_num SERIAL, text VARCHAR)')

        fullpath = os.path.abspath(path)
        yaml = """
        ---
        VERSION: 1.0.0.1
        GPLOAD:
            INPUT:
                - SOURCE:
                    FILE:
                        - {}
                    SSL: false
                - FORMAT: text
                - MAX_LINE_LENGTH: 256000000
                - HEADER: false
            OUTPUT:
                - TABLE: LOAD
                - MODE: insert
        """.format(path)

        yaml = textwrap.dedent(yaml)
        with open('_load.yaml', 'w') as fh:
            fh.write(yaml)

        print 'Running GPLOAD with configuration: '
        print yaml

        rc = os.system('gpload -f _load.yaml')
        if (rc != 0):
            raise RuntimeError('GPLOAD returned error. Review logs')

        self.execute('DROP TABLE IF EXISTS {}'.format(table_name))

        array_stmt = (
            "STRING_TO_ARRAY(text, ',')::DOUBLE PRECISION[{}]".format(ncols))

        stmt = """
            CREATE TABLE {table_name} AS (
                SELECT row_num AS {id_var_name},
                       {array_stmt} AS {row_vec_name}
                  FROM load
            ) DISTRIBUTED BY ({id_var_name})
        """.format(table_name = table_name,
                   id_var_name = id_var_name,
                   array_stmt = array_stmt,
                   row_vec_name = row_vec_name)

        self.execute(stmt)

        stmt = """
            ALTER TABLE {table_name}
            ADD CONSTRAINT pk_{table_name}
            UNIQUE ({id_var_name})
        """.format(table_name = table_name, id_var_name = id_var_name)
        self.execute(stmt)

        self.execute('DROP TABLE LOAD')
        self._allocated_objects.append(table_name)

    def load_sparse_matrix(self, path, table_name,
                           force_reload=False,
                           row_var_name='row_num',
                           col_var_name='col_num',
                           val_var_name='val'):

        # check if it's necessary to load new data
        stmt = "SELECT COUNT(*) FROM pg_tables WHERE tablename = '{}'".format(
            table_name.lower())
        (_,res) = self.execute(stmt)
        tuples = res.fetchall()
        if ((tuples[0][0] > 0) and (force_reload is False)):
            print 'Warning: using existent table'
            self._allocated_objects.append(table_name)
            return None

        # figure out the number of columns

        stmt = (
            'CREATE TABLE {0} ({1} INTEGER, {2} INTEGER, {3} NUMERIC) '
            'DISTRIBUTED BY ({1},{2})').format(
             table_name, row_var_name, col_var_name, val_var_name)
        self.execute(stmt)

        fullpath = os.path.abspath(path)
        yaml = """
        ---
        VERSION: 1.0.0.1
        GPLOAD:
            INPUT:
                - SOURCE:
                    FILE:
                        - {}
                    SSL: false
                - FORMAT: text
                - HEADER: false
                - MAX_LINE_LENGTH: 256000000
                - ERROR_LIMIT: 4
                - DELIMITER: ' '
            OUTPUT:
                - TABLE: {}
                - MODE: insert
        """.format(path, table_name)

        yaml = textwrap.dedent(yaml)
        with open('_load.yaml', 'w') as fh:
            fh.write(yaml)

        print 'Running GPLOAD with configuration: '
        print yaml

        rc = os.system('gpload -f _load.yaml')
        if (rc != 0):
            raise RuntimeError('GPLOAD returned error. Review logs')

        stmt = """
            CREATE UNIQUE INDEX pk_{table} ON {table} ({row_var}, {col_var})
        """.format(table=table_name, row_var=row_var_name, col_var=col_var_name)
        self.execute(stmt)

        self._allocated_objects.append(table_name)

    def vacuum(self):
        orig_level = self._cxn.isolation_level
        self._cxn.set_isolation_level(0)
        self.execute('VACUUM FULL')
        self._cxn.set_isolation_level(orig_level)

    def cleanup(self):
        for obj in self._allocated_objects:
            self.execute('DROP TABLE IF EXISTS {}'.format(obj))

        self._allocated_objects = []
