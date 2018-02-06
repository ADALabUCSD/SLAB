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
import argparse

ROOT = os.getenv('BENCHMARK_PROJECT_ROOT')
if (ROOT is None):
    msg = 'Please set environment variable BENCHMARK_PROJECT_ROOT'
    raise StandardError(msg)

sys.path.append(os.path.join(ROOT,'lib','python'))
import make_utils as utils
from sql_cxn import SQLCxn

parser = argparse.ArgumentParser()
parser.add_argument('--stub', default='_sample', type=str)
parser.add_argument('--sparse', default=False, type=bool)

def main(argv):
    cxn = SQLCxn(timeout=None, username='ubuntu', db='ubuntu')
    coalesce_files(argv.stub, argv.sparse)
    process_madlib(argv.stub, argv.sparse, cxn)

def coalesce_files(stub, sparse_flag):
    hdfs_dir = '/scratch/adclick_clean{}_dense.csv'.format(stub)
    outfile  = '../output/adclick_clean{}_dense.csv'.format(stub)
    utils.coalesce_hdfs_files(hdfs_dir, outfile)

    hdfs_dir = '/scratch/adclick_clean{}_y.csv'.format(stub)
    outfile  = '../output/adclick_clean{}_y.csv'.format(stub)
    utils.coalesce_hdfs_files(hdfs_dir, outfile)

    if sparse_flag:
        hdfs_dir = '/scratch/adclick_clean{}_sparse.mtx'.format(stub)
        outfile  = '../output/adclick_clean{}_sparse.mtx'.format(stub)
        utils.coalesce_hdfs_files(hdfs_dir, outfile)

        hdfs_dir = '/scratch/adclick_clean{}_raw.csv'.format(stub)
        outfile  = '../output/adclick_clean{}_raw.csv'.format(stub)
        utils.coalesce_hdfs_files(hdfs_dir, outfile)

def process_madlib(stub, sparse_flag, cxn):
    cxn.load_dense_matrix('../output/adclick_clean{}_dense.csv'.format(stub),
                          'adclick_clean{}_dense'.format(stub))
    if sparse_flag:
        cxn.load_sparse_matrix(
            '../output/adclick_clean{}_sparse.mtx'.format(stub),
            'adclick_clean{}_sparse_'.format(stub))
        stmt = """
           CREATE TABLE adclick_clean{}_sparse AS (
               SELECT row_num+1 AS row_num, col_num+1 AS col_num, val
                 FROM adclick_clean{}_sparse_
           ) DISTRIBUTED BY (row_num, col_num)
        """.format(stub)
        cxn.execute(stmt)
        cxn.execute('DROP TABLE adclick_clean{}_sparse_')

if __name__ == '__main__':
    argv = parser.parse_args()
    main(argv)
