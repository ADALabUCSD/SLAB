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
import sys
import numpy as np
import argparse
from gslab_make.dir_mod import *
from gslab_make.run_program import *
from gslab_make.make_log import *

# Clean up after previous runs
clear_dirs('../temp/*')
clear_dirs('../external')

# create symlinks to external resources
project_root = os.getenv('BENCHMARK_PROJECT_ROOT')
if (project_root is None):
    msg = 'Pease set environment variable "BENCHMARK_PROJECT_ROOT"'
    raise StandardError(msg)

externals = {'lib' : '/lib/python'}
for name in externals:
    os.symlink(project_root + externals[name], '../external/' + name)
sys.path.append('../external/lib')
from sql_cxn import SQLCxn
import gen_data as data
import make_utils as utils

parser = argparse.ArgumentParser()
parser.add_argument(
    '--msize-sparse', default=100, type=str,
    help='Approximate -logical- size of matrix (default 100G)')

helpstr = 'Approximate fraction nonzero values. Delimited by spaces. Default "0.0001 0.001 0.01 0.1"'
parser.add_argument(
    '--sparsity', default='0.0001 0.001 0.01 0.1', type=str,
    help=helpstr)
args = parser.parse_args()

if not os.path.exists('manifest.txt'):
    with open('manifest.txt','w') as fh:
        fh.write('')

# start logging
start_make_logging()

os.system('hdfs dfs -mkdir -p /scratch')
cxn = SQLCxn(timeout=None, username='ubuntu', db='ubuntu')

sparse_gb = int(args.msize_sparse)
sparsity = map(lambda x: float(x), args.sparsity.split(' '))
for sr in sparsity:
    stub = '{}'.format(sr).replace('0.','_')
    gb_stub = sparse_gb
    fmt = (stub, gb_stub)
    k = int(np.ceil((sparse_gb*1e9)/float(8*100)))

    mpath_tall = os.path.abspath(
        '../output/M{}{}_sparse_tall.mtx'.format(*fmt))
    mpath_wide = os.path.abspath(
        '../output/M{}{}_sparse_wide.mtx'.format(*fmt))
    data.gen_data_sparse(k, 100, sr, 'M{}{}_sparse_tall'.format(*fmt), mpath_tall)
    data.gen_data_sparse(100, k, sr, 'M{}{}_sparse_wide'.format(*fmt), mpath_wide)
    data.gen_data_disk('../output/y{}_sparse.csv'.format(sparse_gb),
                       k, 1, k, True)
    stmt = """
        CREATE VIEW N{0}{1}_sparse_tall AS (
            SELECT * FROM M{0}{1}_sparse_tall
        )
    """.format(*fmt)
    if not cxn.table_exists('N{}{}_sparse_tall'.format(*fmt)):
        cxn.execute(stmt)

    utils.link_if_not('../output/M{}{}_sparse_tall.mtx'.format(*fmt),
                      '../output/N{}{}_sparse_tall.mtx'.format(*fmt))

cxn.load_dense_matrix('../output/y{}_sparse.csv'.format(sparse_gb),
                      'y{}_sparse'.format(sparse_gb))

paths = os.listdir('../output')
paths = filter(
    lambda x: (x != '.gitignore') and ('.log' not in x) and ('.mtd' not in x),
    paths)
paths = map(lambda x: os.path.join('../output', x), paths)

with open('manifest.txt') as fh:
   manifest = fh.read().split('\n')
fh = open('manifest.txt', 'a')

for path in paths:
    dest, ext = path.replace('../output/','').split('.')
    
    data.write_sparse_meta(dest, path, cxn)
    if path in manifest:
        continue
    utils.hdfs_put(path)
    fh.write(path + '\n')
    fh.flush()

fh.close()
# make sure git ignores these files
with open('../output/.gitignore', 'w') as fh:
    fh.write('*.csv\n*.mtd\n*.mtx')

# stop logging
end_make_logging()
