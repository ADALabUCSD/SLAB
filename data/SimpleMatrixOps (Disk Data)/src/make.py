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
    '--msize', default='2 4 8 16', type=str, 
    help='Approximate size of matrix (GB in memory) to generate')
args = parser.parse_args()

if not os.path.exists('manifest.txt'):
    with open('manifest.txt', 'w') as fh:
        fh.write('')

# start logging
start_make_logging()

os.system('hdfs dfs -mkdir -p /scratch')
cxn = SQLCxn(timeout=None, username='ubuntu', db='ubuntu')

approx_gb = [.001]
if len(args.msize) > 0:
    approx_gb += [int(x) for x in args.msize.split()]

mtypes = ['tall', 'wide']
for mtype in mtypes:
    for gb in approx_gb:
        print mtype
        if gb < .1: gb_stub = '_test'
        else: gb_stub = '_small' if gb < 1 else gb
        
        if mtype == 'square':
            k = int(np.ceil(np.sqrt((gb*1e9)/float(8))))
            m = k
            rows = 2**12
        elif mtype == 'tall':
            k = int(np.ceil((gb*1e9)/float(8*100)))
            m = 100
            rows = 2**14
        elif mtype == 'wide':
            k = 100
            m = int(np.ceil((gb*1e9)/float(8*100)))    
            rows = 1

        stub = '_' + mtype
        fmt = (gb_stub, stub)
        data.gen_data_disk('../output/M{}{}.csv'.format(*fmt), k, m, rows)
        if ((mtype == 'wide') and 
            (not cxn.table_exists('M{}{}'.format(*fmt)))):
            print 'CREATING MATRIX: M{}{}'.format(*fmt)
            cxn.randomMatrix(k, m, 'M{}{}'.format(*fmt))
        if mtype != 'tall':
            continue
        mpath = os.path.abspath('../output/M{}{}_sparse.mtx'.format(*fmt))
        data.gen_data_disk('../output/y{}{}.csv'.format(*fmt), k, 1, rows, True)
        utils.link_if_not('../output/M{}{}.csv'.format(*fmt), 
                          '../output/N{}{}.csv'.format(*fmt))
        utils.link_if_not('../output/M{}{}.csv.mtd'.format(*fmt),
                          '../output/N{}{}.csv.mtd'.format(*fmt))

paths = os.listdir('../output')
paths = filter(
    lambda x: (x != '.gitignore') and ('.log' not in x) and ('.mtd' not in x), 
    paths)
paths = map(lambda x: os.path.join('../output', x), paths)

with open('manifest.txt') as fh:
   manifest = fh.read().split('\n')    
fh = open('manifest.txt', 'a')

for path in paths:
    if path in manifest:
        continue
    
    utils.hdfs_put(path)
    dest, ext = path.replace('../output/','').split('.')
    if (ext == 'csv') and ('wide' not in path):
        cxn.load_dense_matrix(path, dest)
    fh.write(path + '\n')
    fh.flush()

fh.close()
# make sure git ignores these files
with open('../output/.gitignore', 'w') as fh:
    fh.write('*.csv\n*.mtd\n*.mtx')

# stop logging
end_make_logging()
