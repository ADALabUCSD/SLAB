import os
from gslab_make.dir_mod import *
from gslab_make.run_program import *
from gslab_make.make_log import *

# Clean up after previous runs
delete_files('../output/*')
clear_dirs('../temp/*')
clear_dirs('../external')

# create symlinks to external resources
project_root = os.getenv('BENCHMARK_PROJECT_ROOT')
if (project_root is None):
    msg = 'Pease set environment variable "BENCHMARK_PROJECT_ROOT"'
    raise StandardError(msg)

externals = {'distributed_dense':
             '/tests/SimpleMatrixOps (Distributed Disk)/output',
             'distributed_sparse':
             '/tests/SimpleMatrixOps (Distributed Sparse)/output',
             'native_algos':
             '/tests/MLAlgorithms (Native Implementations)/output',
             'dense_la_algos':
             '/tests/MLAlgorithms (Distributed Dense LA)/output',
             'sparse_la_algos':
             '/tests/MLAlgorithms (Distributed Sparse LA)/output',
             'pipelines':
             '/tests/Pipelines (Distributed Dense)/output',
             'decompositions':
             '/tests/Decompositions (Distributed Dense)/output',
             'single_node_dense':
             '/tests/SimpleMatrixOps (Single Node Dense)/output',
             'single_node_ml':
             '/tests/MLAlgorithms (Single Node Dense)/output',
             'lib' : '/lib'}
for name in externals:
    os.symlink(project_root + externals[name], '../external/' + name)

# start logging
start_make_logging()

# Run various programs
run_python(program='figures.py')

# stop logging
end_make_logging()
