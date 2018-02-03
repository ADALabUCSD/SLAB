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

# this file defines global parameters that are shared
# by several directories

MAX_ITERATIONS = 10 # number of iterations to run ML algorithms
DEFAULT_COLS = 1024 # number of columns in tall and skinny matrices
DEFAULT_ROWS = 1024 # number of rows in short and wide matrices

SQUARE_MATRIX_MIN = 11 # smallest power of 2 used to generate matrices 
SQUARE_MATRIX_MAX = 14 # largest power of 2 used to generate matrices
#SQUARE_MATRIX_MIN = 1
#SQUARE_MATRIX_MAX = 4

TALL_MATRIX_MIN = 12 # smallest power of 2 used in ML Algorithm tests
TALL_MATRIX_MAX = 15 # largest power of 2 used in ML Algorithm tests
#TALL_MATRIX_MIN = 1
#TALL_MATRIX_MAX = 4

SPARSE_MATRIX_SIZE = 12
DIST_MATRIX_SIZE = 10

WIDE_MATRIX_MIN = 12
WIDE_MATRIX_MAX = 15
#WIDE_MATRIX_MIN = 1
#WIDE_MATRIX_MAX = 4

NPROC_MAX = 3 # largest power of two used to set the number of cores

# Spark settings
NUM_EXECUTORS_LOCAL = 1 # number of executors to launch for a local cluster
EXECUTOR_CORES = 8      # number of cores per executor
EXECUTOR_MEMORY = '30G' # RAM allocated to each executor
DRIVER_MEMORY = '1G'   # RAM allocated to the driver
