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
import time
import json

import numpy as np
import pandas as pd
import tensorflow as tf

ROOT = os.getenv('BENCHMARK_PROJECT_ROOT')
if (ROOT is None):
    msg = 'Please set environment variable BENCHMARK_PROJECT_ROOT'
    raise StandardError(msg)

sys.path.append(os.path.join(ROOT,'lib','python'))
import np_timing_utils as utils

print 'Using TensorFlow: ' + tf.__version__

"""
Pipeline is modeled after the example code here:
https://github.com/tensorflow/models/blob/master/official/wide_deep/wide_deep.py
"""

CONT_COLS = ['V{}'.format(x) for x in range(1,11)]
CAT_COLS = ['V{}'.format(x) for x in range(11,16)]
CONT_COLS_DEFAULTS = [[0.0]]*len(CONT_COLS)
CAT_COLS_DEFAULTS = [['']]*len(CAT_COLS)
CONTINUOUS_ONLY_FLAG = None
CSV_COL_NAMES = None
CSV_COL_DEFAULTS = None
BATCH_SIZE = 100
GLOBAL_COUNTER = 0

def main(args):
    global CONTINUOUS_ONLY_FLAG, CSV_COL_NAMES, CSV_COL_DEFAULTS

    op_type = args['opType']
    nodes = args['nodes']
    stub = args['stub']
    input_path = args['inputPath']
    rank = int(args['rank'])
    instance_type = args['instanceType']

    CONTINUOUS_ONLY_FLAG = 'sparse' not in input_path

    CSV_COL_NAMES = ['y'] + CONT_COLS
    CSV_COL_DEFAULTS = [[0.0]] + CONT_COLS_DEFAULTS
    if not CONTINUOUS_ONLY_FLAG:
        CSV_COL_NAMES += CAT_COLS
        CSV_COL_DEFAULTS += CAT_COLS_DEFAULTS

    colnames = ['nodes','time','variance']
    runTimes = pd.DataFrame(np.zeros((1,len(colnames))))
    runTimes.columns = colnames

    # configure the distributed environment
    cluster = {'chief': ['mycluster-master:9824'],
               'ps': ['mycluster-master:3298'],
               'worker' : ['mycluster-slave-1:9829']}
    os.environ['TF_CONFIG']= json.dumps(
        {'cluster': cluster,
        'task': {'type': instance_type, 'index': rank}})

    # tensorflow requires too many variables to pass them all in a dictionary
    # so need to implement timing from scratch
    times = []
    for ix in range(5):
        start = time.time()
        if op_type == 'logit':
            do_logit(input_path)
        if op_type == 'reg':
            do_regress(input_path)
        stop = time.time()
        times.append(stop-start)

    results = (np.mean(times[1:5]), np.var(times[1:5], ddof=1))
    runTimes.ix[:,['time','variance']] = results
    runTimes.to_csv(
        '../output/tf_{}{}.txt'.format(op_type, int(nodes)), index=False)

def do_logit(data_path):
    indep_vars = build_vars()
    config = tf.estimator.RunConfig()
    lr = tf.estimator.LinearClassifier(model_dir=None,
                                       config = config,
                                       feature_columns=indep_vars)
    train_spec = tf.estimator.TrainSpec(
        input_fn=lambda: input_fn_logit(data_path, 1, BATCH_SIZE),
        max_steps=10)
    eval_spec = tf.estimator.EvalSpec(
        input_fn=lambda: input_fn_logit(data_path, 1, BATCH_SIZE),
        steps=1)
    tf.estimator.train_and_evaluate(lr, train_spec, eval_spec)

def eval_fn_logit(path, num_epochs, batch_size):
    dataset = tf.data.TextLineDataset(path)
    dataset = dataset.map(parse_table, num_parallel_calls=12)
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)

    iterator = dataset.make_one_shot_iterator()
    features, labels = iterator.get_next()
    return features, labels

def input_fn_logit(path, num_epochs, batch_size):
    dataset = tf.data.TextLineDataset(path)
    dataset = dataset.map(parse_table, num_parallel_calls=12)
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)

    iterator = dataset.make_one_shot_iterator()
    features, labels = iterator.get_next()
    return features, labels

def parse_table(value):
    cols = tf.decode_csv(value, record_defaults=CSV_COL_DEFAULTS)
    indep_vars = dict(zip(CSV_COL_NAMES, cols))
    y = indep_vars.pop('y')
    return indep_vars, y

def build_vars():
    continuous_vars = [
        tf.feature_column.numeric_column(x, shape=(1,)) for x in CONT_COLS]
    categorical_vars = []
    if not CONTINUOUS_ONLY_FLAG:
        categorical_vars = [
            tf.feature_column.categorical_column_with_hash_bucket(
                x, hash_bucket_size=1000) for x in CAT_COLS]
    return  continuous_vars + categorical_vars

def do_regress(data_path, cluster):
    indep_vars = build_vars()
    lr = tf.estimator.LinearRegressor(model_dir=None,
                                       feature_columns=indep_vars)
    lr.train(
        input_fn = lambda: input_fn_logit(data_path, 1, BATCH_SIZE),
        max_steps=3)

if __name__ == '__main__':
    argv = utils.parse_cmd_args(sys.argv[1:])
    main(argv)
