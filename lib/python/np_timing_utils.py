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
import time
import logging
import datetime
import numpy as np

def allocMatrix(rows, cols, rng):
    M = rng.rand(int(rows), int(cols))
    return M

def timeOp(string, envr, cleanup=None):
    times = []
    time_stamp = datetime.datetime.fromtimestamp(
        time.time()).strftime('%Y%m%d%H%M%S')
    logname = '../temp/time_{}.log'.format(time_stamp)
    logging.basicConfig(filename=logname, level=logging.INFO)
    for ix in range(5):
        try:
            start = time.time()
            res = eval(string, envr)
            stop = time.time()
            times.append(stop-start)
            logging.info('RAN: {} IN {} SECONDS'.format(
                string.replace('\n',''), stop-start))
        except MemoryError:
            return np.nan
        if cleanup is not None:
            eval(cleanup, envr)
    return times

def parseCMDArg(argStr):
    parse_cmd_arg(argStr)

def parse_cmd_args(argl):
    args = {}
    for arg in argl:
        args.update(parse_cmd_arg(arg))
    return args

def parse_cmd_arg(arg_str):
    arg = arg_str.split('=')
    return {arg[0] : arg[1]}
