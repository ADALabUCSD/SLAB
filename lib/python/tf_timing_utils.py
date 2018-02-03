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
import os
import numpy as np
import pandas as pd
import tensorflow as tf

def timeOp(op, session, feed_dict=None):
    times = []
    for ix in range(5):
        start = time.time()
        session.run(op) if feed_dict is None else session.run(op, feed_dict=feed_dict)
        stop = time.time()
        times.append(stop-start)

    return times

def tf_wait(flag_file='../temp/__tf_done__'):
    while not os.path.exists(flag_file):
        time.sleep(1)

def mark_task_complete(flag_file='../temp/__tf_done__'):
    with open(flag_file, 'w') as fh:
        fh.write('DONE\n')
