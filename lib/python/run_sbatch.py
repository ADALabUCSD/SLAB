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
import subprocess

def run_sbatch(program=None, 
               kwargs=None,
               local=False,
               makelog='../output/make.log'):

    with open(program) as prog:
        contents = prog.read()
        if (kwargs is not None):
            for kw in kwargs:
                contents = contents.replace('%{}%'.format(kw), kwargs[kw])

        tempname = '_{}'.format(program)

        with open(tempname, 'w') as outprog:
            outprog.write(contents)
        with open(makelog, 'a') as log:
            log.write(contents)

    os.system('chmod +x {}'.format(tempname))
    if (local is False):
        call = 'sbatch {}'.format(tempname)
    else:
        call = './{}'.format(tempname)
    res = os.system(call)
    return res

def sbatch_wait(taskNum, username, delay=60):
    complete = False
    call = 'squeue -u {}'.format(username)
    while (not complete):
        procs = subprocess.check_output(call, shell=True)
        complete = str(taskNum) not in procs
        time.s
