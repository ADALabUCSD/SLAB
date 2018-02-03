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
import _memlock as mm

class MemLocker( ):
    
    def __init__(self):
        self._segment = None
        self._counter = 0

    def __enter__(self):
        self._segment = None
        self._counter = 0
        return self

    def lock_mem(self, size):
        if (size > 0):
            self._segment = mm._memlock(size, self._counter)
            self._counter += 1
        else:
            self._segment = None

    def __exit__(self, type, value, traceback):
        if not (self._segment is None):
            call = 'ipcrm -m {}'.format(self._segment)
            os.system(call)
