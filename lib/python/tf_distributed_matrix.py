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
import time

import numpy as np
import pandas as pd
import numpy.linalg as alg
import tensorflow as tf

class partitioned_matrix( object ):

    def __init__(self, data_path=None, 
                 device_str=None,
                 num_devices=None,
                 partition_scheme='round-robin',
                 cluster=None):

        self._num_devices = num_devices
        self._partition_scheme = partition_scheme
        self._device_stub = device_str
        self._data_path = data_path

        if (data_path is not None):
            self._shard_matrix(data_path)
        else:
            self._partitions = None
            self._rows = None
            self._cols = None
            self._stride = None
            self._num_blocks = None

    def _shard_matrix(self, data_path):
        # takes an ND-array and converts it into a bunch of
        # smaller tensors. Vectorization is row-major

        self._partitions = []
        self._rows, self._cols = parse_metadata(data_path)
        assert self._rows == self._cols, "Matrix must be square"

        self._stride = np.int32(np.floor(np.log2(self._rows)))
        
        self._pad_size = np.int32(
            self._rows - (self._stride*np.floor(self._rows/self._stride)))
        extra_blocks = 0 if self._pad_size == 0 else 1
        self._num_blocks = np.int32(np.floor(self._rows/self._stride)) + extra_blocks
        self._num_partitions = self._num_blocks**2
        self._assign_partitions_to_devices()

        col_ptr = 0
        row_ptr = 0
        for ix in range(self._num_partitions):
            shape = self._get_block_shape(row_ptr+1, col_ptr+1)
            with tf.device(self._block_devices[ix]):
                self._partitions.append(tf.placeholder(tf.float32, shape=shape))
            col_ptr = (col_ptr + 1) % self._num_blocks
            if col_ptr == 0:
                row_ptr += 1

    def _get_block_shape(self, row, col):
        corner_size = self._stride if self._pad_size == 0 else self._pad_size
        if (row < self._num_blocks) and (col < self._num_blocks):
            shape = (self._stride, self._stride)
        elif (row < self._num_blocks) and (col == self._num_blocks):
            shape = (self._stride, corner_size)
        elif (row == self._num_blocks) and (col < self._num_blocks):
            shape = (corner_size, self._stride)
        else:
            shape = (corner_size, corner_size)

        return shape

    def _iter_raw_blocks(self):
        iterator = pd.read_csv(self._data_path, header=None, chunksize=self._stride)

        lower_bound = np.arange(0, self._cols, self._stride)
        upper_bound = np.arange(self._stride, self._cols+self._stride, self._stride)
        upper_bound[upper_bound > self._cols] = self._cols
        bounds = zip(lower_bound, upper_bound)

        for chunk in iterator:
            chunk = chunk.values.astype(np.float32)
            for lb, ub in bounds:
                yield chunk[:,lb:ub]

    def get_feed_dict(self):
        blocks = [block for block in self._iter_raw_blocks()]
        return dict(zip(self._partitions, blocks))

    def _assign_partitions_to_devices(self):
        # determines the placement of sub-blocks onto devices
        # choices for partitioning are: 'row-wise', 'column-wise', or
        # 'round-robin' (the default)

        self._block_devices = []
        if (self._partition_scheme == 'round-robin'):
            dev_name_generator = lambda x: self._device_stub.format(
                    dev_num = x % self._num_devices)
            self._block_devices = map(dev_name_generator, range(self._num_partitions))
        elif (self._partition_scheme == 'column-wise'):
            usable_devices = min(self._num_devices, self._num_blocks)
            dev_name_generator = lambda x: self._device_stub.format(
                dev_num = x % usable_devices)
            self._block_devices = map(dev_name_generator, range(self._num_partitions))
        elif (self._partition_scheme == 'row-wise'):
            for ix in range(self._num_blocks):
                dev_ix = ix % self._num_devices
                for part_ix in range(self._num_blocks):
                    device = self._device_stub.format(dev_num = dev_ix)
                    self._block_devices.append(device)

        else:
            raise RuntimeError('Invalid Partitioning Scheme passed')

    def sum(self):
        # computes the sum of the matrix shards contained herein
        return tf.add_n([tf.reduce_sum(p) for p in self._partitions])

    def dot(self, other, partitioning='left'):
        # computes the matrix product of two sharded arrays and return
        # the result as a new sharded array

        if (other._cols != self._cols):
            raise ValueError('Non-Conformable Arrays')

        if (partitioning == 'left'):
            block_devices = self._block_devices
        elif (partitioning == 'right'):
            block_devices = other._block_devices 

        result = []
        for ix in range(len(self._partitions)):
            row_offset = (ix / self._num_blocks) * self._num_blocks
            col_offset = (ix % self._num_blocks)
            sub_products = []
            for ixx in range(self._num_blocks):
                row_ix = row_offset + ixx
                col_ix = col_offset + (ixx*self._num_blocks)
                with tf.device(block_devices[row_ix]):
                    sub_products.append(tf.matmul(self._partitions[row_ix],
                                                  other._partitions[col_ix]))

            #print sub_products
            res = tf.add_n(sub_products)
            result.append(res)

        new_matrix = partitioned_matrix()
        new_matrix._partitions = result
        new_matrix._rows = self._rows
        new_matrix._cols = self._cols
        new_matrix._stride = self._stride
        new_matrix._num_blocks = self._num_blocks
        return new_matrix

    def multiply(self, other):
        # computes the Hadamard product of two arrays and returns the result
        # as a new sharded matrix
        result = [tf.multiply(self._partitions[ix], other._partitions[ix])
                     for ix in range(len(self._partitions))]

        new_matrix = partitioned_matrix()
        new_matrix._partitions = result
        new_matrix._rows = self._rows
        new_matrix._cols = self._cols
        new_matrix._stride = self._stride
        new_matrix._num_blocks = self._num_blocks
        return new_matrix

    def add(self, other):
        #computes the sum of the two arrays
        result = [tf.add(self._partitions[ix], other._partitions[ix])
                     for ix in range(len(self._partitions))]

        new_matrix = partitioned_matrix()
        new_matrix._partitions = result
        new_matrix._rows = self._rows
        new_matrix._cols = self._cols
        new_matrix._stride = self._stride
        new_matrix._num_blocks = self._num_blocks
        return new_matrix

    def norm(self):
        # computes the frobenius norm of the array
        total = tf.add_n(
            [tf.reduce_sum(tf.square(p)) for p in self._partitions])
        return tf.sqrt(total)

    def collect(self):
        # pools the array into a single in-memory tensor
        ix = 0
        row_chunks = []
        while (ix < len(self._partitions)):
            chunk = tf.concat(self._partitions[ix:ix+self._num_blocks], axis=1)
            row_chunks.append(chunk)
            ix += self._num_blocks

        res = tf.concat(row_chunks, axis=0)
        return res

def parse_metadata(path):
    path = '{}.mtd'.format(path) if not '.mtd' in path else path
    with open(path) as fh:
        mtd = fh.read()

    mtd = mtd.split('\n')
    rows = filter(lambda x: 'rows' in x, mtd)
    cols = filter(lambda x: 'cols' in x, mtd)

    rows = int(rows[0].replace(',','').split(':')[1])
    cols = int(cols[0].replace(',','').split(':')[1])

    return (rows, cols)

if __name__=='__main__':
    # test that everything works okay
    
    # set up the cluster (assumes 2 workers)

    from make_utils import parse_hosts
    from gen_data   import gen_data_disk
    from tf_timing_utils import tf_wait, mark_task_complete
    from np_timing_utils import parse_cmd_args

    args = parse_cmd_args(sys.argv[1:])
    rank = int(args['worker-id'])
    print rank

    hosts = parse_hosts()
    worker_names = map(lambda x: x + ':2222', hosts.values())

    cluster = tf.train.ClusterSpec({'worker' : worker_names})
    server = tf.train.Server(cluster, job_name='worker', task_index=rank)

    device_str = '/job:worker/task:{dev_num}'
    num_workers = len(worker_names)

    if rank == 0:
        gen_data_disk('M.csv', 10, 10, 10)
        gen_data_disk('N.csv', 10, 10, 10)
    else:
        while not os.path.exists('N.csv'):
            time.sleep(.01)

    M = np.loadtxt('M.csv', delimiter=',')
    dM = partitioned_matrix('M.csv', device_str=device_str, num_devices=num_workers, cluster=cluster)
    
    N = np.loadtxt('N.csv', delimiter=',')
    dN = partitioned_matrix('N.csv', device_str=device_str, num_devices=num_workers, cluster=cluster)

    sess = tf.Session('grpc://{}:2222'.format(master_dns))

    if rank == 0:
        feed_dict_m = dM.get_feed_dict()
        feed_dict_n = dN.get_feed_dict()

        both = feed_dict_m.copy()
        both.update(feed_dict_n)

        # test collect
        res = sess.run(dM.collect(), feed_dict=feed_dict_m)
        assert (res - M).sum() < 1e-5, 'Error in collect'
        print 'Okay'

        # test dot
        res = sess.run(dM.dot(dN).collect(), feed_dict=both)
        assert (res - M.dot(N)).sum() < 1e-5, 'Error in dot'
        print 'Okay'

        # test multiply
        res = sess.run(dM.multiply(dN).collect(), feed_dict=both)
        assert (res - np.multiply(M,N)).sum() < 1e-5, 'Error in multiply'
        print 'Okay'

        # test norm 
        res = sess.run(dM.norm(), feed_dict=feed_dict_m)
        assert (res - alg.norm(M)) < 1e-5, 'Error in norm'
        print 'Okay'

        # test sum
        res = sess.run(dM.sum(), feed_dict=feed_dict_m)
        assert res - M.sum() < 1e-5, 'Error in sum'
        print 'Okay'

        cleanup = ['M.csv', 'N.csv', 'M.csv.mtd', 'N.csv.mtd']
        map(lambda x: os.unlink(x) if os.path.exists(x) else 1, cleanup)
        mark_task_complete()
    else:
        tf_wait()
