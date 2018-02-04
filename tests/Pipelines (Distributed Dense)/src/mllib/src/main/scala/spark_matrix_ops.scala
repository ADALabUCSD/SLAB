// Copyright 2018 Anthony H Thomas and Arun Kumar
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//     http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import scala.math._
import java.nio.file.{Paths, Files}
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.spark.rdd._
import breeze.{linalg => bAlg, numerics => bNum}
import breeze.linalg.{Vector => bVector,
                      Matrix => bMatrix,
                      SparseVector => bSparseVector,
                      DenseVector => bDenseVector,
                      CSCMatrix => bSparseMatrix,
                      DenseMatrix => bDenseMatrix}
import org.apache.spark.storage.StorageLevel._
import org.apache.spark.mllib.linalg._
import org.apache.spark.mllib.linalg
import org.apache.spark.mllib.random.RandomRDDs._
import org.apache.spark.mllib.linalg.distributed._
import org.apache.spark.sql._
import scala.tools.nsc.io._
import scala.io.Source
import java.util.Random
import scala.collection.immutable._

object SparkPipelines {
    def main(args: Array[String]) {
        val conf = new SparkConf().setAppName("MLLibMatrixOps")
        val sc = new SparkContext(conf)

        val root = sys.env("BENCHMARK_PROJECT_ROOT")

        val stub = s"/tests/Pipelines (Distributed Dense)/output/"
        val base = s"mllib_pipelines.txt"
        val path = root + stub + base

        File(path).writeAll("rows,time1,time2,time3,time4,time5\n")

        val nrows = Array[Int](1000,10000,100000,1000000)
        for (r <- nrows) {
            val t = random_matrix(r, 1, 1024, 1, sc)
            val u = random_matrix(1, r, 1, 1024, sc)
            val v = random_matrix(r, 1, 1024, 1, sc)

            val times = Array[Double](0,0,0,0,0)
            for (ix <- 0 to 4) {
                val start = System.nanoTime()

                val res = t.multiply( u, 500 ).multiply( v, 500 )
                println(res.blocks.count)
                
                val stop = System.nanoTime()
                times(ix) = (stop - start)/1e9
            }

            File(path).appendAll(s"${r}" + "," + times.mkString(",") + '\n')

        }
    }

    def random_matrix(N: Int, K: Int,
                      r: Int, c: Int, sc: SparkContext) : BlockMatrix = {
        val tmp = new IndexedRowMatrix(
            normalVectorRDD(sc, N.toLong, K).zipWithIndex().map(
                tup => new IndexedRow(tup._2, tup._1))).toBlockMatrix(r, c)
        val MM = new BlockMatrix(tmp.blocks.repartition(1000), 
                                 tmp.rowsPerBlock, tmp.colsPerBlock,
                                 tmp.numRows, tmp.numCols)
        MM.persist(MEMORY_AND_DISK_SER)
        MM.blocks.count
        return MM
    }
}
