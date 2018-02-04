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
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.linalg._
import org.apache.spark.mllib.linalg
import org.apache.spark.mllib.random.RandomRDDs._
import org.apache.spark.mllib.linalg.distributed._
import org.apache.spark.sql._
import scala.tools.nsc.io._
import scala.io.Source
import java.util.Random
import scala.collection.immutable._

object SparkMatrixOps {
    def main(args: Array[String]) {
        val conf = new SparkConf().setAppName("MLLibMatrixOps")
        val sc = new SparkContext(conf)

        val root = sys.env("BENCHMARK_PROJECT_ROOT")

        val argMap = Map[String,String](
                args.map(_.split("=")).map({
                    case Array(x,y) => (x -> y)
                }):_*
            )

        val mattype = argMap("mattype")
        val opType = argMap("opType")
        val mPath = argMap("Mpath")
        val nPath = argMap("Npath")
        val savestub = argMap("savestub")
        val passPath = argMap("passPath")
        val nodes = argMap("nodes")
        val outdir = argMap("outdir")
        val sr = argMap("sr")

        val stub = s"/tests/SimpleMatrixOps (Distributed Sparse)/output/${outdir}"
        val base = s"/mllib_${mattype}_${opType}${nodes}.txt"
        val path = root + stub + base

        if (!Files.exists(Paths.get(path))) {
          File(path).writeAll("nodes,sr,time1,time2,time3,time4,time5\n")
        }

        val results : Array[Double] = opType match {
            case "TRANS" => doBlockMatrixOp(mPath, passPath, "TRANS", sc)
            case "GMM"   => doBlockMatrixOp(mPath, nPath,  "GMM", sc)
            case "MVM"   => doRowMatrixOp(mPath, passPath, "MVM", sc)
            case "NORM"  => doRowMatrixOp(mPath, passPath, "NORM", sc)
            case "ADD"   => doRowMatrixOp(mPath, nPath,    "ADD", sc)
            case "TSM"   => doRowMatrixOp(mPath, passPath, "TSM", sc)
            case _       => Array[Double](0.0, 0.0)
        }

        File(path).appendAll(
            nodes + "," + sr + "," + results.mkString(",") + '\n')
    }

    def doBlockMatrixOp(mPath: String,
                        nPath: String,
                        opType: String,
                        sc: SparkContext) : Array[Double] = {
        val M = readSMM(mPath, sc).toBlockMatrix(100,100)
        M.blocks.count
        val N = readSMM(nPath, sc).toBlockMatrix(100,100)
        N.blocks.count

        val times = Array[Double](0,0,0,0,0)
        for (ix <- 0 to 4) {
            val start = System.nanoTime()
            if (opType == "GMM") {
                val res = M.multiply( N, 500 )
                res.blocks.count
            } else if (opType == "TRANS") {
                val res = M.transpose
                res.blocks.count
            }
            val stop = System.nanoTime()
            times(ix) = (stop - start)/1e9
        }
        return times
    }

    def doRowMatrixOp(Mpath: String,
                      Npath: String,
                      opType: String,
                      sc: SparkContext) : Array[Double] = {

        val M = readSMM(Mpath, sc)
        val N_dist = readSMM(Npath, sc)
        val N = opType match {
            case "MVM" => random_local_matrix(M.numCols.toInt, 1, .10)
            case _     => random_local_matrix(1, 1, .10)
        }
        println( M.rows.count )

        val times = Array[Double](0,0,0,0,0)
        for (ix <- 0 to 4) {
            val start = System.nanoTime()
            if (opType == "MVM") {
                val res = M.multiply( N )
                res.rows.count
            } else if (opType == "ADD") {
                val res = add_row_matrices(M, N_dist)
                res.rows.count
            } else if (opType == "TSM") {
                M.computeGramianMatrix
            } else if (opType == "NORM") {
                println(norm( M ))
            }
            val stop = System.nanoTime()
            times(ix) = (stop - start)/1e9
        }
        return times
    }

    def readSMM(path: String, sc: SparkContext) : IndexedRowMatrix = {
        if (path.contains("pass")) {
            val rows = sc.parallelize(
                Seq(new IndexedRow(0, Vectors.sparse(
                        3, Array(0, 2), Array(1.0, 3.0)))))
            return new IndexedRowMatrix(rows)
        }

        val M = sc.textFile(path, 500).
                   zipWithIndex().filter(tup => tup._2 > 2).
                   map(tup => tup._1.trim.split(" ")).
                   map(row => new MatrixEntry(
                       row(0).toLong, row(1).toLong, row(2).toDouble))
        val SM = new CoordinateMatrix( M )
        SM.entries.persist(MEMORY_AND_DISK_SER)
        println( SM.numCols() + " " + SM.numRows())
        return SM.toIndexedRowMatrix()
    }

    // Spark annoyingly does not expose any of these primitive methods
    // which makes their library not very useful
    def as_breeze(v: linalg.Vector) : bVector[Double] = v match {
        case v: linalg.SparseVector =>
            return new bSparseVector[Double](v.indices, v.values, v.size)
        case v: linalg.DenseVector  =>
            return new bDenseVector[Double](v.values)
    }

    def from_breeze(v: bVector[Double]) : linalg.Vector = v match {
        case v: bSparseVector[Double] =>
            return Vectors.sparse(v.length, v.activeIterator.toSeq)
        case v: bDenseVector[Double] =>
            return Vectors.dense(v.data)
    }

    def as_breeze(m: linalg.Matrix) : bMatrix[Double] = m match {
        case m: linalg.DenseMatrix =>
            return new bDenseMatrix(m.numRows, m.numCols, m.toArray)
        case m: linalg.SparseMatrix =>
            return new bSparseMatrix(
                m.values, m.numRows, m.numCols,
                m.colPtrs, m.numActives, m.rowIndices)
    }

    def add_row_matrices(A: IndexedRowMatrix,
                         B: IndexedRowMatrix) : IndexedRowMatrix = {
        val both = join_row_matrices(A, B)
        return new IndexedRowMatrix(
            both.map(tup => new IndexedRow(tup._1, 
                from_breeze(as_breeze(tup._2._1) + as_breeze(tup._2._2)))))
    }

    def join_row_matrices(A: IndexedRowMatrix,
                          B: IndexedRowMatrix) :
            RDD[(Long, (linalg.Vector,linalg.Vector))] = {
        val pair_A = A.rows.map(row => (row.index, row.vector))
        val pair_B = B.rows.map(row => (row.index, row.vector))
        return pair_A.join(pair_B)
    }

    def from_breeze(m: bMatrix[Double]) : linalg.Matrix = m match {
        case m: bDenseMatrix[Double] =>
            return Matrices.dense(m.rows, m.cols, m.toDenseMatrix.data)
        case m: bSparseMatrix[Double] =>
            return Matrices.sparse(m.rows, m.cols,
                    m.colPtrs, m.rowIndices, m.data)
    }

    def random_local_matrix(rows: Int, cols: Int, den: Double) : Matrix = {
        val rng = new Random()
        return Matrices.rand(rows, cols, rng)
    }

    def norm(M: IndexedRowMatrix) : Double = {
        val temp = M.rows.map(row => bAlg.sum(bNum.pow(as_breeze(row.vector),2)))
        val norm = sqrt(temp.sum)
        return norm
    }

    def mean(A: Array[Double]) : Double = {
        return A.sum / A.length
    }

    def variance(A: Array[Double]) : Double = {
        val abar = mean(A)
        val accum = A.map(e => pow(e-abar,2)).sum
        return accum / (A.length - 1)
    }
}
