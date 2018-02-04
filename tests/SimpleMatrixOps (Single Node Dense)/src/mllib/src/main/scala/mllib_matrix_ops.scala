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
import org.apache.spark.sql._
import scala.tools.nsc.io._
import java.util.Random
import org.apache.spark.mllib.linalg._
import org.apache.spark.mllib.linalg
import breeze.{linalg => bAlg, numerics => bNum}
import breeze.linalg.{Vector => bVector,
                      Matrix => bMatrix,
                      SparseVector => bSparseVector,
                      DenseVector => bDenseVector,
                      CSCMatrix => bSparseMatrix,
                      DenseMatrix => bDenseMatrix}

object MLLibMatrixOps extends App {

    override def main(args: Array[String]) {
        val conf = new SparkConf().setAppName("MLLibMatrixOps")
        val sc = new SparkContext(conf)
        val root = sys.env("BENCHMARK_PROJECT_ROOT")
        
        val argMap = Map[String,String](
                args.map(_.split("=")).map({
                    case Array(x,y) => (x -> y)
                }):_*
            )

        val nrows = argMap("nrows").split(" ").map(x => x.toInt)
        val mattype = argMap("mattype")
        val opType = argMap("opType")
        val fixedAxis = argMap("fixedAxis").toInt
        val nproc = argMap.get("nproc")

        val stub = "/tests/SimpleMatrixOps (Single Node Dense)/output/"
        val base = nproc match {
            case Some(np) => s"mllib_cpu_${opType}_scale.txt"
            case None => s"mllib_${mattype}_${opType}.txt"
        }
        val path = root + stub + base
        
        if (!Files.exists(Paths.get(path)))
            File(path).writeAll("rows,time1,time2,time3,time4,time5\n")

        for (nr <- nrows) {
            val nrow = opType match {
                case "GMM" => fixedAxis
                case _     => nr
            }
            val ncol = opType match {
                case "GMM" => nr
                case _     => fixedAxis
            }
            val major_axis = opType match {
                case "GMM" => ncol
                case _     => nrow
            }

            val M = as_dense(Matrices.rand(nrow, ncol, new Random()))
            val N = opType match {
                case "GMM" => as_dense(Matrices.rand(ncol, nrow, new Random()))
                case "ADD" => as_dense(Matrices.rand(nrow, ncol, new Random()))
                case _     => as_dense(Matrices.rand(1, 1, new Random()))
            }
            val v = opType match {
                case "MVM" => vectorize(Matrices.rand(ncol, 1, new Random()))
                case _     => vectorize(Matrices.rand(1, 1, new Random()))
            }

            val times = Array[Double](0,0,0,0,0)
            for (ix <- 0 to 4) {
                println(s"Test => ${ix}")
                val start = System.nanoTime()

                val action = opType match {
                    case "TRANS" => do_trans( M )
                    case "NORM"  => do_norm( M )
                    case "GMM"   => do_gmm( M, N )
                    case "MVM"   => do_mvm( M, v )
                    case "TSM"   => do_tsm( M )
                    case "ADD"   => do_add( M, N )
                    case _       => throw new Exception("Invalid Operator")
                }

                val stop = System.nanoTime()
                times(ix) = (stop - start) / 1e9
            }
            val pref = nproc match {
                case Some(np) => s"${np}"
                case None     => s"${nr}"
            }
            File(path).appendAll(s"${pref}"+","+times.mkString(",")+'\n')
        } 
    }

    def do_trans(M: DenseMatrix) {
        return M.transpose
    }

    def do_norm(M: DenseMatrix) {
        return M.toArray.map(x => pow(x, 2)).sum
    }

    def do_gmm(M: DenseMatrix, N: DenseMatrix) {
        return M.multiply( N )
    }

    def do_tsm(M: DenseMatrix) {
        return M.transpose.multiply( M )
    }

    def do_add(M: DenseMatrix, N: DenseMatrix) {
        return from_breeze( as_breeze( M ) + as_breeze( N ) )
    }

    def do_mvm(M: DenseMatrix, v: DenseVector) {
        return M.multiply( v )
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

    def from_breeze(m: bMatrix[Double]) : linalg.Matrix = m match {
        case m: bDenseMatrix[Double] =>
            return Matrices.dense(m.rows, m.cols, m.toDenseMatrix.data)
        case m: bSparseMatrix[Double] =>
            return Matrices.sparse(m.rows, m.cols,
                    m.colPtrs, m.rowIndices, m.data)
    }

    def as_dense(M: Matrix) : DenseMatrix = {
        return M.asInstanceOf[DenseMatrix]
    }

    def vectorize(M: Matrix) : DenseVector = {
        return as_dense( Vectors.dense(M.toArray) )
    }

    def as_dense(v: Vector) : DenseVector = {
        return v.asInstanceOf[DenseVector]
    } 
}
