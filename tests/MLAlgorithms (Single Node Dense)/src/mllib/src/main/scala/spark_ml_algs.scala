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
import breeze.linalg.*
import breeze.{math => bMath,
               numerics => bNum,
               linalg => bAlg}
import breeze.linalg.{Vector => bVector,
                      Matrix => bMatrix,
                      SparseVector => bSparseVector,
                      DenseVector => bDenseVector,
                      CSCMatrix => bSparseMatrix,
                      DenseMatrix => bDenseMatrix}
import org.apache.spark.storage.StorageLevel._
import org.apache.spark.mllib.linalg._
import org.apache.spark.mllib.linalg
import org.apache.spark.ml.{linalg => alg}
import org.apache.spark.mllib.random.RandomRDDs._
import org.apache.spark.mllib.linalg.distributed._
import org.apache.spark.sql._
import scala.tools.nsc.io._
import scala.io.Source
import java.util.Random

object SparkMLAlgorithms {
    def main(args: Array[String]) {
        val conf = new SparkConf().setAppName("MLLibMatrixOps")
        val sc = new SparkContext(conf)
        val spark = SparkSession.builder.getOrCreate()

        val root = sys.env("BENCHMARK_PROJECT_ROOT")

        val argMap = Map[String,String](
                args.map(_.split("=")).map({
                    case Array(x,y) => (x -> y)
                }):_*
            )

        val mattype = argMap("mattype")
        val opType = argMap("opType")
        val nrow = argMap("nrow").toInt
        val ncol = argMap("ncol").toInt
        val nproc = argMap("nproc").toInt

        val stub = "/tests/MLAlgorithms (Single Node Dense)/output/"
        val base = s"mllib_${opType}.txt"
        val path = root + stub + base

        if (!Files.exists(Paths.get(path))) {
          File(path).writeAll("nproc,time1,time2,time3,time4,time5\n")
        }

        val x = as_dense( Matrices.rand(nrow, ncol, new Random()) )
        val y = opType match {
            case "gnmf" => vectorize( Matrices.rand(1, 1, new Random()) )
            case _ => vectorize( Matrices.rand(nrow, 1, new Random()) )
        }

        val r2 = opType match {
            case "robust" => compute_r2(x, y)
            case _ => vectorize( Matrices.rand(1, 1, new Random()) )
        }

        val times = Array[Double](0,0,0,0,0)
        for (ix <- 0 to 4) {
            println(s"Test: ${ix}")
            val start = System.nanoTime()

            if (opType == "logit")
                logit(x, y, 3)
            if (opType == "gnmf")
                gnmf(x, 10, 3)
            if (opType == "reg")
                reg(x, y)
            if (opType == "robust")
                robust_se(x, r2)

            val stop = System.nanoTime()
            times(ix) = (stop - start)/1e9
        }

        File(path).appendAll(
            nproc + "," + times.mkString(",") + "\n")
    }

    def compute_r2(X: DenseMatrix, y: DenseVector) : DenseVector = {
        val b = reg(X, y)
        val y_hat = X.multiply(b)
        val eps = as_breeze( y ) - as_breeze( y_hat )
        return from_breeze(eps:^2.0)
    }

    def logit(X: DenseMatrix, y: DenseVector, max_iter: Int) : DenseVector = {
        val N = X.numRows
        val K = X.numCols

        var w = vectorize( Matrices.rand(K, 1, new Random()) )
        var iteration = 0
        val stepSize = 0.001

        while (iteration < max_iter) {
            val xb = X.multiply(w)
            val eps = from_breeze(
                as_breeze( y ) - bNum.sigmoid( as_breeze( xb )))
            val delta = as_breeze(
                X.transpose.multiply(eps)):*(stepSize/N.toDouble)
            w = from_breeze( as_breeze( w ) + delta )
            iteration = iteration + 1
        }

        return w
    }

    def reg(X: DenseMatrix, y: DenseVector) : DenseVector = {
        return as_dense(from_breeze((as_breeze( X.transpose.multiply(X) ).
                    asInstanceOf[bDenseMatrix[Double]] \ 
                as_breeze( X.transpose.multiply(y) ).
                    asInstanceOf[bDenseVector[Double]])))
    }

    def gnmf(X: DenseMatrix, r: Int, iterations: Int) : 
        (DenseMatrix, DenseMatrix) = {
        val N = X.numRows
        val K = X.numCols
        var W = as_dense(Matrices.rand(N, r, new Random()))
        var H = as_dense(Matrices.rand(r, K, new Random()))

        var iteration = 0
        while (iteration < iterations) {
            W = from_breeze( 
                as_breeze(W) :* (
                    as_breeze(X.multiply(H.transpose)) :/
                    as_breeze(W.multiply(H.multiply(H.transpose))) )
            )
            H = from_breeze(
                as_breeze(H) :* (
                    as_breeze(W.transpose.multiply(X)) :/
                    as_breeze(W.transpose.multiply(W).multiply(H))
                )
            )
            iteration = iteration + 1
        }
        return (W,H)
    }

    def robust_se(X: DenseMatrix, eps: DenseVector) : DenseMatrix = {
        val XTX_INV = from_breeze(bAlg.inv(as_breeze(X.transpose.multiply(X))))
        val XBT = as_breeze( X.transpose )
        val S = XBT(*,::)*as_breeze(eps)
        val VAR = XTX_INV.multiply(from_breeze(S).
                                   multiply(X).
                                   multiply(XTX_INV))
        return VAR
    }

    def as_breeze(v: linalg.DenseVector) : bDenseVector[Double] = {
        return new bDenseVector[Double](v.values)
    }

    def from_breeze(v: bDenseVector[Double]) : linalg.DenseVector = {
        return as_dense(Vectors.dense(v.data))
    }

    def as_breeze(m: linalg.DenseMatrix) : bDenseMatrix[Double] = {
        return new bDenseMatrix(m.numRows, m.numCols, m.toArray)
    }

    def from_breeze(m: bDenseMatrix[Double]) : linalg.DenseMatrix = {
        return as_dense(Matrices.dense(m.rows, m.cols, m.toDenseMatrix.data))
    }

    def vectorize(M: Matrix) : DenseVector = {
        return as_dense( Vectors.dense( M.toArray ) )
    }

    def as_dense(M: Matrix) : DenseMatrix = {
        return M.asInstanceOf[DenseMatrix]
    }

    def as_dense(v: Vector) : DenseVector = {
        return v.asInstanceOf[DenseVector]
    }
}
