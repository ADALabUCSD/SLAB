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
import scala.collection.immutable._

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

        val opType = argMap("opType")
        val Xpath = argMap("Xpath")
        val Ypath = argMap("Ypath")
        val passPath = argMap("passPath")
        val nodes = argMap("nodes")
        val mattype = argMap("mattype")
        val dataPath = argMap.get("dataPath")       

        val stub = "/tests/MLAlgorithms (Distributed Dense LA)/output/"
        val base = s"mllib_${mattype}_${opType}${nodes}.txt"
        val path = root + stub + base

        if (!Files.exists(Paths.get(path))) {
          File(path).writeAll("nodes,time1,time2,time3,time4,time5\n")
        }

        val (x, y) = dataPath match {
            case Some(path) => parquet_to_irm(path, spark)
            case None => (readMM(Xpath, sc), readMM(Ypath, sc))
        }

        val r2: IndexedRowMatrix = opType match {
            case "robust" => compute_r2( x, y )
            case _        => readMM(passPath, sc)
        }

        val xr : RowMatrix = opType match {
            case "pca" => x.toRowMatrix
            case _     => random_matrix(10, 10, 10, 10, sc).
                            toIndexedRowMatrix.
                            toRowMatrix
        }
        xr.rows.persist(MEMORY_AND_DISK_SER)
        xr.rows.count

        val times = Array[Double](0,0,0,0,0)
        for (ix <- 0 to 4) {
            println(s"Test: ${ix}")
            val start = System.nanoTime()

            if (opType == "logit")
                logit(x, y, 3, sc)
            if (opType == "gnmf")
                gnmf(x, 10, 3, sc)
            if (opType == "reg")
                reg(x, y)
            if (opType == "robust")
                robust_se(x, r2)
            if (opType == "pca")
                pca(xr, 5)

            val stop = System.nanoTime()
            times(ix) = (stop - start)/1e9
        }

        File(path).appendAll(
            nodes + "," + times.mkString(",") + "\n")
    }

    def compute_r2(x: IndexedRowMatrix, y: IndexedRowMatrix) : 
        IndexedRowMatrix = {
           val b = reg(x, y)
           val y_hat = x.multiply( b )
           val r2 = elem_pow( elem_subtract( y, y_hat ), 2.0 )
           return r2 
        }

    def readMM(path: String, sc: SparkContext) :
            IndexedRowMatrix = {
        val M = sc.textFile(path, 500).zipWithIndex().
                         map(
                            tup => new IndexedRow(
                                tup._2.toInt,
                                new DenseVector(tup._1.split(",").
                                    map(_.toDouble))))
        val BM = new IndexedRowMatrix(M)
        BM.rows.persist(MEMORY_AND_DISK_SER)
        return BM
    }

    def parquet_to_irm(path: String, spark: SparkSession) :
        (IndexedRowMatrix, IndexedRowMatrix) = {
        val df = spark.read.parquet(path)
        df.persist(MEMORY_AND_DISK_SER)

        val X_rows = df.repartition(1000).
            select("dense_features_scaled").rdd.zipWithIndex().
            map(tup => new IndexedRow(tup._2.toInt,
                                      DenseVector.fromML(
                                          tup._1.getAs[alg.Vector](0).toDense)))
        val X = new IndexedRowMatrix(X_rows)
        X.rows.persist(MEMORY_AND_DISK_SER)

        val y_rows = df.repartition(1000).
            select("y").rdd.zipWithIndex().
            map(tup => new IndexedRow(tup._2.toInt,
                                      Vectors.dense(tup._1.getAs[Int](0))))
        val y = new IndexedRowMatrix(y_rows)
        y.rows.persist(MEMORY_AND_DISK_SER)
        return (X, y)
    }

    def logit(X: IndexedRowMatrix, y: IndexedRowMatrix,
              max_iter: Int = 3, sc: SparkContext) : Matrix = {
        var iteration = 0
        var step_size = 0.001
        val N = X.numRows.toInt
        val K = X.numCols.toInt
        var w = Matrices.rand(K, 1, new Random())
        val XT = X.toBlockMatrix(1024,X.numCols.toInt).transpose
        XT.persist(MEMORY_AND_DISK_SER)

        while (iteration < max_iter) {
            println(s"Iteration => ${iteration}")
            val xb = X.multiply( w )
            val gg = new IndexedRowMatrix(
                xb.rows.map(row => new IndexedRow(
                    row.index, from_breeze(
                        bNum.sigmoid(as_breeze(row.vector))))
            ))
            val eps = elem_subtract(gg, y).toBlockMatrix(XT.colsPerBlock,1)
            val XTe = XT.multiply( eps, 500 ).toLocalMatrix
            val w_update = (step_size/N.toDouble)*as_breeze( XTe ) 
            w = from_breeze( as_breeze( w ) :- w_update )
            step_size /= 2.0
            iteration += 1
        }

        return w
    }

    def vectorize(M: IndexedRowMatrix) : Matrix = {
        val loc = M.toCoordinateMatrix.toBlockMatrix()
        return loc.toLocalMatrix()
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

    def elem_subtract(A: IndexedRowMatrix,
                      B: IndexedRowMatrix) : IndexedRowMatrix = {
        val both = join_row_matrices(A,B)
        val AB = both.map(tup => new IndexedRow(tup._1,
            from_breeze(as_breeze(tup._2._1) :- as_breeze(tup._2._2)))
        )
        return new IndexedRowMatrix(AB)
    }

    def transpose_row_matrix(A: IndexedRowMatrix) : IndexedRowMatrix = {
        // hacky way to transpose an IRM
        return A.toCoordinateMatrix.transpose.toIndexedRowMatrix
    }

    def join_row_matrices(A: IndexedRowMatrix,
                          B: IndexedRowMatrix) :
            RDD[(Long, (linalg.Vector,linalg.Vector))] = {
        val pair_A = A.rows.map(row => (row.index, row.vector))
        val pair_B = B.rows.map(row => (row.index, row.vector))
        return pair_A.join(pair_B)
    }

    def scalar_multiply(v: Double, M: IndexedRowMatrix) : IndexedRowMatrix = {
        val rows_rdd = M.rows.map(row => new IndexedRow(row.index,
                from_breeze(v*as_breeze(row.vector))))
        return new IndexedRowMatrix(rows_rdd)
    }

    def reg(X: IndexedRowMatrix, y: IndexedRowMatrix) : Matrix = {
        val XTX = X.computeGramianMatrix()
        val XTY = X.toBlockMatrix(1024,X.numCols.toInt).transpose.
                    multiply( y.toBlockMatrix(1024,1), 500 ).
                    toLocalMatrix
        val b = from_breeze( to_dense( as_breeze( XTX ) ) \
                             to_dense( as_breeze( XTY ) ) )
        return b
    }

    def to_dense(M: bMatrix[Double]) : bDenseMatrix[Double] = {
        return M.asInstanceOf[bDenseMatrix[Double]]
    }

    def robust_se(X: IndexedRowMatrix, 
                  r2: IndexedRowMatrix) : Matrix = {
        val XTX_INV = from_breeze( bAlg.inv( 
            to_dense( as_breeze( X.computeGramianMatrix() ) ) ) 
        ).asInstanceOf[DenseMatrix]
        val XB = X.toBlockMatrix(1024,X.numCols.toInt)
        val S = diagonalize( r2 )
        val SW = XB.transpose.multiply( S, 500 ).multiply( XB, 500 ).
                    toLocalMatrix
        return XTX_INV.multiply( SW.asInstanceOf[DenseMatrix] ).multiply( XTX_INV )
    }

    def pca(X: RowMatrix, k: Int) : RowMatrix = {
        val S = X.computeCovariance()
        val eigs = bAlg.eigSym( to_dense( as_breeze( S ) ) )
        val eigenvectors = eigs.
                eigenvectors(::,bAlg.argsort(eigs.eigenvalues).reverse)
        val U = from_breeze( eigenvectors(::,0 to k-1).toDenseMatrix )
        val PRJ = X.multiply( U ) // note: we really should subtract the mean
        return PRJ
    }

    def elem_pow(M: IndexedRowMatrix, v: Double) : IndexedRowMatrix = {
        val rows_rdd = M.rows.map(row => new IndexedRow(row.index,
                from_breeze(bNum.pow(as_breeze(row.vector), v))))
        return new IndexedRowMatrix(rows_rdd)
    }

    // converts a distributed vector (as an IRM) to a sparse BlockMatrix
    def diagonalize(M: IndexedRowMatrix) : BlockMatrix = {
        val elements = M.rows.map(row =>
            new MatrixEntry(row.index, 
                            row.index,
                            row.vector.toArray(0)))
        return new CoordinateMatrix(elements).toBlockMatrix
    }

    def gnmf(Xr: IndexedRowMatrix, r: Int, max_iter: Int, sc: SparkContext) :
            (BlockMatrix, BlockMatrix) = {
        val X = Xr.toBlockMatrix(100, 100)
        X.blocks.persist(MEMORY_AND_DISK_SER)
        val RNG = new Random()
        val N = X.numRows.toInt
        val K = X.numCols.toInt
        var W = random_matrix(N, r, 100, 100, sc)
        var H = random_matrix(r, K, 100, 100, sc)

        W.blocks.persist(MEMORY_AND_DISK_SER)
        H.blocks.persist(MEMORY_AND_DISK_SER)

        var iteration = 0
        while (iteration < max_iter) {
            println(s"Iteration => ${iteration}")
            W = elem_multiply(W, elem_divide( X.multiply( H.transpose, 500 ),
                              W.multiply( H.multiply( H.transpose, 500 ), 500)))
            H = elem_multiply(H, elem_divide( W.transpose.multiply( X, 500 ),
                              (W.transpose.multiply( W, 500 ).multiply( H, 500 ))))
            iteration = iteration + 1
        }

        return (W, H)
    }

    def random_matrix(N: Int, K: Int,
                      r: Int, c: Int, sc: SparkContext) : BlockMatrix = {
        val MM = new IndexedRowMatrix(
            normalVectorRDD(sc, N.toLong, K).zipWithIndex().map(
                tup => new IndexedRow(tup._2, tup._1))).toBlockMatrix(r, c)
        return MM
    }

    def join_block_matrices(A: BlockMatrix,
                            B: BlockMatrix) :
            RDD[((Int, Int), (linalg.Matrix,linalg.Matrix))] = {
        val pair_A = A.blocks.map(block => block)
        val pair_B = B.blocks.map(block => block)
        return pair_A.join(pair_B)
    }

    def elem_divide(A: BlockMatrix, B: BlockMatrix) : BlockMatrix = {
        val both = join_block_matrices( A, B )
        val new_blocks = both.map(block => ((block._1._1, block._1._2),
            from_breeze( as_breeze( block._2._1 ) :/ as_breeze( block._2._1 ))))
        return new BlockMatrix(new_blocks, A.rowsPerBlock, B.rowsPerBlock)
    }

    def elem_multiply(A: BlockMatrix, B: BlockMatrix) : BlockMatrix = {
        val both = join_block_matrices( A, B )
        val new_blocks = both.map(block => (block._1,
            from_breeze( as_breeze( block._2._1 ) :* as_breeze( block._2._1 ))))
        return new BlockMatrix(new_blocks, A.rowsPerBlock, B.rowsPerBlock)
    }
}

