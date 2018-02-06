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

        val stub = "/tests/MLAlgorithms (Distributed Sparse LA)/output/"
        val base = s"mllib_${mattype}_${opType}${nodes}.txt"
        val path = root + stub + base

        if (!Files.exists(Paths.get(path))) {
          File(path).writeAll("nodes,time1,time2,time3,time4,time5\n")
        }

        val (x, y) = dataPath match {
            case Some(path) => parquet_to_bmm(path, spark)
            case None => (readSMM(Xpath, sc), readMM(Ypath, sc))
        }

        // BlockMatrix.toIndexedRowMatrix does not correctly preserve dimensions
        val (xir, yir) = opType match {
            case "reg" | "robust" => (x.toCoordinateMatrix.toIndexedRowMatrix, 
                                      y.toCoordinateMatrix.toIndexedRowMatrix)
            case _     => (empty_IRM(sc), empty_IRM(sc))
        }
        xir.rows.persist(MEMORY_AND_DISK_SER)
        yir.rows.persist(MEMORY_AND_DISK_SER)

        val r2 = random_matrix(xir.numRows.toInt, 1, 100, 1, sc).toIndexedRowMatrix

        val xr : RowMatrix = opType match {
            case "pca" => x.toIndexedRowMatrix.toRowMatrix
            case _     => empty_IRM(sc).toRowMatrix
        }
        xr.rows.persist(MEMORY_AND_DISK_SER)
        xr.rows.count

        val times = Array[Double](0,0,0,0,0)
        for (ix <- 0 to 4) {
            println(s"Test => ${ix}")
            val start = System.nanoTime()

            if (opType == "logit")
                logit(x, y, 3, sc)
            if (opType == "gnmf")
                gnmf(x, 10, 3, sc)
            if (opType == "reg")
                reg(xir, yir)
            if (opType == "robust")
                robust_se(xir, r2)
            if (opType == "pca")
                pca(xr, 5)

            val stop = System.nanoTime()
            times(ix) = (stop - start)/1e9
        }

        File(path).appendAll(
            nodes + "," + times.mkString(",") + "\n")
    }

    def empty_IRM(sc: SparkContext) : IndexedRowMatrix = {
        val CMM = new CoordinateMatrix(
            sc.parallelize(Seq(new MatrixEntry(1,1,1)))
        )
        return CMM.toIndexedRowMatrix()
    }

    def readSMM(path: String, sc: SparkContext) : BlockMatrix = {
        val root = sys.env("BENCHMARK_PROJECT_ROOT")
        val meta_name = path.split("/").last
        val meta_path = s"${root}/data/SimpleMatrixOps (Sparse Data)/output/${meta_name}.mtd"
        val (rows, cols, nnz) = parse_meta(meta_path)

        val M = sc.textFile(path, 1000).zipWithIndex().filter(tup => tup._2 > 2).
            map(tup => tup._1.trim.split(" ")).
            map(row => new MatrixEntry(row(0).toLong-1,
                                       row(1).toLong-1,
                                       row(2).toDouble))
        val BM = new CoordinateMatrix(M, rows, cols).toBlockMatrix(100,100)
        BM.persist(MEMORY_AND_DISK_SER)
        println(BM.blocks.count)
        return BM
    }

    def parse_meta(path: String) : (Long, Long, Long) = {
        val meta = Source.fromFile(path).
            mkString.replace("{","").
                replace("}","").
                replace(" ","").
                split(",\n").map(x => x.split(":")).
                map({case Array(x,y) => (x -> y)}).toMap
        return (meta("\"rows\"").toLong,
                meta("\"cols\"").toLong,
                meta("\"nnz\"").toLong)
    }

    def readMM(path: String, sc: SparkContext) : BlockMatrix = {
        val M = sc.textFile(path, 1000).zipWithIndex().
                         map(
                            tup => new IndexedRow(
                                tup._2.toInt,
                                new DenseVector(tup._1.split(",").
                                    map(_.toDouble))))
        val IRM = new IndexedRowMatrix(M)
        val BM = IRM.toBlockMatrix(100,IRM.numCols.toInt)
        BM.persist(MEMORY_AND_DISK_SER)
        println(BM.blocks.count)
        return BM
    }

    def parquet_to_bmm(path: String, spark: SparkSession) :
        (BlockMatrix, BlockMatrix) = {
        println("READING PARQUET")
        val df = spark.read.parquet(path).repartition(1000)
        df.persist(MEMORY_AND_DISK_SER)

        val X_rows = df.select("features").rdd.zipWithIndex().
            map(tup => new IndexedRow(
                tup._2.toInt, SparseVector.fromML(
                    tup._1.getAs[alg.Vector](0).toSparse)))
        val X = new IndexedRowMatrix(X_rows).toBlockMatrix(100,100)
        X.persist(MEMORY_AND_DISK_SER)
        println(X.blocks.count)
        println("NUM PARTITIONS")
        println(X.blocks.getNumPartitions)

        val y_rows = df.select("y").rdd.zipWithIndex().
            map(tup => new IndexedRow(tup._2.toInt,
                                      Vectors.dense(tup._1.getAs[Int](0)).
                                      toSparse))
        val y = new IndexedRowMatrix(y_rows).toBlockMatrix(100,1)
        y.persist(MEMORY_AND_DISK_SER)
        println(y.blocks.count)
        return (X, y)
    }

    def logit(X: BlockMatrix, y: BlockMatrix,
              max_iter: Int = 3, sc: SparkContext) : BlockMatrix = {

        println("--------> LOGIT <---------")
        var iteration = 0
        var step_size = 0.001
        val N = X.numRows.toInt
        val K = X.numCols.toInt
        var w = random_matrix(K, 1, y.rowsPerBlock, y.colsPerBlock, sc)
        val XT = X.transpose
        XT.persist(MEMORY_AND_DISK_SER)

        while (iteration < max_iter) {
            println( s"Test => ${iteration}" )
            val xb = X.multiply( w )
            val gg = new BlockMatrix(xb.blocks.map({
                case ((x,y), mat) => (
                        (x,y), from_breeze(
                    bNum.sigmoid( to_dense( as_breeze( mat ) ) ) )
                )
            }), y.rowsPerBlock, y.colsPerBlock)
            val eps = gg.subtract( y )
            val w_update = scalar_multiply((step_size/N.toDouble),
                XT.multiply( eps, 500 ) ) 
            w = w.subtract( w_update )
            step_size /= 2.0
            iteration += 1
        }

        return w
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
        return new IndexedRowMatrix(AB, A.numRows.toInt, A.numCols.toInt)
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

    def scalar_multiply(v: Double, M: BlockMatrix) : BlockMatrix = {
        return new BlockMatrix(M.blocks.map({
            case ((x,y), mat) => ((x,y), from_breeze(v*as_breeze(mat)))
        }), M.rowsPerBlock, M.colsPerBlock)
    }

    def reg(X: IndexedRowMatrix, y: IndexedRowMatrix) : Matrix = {
        // here we use Tikhonov regularization to ensure the sparse system is nonsingular
        println("--------> REG <---------")

        val XTX = X.computeGramianMatrix()
        val XTY = X.toBlockMatrix(100,100).transpose.multiply(
                      y.toBlockMatrix(100,1), 500).toLocalMatrix
        val T = to_sparse( as_breeze( Matrices.speye( XTX.numCols.toInt ) ) ):*0.001
        val b = from_breeze( (( to_dense( as_breeze( XTX ) ) + (T.t * T)) \
                               as_breeze( XTY ).toDenseMatrix.
                                                toDenseVector ).toDenseMatrix )
        return b.transpose
    }

    def to_dense(M: bMatrix[Double]) : bDenseMatrix[Double] = {
        return M.asInstanceOf[bDenseMatrix[Double]]
    }

    def to_sparse(M: bMatrix[Double]) : bSparseMatrix[Double] = {
        return M.asInstanceOf[bSparseMatrix[Double]]
    }

    def robust_se(X: IndexedRowMatrix, r2: IndexedRowMatrix) : Matrix = {
        println("--------> ROBUST <---------")

        val XTX_INV = from_breeze( bAlg.inv(
            to_dense( as_breeze( X.computeGramianMatrix() ) ) )
        ).asInstanceOf[DenseMatrix]
        val XB = X.toBlockMatrix(100,100)
        val S = diagonalize( r2 )
        val SW = XB.transpose.multiply( S, 500 ).multiply( XB, 500 ).toLocalMatrix
        return XTX_INV.multiply( SW.asInstanceOf[DenseMatrix] ).multiply( XTX_INV )
    }

    def pca(X: RowMatrix, k: Int) : RowMatrix = {
        val S = X.computeCovariance()
        val eigs = bAlg.eigSym( to_dense( as_breeze( S ) ) )
        val ixs = bAlg.argsort(eigs.eigenvalues).reverse.slice(0,k)
        val U = from_breeze( eigs.eigenvectors(::,ixs).toDenseMatrix )
        val PRJ = X.multiply( U ) // note: we really should subtract the mean...
        return PRJ
    }

    def elem_pow(M: IndexedRowMatrix, v: Double) : IndexedRowMatrix = {
        val rows_rdd = M.rows.map(row => new IndexedRow(row.index,
                from_breeze(bNum.pow(as_breeze(row.vector), v))))
        return new IndexedRowMatrix(rows_rdd, M.numRows.toInt, M.numCols.toInt)
    }

    // converts a distributed vector (as an IRM) to a sparse BlockMatrix
    def diagonalize(M: IndexedRowMatrix) : BlockMatrix = {
        val elements = M.rows.map(row =>
            new MatrixEntry(row.index,
                            row.index,
                            row.vector.toArray(0)))
        return new CoordinateMatrix(
            elements, M.numRows.toInt, M.numRows.toInt).toBlockMatrix(100,100)
    }

    def gnmf(X: BlockMatrix, r: Int, max_iter: Int, sc: SparkContext) :
            (BlockMatrix, BlockMatrix) = {

        val RNG = new Random()
        val N = X.numRows.toInt
        val K = X.numCols.toInt
        var W = random_matrix(N, r, X.rowsPerBlock, X.colsPerBlock, sc)
        var H = random_matrix(r, K, X.rowsPerBlock, X.colsPerBlock, sc)

        println("--------> GNMF <---------")

        W.blocks.persist(MEMORY_AND_DISK_SER)
        H.blocks.persist(MEMORY_AND_DISK_SER)

        var iteration = 0
        while (iteration < max_iter) {
            println(s"Iteration => ${iteration}")
            W = elem_multiply(W, elem_divide( X.multiply( H.transpose, 500 ),
                              W.multiply( H.multiply( H.transpose, 500 ), 500)))
            H = elem_multiply(H, elem_divide( W.transpose.multiply( X, 500 ),
                              (W.transpose.multiply( W, 500 ).multiply( H, 500 )) ))
            iteration = iteration + 1
        }

        return (W, H)
    }

    def random_matrix(N: Int, K: Int,
                      r: Int, c: Int, sc: SparkContext) : BlockMatrix = {
        val MM = new IndexedRowMatrix(
            normalVectorRDD(sc, N.toLong, K).
                repartition(1000).zipWithIndex().map(
                tup => new IndexedRow(tup._2, tup._1))).toBlockMatrix(r, c)
        return new BlockMatrix( MM.blocks.map(
            {case ((x,y), mat) => ((x,y),
             mat.asInstanceOf[SparseMatrix])}), r, c)
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
