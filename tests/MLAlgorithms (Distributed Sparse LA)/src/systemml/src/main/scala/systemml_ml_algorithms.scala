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
import org.apache.sysml.api.mlcontext._
import org.apache.sysml.api.mlcontext.ScriptFactory._
import org.apache.sysml.api.mlcontext.MatrixFormat._
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import scala.io.Source
import breeze.{math => bMath, 
               numerics => bNum,
               linalg => bAlg}
import breeze.linalg.{Vector => bVector,
                      Matrix => bMatrix,
                      SparseVector => bSparseVector, 
                      DenseVector => bDenseVector,
                      CSCMatrix => bSparseMatrix,
                      DenseMatrix => bDenseMatrix}
import org.apache.spark.sql._
import org.apache.spark.sql.types._
import scala.tools.nsc.io._
import scala.collection.immutable._
import org.apache.spark.storage.StorageLevel._
import org.apache.spark.ml.{linalg => alg}
import org.apache.spark.mllib.{linalg => malg}
import org.apache.spark.mllib.linalg.distributed._
import org.apache.spark.ml.linalg.SQLDataTypes.VectorType

object SystemMLMLAlgorithms extends App {

    override def main(args: Array[String]) {
        val conf = new SparkConf().setAppName("MLLibMatrixOps")
        val sc = new SparkContext(conf)
        val ml = new MLContext(sc)
        ml.setExplain(true)
        ml.setExplainLevel(MLContext.ExplainLevel.RUNTIME)
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
        val sr = argMap.get("sr")
        val ncol = argMap.get("ncol")

        val (x, y) = dataPath match {
            case Some(path) => readParquet(path, spark)
            case None => (readSMM(Xpath, sc, spark),
                          readMM(Ypath, spark))
        }

        val pathname = dataPath match {
            case Some(path) => path
            case None => Xpath
        }

        val isSparse = pathname.contains("sparse")
        val stub = "/tests/MLAlgorithms (Distributed Sparse LA)/output/"
        val base = s"systemml_${mattype}_${opType}${nodes}.txt"
        val path = root + stub + base
        if (!File(path).exists) {
            File(path).writeAll("nodes,sr,ncol,time1,time2,time3,time4,time5\n")
        }

        val meta_stub = dataPath match {
            case Some(path) => "data/Criteo (Build Derived Tables)"
            case None => "data/SimpleMatrixOps (Sparse Data)"
        }

        val meta_name = dataPath match {
            case Some(path) => pathname.split("/").last.replace("parquet","csv")
            case None => pathname.split("/").last
        }

        val meta_path = s"${root}/${meta_stub}/output/${meta_name}.mtd"
        val (rows, cols, nnz) = parse_meta(meta_path)
        val x_meta = new MatrixMetadata(DF_VECTOR, rows, cols, nnz)
        val y_meta = new MatrixMetadata(DF_DOUBLES, rows, 1, nnz)

        val call = opType match {
            case "logit"  => "logit(X, y, 10)"
            case "gnmf"   => "gnmf(X, 10, 10)"
            case "reg"    => "reg(X, y)"
            case "robust" => "robust_se(X, r2)"
            case "pca"    => "pca(X, 5)"
            case _        => throw new Exception("Invalid Operator")
        }

        val print_op = opType match {
           case "gnmf" => ""
           case _      => "res = utils::printRandElements(tmp, 10)"
        }

        val preprocess_string = opType match {
            case "robust" => s"r2 = rand(rows=${rows},cols=1,pdf='normal')"
            case _        => ""
        }

        val libDir = root + "/lib/dml"
        val dmlText =
          s"""
            | setwd('${libDir}')
            | source('utils.dml') as utils
            |
            | p = sum( X )
            | q = sum( y )
            | print(p)
            | print(q)
            |
            | ${preprocess_string}
            |
            | times = matrix(0.0, rows = 5, cols = 1)
            | for (ix in 1:5) {
            |   if ((p != 0) & (q != 0)) {
            |       start = utils::time(1)
            |   }
            |   tmp = ${call}
            |   if ((p != 0) & (q != 0)) {
            |       ${print_op}
            |       stop = utils::time(1)
            |   }
            |   times[ix,1] = (stop - start) / 1000
            | }
            | times = t(times)
            |
            | logit = function(matrix[double] X, 
            |                  matrix[double] y, 
            |                  Integer iterations)
            |     return (matrix[double] w) {
            |
            |     N = nrow(X)
            |     w = matrix(0, rows=ncol(X), cols=1)
            |     iteration = 0
            |     stepSize = 10
            |
            |     while (iteration < iterations) {
            |         xb = X %*% w
            |         delta = 1/(1+exp(-xb)) - y
            |         stepSize = stepSize / 2
            |         w = w - ((stepSize * t(X) %*% delta)/N)
            |
            |         iteration = iteration+1
            |     }
            | }
            |
            | gnmf = function(matrix[double] X, Integer r, Integer iterations)
            |     return (integer iteration) {
            |     W = rand(rows = nrow(X), cols = r, pdf = 'uniform')
            |     H = rand(rows = r, cols = ncol(X), pdf = 'uniform')
            |
            |     for (i in 1:3) {
            |         W = W * ((X %*% t(H)) / (W %*% (H %*% t(H))))
            |         H = H * ((t(W) %*% X) / ((t(W) %*% W) %*% H))
            |     }
            |     if ((as.scalar(W[1,1]) >  0) & (as.scalar(H[1,1]) > 0)) {
            |         print(as.scalar(H[1,1]))
            |         print(as.scalar(W[1,1]))
            |     }
            |
            |     iteration = 0
            | }
            |
            | reg = function(matrix[double] X, matrix[double] y)
            |     return (matrix[double] b) {
            |     # ensure XTX is nonsingular
            |     T = 0.001*diag( matrix( 1, rows=ncol(X), cols=1 ) )
            |     XTX = (t(X) %*% X) + (t(T) %*% T)
            |     b = solve(XTX, t(X) %*% y)
            | }
            |
            | robust_se = function(matrix[double] X, 
            |                      matrix[double] r2) 
            |     return (matrix[double] se) {
            |     # NOTE: SVD is cheap since XTX is small!
            |     [U,H,V] = svd( t(X) %*% X )
            |     h = diag( H )
            |     XTX_INV = U %*% diag(h^-1) %*% t(V)
            |     S = diag( r2 )
            |     se = XTX_INV %*% (t(X) %*% S %*% X) %*% XTX_INV
            | }
            |
            | pca = function(matrix[double] X, Integer k) 
            |   return (matrix[double] PRJ) {
            |     N = nrow( X )
            |     K = ncol( X )
            |     XS = X - colMeans( X )
            |     S = (1/(N-1))*(t( XS ) %*% XS)
            |     [eigvals, eigvects] = eigen( S )
            |     
            |     # Thanks to the Sysml implementation for this helpful bit 
            |     # of code to sort the eigenvectors
            |
            |     eigssorted = order(target=eigvals,by=1, 
            |                        decreasing=TRUE,
            |                        index.return=TRUE)
            |     diagmat = table(seq(1,K), eigssorted)
            |     eigvals = diagmat %*% eigvals
            |     eigvects = eigvects %*% diagmat
            |     eigvects = eigvects[,1:k]
            |
            |     PRJ = XS %*% eigvects
            | }
        """.stripMargin
        println("Running DML: ")
        println(dmlText)
        val script = dml(dmlText).in(Seq(("X", x, x_meta),
                                         ("y", y, y_meta))).out("times")
        val res = ml.execute(script)
        val results = res.getTuple[Matrix]("times")._1.to2DDoubleArray

        val prefix = Array(nodes, sr, ncol)
        File(path).appendAll(
            prefix.mkString(",") + "," + results(0).mkString(",") + '\n')
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

    def readSMM(path: String, 
                sc: SparkContext,
                spark: SparkSession) : DataFrame = {

        val schema = StructType(Seq(StructField("_c0", VectorType)))
        val M = sc.textFile(path, 500).
                   zipWithIndex().filter(tup => tup._2 > 2).
                   map(tup => tup._1.trim.split(" ")).
                   map(row => new MatrixEntry(
                       row(0).toLong, row(1).toLong, row(2).toDouble))
        val RM = new CoordinateMatrix( M ).toRowMatrix()
        val df = spark.createDataFrame(RM.rows.map(x => Row(x.asML)), schema)
        df.persist(MEMORY_AND_DISK_SER)
        df.count
        return df
    }

    def readMM(path: String, spark: SparkSession) : DataFrame = {
        val M = spark.read.option("header", false).
            option("inferSchema",true).csv(path).repartition(1000)
        M.persist(MEMORY_AND_DISK_SER)
        M.count
        return M
    }

    def readParquet(path: String, spark: SparkSession) : (DataFrame, DataFrame) = {
        val M = spark.read.parquet(path).repartition(1000)
        M.persist(MEMORY_AND_DISK_SER)

        val X = M.select("features")
        val y = M.select("y")
        return (X,y)
    }

}
