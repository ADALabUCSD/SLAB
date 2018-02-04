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
import org.apache.spark.sql.SparkSession
import org.apache.spark.SparkContext
import org.apache.spark.storage.StorageLevel._
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.mllib.linalg.distributed._
import org.apache.spark.mllib.linalg._
import org.apache.spark.ml.{linalg => alg}
import org.apache.spark.sql._
import scala.tools.nsc.io._

object SparkMLAlgorithms {
    def main(args: Array[String]) {
        val spark = SparkSession.builder.getOrCreate()

        val root = sys.env("BENCHMARK_PROJECT_ROOT")
        val argMap = Map[String,String](
                args.map(_.split("=")).map({
                    case Array(x,y) => (x -> y)
                }):_*
            )

        val opType = argMap("opType")
        val inputPath = argMap("inputPath")
        val nodes = argMap("nodes")
        val featureNames = argMap("featureNames")
        val isSparse = inputPath.contains("sparse")

        val sparse_stub = if (isSparse) "_sparse" else "_dense"
        val stub = "/tests/MLAlgorithms (Native Implementations)/output/"
        val base = s"mllib_${opType}${nodes}${sparse_stub}.txt"
        val path = root + stub + base

        val input = spark.read.parquet(inputPath).
            withColumnRenamed(featureNames,"features")
        input.persist(MEMORY_AND_DISK_SER)
        println(input.count)

        val xRM = opType match {
            case "pca" => parquet_to_rm(input.select("features"))
            case _     => blank_row_matrix(spark.sparkContext)
        }

        val times = Array[Double](0,0,0,0,0)
        for (ix <- 0 to 4) {
            println(s"Test => ${ix}")
            val start = System.nanoTime()

            if (opType == "logit") {
                val lr = new LogisticRegression().
                    setMaxIter(3).
                    setLabelCol("y")
                val params = lr.fit(input)
                println(params.coefficients.size)
            } else if (opType == "reg") {
                val reg = new LinearRegression().
                    setMaxIter(3).
                    setLabelCol("y")
                val params = reg.fit(input)
                println(params.coefficients.size)
            } else if (opType == "pca") {
                val prcomp = xRM.computePrincipalComponents( 5 )
                val prj = xRM.multiply(prcomp)
                prj.rows.count
            } else {
                throw new Exception("Invalid operator")
            }
            val stop = System.nanoTime()
            times(ix) = (stop - start)/1e9
        }

        File(path).writeAll("nodes,time1,time2,time3,time4,time5\n")
        File(path).appendAll(
            nodes + "," + times.mkString(",") + '\n')
    }

    def parquet_to_rm(df: DataFrame) : RowMatrix = {
        val X_rows = df.repartition(1000).rdd.
            map(tup => DenseVector.fromML(
                tup.getAs[alg.Vector](0).toDense).asInstanceOf[Vector])
        X_rows.persist(MEMORY_AND_DISK_SER)
        return new RowMatrix(X_rows)
    }

    def blank_row_matrix(sc: SparkContext) : RowMatrix = {
        val tmp = sc.parallelize(
            Seq(Vectors.dense(1,2,3), Vectors.dense(1,2,3)))
        return new RowMatrix(tmp)
    }
}
