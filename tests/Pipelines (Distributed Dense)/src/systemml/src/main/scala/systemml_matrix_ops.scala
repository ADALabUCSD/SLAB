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
import org.apache.sysml.api.mlcontext.MatrixFormat._
import org.apache.sysml.api.mlcontext.ScriptFactory._
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.spark.sql._
import scala.tools.nsc.io._
import scala.collection.immutable._
import org.apache.spark.sql.types._
import org.apache.spark.storage.StorageLevel._
import org.apache.spark.mllib.random.RandomRDDs._
import org.apache.spark.ml.linalg.SQLDataTypes.VectorType

object SystemMLPipelines extends App {

    override def main(args: Array[String]) {
        val conf = new SparkConf().setAppName("SystemMLMatrixOps")
        val sc = new SparkContext(conf)
        val spark = SparkSession.builder.getOrCreate()
        val ml = new MLContext(sc)
        ml.setExplain(true)
        ml.setExplainLevel(MLContext.ExplainLevel.RUNTIME)
        val root = sys.env("BENCHMARK_PROJECT_ROOT")

        val argMap = Map[String,String](
                args.map(_.split("=")).map({
                    case Array(x,y) => (x -> y)
                }):_*
            )

        val stub = s"/tests/Pipelines (Distributed Dense)/output/"
        val base = s"systemml_pipelines.txt"
        val path = root + stub + base

        File(path).writeAll("nodes,rows,cols,time1,time2,time3,time4,time5\n")
        val libDir = root + "/lib/dml"
        val dmlText =
        s"""
          | setwd('${libDir}')
          | time = externalFunction(Integer i) return (Double B)
          |      implemented in (classname="org.apache.sysml.udf.lib.TimeWrapper", exectype="mem");
          | q = sum( t )
          | r = sum( u )
          | s = sum( v )
          |
          | times = matrix(0.0, rows = 5, cols = 1)
          | for (ix in 1:5) {
          |   if ((q != 0) & (r != 0) & (s != 0)) {
          |       start = time(1)
          |   }
          |   res = t %*% u %*% v
          |   if ((q != 0) & (r != 0) & (s != 0)) {
          |       print(as.scalar(res[1,1]))
          |   }
          |   if ((q != 0) & (r != 0) & (s != 0)) {
          |       stop = time(1)
          |   }
          |   times[ix,1] = (stop - start) / 1000.0
          | }
          | times = t(times)
        """.stripMargin

        val nrows = Array[Int](1000,10000,100000,1000000)
        for (r <- nrows) {
            val t = random_matrix(r, 1, spark)
            val u = random_matrix(1, r, spark)
            val v = random_matrix(r, 1, spark)
            
            val meta_t = new MatrixMetadata(DF_VECTOR, r, 1, r)
            val meta_u = new MatrixMetadata(DF_VECTOR, 1, r, r)
            val meta_v = new MatrixMetadata(DF_VECTOR, r, 1, r)

            val script = dml(dmlText).in(Seq(("t", t, meta_t),
                                             ("u", u, meta_u),
                                             ("v", v, meta_v))).out("times")
            val res = ml.execute(script)
            val results = res.getTuple[Matrix]("times")._1.to2DDoubleArray
            File(path).appendAll(s"${r}" + "," + results(0).mkString(",") + '\n')
        }
    }

    def random_matrix(N: Int, K: Int, spark: SparkSession) : DataFrame = {
        val schema = StructType(Seq(StructField("_c0", VectorType)))
        val df = spark.createDataFrame(
            normalVectorRDD(spark.sparkContext, N.toLong, K).
                map(x => Row(x.asML)), schema)
        df.persist(MEMORY_AND_DISK_SER)
        df.count
        return df
    }
}
