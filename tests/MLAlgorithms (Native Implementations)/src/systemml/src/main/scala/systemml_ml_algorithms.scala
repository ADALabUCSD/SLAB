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
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.SQLContext
import org.apache.sysml.api.mlcontext._
import org.apache.sysml.api.mlcontext.MatrixFormat._
import org.apache.sysml.api.mlcontext.ScriptFactory._
import org.apache.sysml.runtime.instructions.spark.utils._
import org.apache.sysml.runtime.matrix.MatrixCharacteristics
import org.apache.spark.storage.StorageLevel._
import scala.tools.nsc.io._
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.sql.DataFrame
import org.apache.spark.mllib.linalg.distributed._

object SystemMLMLAlgorithms {
    def main(args: Array[String]) {
        val spark = SparkSession.builder.getOrCreate()
        val sc = spark.sparkContext
        val ml = new MLContext(sc)

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
        val execSpark = argMap.get("execSpark")

        val execution_type = execSpark match {
            case Some(arg) => MLContext.ExecutionType.SPARK
            case None => MLContext.ExecutionType.DRIVER_AND_SPARK
        }

        val exec_type_stub = execSpark match {
            case Some(arg) => "spark"
            case None      => "spark_and_driver"
        }

        println(execution_type)
        ml.setExecutionType(execution_type)
        ml.setExplain(true)
        ml.setExplainLevel(MLContext.ExplainLevel.RUNTIME)

        val sparse_stub = if (isSparse) "_sparse" else "_dense"
        var stub = "/tests/MLAlgorithms (Native Implementations)/output/"
        val base = s"systemml_${opType}${nodes}${sparse_stub}_${exec_type_stub}.txt"
        val path = root + stub + base

        if (!Files.exists(Paths.get(path))) {
          File(path).writeAll("nodes,time1,time2,time3,time4,time5\n")
        }

        val script_stub = "tests/MLAlgorithms (Native Implementations)/src"
        val script_path = s"${root}/${script_stub}/systemml/src/main/dml"
        val input_df = spark.read.parquet(inputPath)
        val x = input_df.select(featureNames).repartition(1000)
        val y = input_df.select("y")
        
        println( x.count )
        var times = Array.ofDim[Double](5,1)
        if (opType == "logit") {
            val script = dmlFromFile(s"${script_path}/MultiLogitReg.dml").
                 in(Map("X"     -> x,
                        "Y_vec" -> y,
                        "$moi"  -> 3,
                        "$mii"  -> 0)).
                 out("times")
             val res = ml.execute(script)
             times = res.getTuple[Matrix]("times")._1.to2DDoubleArray
         } else if (opType == "reg") {
             val script = dmlFromFile(s"${script_path}/LinearRegCG.dml").
                 in("X", x).
                 in("$maxi", 3).
                 in("y", y).
                 out("times")
             val res = ml.execute(script)
             times = res.getTuple[Matrix]("times")._1.to2DDoubleArray
         } else if (opType == "pca") {
             val script = dmlFromFile(s"${script_path}/PCA.dml").
                 in("A", x).
                 in("K", 5).
                 out("times")
             val res = ml.execute(script)
             times = res.getTuple[Matrix]("times")._1.to2DDoubleArray
         } else {
             throw new Exception("Invalid operator")
        }

        File(path).appendAll(
            nodes + "," + times(0).map(x => x/1000.0).mkString(",") + '\n')
    }
}
