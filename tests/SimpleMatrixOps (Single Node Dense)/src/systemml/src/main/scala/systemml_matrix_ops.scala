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
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.spark.sql._
import scala.tools.nsc.io._
import scala.collection.immutable._

object SystemMLMatrixOps extends App {

    override def main(args: Array[String]) {
        val conf = new SparkConf().setAppName("SystemMLMatrixOps")
        val sc = new SparkContext(conf)
        val ml = new MLContext(sc)
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

        ml.setExplain(true)
        ml.setExplainLevel(MLContext.ExplainLevel.RUNTIME)
        ml.setExecutionType(MLContext.ExecutionType.DRIVER)
        
        val opString = opType match {
            case "TRANS" => "t(M)"
            case "NORM"  => "sqrt(sum(M^2))"
            case "GMM"   => "M %*% N"
            case "MVM"   => "M %*% w"
            case "TSM"   => "t(M) %*% M"
            case "ADD"   => "M + X"
            case _       => throw new Exception("Invalid Operator")
        }

        println("OP STRING => " + opString)

        val allocOp = opType match {
            case "GMM"   => "N = utils::allocMatrix(ncol, nrow)"
            case "MVM"   => "w = utils::allocMatrix(ncol, 1)"
            case "ADD"   => "X = utils::allocMatrix(nrow, ncol)"
            case _       => ""
        }

        val printOp = opType match {
            case "NORM" => "print(R)"
            case _      => "tmp = utils::printRandElements(R,10)"
        }

        val stub = "/tests/SimpleMatrixOps (Single Node Dense)/output/"
        val base = nproc match {
            case Some(np) => s"systemml_cpu_${opType}_scale.txt"
            case None => s"systemml_${mattype}_${opType}.txt"
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

            val libDir = root + "/lib/dml"
            val dmlText = 
            s"""
              | setwd('$libDir')
              | source('utils.dml') as utils
              |
              | M = utils::allocMatrix(nrow, ncol)
              | $allocOp
              | k = sum( M )
              | times = matrix(0.0, rows = 5, cols = 1)
              | for (ix in 1:5) {
              |   if (k != 0) {
              |     start = utils::time(1)
              |   }
              |   R = $opString
              |   if (k != 0) {
              |     $printOp
              |   }
              |   if (k != 0) {
              |     stop = utils::time(1)
              |   }
              |   times[ix,1] = (stop - start) / 1000
              | }
              | times = t( times )
            """.stripMargin
            println("Executing DML: ")
            println(dmlText)

            val script = dml(dmlText).
                         in("nrow", nrow).
                         in("ncol", ncol).
                         out("times")
            val res = ml.execute(script)
            val results = res.getTuple[Matrix]("times")._1.to2DDoubleArray
            val pref = nproc match {
                case Some(np) => s"${np}"
                case None     => s"${nr}"
            }
            File(path).appendAll(
              s"${pref}" + "," + results(0).mkString(",") + '\n')
        }
    }
}
