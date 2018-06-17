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

        ml.setExplain(true)
        ml.setExplainLevel(MLContext.ExplainLevel.RUNTIME)

        val argMap = Map[String,String](
                args.map(_.split("=")).map({
                    case Array(x,y) => (x -> y)
                }):_*
            )

        var stub = "/tests/SimpleMatrixOps (Single Node Disk)/"
        val mattype = argMap("mattype")
        val opType = argMap("opType")
        val Mpath = argMap("Mpath")
        val Npath = argMap("Npath")
        val wPath = argMap("wPath")
        val savestub = argMap("savestub")
        val nodes = argMap("nodes")
        val outdir = argMap("outdir")
        val sr = argMap("sr")

        println("Evaluating: " + opType)

        val opString = opType match {
            case "TRANS" => "t(M)"
            case "NORM"  => "sqrt(sum(M^2))"
            case "GMM"   => "M %*% N"
            case "MVM"   => "M %*% w"
            case "TSM"   => "t(M) %*% M"
            case "ADD"   => "M + N"
            case _       => println("Invalid operation")
        }

        val allocOp = opType match {
            case "GMM" => s"N = read('${Npath}')"
            case "MVM" => s"w = rand(rows=ncol(M), cols=1)"
            case "ADD" => s"N = read('${Npath}')"
            case _     => ""
        }

        val extraPrintOp = opType match {
            case "GMM" => s"print(sum(N))"
            case "MVM" => s"print(sum(w))"
            case "ADD" => s"print(sum(N))"
            case _     => ""
        }

        val printOp = opType match {
            case "NORM" => "print(R)"
            case _      => "tmp = utils::printRandElements(R, 10)"
        }

        stub = s"/tests/SimpleMatrixOps (Distributed Sparse)/output/${outdir}/"
        val base = s"systemml_${mattype}_${opType}${nodes}.txt"
        val path = root + stub + base

        if (!Files.exists(Paths.get(path))) {
          File(path).writeAll("nodes,sr,time1,time2,time3,time4,time5\n")
        }

        val libDir = root + "/lib/dml"
        val dmlText =
        s"""
          | setwd('$libDir')
          | source('utils.dml') as utils
          |
          | M = read(Mpath)
          | K = sum( M )
          | print(K)
          | $allocOp
          | $extraPrintOp
          | times = matrix(0.0, rows = 5, cols = 1)
          | for (ix in 1:5) {
          |   if (K > 0.0) {
          |       start = utils::time(1)
          |   }
          |   R = $opString
          |   if (K > 0.0) {
          |       $printOp
          |   }
          |   if (K > 0.0) {
          |       stop = utils::time(1)
          |   }
          |   times[ix,1] = (stop - start) / 1000
          | }
          | times = t(times)
        """.stripMargin
        println("Executing DML: ")
        println(dmlText)

        val script = dml(dmlText).
                     in("Mpath", Mpath).
                     out("times")
        val res = ml.execute(script)
        val results = res.getTuple[Matrix]("times")._1.to2DDoubleArray
        File(path).appendAll(
            nodes + "," + sr + "," + results(0).mkString(",") + '\n')
    }
}
