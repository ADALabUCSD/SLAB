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

object SystemMLDecompositions extends App {

    override def main(args: Array[String]) {
        val conf = new SparkConf().setAppName("SystemMLDecompositions")
        val sc = new SparkContext(conf)
        val ml = new MLContext(sc)
        ml.setExplain(true)
        ml.setExplainLevel(MLContext.ExplainLevel.RUNTIME)
        val root = sys.env("BENCHMARK_PROJECT_ROOT")

        val argMap = Map[String,String](
                args.map(_.split("=")).map({
                    case Array(x,y) => (x -> y)
                }):_*
            )

        var stub = "/tests/Decompositions (Distributed Dense)/"
        val mattype = argMap("mattype")
        val opType = argMap("opType")
        val Mpath = argMap("Mpath")
        val Npath = argMap("Npath")
        val wPath = argMap("wPath")
        val nodes = argMap("nodes")
        val outdir = argMap("outdir")

        println("Evaluating: " + opType)

        val opString = opType match {
            case "SVD" => "[U,D,V] = svd( M )"
            case _       => println("Invalid operation")
        }

        val extraPrintOp = opType match {
            case "GMM" => s"print(sum(N))"
            case "LMM" => s"print(sum(N))"
            case "MVM" => s"print(sum(w))"
            case _     => ""
        }

        val printOp = opType match {
            case "SVD" => """
                | print(as.scalar(U[1,1]))
                | print(as.scalar(D[1,1]))
                | print(as.scalar(V[1,1]))
              """.stripMargin
            case _      => "print(as.scalar(R[1,1]))"
        }

        stub = s"/tests/SimpleMatrixOps (Distributed Disk)/output/${outdir}/"
        val base = s"systemml_${mattype}_${opType}${nodes}.txt"
        val path = root + stub + base

        if (!File(path).exists)
            File(path).writeAll(
              "nodes,rows,cols,time1,time2,time3,time4,time5\n")
        val libDir = root + "/lib/dml"
        val dmlText =
        s"""
          | setwd('${libDir}')
          | time = externalFunction(Integer i) return (Double B)
          |      implemented in (classname="org.apache.sysml.udf.lib.TimeWrapper", exectype="mem");
          |
          | t = time(1)
          | print("Time " + t)
          | M = read(Mpath)
          | K = sum(M)
          | print(K)
          | times = matrix(0.0, rows = 5, cols = 1)
          | for (ix in 1:5) {
          |   if (K > 0) {
          |       start = time(1)
          |   }
          |
          |   ${opString}
          |   if (K > 0) {
          |       ${printOp}
          |   }
          |   if (K > 0) {
          |       stop = time(1)
          |   }
          |   times[ix,1] = (stop - start) / 1000.0
          | }
          |
          | results = matrix(0.0, 1, 7)
          | results[1,1] = nrow(M)
          | results[1,2] = ncol(M)
          | results[1,3:7] = t(times)
        """.stripMargin
        println("Executing DML: ")
        println(dmlText)

        val script = dml(dmlText).
                     in("Mpath", Mpath).
                     out("results")
        val res = ml.execute(script)
        val results = res.getTuple[Matrix]("results")._1.to2DDoubleArray
        File(path).appendAll(
            nodes + "," + results(0).mkString(",") + '\n')
    }
}
