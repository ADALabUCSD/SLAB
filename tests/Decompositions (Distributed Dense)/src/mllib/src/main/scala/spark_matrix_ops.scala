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
import org.apache.spark.storage.StorageLevel._
import org.apache.spark.mllib.linalg._
import org.apache.spark.mllib.linalg
import org.apache.spark.mllib.random.RandomRDDs._
import org.apache.spark.mllib.linalg.distributed._
import org.apache.spark.sql._
import scala.tools.nsc.io._
import scala.io.Source
import java.util.Random
import scala.collection.immutable._

object SparkDecompositions {
    def main(args: Array[String]) {
        val conf = new SparkConf().setAppName("MLLibDecompositions")
        val sc = new SparkContext(conf)

        val root = sys.env("BENCHMARK_PROJECT_ROOT")

        val argMap = Map[String,String](
                args.map(_.split("=")).map({
                    case Array(x,y) => (x -> y)
                }):_*
            )

        val mattype = argMap("mattype")
        val opType = argMap("opType")
        val mPath = argMap("Mpath")
        val nPath = argMap("Npath")
        val passPath = argMap("passPath")
        val nodes = argMap("nodes")
        val outdir = argMap("outdir")

        val stub = s"/tests/Decompositions (Distributed Dense)/output/${outdir}/"
        val base = s"mllib_${mattype}_${opType}${nodes}.txt"
        val path = root + stub + base

        if (!Files.exists(Paths.get(path))) {
          File(path).writeAll("nodes,rows,cols,time1,time2,time3,time4,time5\n")
        }

        val results : Array[Double] = opType match {
            case "SVD" => doRowMatrixOp(mPath, "SVD", sc)
            case _       => Array[Double](0.0, 0.0, 0.0, 0.0, 0.0)
        }

        File(path).appendAll(
            nodes + "," + results.mkString(",") + '\n')
    }

    def doRowMatrixOp(Mpath: String,
                      opType: String,
                      sc: SparkContext) : Array[Double] = {

        val M = readMM(Mpath, sc)
        val rows = M.numRows()
        val cols = M.numCols()
        println( M.rows.count )

        val times = Array[Double](0,0,0,0,0)
        for (ix <- 0 to 4) {
            val start = System.nanoTime()
            
            if (opType == "SVD") {
                val res = M.computeSVD(10, true)
                println(res.U.rows.count)
                println(res.V.toArray.length)
            }
            val stop = System.nanoTime()
            times(ix) = (stop - start)/1e9
        }

        val results = Array[Double](rows, cols) ++ times
        return results
    }

    def readMM(path: String, sc: SparkContext) : IndexedRowMatrix = {
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
    
}
