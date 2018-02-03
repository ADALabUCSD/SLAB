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

import sys.process._
import java.io.File
import java.io.PrintWriter
import org.apache.spark.sql.Row
import org.apache.spark.sql.DataFrame
import org.apache.spark.SparkContext
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.PipelineStage
import org.apache.spark.storage.StorageLevel._
import org.apache.spark.sql.types._
import org.apache.spark.ml.feature._
import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.functions._
import org.apache.spark.mllib.linalg.distributed._
import org.apache.spark.ml.linalg._
import org.apache.spark.mllib.{linalg => alg}
import org.apache.spark.mllib.util.MLUtils
import org.apache.hadoop.fs.{FileSystem, Path}

object SparkPreclean {
    def main(args: Array[String]) {
        val path_stub = args(0)
        val sparse_flag = (args.size > 1)
        val spark = SparkSession.builder.getOrCreate()
        val hdfs = FileSystem.get(spark.sparkContext.hadoopConfiguration)
        val dense_path = s"/scratch/adclick_clean${path_stub}_dense.parquet"

        process_dense(dense_path, path_stub, hdfs, spark)
        if (sparse_flag) {
            val sparse_path = dense_path.replace("_dense", "_sparse")
            process_sparse(sparse_path, path_stub, hdfs, spark)
        }
    }

    def process_dense(path: String,
                      stub: String,
                      hdfs: FileSystem,
                      spark: SparkSession) {
        val processed = spark.read.parquet(path).repartition(500)
        processed.persist(MEMORY_AND_DISK_SER)
        val nrow = processed.count
        val ncol = processed.take(1)(0).getAs[Vector](1).size
        val base = sys.env("BENCHMARK_PROJECT_ROOT")
        val loc = "/data/Criteo (Build Derived Tables)/output"
        val outfile = s"adclick_clean${stub}_dense.csv"
        val meta_path = s"${base}/${loc}/${outfile}.mtd"
        write_meta(meta_path, nrow, ncol, nrow*ncol, "csv", ",")

        drop_if_exists(s"/scratch/${outfile}", hdfs)
        processed.rdd.map(
            row => (row.getAs[Float](0),
                    row.getAs[Vector](1).toArray.mkString(","))
        ).map(
            row => s"${row._1},${row._2}"
        ).saveAsTextFile(s"/scratch/${outfile}")

        processed.select("y").
            write.mode("overwrite").
            csv(s"/scratch/adclick_clean${stub}_y.csv")
    }

    def drop_if_exists(path: String, hdfs: FileSystem) {
        if (hdfs.exists(new Path(path)))
            hdfs.delete(new Path(path))
    }

    def write_meta(path: String, rows: Long,
                   cols: Long, nnz: Long,
                   fmt: String, sep: String) {
        val writer = new PrintWriter(new File(path))
        val metadata = s"""|{
            |    "data_type": "matrix",
            |    "value_type": "double",
            |    "rows": ${rows},
            |    "cols": ${cols},
            |    "nnz": ${nnz},
            |    "format": "${fmt}",
            |    "header": false,
            |    "sep": ","
        |}""".stripMargin
        writer.write(metadata)
        writer.close
    }

    def process_sparse(path: String,
                       stub: String,
                       hdfs: FileSystem,
                       spark: SparkSession) {
        val processed = spark.read.parquet(path)
        val base = processed.select("features").rdd.
            zipWithIndex().map(row =>
                new IndexedRow(row._2, alg.SparseVector.fromML(
                    row._1.getAs[SparseVector](0))))

        val M = new IndexedRowMatrix(base).toCoordinateMatrix()
        M.entries.persist(MEMORY_AND_DISK_SER)

        val nrow = M.numRows()
        val ncol = M.numCols()
        val nnz = M.entries.count
        val path_base = sys.env("BENCHMARK_PROJECT_ROOT")
        val loc = "/data/Criteo (Build Derived Tables)/output"
        val outfile = s"adclick_clean${stub}_sparse.csv"
        val meta_path = s"${path_base}/${loc}/${outfile}.mtd"
        write_meta(meta_path, nrow, ncol, nnz, "text", " ")

        drop_if_exists(s"/scratch/adclick_clean${stub}_sparse.mtx", hdfs)
        M.entries.map(
            row => s"${row.i} ${row.j} ${row.value}"
        ).saveAsTextFile(s"/scratch/adclick_clean${stub}_sparse.mtx")

        val scaled_raw = spark.read.
                parquet(s"/scratch/adclick_clean${stub}_raw.parquet")
        drop_if_exists(s"/scratch/adclick_clean${stub}_raw.csv", hdfs)
        scaled_raw.rdd.map(
            row => (row.getAs[Float](0),
                    row.getAs[Vector](1).toArray.mkString(","),
                    (2 to row.size-1).map(
                        i => row.get(i)).toArray.mkString(","))
        ).map(
            row => s"${row._1},${row._2},${row._3}"
        ).saveAsTextFile(s"/scratch/adclick_clean${stub}_raw.csv")
    }

}
