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
import org.apache.spark.sql.functions._
import org.apache.spark.mllib.linalg.distributed._
import org.apache.spark.ml.linalg._
import org.apache.spark.mllib.util.MLUtils
import org.apache.hadoop.fs.{FileSystem, Path}

object SparkPreclean {
    def main(args: Array[String]) {
        val path = args(0)
        val sparse_flag = (args.size > 1)
        val spark = SparkSession.builder.getOrCreate()

        val raw_df = preclean(spark, path, sparse_flag)
        encode_features(raw_df, sparse_flag)
    }

    def preclean(spark: SparkSession,
                 path: String,
                 sparse_flag: Boolean) : DataFrame = {

        val path_stub = sys.env("SAVE_STUB")
        val continuous_cols = (0 to 13).map(
            x => StructField(s"V${x}", IntegerType))
        val categorical_cols = (14 to 39).map(
            x => StructField(s"V${x}", StringType))
        val schema = StructType(continuous_cols ++ categorical_cols)

        var cols_to_drop = ((11 to 14) ++ (20 to 39)).map(x => s"V${x}")
        if (!sparse_flag)
            cols_to_drop = (14 to 39).map(x => s"V${x}")

        // NOTE: Probably not a good idea for real analysis!!!
        // but saves us some hassle when downloading data...
        val mv_map = ((1 to 10).map(x => s"V${x}" -> 1) ++
                      (15 to 19).map(x => s"V${x}" -> "NA")).toMap

        val raw_df = spark.read.option("sep", "\t").
                                schema(schema).
                                csv(path).
                                drop(cols_to_drop: _*).
                                na.fill(mv_map).withColumnRenamed(
                                    "V0", "y"
                                )

        raw_df.persist(MEMORY_AND_DISK_SER)
        return raw_df
    }

    def encode_features(df: DataFrame,
                        sparse_flag: Boolean) {

        // They're called dummy variables not "one hot encodings"
        val path_stub = sys.env("SAVE_STUB")
        val cat_col_names = (15 to 19).map(x => s"V${x}")

        val indexers = cat_col_names.map(
                col => new StringIndexer().
                               setInputCol(col).
                               setOutputCol(s"${col}_ix")
            )

        val encoders = cat_col_names.map(
                col => new OneHotEncoder().
                        setInputCol(s"${col}_ix").
                        setOutputCol(s"${col}_vect")
            )

        val assemble_base = (1 to 10).map(x => s"V${x}").toArray
        val assemble_cat = (15 to 19).map(x => s"V${x}_vect").toArray

        val dense_assembler = new VectorAssembler().
                setInputCols(assemble_base).
                setOutputCol("dense_features")
        val sparse_assembler = new VectorAssembler().
                setInputCols(assemble_cat).
                setOutputCol("cat_features")
        val scalar = new StandardScaler().
            setInputCol("dense_features").
            setOutputCol("dense_features_scaled")
        val assembler = new VectorAssembler().
            setInputCols(Array("dense_features_scaled",
                               "cat_features")).
            setOutputCol("features")

        val pipeline_stages = sparse_flag match {
            case true => (indexers ++ encoders).toArray ++
                         Array(dense_assembler,
                               sparse_assembler,
                               scalar,
                               assembler)
            case _    => Array(dense_assembler, scalar)
        }

        val transform_pipeline = new Pipeline().setStages(pipeline_stages)
        val model: PipelineModel = transform_pipeline.fit(df)
        val transformed: DataFrame = model.transform(df)

        //always write dense features
        transformed.select("y","dense_features_scaled").
            write.mode("overwrite").
            parquet(s"/scratch/adclick_clean${path_stub}_dense.parquet")

        if (sparse_flag) {
            val varnames = Array("y", "dense_features_scaled") ++ cat_col_names
            transformed.select(varnames.head, varnames.tail: _*).
                write.mode("overwrite").
                parquet(s"/scratch/adclick_clean${path_stub}_raw.parquet")

            transformed.select("y","features").
                write.mode("overwrite").
                parquet(s"/scratch/adclick_clean${path_stub}_sparse.parquet")
        }
    }
}
