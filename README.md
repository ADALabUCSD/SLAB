# SLAB

The Scalable Linear Algebra Benchmark

# NOTE: This repo is currently being populated. Some referenced contents may not be present and will be uploaded within the next couple days.

## Instructions

### Descrition of Directory Structure

All directories follow an identical configuration. Each directory consists of two folders: `src` and `output`. `src` contains all code for that directory and `output` contains output (mostly logs) produced by code in `src`. Each `src` folder contains a script `make.py` which will run all code for that directory. `make.py` often requires command line arguments which can be used to control which tests are run, what datasets are used, etc... To see the command line arguments needed by a particular `make.py` simply run `python make.py -h` and a help message will be printed explain what options are required/present. By default, `make.py` does not capture the `stdout` and `stderr` of child processes. It is often useful to log `stdout` and `stderr`. To do so, simply pipe the output of `make.py` to a log file: `unbuffer python make.py <options> 2>&1 | tee make.out`.

### Configure Cluster Node(s)

All tests are designed to run on Ubuntu 16.04. The disk image used to create cluster nodes is available [here](http://cloud-images.ubuntu.com/xenial/current/xenial-server-cloudimg-amd64-disk1.img). An automated script for creating a cluster on top of OpenStack with all tools installed is available [here](https://github.com/thomas9t/spark-openstack). To prepare a single node cluster, simply run the script `/config/setup-nodes.sh`. The script will download and install dependencies. We recommend creating a VM, or a fresh instance in your favorite cluster manager and simply running the config script rather than trying to install dependencies manually. The `/conf` directories for our Spark and Hadoop clusters are available under the `/configs` folder of this repository. The testing environment assumes the following (see the tech report for versions of software):

1. Spark and Hadoop are installed and their respective `/bin` directories are on the system `PATH`.
2. Greenplum is installed and accepting connections on port `5432` (the default). We used Greenplum for all tests although they -should- also work with vanilla Postgres.
3. MADLib has been installed and loaded into the RDBMS.
3. Scripts assume the existance of a user and database `ubuntu` in the RDBMS.
4. Python has been installed along with Pandas, NumPy, psycopg2, gslab_econ, and TensorFlow. All such packages can be installed through PiP.
5. R has been installed along with the pbdDMAT package (and its dependencies) and OpenMPI
6. OpenBLAS has been installed (either through apt-get or from sources) and can be loaded by R.
7. Scala and SBT are installed 

### NFS Share Relevant Directories

pbdR expects that all code files are available to each local R process at run time. Ensure that the directory containing the `SLAB` repo has been NFS shared to all nodes in the cluster along with the directory containing the pbdR library. For example, if the path to `my_test.R` on the master node is `/home/me/SLAB/tests/mytest.R`, then NFS share `SLAB` to `/home/me` on worker nodes in the cluster.

### Generate Test Data

Tests expect that data has been pre-generated and loaded into HDFS/Greenplum. Fortunately, we have provided scripts which automate this process. There are four directories which manage building data. As described above, each contains a `make.py` which will build that directory. The make file will handle generating data to your specifications as well as loading it into Greenplum and HDFS. Note that data generators use SQL internally so it's important to have one installed and configured as described above even if you don't want to run any of the MADLib tests. 

 1. `/data/Criteo (Process Raw Data)` - This directory will download the Criteo dataset and run preprocessing scripts to produce a `parquet` file containing cleaned output. Note that you must accept the Critro data useage agreement before downloading data.
 2. `/data/Criteo (Build Derived Tables)` - This directory consumes the output of (1) and produces derived tables which are used by tests. Specifically, this directory will split the raw data into a "dense" and "sparse" version, impute missing values with zeros, and transform categorical variables to dummy vector (one-hot) representation.
 3. `/data/SimpleMatrixOps (Disk Data)` - This directory generates dense synthetic datasets to use for experiments. The make file takes an option which can be used to stiplate the approximate size of data generated. Note that sizes are calculated somewhat unrealisitically assuming that each double uses exactly 8 bytes. The size of files generated is also larger on disk, so generating a matrix with an approximate size of 16GB will result in a matrix on disk which is about 30GB. 
 4. `/data/SimpleMatrixOps (Sparse Data)` - This directory generates sparse synthetic data in `i,j,v` format. The make file for this directory takes two options which can set the *logical* size of matrices generated (e.g. 100GB if fully materialized with zeros on disk) and the fraction of values which are nonzero. Again, run the make file with the `-h` flag to see the appropriate syntax.

### Run Tests

TODO

#### Output File Format

TODO

## Utility Code

TODO
