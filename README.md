# SLAB

The Scalable Linear Algebra Benchmark

# NOTE: This repo is currently being populated. Some referenced contents may not be present and will be uploaded within the next couple days.

## Instructions

### Descrition of Directory Structure

All directories follow an identical configuration. Each directory consists of two folders: `src` and `output`. `src` contains all code for that directory and `output` contains output (mostly logs) produced by code in `src`. Each `src` folder contains a script `make.py` which will run all code for that directory. `make.py` often requires command line arguments which can be used to control which tests are run, what datasets are used, etc... To see the command line arguments needed by a particular `make.py` simply run `python make.py -h` and a help message will be printed explain what options are required/present. By default, `make.py` does not capture the `stdout` and `stderr` of child processes. It is often useful to log `stdout` and `stderr`. To do so, simply pipe the output of `make.py` to a log file: `unbuffer python make.py <options> 2>&1 | tee make.out`.

### Configure Cluster Node(s)

#### Using the AWS AMI

We have released an AWS AMI based on Ubuntu 16.04 with all dependencies pre-installed. The AMI is available here. You can use this AMI to run tests in the singlenode setting or create your own cluster. A couple relevant pieces of information:

1. You will need to configure Spark and Haddop to run on your cluster. The AMI has them set up to run in single node mode. There are many online tutorials explaining how to do this.
2. We have built Greenplum using six segments. If using in the single node setting you will likely want to increase this to 16-24 (we use 24) depending on the number of cores on your machine. To do so you can use the `gpexpand` command line utility. This utility can be used to expand Greenplum to new nodes as well. Consult the documentation available [here](http://gpdb.docs.pivotal.io/520/utility_guide/admin_utilities/gpexpand.html).
3. If you wish to see the specific configuration settings we used for Spark and Hadoop (hdfs), the configuration directories we used are available as zip files in the `/config/` subfolder of this repository.

#### Building From Source

If you don't want to use the provided AMI, you can create a fresh cluster using the compilation scripts provided in `/config`. First run `setup-nodes.sh`. This script will install software and perform basic system configuration. The script takes the following parameters in the form of environment variables

1. `COMPILE_OPENBLAS=1` - Set this environment variable to compile OpenBLAS from source. If this variable is unset then OpenBLAS will be installed from `apt-get`
2. `INSTALL_TENSORFLOW=1` - Set this environment variable to install TensorFlow. If this variable is unset then TensorFlow will not be installed.
3. `COMPILE_SPARK=1` - Set this environment variable to download Spark and compile from Source. Spark will be compiled to support linking OpenBLAS. If this variable unset then we assume you will download and install your own version of Spark.

After running `setup-nodes.sh` run `install-gpdb.sh` to build the Greenplum database. The script will create a user `ubuntu` and a corresponding databas. Finally run `install-madlib.sh` to download and install MADLib. Greenplum will be installed with six segments. You will need to `gpexpand` Greenplum if you wish to use more segments or extend Greenplum to more nodes. 

All tests are designed to run on Ubuntu 16.04. The disk image used to create cluster nodes is available [here](http://cloud-images.ubuntu.com/xenial/current/xenial-server-cloudimg-amd64-disk1.img). An automated script for creating a cluster on top of OpenStack with all tools installed is available [here](https://github.com/thomas9t/spark-openstack). To prepare a single node cluster, simply run the script `/config/setup-nodes.sh`. The script will download and install dependencies.

For the intrepid used who wishes to go it alone setting up the testing environment we assume the following (see the tech report for all versions of software used):

1. Spark and Hadoop are installed and their respective `/bin` directories are on the system `PATH`.
2. Greenplum is installed and accepting connections on port `5432` (the default). We used Greenplum for all tests although they -should- also work with vanilla Postgres.
3. MADLib has been installed and loaded into the RDBMS.
3. **Scripts assume the existance of a user and database `ubuntu` in the RDBMS.**
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

Running tests is very straightforward! Just `cd` to the appropriate directory and run `make.py`. Each system will do its thing and log runtimes to the test directory's `/output` subfolder. It should be clear from filenames which log corresponds to each test. As before each `make.py` script takes command line arguments which can be used to adjust the parameters of each test. Use `python make.py -h` to see the specific options supported by each script.  

## Details

Most support and utility code lives in the `/lib` subfolder of this repository. The programs likely to be of interest are:

1. `gen_data.py` - Contains routines used to generate synthetic data.
2. `sql_cxn.py` - Contains a class which makes interacting with Greenplum over Python more pleasant.
3. `make_utils.py` - Contains the code used by `make.py` files.
