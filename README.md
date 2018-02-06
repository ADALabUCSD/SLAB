# SLAB

The Scalable Linear Algebra Benchmark

### If you find a bug please open an issue!

## Instructions

### Description of Directory Structure

All directories follow an identical configuration. Each directory consists of two folders: `src` and `output`. `src` contains all code for that directory and `output` contains output (mostly logs) produced by code in `src`. Each `src` folder contains a script `make.py` which will run all code for that directory. `make.py` often requires command line arguments which can be used to control which tests are run, what datasets are used, etc... To see the command line arguments needed by a particular `make.py` simply run `python make.py -h` and a help message will be printed explain what options are required/present. By default, `make.py` does not capture the `stdout` and `stderr` of child processes. It is often useful to log `stdout` and `stderr`. To do so, simply pipe the output of `make.py` to a log file: `unbuffer python make.py <options> 2>&1 | tee make.out`.

### (1) Configure Cluster Node(s)

The repo linked [here](https://github.com/thomas9t/spark-openstack.git) can be used to automatically set up a cluster using OpenStack. Unless you have a good reason not to, you are advised to use this automated script.

You can create a fresh cluster using the compilation scripts provided in `/config`. Be aware that the provided scripts will install some packages from 3rd party Ubuntu repos. These are all legit (e.g. Rstudio and SBT) but if you're suspicious of such things you may want to comment out these lines and install on your own. First run `setup-nodes.sh`. This script will install software and perform basic system configuration. The script takes the following parameters in the form of environment variables

1. `COMPILE_OPENBLAS=1` - Set this environment variable to compile OpenBLAS from source. If this variable is unset then OpenBLAS will be installed from `apt-get`
2. `INSTALL_TENSORFLOW=1` - Set this environment variable to install TensorFlow. If this variable is unset then TensorFlow will not be installed.
3. `COMPILE_SPARK=1` - Set this environment variable to download Spark and compile from Source. Spark will be compiled to support linking OpenBLAS. If this variable unset then we assume you will download and install your own version of Spark.

After running `setup-nodes.sh` run `install-gpdb.sh` to build the Greenplum database. Before doing so, ensure that you have enabled passwordless SSH between the nodes in you cluster (including `localhost`!) and create a database and user "ubuntu" in Greenplum. The create an "ubuntu" user, run the following lines:

    source /usr/local/gpdb/greenplum_path.sh
    createdb ubuntu

To enable passwordless SSH you can use the following lines. We do not do this by default because some cluster managers already configure passwordless SSH and we don't want to overwrite whatever they're doing.

    printf '\n\n\n' | ssh-keygen -t rsa
    cat ~/.ssh/id_rsa.pub >> ~/.ssh/authorized_keys
    chmod 600 ~/.ssh/authorized_keys
    sudo service ssh restart

The script will create a user `ubuntu` and a corresponding databas. Finally run `install-madlib.sh` to download and install MADLib. Greenplum will be installed with six segments. You will need to `gpexpand` Greenplum if you wish to use more segments or extend Greenplum to more nodes. You should also take care that system level configuration parameters are set as described in the technical report.

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
8. System Parameters have been configured as described in the technical report
9. Our configuration files for Spark and Hadoop **assume the hostname is mycluster-master** and workers are named `mycluster-slave-${i}` you may need to modify Spark's `slaves`, `spark-defaults.conf` and `spark-env.sh` to suit the hostname of your cluster master. You will additionally need to modify Hadoop's `masters`, `slaves` and `core-site.xml`.

### (2) Additional System Configuration

Some additional manual configuration steps are necessary to replicate our testing environment.

1. Install `pbdDMAT` - Open R (type `R` on the command line) and run `install.packages('pbmDMAT')`
1. You will need to configure Spark and Hadoop to run on your cluster. Our configuration directories are available under the `/config` folder of this repo. You can tweak them to suit the resources available on your system and your hostnames/IPs etc... Our configuration files for Spark and Hadoop **assume the hostname is mycluster-master** and workers are named `mycluster-slave-${i}` you may need to modify Spark's `slaves`, `spark-defaults.conf` and `spark-env.sh` to suit the hostname of your cluster master. You will additionally need to modify Hadoop's `masters`, `slaves` and `core-site.xml`. 
2. We have built Greenplum using six segments. If using in the single node setting you will likely want to increase this to 16-24 (we use 24) depending on the number of cores on your machine. To do so you can use the `gpexpand` command line utility. This utility can be used to expand Greenplum to new nodes as well. Consult the documentation available [here](http://gpdb.docs.pivotal.io/520/utility_guide/admin_utilities/gpexpand.html).
3. If you wish to see the specific configuration settings we used for Spark and Hadoop (hdfs), the configuration directories we used are available as zip files in the `/config/` subfolder of this repository.
4. You may need to modify some system level parameters in `/etc/security/limits.conf` and `/etc/sysctl.conf` to ensure your system is properly configured. We have provided our versions of these files in the `/config` directory of this repo. You can modify them based on the resources available to you.
5. If you wish to run tests which compare results for varying numbers of Greenplum segments you will need to manually create new Greenplum database instances and import data. Our test scripts assume that the Greenplum master directory is named according to: `/gpsegs/gpdb-<num_segments>/master/gpseg-1` and that instances are bound to ports under the following mapping:

    {'1': 5481, '2': 6431, '4': 6431, 
     '8': 6431, '16': 6431, '24': 5432}

### (3) NFS Share Relevant Directories

pbdR expects that all code files are available to each local R process at run time. Ensure that the directory containing the `SLAB` repo has been NFS shared to all nodes in the cluster along with the directory containing the pbdR library. For example, if the path to `my_test.R` on the master node is `/home/me/SLAB/tests/mytest.R`, then NFS share `SLAB` to `/home/me` on worker nodes in the cluster.

### (4) Generate Test Data

Tests expect that data has been pre-generated and loaded into HDFS/Greenplum. Fortunately, we have provided scripts which automate this process. There are four directories which manage building data. As described above, each contains a `make.py` which will build that directory. The make file will handle generating data to your specifications as well as loading it into Greenplum and HDFS. Note that data generators use SQL internally so it's important to have one installed and configured as described above even if you don't want to run any of the MADLib tests. 

 1. `/data/Criteo (Process Raw Data)` - This directory will download the Criteo dataset and run preprocessing scripts to produce a `parquet` file containing cleaned output. Note that you must accept the Critro data useage agreement before downloading data.
 2. `/data/Criteo (Build Derived Tables)` - This directory consumes the output of (1) and produces derived tables which are used by tests. Specifically, this directory will split the raw data into a "dense" and "sparse" version, impute missing values with zeros, and transform categorical variables to dummy vector (one-hot) representation.
 3. `/data/SimpleMatrixOps (Disk Data)` - This directory generates dense synthetic datasets to use for experiments. The make file takes an option which can be used to stiplate the approximate size of data generated. Note that sizes are calculated somewhat unrealisitically assuming that each double uses exactly 8 bytes. The size of files generated is also larger on disk, so generating a matrix with an approximate size of 16GB will result in a matrix on disk which is about 30GB. 
 4. `/data/SimpleMatrixOps (Sparse Data)` - This directory generates sparse synthetic data in `i,j,v` format. The make file for this directory takes two options which can set the *logical* size of matrices generated (e.g. 100GB if fully materialized with zeros on disk) and the fraction of values which are nonzero. Again, run the make file with the `-h` flag to see the appropriate syntax.

### (5) Run Tests

Running tests is very straightforward! Just `cd` to the appropriate directory and run `make.py`. Each system will do its thing and log runtimes to the test directory's `/output` subfolder. It should be clear from filenames which log corresponds to each test. As before each `make.py` script takes command line arguments which can be used to adjust the parameters of each test. Use `python make.py -h` to see the specific options supported by each script. The following points detail the meaning of common command line arguments. This is not an exhaustive list. Be sure to consult the help output for each test:

1. `systems` - A space delimited list of systems to compare. In the distributed setting may be any of `"MADLIB SYSTEMML MLLIB R"`. In the single node setting may be any of `"MADLIB SYSTEMML MLLIB R TF NUMPY"`.
2. `operators` - For `SimpleMatrixOps` tests a space delimited list of operators to evaluate. May be any of `TRANS MVM NORM TSM GMM ADD`.
3. `msize` - The size of matrix on which to perform tests. **Must conform with the data geenrated in (4)**. That is, passing `--msize 4` means that you should generate data with `--msize 4`. 
4. `test-type` - Which type of test to run. Several directories can run multiple tests. Consult the output of `python make.py -h` for the options available for each directory.
3. `algorithms` - For `MLAlgorithms` tests a space delimited list of algorithms to run. For LA based tests may be any of `robust gnmf reg logit`. For native implementation tests may be any of `reg logit pca`.
4. `nodes` - For distributed tests the number of nodes on which the test is being run. E.g. if you're running the test on a two node cluster this would be 2.

## Examples:

The following example details how to build data and run tests to compare system performance for dense primitive matrix operators and dense LA based ML algorithms on matrices which will be approximately 8 and 30 GB on disk as CSV files.

    cd "${BENCHMARK_PROJECT_ROOT}/data/SimpleMatrixOps (Disk Data)/src"
    python make.py --msize "4 16"
    
    cd "${BENCHMARK_PROJECT_ROOT}/tests/SimpleMatrixOps (Distributed Disk)/src"
    # Note that --msize must agree with the sizes used above
    unbuffer python make.py --nodes 8 --msize "4 16" --systems "MLLIB R" --operators "NORM GMM" --test-type scale_mat 2>&1 | tee make8.out
    cd ../output
    tail -vn +1 *.txt
    
    cd "${BENCHMARK_PROJECT_ROOT}/tests/MLAlgorithms (Distributed Dense LA)/src"
    unbuffer python make.py --nodes 8 --msize "4 16" --systems "MLLIB R" --algorithms "logit gnmf" --test-type scale 2>&1 | tee make8.out
    cd ../output
    tail -vn +1 *.txt

## Details

Most support and utility code lives in the `/lib` subfolder of this repository. The programs likely to be of interest are:

1. `gen_data.py` - Contains routines used to generate synthetic data.
2. `sql_cxn.py` - Contains a class which makes interacting with Greenplum over Python more pleasant.
3. `make_utils.py` - Contains the code used by `make.py` files.
