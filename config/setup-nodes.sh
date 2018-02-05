#!/bin/bash

############################
# install java related stuff
############################

set -e
sudo apt-get -y update
echo debconf shared/accepted-oracle-license-v1-1 select true | \
  sudo debconf-set-selections
echo debconf shared/accepted-oracle-license-v1-1 seen true | \
  sudo debconf-set-selections
yes 'n' | sudo apt-add-repository ppa:webupd8team/java
sudo apt-get update
sudo apt-get -y install oracle-java8-installer

if [ -f ~/SETUP_COMPLETE ]; then
    exit 0
fi

#git config --global user.email "nunya@business.com"
#git config --global user.name "R Daneel Olivaw"

#################
# Install OpenMPI
#################

sudo apt-get -y install openmpi-bin openmpi-doc libopenmpi-dev
echo "export PATH=/usr/lib64/openmpi/bin:$PATH" >> ~/.bashrc
echo "export LD_LIBRARY_PATH=/usr/lib64/openmpi/lib" >> ~/.bashrc
source ~/.bashrc

#############
# Install NFS
#############

sudo apt-get -y install nfs-kernel-server
sudo apt-get -y install nfs-common

###########
# Install R
###########

sudo echo "deb http://cran.rstudio.com/bin/linux/ubuntu xenial/" | sudo tee -a /etc/apt/sources.list
gpg --keyserver keyserver.ubuntu.com --recv-key E084DAB9
gpg -a --export E084DAB9 | sudo apt-key add -
sudo apt-get -y update
sudo apt-get -y install r-base r-base-dev

##################
# Install OpenBLAS
##################

mkdir -p ~/Software
cd ~/Software
if [ "${COMPILE_OPENBLAS}" != "" ]; then
    # see: https://apache.github.io/systemml/native-backend
    git clone https://github.com/xianyi/OpenBLAS.git
    cd OpenBLAS
    git checkout 114fc0bae3a
    make clean
    export OMP_NUM_THREADS=`nproc`
    make USE_OPENMP=1
    sudo make install
    echo "export LD_LIBRARY_PATH=/opt/OpenBLAS/lib:${LD_LIBRARY_PATH}" >> ~/.bashrc

    # A bit of hackiness is needed to get R to use this blas
    if [ -d /usr/lib/libblas ]; then
        sudo mv /usr/lib/libblas /usr/lib/_libblas
        sudo ln -s /opt/OpenBLAS/lib /usr/lib/libblas
        sudo ln -s /opt/OpenBLAS/lib/libopenblas.so /opt/OpenBLAS/lib/libblas.so.3
    fi
else
    sudo apt-get -y install libopenblas-dev
fi

##############################
# Install Python related stuff
##############################

cd ~/Software
sudo apt-get -y install python-dev
sudo apt-get -y install python-wheel
wget https://bootstrap.pypa.io/get-pip.py
sudo -H python get-pip.py
sudo -H pip install --upgrade pip
sudo -H pip install --upgrade numpy
sudo -H pip install --upgrade scipy
sudo -H pip install --upgrade pandas
sudo -H pip install --upgrade matplotlib
sudo -H pip install --upgrade ipython
sudo -H pip install --upgrade psycopg2
sudo -H pip install --upgrade gslab_tools

##############################
# Install Bazel and TensorFlow
##############################

if [ "${INSTALL_TENSORFLOW}" != ""]; then
    # Bazel and TensorFlow have problems all the time so need to get consistent version of Bazel
    # this is very annoying
    mkdir ~/Software/bazel
    cd ~/Software/bazel
    wget https://github.com/bazelbuild/bazel/releases/download/0.9.0/bazel_0.9.0-linux-x86_64.deb
    sudo dpkg -i bazel_0.9.0-linux-x86_64.deb
    sudo apt-get install -f

    cd ~/Software
    git clone https://github.com/tensorflow/tensorflow
    cd tensorflow
    git checkout r1.4
    printf '\n\nn\nn\nn\nn\nn\nn\nn\nn\n' | ./configure
    bazel build --config=opt --incompatible_load_argument_is_label=false //tensorflow/tools/pip_package:build_pip_package
    bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
    sudo -H pip install /tmp/tensorflow_pkg/`ls /tmp/tensorflow_pkg`
fi

#######################
# Install Scala and SBT
#######################

echo "deb https://dl.bintray.com/sbt/debian /" | sudo tee -a /etc/apt/sources.list.d/sbt.list
sudo apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv 2EE0EA64E40A89B84B2DF73499E82A75642AC823
sudo apt-get update
sudo apt-get install sbt
sudo apt-get install bc

############
# Get Eigen3
############

cd ~/Software
wget http://bitbucket.org/eigen/eigen/get/3.3.3.tar.gz
tar -xvf 3.3.3.tar.gz
cd eigen-eigen-67e894c6cd8f
echo "export EIGEN3_INCLUDE_DIR=`pwd`" >> ~/.bashrc

##################
# Get Commons Math
##################

sudo apt-get -y install maven

cd ~/Software
wget http://mirror.stjschools.org/public/apache/commons/math/source/commons-math3-3.6.1-src.tar.gz
tar -xvf commons-math3-3.6.1-src.tar.gz
cd commons-math3-3.6.1-src
mvn clean install
cd target
echo "export COMMONS_MATH_JAR=`pwd`/commons-math3-3.6.1.jar" >> ~/.bashrc

source ~/.bashrc

############################
# Install Greenplum Database
############################

sudo apt-get -qq update
sudo apt-get -y install python-dev cmake git-core ccache libreadline-dev \
      bison flex zlib1g-dev openssl libssl-dev libpam-dev libcurl4-openssl-dev \
      libbz2-dev build-essential libapr1-dev libevent-dev libffi-dev libyaml-dev \
      libperl-dev

mkdir -p ~/gpdb/build
cd ~/gpdb/build
git clone https://github.com/greenplum-db/gp-xerces
cd ~/gpdb/build/gp-xerces
mkdir build
cd ~/gpdb/build/gp-xerces/build
sudo ../configure --prefix=/usr/local
sudo make
sudo make install

sudo -H pip install --upgrade lockfile paramiko setuptools epydoc
sudo -H pip install --upgrade psutil
sudo -H pip install conan==0.30.3

cd ~/gpdb/build
git clone https://github.com/greenplum-db/gpdb
sudo apt-get install -y libxml2-dev vim iputils-ping
cd ~/gpdb/build/gpdb/depends
git checkout 5.1.0
sudo conan remote add conan-gpdb https://api.bintray.com/conan/greenplum-db/gpdb-oss
sudo conan install --build
cd ~/gpdb/build/gpdb
sudo ./configure --with-perl --with-python --with-libxml --enable-mapreduce --enable-orca --prefix=/usr/local/gpdb
sudo make
sudo make install
sudo apt-get install -y libboost-dev

export me=`whoami`
sudo chown -R ${me}:${me} ~/gpdb

sudo mkdir -p /data/master
sudo mkdir -p /data1/primary
sudo mkdir -p /data2/primary
sudo chown -R $me:$me /data/master
sudo chown -R $me:$me /data1
sudo chown -R $me:$me /data2
sudo ldconfig

cd ~/
echo "export BENCHMARK_PROJECT_ROOT=/home/ubuntu/SLAB" >> ~/.bashrc
echo "COMPLETE" > ~/SETUP_COMPLETE

if [ "${COMPILE_SPARK}" != "" ]; then
    cd ~/Software
    # we can use the version I have precompiled on Ubuntu 16.04
    wget http://slab:eigen@souchong.ucsd.edu/spark-2.2.0.tar.gz
    tar -xvf spark-2.2.0.tar.gz
    sudo mv spark-2.2.0 /usr/local/spark
    export me=`whoami`
    sudo chown -R ${me}:${me} /usr/local/spark
    echo "export PATH=${PATH}:/usr/local/spark/bin" >> ~/.bashrc
    source ~/.bashrc

    # same for Hadoop
    wget http://slab:eigen@souchong.ucsd.edu/hadoop.tar.gz
    tar -xvf hadoop.tar.gz
    sudo mv hadoop /usr/local/hadoop
    sudo chown -R "${me}:${me}" /usr/local/hadoop
    echo "export PATH=${PATH}:/usr/local/hadoop/bin" >> ~/.bashrc
    source ~/.bashrc

    sudo mkdir /mnt/hdfs
    sudo chown ${me}:${me} /mnt/hdfs
    /usr/local/hadoop/bin/hdfs namenode format
fi
