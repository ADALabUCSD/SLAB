cd ~/gpdb/build

set -e
echo "export MASTER_DATA_DIRECTORY=/data/master/gpseg-1" >> ~/.bashrc
wget 'https://archive.apache.org/dist/madlib/1.12/apache-madlib-1.12-src.tar.gz' -O madlib.tar.gz
sudo apt-get -y install gcc-4.9 g++-4.9
export CC=/usr/bin/gcc-4.9
export CXX=/usr/bin/gcc-4.9
tar -xvf madlib.tar.gz
cd ~/gpdb/build/apache-madlib-1.12-src
mkdir build
cd build
cmake ../
sudo make

# cd ~/gpdb/build
# for name in `awk '/^[[:space:]]*($|#)/{next} /mycluster/{print $2;}' /etc/hosts`; do
#     if [ "${name}" == "mycluster-master" ]; then
#         echo "Not Copying to Master"
#         continue
#     fi
#
#     scp -r apache-madlib-1.12-src ${name}:/home/ubuntu/gpdb/build/
# done

cd ~/gpdb/build/apache-madlib-1.12-src/build
echo "source /usr/local/gpdb/greenplum_path.sh" >> ~/.bashrc
source ~/.bashrc
source /usr/local/gpdb/greenplum_path.sh
./src/bin/madpack -p greenplum -s madlib install
