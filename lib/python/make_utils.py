# Copyright 2018 Anthony H Thomas and Arun Kumar
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
import time
import resource
import textwrap
import subprocess32
import multiprocessing

MAX_PROC = None
counter = 0
cluster_unique_ids = {}

def _do_process(call, log, timeout):
    log.flush()
    os.fsync(log)

    if (MAX_PROC is not None):
        procset = '0' if (MAX_PROC == 1) else '0-{}'.format(MAX_PROC-1)
        call = 'taskset -c {} {}'.format(procset, call)
        ncore = MAX_PROC
    else:
        ncore = multiprocessing.cpu_count()

    print 'Running: {}'.format(call)
    log.write('\nRunning: {}\n'.format(call))
    log.write('\nCPU count capped at: {}\n'.format(MAX_PROC))
    log.write('Memory use capped at: {}GB\n'.format(
        resource.getrlimit(resource.RLIMIT_AS)[0]/float(1e9)))
    log.write('CPU Time capped at: {} seconds\n'.format(
        resource.getrlimit(resource.RLIMIT_CPU)[0]
      ))
    log.flush()
    os.fsync(log)
    os.system(call)

def set_memcap(limit):
    resource.setrlimit(resource.RLIMIT_AS, (limit, limit))

def set_nproc(limit):
    global MAX_PROC
    nproc = multiprocessing.cpu_count()
    if (limit > nproc):
        print 'Warning: requested CPU count exceeds number available'
        print 'Setting CPU count to: {}'.format(nproc)
        limit = nproc

    MAX_PROC = limit

def set_timeout(limit):
    resource.setrlimit(resource.RLIMIT_CPU, (limit, limit))

def run_R(program = None,
          cmd_args = '',
          timeout = None,
          makelog = '../output/make.log'):

    call = 'Rscript {} {}'.format(program, cmd_args)
    with open(makelog, 'a') as log:
        with open(program) as prog:
            log.write(prog.read())
        _do_process(call, log, timeout)

def parse_hosts():
    with open('/etc/hosts') as fh:
        pairs = []
        for line in fh:
            if (len(line.strip()) > 0) and ('ip6-' not in line) and ('#' not in line):
                print line
                pairs.append(line.split())

    hosts = dict(map(lambda x: (x[1],x[0]), pairs))
    hosts.pop('localhost', None)
    return hosts

def link_if_not(source, dest):
    try:
        os.symlink(source, dest)
    except OSError:
        pass

def run_pbdR(program = None,
             nproc = None,
             cmd_args = '',
             timeout = None,
             local = False,
             makelog = '../output/make.log'):

    hosts = parse_hosts()
    if (local is True):
        host_arg = ''
    else:
        host_arg = '-host {}'.format(','.join(hosts.keys()))

    if (nproc is None) and (local is False):
        nproc = multiprocessing.cpu_count() * len(hosts.keys())
        nproc -= 2*len(hosts.keys())
    if (nproc is None) and (local is True):
        nproc = multiprocessing.cpu_count()
        nproc -= 2

    call = ('mpirun -x BENCHMARK_PROJECT_ROOT -x HADOOP_CMD '
            '-np {nproc} {host_arg} '
            '-mca btl ^openib --oversubscribe '
            'Rscript {program} {cmd_args}').format(
               nproc = nproc, host_arg = host_arg,
               program = program, cmd_args = cmd_args
           )
    with open(makelog, 'a') as log:
        with open(program) as prog:
            log.write(prog.read())
        _do_process(call, log, timeout)

def run_python(program = None,
               params = None,
               cmd_args = '',
               timeout = None,
               makelog = '../output/make.log'):

    with open(program) as prog:
        contents = prog.read()
        if (params is not None):
            for param in params:
                contents = contents.replace('%{}%'.format(param), params[param])

    tempname = '_{}'.format(program)

    with open(tempname, 'w') as outprog:
        outprog.write(contents)

    call = 'python {} {}'.format(tempname, cmd_args)
    with open(makelog, 'a') as log:
        log.write(contents)
        _do_process(call, log, timeout)

    os.system('rm {}'.format(tempname))

def run_sbt(sbt_dir = None,
            cmd_args = '',
            timeout = None,
            makelog = '../output/make.log'):

    beginDir = os.getcwd()
    if (sbt_dir is not None):
        os.chdir(sbt_dir)

    prog = filter(lambda x: '.sbt' in x, os.listdir('./'))
    if (len(prog) != 1):
        raise StandardError('Incorrect number of SBT files found')
    prog = prog[0]

    call = 'sbt -Dsbt.log.noformat=true assembly {}'.format(cmd_args)
    with open(makelog, 'a') as log:
        with open(prog) as fh:
            log.write(fh.read())
        _do_process(call, log, timeout)

    os.chdir(beginDir)

def _find_spark_files(sbt_dir):
    jarDir = filter(lambda x: 'scala' in x,
    os.listdir(os.path.join(sbt_dir,'target')))[0]
    jarPath = os.path.join(sbt_dir, 'target', jarDir)
    jarFile = filter(lambda x: '.jar' in x,
        os.listdir(os.path.join(jarPath)))[0]
    jarFile = os.path.join(jarPath, jarFile)

    scalaPath = os.path.join(sbt_dir, 'src', 'main', 'scala')
    scalaFiles = os.listdir(scalaPath)

    dmlPath = os.path.join(sbt_dir, 'src', 'main', 'dml')
    dmlFiles = os.listdir(dmlPath) if (os.path.exists(dmlPath)) else []

    scalaFiles = map(lambda x: os.path.join(scalaPath, x), scalaFiles)
    dmlFiles = map(lambda x: os.path.join(dmlPath, x), dmlFiles)

    paths = {'jarFile' : jarFile,
             'scalaFiles' : scalaFiles,
             'dmlFiles' : dmlFiles}
    return paths

def run_systemml(**kwargs):
    # only exists for compatability
    run_spark(**kwargs)


def run_spark(**kwargs):
    sbt_dir = kwargs.pop('sbt_dir', None)
    if sbt_dir is None:
        print "Error: Must specify argument 'sbt_dir'"
        raise Exception("Error: Must specify argument 'sbt_dir'")

    program = kwargs.pop('program', None)
    if program is None:
        print "Error: Must specify argument 'program'"
        raise Exception("Error: Must specify argument 'program'")

    cmd_args = kwargs.pop('cmd_args')

    paths      = _find_spark_files(sbt_dir)
    jarFile    = paths['jarFile']
    scalaFiles = paths['scalaFiles']
    dmlFiles   = paths['dmlFiles']

    arg_formatter = (
        lambda x,y: ' --{} {} '.format(x, kwargs.pop(y)) if y in kwargs else '')

    argl = [('master', 'master'),
            ('driver-cores', 'driver_cores'),
            ('driver-memory', 'driver_memory'),
            ('executor-memory', 'executor_memory'),
            ('num-executors', 'num_executors')]
    extra_args = map(lambda x: arg_formatter(x[0], x[1]), argl)

    spark_args = kwargs.pop('spark_args', '')
    pretty_call = ('spark-submit --class {classname} \n'
                   '{extra_args} {spark_args} \n'
                   '{jarFile} {cmd_args}').format(
                        classname  = program,
                        extra_args = '\n'.join(extra_args),
                        spark_args = spark_args,
                        jarFile    = jarFile,
                        cmd_args   = cmd_args
                   )

    call = pretty_call.replace('\n', '')
    print 'Running: {}'.format(pretty_call)

    makelog = kwargs.pop('makelog', '../output/make.log')
    with open(makelog, 'a') as log:
        for prog in scalaFiles:
            log.write(prog + '\n' + '=' * 80 + '\n')
            with open(prog) as fh:
                log.write(fh.read())

        for prog in dmlFiles:
            if (prog != 'scratch_space'):
                log.write(prog + '\n' + '=' * 80 + '\n')
                with open(prog) as fh:
                    log.write(fh.read())
        log.flush()
        os.fsync(log)
        _do_process(call, log, None)


def run_tf_cluster(program = None,
                   params = None,
                   cmd_args = '',
                   timeout = None,
                   makelog = '../output/make.log'):

    # it must be pre-prdained that the master writes a file indicating
    # completion. If the file already exists remove it
    if (os.path.exists('../temp/done')):
        os.unlink('../temp/done')

    me = subprocess32.check_output('hostname').replace('\n','')
    hosts = parse_hosts()
    hosts.pop(me, None)
    path = os.path.abspath(program)
    dirname = os.path.dirname(path)

    if (len(hosts) == 0):
        raise RuntimeError('No remote hosts detected. Use run_python')

    worker_id = 1
    handles = []
    logfiles = []
    calls = []

    # also run the program locally. Cannot block though so we need to use
    # a slightly modified run framework
    logname = '../temp/server_log{}.txt'.format(worker_id)
    fh = open(logname, 'w')
    handles.append(fh)
    logfiles.append(logname)

    procs = []
    tf_cluster_args = 'dirname={} job-name=worker worker-id={}'.format(
        dirname, worker_id)
    call = 'python {} {} {}'.format(program, cmd_args, tf_cluster_args)
    print 'Running: {} locally'.format(call)
    p = subprocess32.Popen(call, shell=True, stdout=fh, stderr=fh)
    calls.append(call)
    procs.append(p)

    # start the remotes
    worker_ids = [0] if len(hosts) == 1 else [0] + range(2, len(hosts)-1)

    for ix, host in enumerate(hosts.keys()):
        worker_id = worker_ids[ix]
        logname = '../temp/server_log{}.txt'.format(worker_id)
        fh = open(logname, 'w')
        handles.append(fh)

        # then run the command remotely on the executor
        tf_cluster_args = 'dirname= {}job-name=worker worker-id={}'.format(
            dirname, worker_id)
        call = 'python {} {} {}'.format(path, cmd_args, tf_cluster_args)
        print 'Running: {} on {}'.format(call, host)
        p = subprocess32.Popen(
            'ssh {} "{}"'.format(host, call), stdout=fh, stderr=fh, shell=True)
        calls.append(call)
        procs.append(p)

    # check periodically to see if the master is done...
    while not (os.path.exists('../temp/done')):
        time.sleep(1)

    # wait a bit to make sure the workers have shut down
    time.sleep(5)

    # close logs and kill any straggling processes
    map(lambda x: x.kill(), procs)
    map(lambda x: x.close(), handles)


    # do the deal with the make library
    with open(makelog, 'a') as log_fh:
        for call,log in zip(calls, logfiles):
            log_fh.write('\nRan: {}\n'.format(call))
            with open(log) as fh:
                log_fh.write(fh.read)

def clean_output(filter_text, dirname='../output'):
    alloutput = set(os.listdir('../output'))
    relevant_files = filter(lambda x: str(filter_text) in x, alloutput)
    for output in alloutput:
        if output in relevant_files:
            os.unlink('../output/{}'.format(output))

def init_greenplum(nodes = 1, start_only=False,
                   existing_hosts = None, check_performance = False,
                   makelog = '../output/make.log'):

    cwd = os.getcwd()

    MAKELOG = open(makelog, 'a')
    print 'Starting Greenplum Database'
    MAKELOG.write('\nStarting Greenplum Database\n')
    MAKELOG.flush()

    p = subprocess32.Popen(
            'gpstart', stdout=MAKELOG, stderr=MAKELOG,
            stdin=subprocess32.PIPE, shell=True)
    p.communicate(input='Y\n')

    if (p.returncode != 0):
        msg = 'gpstart returned non-zero exit status. Consult makelog.'
        print msg
        raise StandardError(msg)

    print 'Stared Greenplum database'
    MAKELOG.write('Started Greenplum')
    if (start_only):
        MAKELOG.close()
        return

    if (nodes > 1):
        msg = 'Expanding Greenplum Database to {} nodes'.format(nodes)
        print msg
        MAKELOG.write('\n{}\n'.format(msg))

        # remove existing hosts
        hosts = parse_hosts()
        hosts.pop(whoami())
        hosts.pop('localhost', None)

        if not (hasattr(existing_hosts, '__iter__')):
            existing_hosts = [existing_hosts]
        map(lambda x: hosts.pop(x, None), existing_hosts)

        hostnames = '\n'.join(hosts.keys())

        # create a temporary directory to begin the expansion
        expand_dir = '_gpExpand'
        cntr = 0
        while (os.path.exists(expand_dir)):
            cntr += 1
            expand_dir += str(cntr)

        os.makedirs(expand_dir)
        os.chdir(expand_dir)
        with open('new_hosts_file', 'w') as fh:
            fh.write(hostnames)

        # run the expansion utility
        MAKELOG.write('\ngpexpand -D ubuntu -f new_hosts_file\n')
        p = subprocess32.Popen(
                'gpexpand -D ubuntu -f new_hosts_file',
                stdout=MAKELOG, stdin=subprocess32.PIPE,
                stderr=MAKELOG, shell=True)
        res = p.communicate(input='Y\n0\n')

        if (p.returncode != 0):
            print "Error in gpexpand. Consult makelog."
            raise StandardError('gpexpand returned non-zero exit status')

        expand_file_name = filter(lambda x: 'gpexpand' in x, os.listdir('./'))

        MAKELOG.write('\ngpexpand -i {} -D ubuntu\n'.format(expand_file_name[0]))
        p = subprocess32.Popen(
                'gpexpand -i {} -D ubuntu'.format(expand_file_name[0]),
                stdin=subprocess32.PIPE, stdout=MAKELOG,
                stderr=MAKELOG, shell=True)
        res = p.communicate('y\n')
        if (p.returncode != 0):
            print 'Warning: gpexpand returned errors. Consult logs'
            raise StandardError('gpexpand returned non-zero exist status')

        MAKELOG.write('\ngpexpand -d 60:00:00 -D ubuntu\n')
        p = subprocess32.Popen(
            'gpexpand -d 60:00:00 -D ubuntu', stdout=MAKELOG, shell=True)
        p.wait()

        if (p.returncode != 0):
            print 'Error in table redistribution. Consult makelog'
            raise StandardError('gpexpand returned non-zero exit status')

        MAKELOG.write('\nExpanded Greenplum Database to {} nodes\n'.format(nodes))
        print 'Expanded Greenplum Database to {} nodes'.format(nodes)

    # Check the performance of the Greenplum system
    if (check_performance):
        hosts = parse_hosts()
        hosts.pop('localhost', None)
        with open('_hosts', 'w') as fh:
            fh.write('\n'.join(hosts.keys()))

        MAKELOG.write('\nValidating Greenplum Database Instance:\n')

        call = 'sudo gpcheckperf -f _hosts -d /data1 -d /data2 -d /data3 -d /data'
        MAKELOG.write('Running: {}\n'.format(call))
        p = subprocess32.Popen(call,
            stdout=MAKELOG, stderr=MAKELOG, shell=True)
        if (p.returncode != 0):
            print "Error in gpcheckperf. Consult makelog."
            raise StandardError("Error in gpcheckperf")

        os.unlink('_hosts')

    os.chdir(cwd)
    MAKELOG.close()

def init_hadoop(nodes = 1, makelog = '../output/make.log', start_only=False):
    with open(makelog, 'a') as MAKELOG:
        print 'Initializing Hadoop cluster with {} nodes'.format(nodes)
        MAKELOG.write('\nInitializing Hadoop cluster with {} nodes\n'.format(nodes))
        hostnames = parse_hosts()
        hadoop_conf_dir = os.getenv('HADOOP_CONF_DIR')
        if (hadoop_conf_dir is None):
            print 'Please set environment variable "HADOOP_CONF_DIR"'
            raise StandardError('Unset environment variable')

        if (start_only is False):
            if (nodes > 1):
                hostnames.pop(whoami(), None)
                slaves = hostnames.keys()
            else:
                slaves = hostnames.keys()
            slaves_file = '{}/slaves'.format(hadoop_conf_dir)
            with open(slaves_file, 'w') as fh:
                fh.write('\n'.join(slaves))

            # wipe the existing data in the data directories (if any)
            MAKELOG.write('Wiping data directory')
            os.system('sudo rm -rf /hadoop_data/hdfs/datanode/*')

            p = subprocess32.Popen(
                'hdfs namenode -format', stdout=MAKELOG,
                stderr=MAKELOG, stdin=subprocess32.PIPE, shell=True)
            p.communicate(input='Y\n')

            if (p.returncode != 0):
                print 'Error in hdfs format. Consult makelog'
                raise StandardError('hdfs format returned non-zero exit status')

        hadoop_home = os.getenv('HADOOP_HOME')
        if (hadoop_home is None):
            print 'Please set environment variable "HADOOP_HOME"'
            raise StandardError('Unset environment variable')

        MAKELOG.write('\nStarting HDFS\n')
        p = subprocess32.Popen(
                '{}/sbin/start-dfs.sh'.format(hadoop_home), stdout=MAKELOG,
                stderr=MAKELOG, stdin=subprocess32.PIPE, shell=True)

        say_yes = ['yes']*nodes
        p.communicate(input='\n'.join(say_yes))
        if (p.returncode != 0):
            print 'Error in start-dfs.sh. Consult makelog'
            raise StandardError('start-dfs returned non-zero exit status')

        MAKELOG.write('\nStarting YARN\n')
        p = subprocess32.Popen(
            '{}/sbin/start-yarn.sh'.format(hadoop_home), stdout=MAKELOG,
            stderr=MAKELOG, shell=True)
        p.wait()
        if (p.returncode != 0):
            print 'Error in start-yarn.sh. Consult makelog'
            raise StandardError('start-yarn returned non-zero exit status')

        if (start_only is False):
            MAKELOG.write('\nCreating scratch directory /scratch:\n')
            p = subprocess32.Popen('hdfs dfs -mkdir /scratch',
                stdout=MAKELOG, stderr=MAKELOG, shell=True)
            p.wait()

def init_spark(nodes = 1, makelog = '../output/make.log'):
    with open(makelog, 'a') as MAKELOG:
        MAKELOG.write('\nLaunching Spark cluster with {} nodes\n'.format(nodes))
        hosts = parse_hosts()

        if (nodes > 1):
            hosts.pop(whoami(), None)
            slaves = hosts.keys()
        else:
            slaves = hosts.keys()

        spark_home = os.getenv('SPARK_HOME')
        if (spark_home is None):
            print 'Please set environment variable "SPARK_HOME"'
            raise StandardError('Unset environment variable')

        with open('{}/conf/slaves'.format(spark_home), 'w') as fh:
            fh.write('\n'.join(slaves))

        p = subprocess32.Popen(
            '{}/sbin/start-all.sh'.format(spark_home),
            stdout=MAKELOG, stderr=MAKELOG, shell=True)
        p.wait()

        MAKELOG.write('Launched Spark cluster with {} nodes'.format(nodes))

def all_hosts_but_me():
    me = subprocess32.check_output('hostname').replace('\n','')
    hosts = parse_hosts()
    hosts.pop(me, None)
    hosts.pop('localhost', None)

    return hosts.keys()

def whoami():
    me = subprocess32.check_output('hostname').replace('\n','')
    return me

def safe_mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)

def hdfs_path_exists(path):
    rc = os.system('hdfs dfs -ls {}'.format(path))
    exists = (rc == 0)
    return exists

def hdfs_put(paths=None, hdfs_stubs=None,
             overwrite=False, makelog='../output/make.log'):
    if not (hasattr(paths, '__iter__')):
        paths = [paths]

    if (hdfs_stubs is None):
        hdfs_stubs = ['/scratch/']*len(paths)

    with open(makelog, 'a') as MAKELOG:
        for path,stub in zip(paths,hdfs_stubs):
            filename = os.path.basename(path)
            dest = '{}/{}'.format(stub, filename)
            if hdfs_path_exists(dest) and (overwrite is False):
                print 'WARNING: File exists. No data copied'
                continue

            call = 'hdfs dfs -copyFromLocal -f "{}" {}/{}'.format(
                path, stub, filename)
            MAKELOG.write('\n{}\n'.format(call))
            p = subprocess32.Popen(call,
                stdout=MAKELOG, stderr=MAKELOG, shell=True)
            p.wait()
            if (p.returncode != 0):
                print 'Error in hdfs dfs -copyFromLocal'
                raise StandardError(
                    'Command {} returned non-zero exit status'.format(call))

            # also put mtd files if present
            if (os.path.exists('{}.mtd'.format(path))):
                call = 'hdfs dfs -copyFromLocal -f {}.mtd {}/{}.mtd'.format(
                    path, stub, filename)
                MAKELOG.write('\n{}\n'.format(call))
                p = subprocess32.Popen(call,
                    stdout=MAKELOG, stderr=MAKELOG, shell=True)
                p.wait()
                if (p.returncode != 0):
                    print 'Error in hdfs dfs -copyFromLocal'
                    raise StandardError(
                        'Command {} returned non-zero exit status'.format(call))

def coalesce_hdfs_files(hdfs_dir, out_path):
    os.system('hdfs dfs -rm {}/_SUCCESS'.format(hdfs_dir))
    rc = os.system('hdfs dfs -getmerge {} {}'.format(hdfs_dir, out_path))
    if rc != 0:
        raise StandardError('Could not coalesce files - HDFS returned error')

def exit_systems(makelog = '../output/make.log'):
    stop_greenplum(makelog = makelog)
    stop_spark(makelog = makelog)
    stop_hadoop(makelog = makelog)

def stop_greenplum(makelog = '../output/make.log'):
    with open(makelog, 'a') as MAKELOG:
        p = subprocess32.Popen('gpstop',
            stdout=MAKELOG, stderr=MAKELOG, stdin=subprocess32.PIPE, shell=True)
        p.communicate(input='Y\n')

        if (p.returncode != 0):
            print 'Error in call: gpstop. Consult make.log'

def stop_hadoop(makelog = '../output/make.log'):
    hadoop_home = os.getenv('HADOOP_HOME')
    calls = ['{}/sbin/stop-yarn.sh'.format(hadoop_home),
             '{}/sbin/stop-dfs.sh'.format(hadoop_home)]

    with open(makelog, 'a') as MAKELOG:
        for call in calls:
            p = subprocess32.Popen(call,
                stdout=MAKELOG, stderr=MAKELOG, shell=True)
            p.wait()
            if (p.returncode != 0):
                print 'Error in call {}. Consult makelog'.format(call)

def stop_spark(makelog = '../output/make.log'):
    spark_home = os.getenv('SPARK_HOME')
    call = '{}/sbin/stop-all.sh'.format(spark_home)
    with open(makelog, 'a') as MAKELOG:
        p = subprocess32.Popen(call,
                stdout=MAKELOG, stderr=MAKELOG, shell=True)
        p.wait()
        if (p.returncode != 0):
            print 'Error in call {}. Consult makelog'.format(call)

def symlink(source, link_name):
    os_symlink = getattr(os, "symlink", None)
    if callable(os_symlink):
        os_symlink(source, link_name)
    else:
        import ctypes
        csl = ctypes.windll.kernel32.CreateSymbolicLinkW
        csl.argtypes = (ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_uint32)
        csl.restype = ctypes.c_ubyte
        flags = 1 if os.path.isdir(source) else 0
        if csl(link_name, source, flags) == 0:
            raise ctypes.WinError()
