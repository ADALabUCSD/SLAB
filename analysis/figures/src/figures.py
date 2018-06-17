from plot_utils import *

def main():
    sparse_matrix_plots()
    dense_matrix_plots()
    dist_svd_plots()
    criteo_plots()
    ml_dense_plots()
    ml_sparse_plots()
    single_node_ml_plots()
    pipelines_plots()

def sparse_matrix_plots():
    sparse_scale_dir = '../external/distributed_sparse/scale_mat_size'
    sparse_nodes_dir = '../external/distributed_sparse/scale_nodes'
    #legend_flag = True

    filenames = os.listdir(sparse_scale_dir)
    filenames = filter(lambda x: '.log' not in x, filenames)
    sparse_scale_paths = map(
        lambda x: os.path.join(sparse_scale_dir, x), filenames)

    filenames = os.listdir(sparse_nodes_dir)
    filenames = filter(lambda x: '.log' not in x, filenames)
    sparse_nodes_paths = map(
        lambda x: os.path.join(sparse_nodes_dir, x), filenames)

    xaxis_label = 'Percentage nonzero values'
    yaxis_label = 'Seconds'

    scale_stubs = ['A','B','C','D','A','B']
    nodes_stubs = ['A','B','C','D','E','F']
    ix = 0
    for op in ['NORM','MVM','ADD','GMM','TRANS','TSM']:
        legend_flag_scale = op in ['NORM','TRANS']
        legend_flag_nodes = op == 'NORM'
        yaxt_nodes = yaxis_label if legend_flag_nodes else None
        yaxt_scale = yaxis_label if legend_flag_scale else None

        title_stub_scale = '({}) '.format(scale_stubs[ix])
        title_stub_nodes = '({}) '.format(nodes_stubs[ix])

        data_scale, sysnames = merge_data(
            sparse_scale_paths, op, merge_var='sr', dtypes={'sr': 'O'})
        data_scale['sr'] = ('0.' + data_scale['sr']).astype(np.float64)*100
        make_plot(op, data_scale, sysnames,
                  stub='_sparse_scale',
                  depvar_name='sr',
                  logx=True,
                  logy=True,
                  xlab=xaxis_label,
                  ylab=yaxt_scale,
                  title_pref=title_stub_scale,
                  legend=True)

        if op == 'ADD':
            make_plot(op, data_scale, sysnames,
                  stub='_sparse_scale_with_legend',
                  depvar_name='sr',
                  logx=True,
                  logy=True,
                  xlab=xaxis_label,
                  ylab=yaxis_label,
                  legend=True)

        data_nodes, sysnames = merge_data(sparse_nodes_paths, op,
                                          merge_var='nodes',
                                          dtypes=np.float64,
                                          filter_text_not='8')

        # we can use the data from 8 node sparsity scaling tests here
        chunk = data_scale.ix[data_scale['sr'] == 1.0,1:]
        chunk['nodes'] = 8.0
        assert chunk.shape[0] == 1, 'Invalid shape'
        data_nodes = data_nodes.append(chunk)

        data_nodes['plotvar'] = np.log2(data_nodes['nodes'])
        ticks = np.log2(data_nodes['nodes'])
        labels = data_nodes['nodes']
        title_stub = '({}) '.format(nodes_stubs[ix])
        make_plot(op, data_nodes, sysnames,
                  stub='_sparse_nodes',
                  depvar_name='plotvar',
                  logy=True,
                  ylab=yaxt_nodes,
                  xlab='Nodes',
                  xticks=(ticks,labels),
                  title_pref=title_stub_nodes,
                  legend=True)

        ix += 1

def dense_matrix_plots():
    dense_scale_dir = '../external/distributed_dense/scale_mat_size'
    dense_nodes_dir = '../external/distributed_dense/scale_nodes'
    single_node_dir = '../external/single_node_dense'

    filenames = os.listdir(dense_scale_dir)
    filenames = filter(lambda x: '.log' not in x, filenames)
    dense_scale_paths = map(
        lambda x: os.path.join(dense_scale_dir, x), filenames)

    filenames = os.listdir(dense_nodes_dir)
    filenames = filter(lambda x: '.log' not in x, filenames)
    dense_nodes_paths = map(
        lambda x: os.path.join(dense_nodes_dir, x), filenames)

    filenames = os.listdir(single_node_dir)
    filenames = filter(lambda x: '.log' not in x, filenames)
    single_node_paths = map(
        lambda x: os.path.join(single_node_dir, x), filenames)
    yaxis_label = 'Seconds'

    stub_scale = ['A','B','C','D','A','B']
    stub_nodes = ['A','B','C','D','E','F']
    ops = ['NORM','MVM','ADD','GMM','TRANS','TSM']
    for stub_scale, stub_nodes, op in zip(stub_scale, stub_nodes, ops):
        legend_flag_scale = op in ['NORM','TRANS']
        legend_flag_nodes = op == 'NORM'
        yaxt_nodes = yaxis_label if legend_flag_nodes else None
        yaxt_scale = yaxis_label if legend_flag_scale else None

        title_stub_scale = '({}) '.format(stub_scale)
        title_stub_nodes = '({}) '.format(stub_nodes)
        depvar_name = 'rows' if op != 'GMM' else 'cols'
        xaxis_label = 'Million Rows' if op != 'GMM' else 'Million Columns'
        data_scale, sysnames = merge_data(
            dense_scale_paths, op, merge_var=depvar_name, dtypes=np.float64)
        data_scale['depvar_rescaled'] = np.log10(data_scale[depvar_name])
        ticks = data_scale['depvar_rescaled']
        labels = data_scale[depvar_name]*1e-6
        make_plot(op, data_scale, sysnames,
                  stub='_dense_scale',
                  depvar_name='depvar_rescaled',
                  logx=False,
                  logy=True,
                  title_pref=title_stub_scale,
                  xticks=(ticks,labels),
                  xlab=xaxis_label,
                  ylab=yaxt_scale,
                  legend=True)

        if op == 'ADD':
            make_plot(op, data_scale, sysnames,
                  stub='_dense_scale_with_legend',
                  depvar_name='depvar_rescaled',
                  logx=False,
                  logy=True,
                  xticks=(ticks,labels),
                  xlab=xaxis_label,
                  ylab=yaxis_label,
                  legend=True)
        data_scale = data_scale.drop('depvar_rescaled', axis='columns')

        data_nodes, sysnames = merge_data(dense_nodes_paths, op,
                                          merge_var='nodes',
                                          dtypes=np.float64,
                                          filter_text_not='8')

        # we can use the data from 8 node sparsity scaling tests here
        maxval = data_scale[depvar_name].max()
        chunk = data_scale.ix[data_scale[depvar_name] == maxval,1:]
        chunk['nodes'] = 8.0
        assert chunk.shape[0] == 1, 'Invalid shape'
        data_nodes = data_nodes.append(chunk)

        data_nodes['plotvar'] = np.log2(data_nodes['nodes'])
        ticks = np.log2(data_nodes['nodes'])
        labels = data_nodes['nodes']
        make_plot(op, data_nodes, sysnames,
                  stub='_dense_nodes',
                  depvar_name='plotvar',
                  logy=True,
                  xlab='Nodes',
                  xticks=(ticks,labels),
                  title_pref=title_stub_nodes,
                  ylab=yaxt_nodes,
                  legend=True)

        data_single, sysnames = merge_data(single_node_paths, op,
                                           merge_var='rows',
                                           dtypes=np.float64,
                                           filter_text_not='cpu')
        data_single['depvar_rescaled'] = np.log10(data_single['rows'])
        ticks = data_single['depvar_rescaled']
        labels = data_single['rows']*1e-6
        make_plot(op, data_single, sysnames,
                  stub='_single_node_dense',
                  depvar_name='depvar_rescaled',
                  logx=False,
                  logy=True,
                  legend_cols=2,
                  use_pbdr=False,
                  title_pref=title_stub_nodes,
                  xticks=(ticks,labels),
                  xlab=xaxis_label,
                  ylab=yaxt_nodes,
                  legend=True)

        # do CPU scaling tests too
        if op in ['TSM','ADD']:
            title_stub = '(A) ' if op == 'TSM' else '(B) '
            logx = op == 'ADD'
            data_cpu, sysnames = merge_data(single_node_paths, op,
                                            merge_var='rows',
                                            dtypes=np.float64,
                                            filter_text_yes='cpu')

            # we can use the full tests on 24 cores here
            maxval = data_single['rows'].max()
            chunk = data_single.ix[data_single['rows'] == maxval,1:]
            chunk['rows'] = 24
            assert chunk.shape[0] == 1, 'Invalid shape'
            data_cpu = data_cpu.append(chunk)

            ticks = data_cpu['rows']
            labels = ['{}'.format(int(x)) for x in data_cpu['rows']]
            lgd = (op == 'TSM')
            make_plot(op, data_cpu, sysnames,
                      stub='_single_node_cpu',
                      depvar_name='rows',
                      logx=logx,
                      logy=True,
                      use_pbdr=False,
                      title_pref=title_stub,
                      xticks=(ticks,labels),
                      xlab='Number of Cores',
                      ylab='Seconds',
                      legend=True)

            # make speedup cols
            title_stub = '(C) ' if op == 'TSM' else '(D) '
            median_only = filter(lambda x: 'median' in x, data_cpu.columns)
            data_speedup = data_cpu.ix[:,['rows'] + median_only]
            for system in sysnames:
                varname = 'median_{}'.format(system)
                if np.isnan(data_speedup.ix[0,varname]):
                    base = data_speedup.ix[1,varname]
                else:
                    base = data_speedup.ix[0,varname]
                data_speedup[varname] = data_speedup[varname] / base
                data_speedup[varname] = 1.0/data_speedup[varname]

            make_plot(op, data_speedup, sysnames,
                      stub='_single_node_cpu_speedup',
                      depvar_name='rows',
                      errbars=False,
                      logx=True,
                      logy=False,
                      legend_cols=2,
                      use_pbdr=False,
                      legend_space=0.1,
                      title_pref=title_stub,
                      xticks=(ticks,labels),
                      xlab='Number of Cores',
                      ylab='Speedup',
                      legend=True)

            not_madlib = filter(lambda x: 'madlib' not in x, data_cpu.columns)
            ixm = sysnames.index('madlib')
            sysnames.pop(ixm)
            sub_cols = data_cpu.ix[:,not_madlib]
            make_plot(op, sub_cols, sysnames,
                      stub='_single_node_cpu_nomadlib',
                      depvar_name='rows',
                      logx=logx,
                      logy=True,
                      use_pbdr=False,
                      title_pref=title_stub,
                      xticks=(ticks,labels),
                      xlab='Number of Cores',
                      ylab='Seconds',
                      legend=True)


def dist_svd_plots():
    in_dir = '../external/decompositions/scale_mat_size'
    filenames = os.listdir(in_dir)
    filenames = filter(lambda x: '.log' not in x, filenames)
    paths = map(lambda x: os.path.join(in_dir, x), filenames)
    data, sysnames = merge_data(paths, 'SVD',
                                merge_var='rows',
                                dtypes=np.float64)

    # R annoyingly reported times in minutes...
    r_cols = filter(lambda x: 'R' in x, data.columns)
    data.ix[:,r_cols] = data.ix[:,r_cols]*60.0
    labels = data['rows']*1e-6
    data['rows'] = np.log10(data['rows'])
    ticks = data['rows']
    make_plot('SVD', data, sysnames,
              depvar_name='rows',
              stub='_dist',
              logy=True,
              title_pref=' (A)',
              xticks=(ticks,labels),
              xlab='Million Rows',
              ylab='Seconds',
              legend=True)

def single_node_ml_plots():
    in_dir = '../external/single_node_ml'
    filenames = os.listdir(in_dir)
    filenames = filter(lambda x: '.log' not in x, filenames)
    paths = map(lambda x: os.path.join(in_dir, x), filenames)

    ylab = 'Seconds'
    legend_flag = True
    title_stubs = ['A','B','C','D']
    ix = 0
    for op in ['reg','logit','gnmf','robust']:
        title_stub = '({}) '.format(title_stubs[ix])
        ix += 1
        average_iters = op in ['logit','gnmf']
        data, sysnames = merge_data(paths, op,
                                    merge_var='nproc',
                                    average_iters=average_iters,
                                    dtypes=np.float64)

        data = data.rename(columns={'nproc': 'num_procs'})
        ticks = data['num_procs']
        labels = ['{}'.format(int(x)) for x in data['num_procs']]
        make_plot(op, data, sysnames,
                  depvar_name='num_procs',
                  stub='_cpu',
                  logx=True,
                  use_pbdr=False,
                  logy=True,
                  title_pref=title_stub,
                  xticks=(ticks,labels),
                  xlab='Number of Cores',
                  ylab=ylab,
                  legend_cols=2,
                  legend_space=0.1,
                  legend=True)

        median_only = filter(lambda x: 'median' in x, data.columns)
        data_speedup = data.ix[:,['num_procs'] + median_only]
        for system in sysnames:
            varname = 'median_{}'.format(system)
            if np.isnan(data_speedup.ix[0,varname]):
                base = data_speedup.ix[1,varname]
            else:
                base = data_speedup.ix[0,varname]
            data_speedup[varname] = base / data_speedup[varname]

        make_plot(op, data_speedup, sysnames,
                  stub='_cpu_speedup',
                  depvar_name='num_procs',
                  errbars=False,
                  logx=True,
                  use_pbdr=False,
                  xticks=(ticks,labels),
                  title_pref=title_stub,
                  legend_space=0.1,
                  legend_cols=2,
                  logy=False,
                  xlab='Number of Cores',
                  ylab='Speedup',
                  legend=True)

        # madlib_cols = filter(lambda x: 'madlib' in x, data.columns)
        # data_nomadlib = data.drop(madlib_cols, axis='columns')
        # madlib_ix = sysnames.index('madlib')
        # sysnames.pop(madlib_ix)
        
        #varnames = filter(lambda x: 'num_procs' not in x, data.columns)
        #data.loc[:,varnames] = np.log(data.loc[:,varnames])

        logy = True
        make_plot(op, data, sysnames,
                  depvar_name='num_procs',
                  stub='_cpu_nomadlib',
                  logx=True,
                  logy=logy,
                  legend_cols=2,
                  use_pbdr=False,
                  title_pref=title_stub,
                  xticks=(ticks,labels),
                  xlab='Number of Cores',
                  ylab=ylab,
                  legend=True)

        if legend_flag:
            legend_flag = False
            ylab = None

def criteo_plots():
    native_dir = '../external/native_algos'
    la_algo_dir = '../external/dense_la_algos'

    filenames = os.listdir(native_dir)
    cond = lambda x: ('.log' not in x) and ('dense' in x) and ('systemml' not in x)
    paths_native_dense = map(
        lambda x: os.path.join(native_dir, x), filter(cond, filenames))

    cond = lambda x: ('driver' not in x) and ('systemml' in x) and ('dense' in x)
    paths_sysml_spark = map(
        lambda x: os.path.join(native_dir, x), filter(cond, filenames))

    cond = lambda x: ('spark_and_driver' in x) and ('dense' in x)
    paths_sysml_driver = map(
        lambda x: os.path.join(native_dir, x), filter(cond, filenames))

    cond = lambda x: ('.log' not in x) and ('sparse' in x) and ('spark' not in x[-12:])
    paths_native_sparse = map(
        lambda x: os.path.join(native_dir, x), filter(cond, filenames))

    filenames = os.listdir(la_algo_dir)
    filenames = filter(
        lambda x: ('.log' not in x) and ('adclick' in x), filenames)
    paths_la_dense = map(lambda x: os.path.join(la_algo_dir, x), filenames)

    filenames = os.listdir(la_algo_dir)
    filenames = filter(
        lambda x: ('.log' not in x) and ('adclick' in x) and ('systemml' in x), filenames)
    paths_la_sparse = map(lambda x: os.path.join('../external/sparse_la_algos', x), filenames)

    yaxis_label = 'Seconds'
    legend_flag = True

    stubs = ['A','B','A']
    ix = 0
    for op in ['logit','reg','pca']:
        average_iters = op in ['logit','reg']
        title_stub = '({}.i) '.format(stubs[ix])
        exclude = 'madlib' if (op == 'reg') and average_iters else None
        data_native_dense, sysnames = merge_data(paths_native_dense, op,
                                                 merge_var='nodes',
                                                 average_iters=average_iters,
                                                 exclude_from_avg=exclude,
                                                 dtypes=np.float64)
        data_sysml_driver, _ = merge_data(paths_sysml_driver, op,
                                          merge_var='nodes',
                                          average_iters=average_iters,
                                          exclude_from_avg=exclude,
                                          dtypes=np.float64)
        data_sysml_driver.columns = map(
            lambda x: x.replace('systemml', 'systemml_driver'),
            data_sysml_driver.columns)

        data_sysml_spark, _ = merge_data(paths_sysml_spark, op,
                                         merge_var='nodes',
                                         average_iters=average_iters,
                                         exclude_from_avg=exclude,
                                         dtypes=np.float64)
        data_sysml_spark.columns = map(
            lambda x: x.replace('systemml', 'systemml_spark'),
            data_sysml_spark.columns)

        data_native_dense = data_native_dense.set_index('nodes').join(
                                data_sysml_driver.set_index('nodes')
                            ).join(
                                data_sysml_spark.set_index('nodes')
                            ).reset_index()

        sysnames = sysnames + ['systemml_driver_native','systemml_spark_native']

        data_native_both = data_native_dense.copy()
        data_native_both.columns = map(
            lambda x: '{}_native'.format(x), data_native_both.columns)
        data_la, sysnames = merge_data(paths_la_dense, op,
                                       merge_var='nodes',
                                       average_iters=(op == 'logit'),
                                       dtypes=np.float64)

        data_la_both = data_la.copy()
        data_la_both.columns = map(
            lambda x: '{}_la'.format(x), data_la_both.columns)
        data_both = data_native_both.set_index('nodes_native').join(
                        data_la_both.set_index('nodes_la')
                    ).reset_index().rename(
                        columns={'nodes_native':'nodes'}
                    )
        sysnames = filter(lambda x: 'median' in x, data_both.columns)
        sysnames = map(lambda x: x.replace('median_',''), sysnames)
        if op == 'logit':
            make_plot(op, data_both, sysnames, 
                use_stub=True, 
                legend_only=True,
                legend_cols=7)
        print data_both

        data_both['plotvar'] = np.log2(data_both['nodes'])
        ticks = data_both['plotvar']
        labels = data_both['nodes']
        make_plot(op, data_both, sysnames,
                  depvar_name='plotvar',
                  stub='_adclick_both_dense',
                  logy=True,
                  xlab='Nodes',
                  text_pch=24,
                  axis_pch=24,
                  figsize=(7,5),
                  title_pref=title_stub,
                  ylab=yaxis_label,
                  xticks=(ticks,labels),
                  legend=False)

        title_stub = '({}.ii) '.format(stubs[ix])
        if not op == 'pca':
            data_native_sparse, sysnames = merge_data(paths_native_sparse, op,
                                                      merge_var='nodes',
                                                      average_iters=average_iters,
                                                      exclude_from_avg='madlib',
                                                      dtypes=np.float64)
            # data_la_sparse, _ = merge_data(paths_la_sparse, op,
            #                                merge_var='nodes',
            #                                average_iters=average_iters,
            #                                exclude_from_avg='madlib',
            #                                dtypes=np.float64)
            # sysml = filter(lambda x: 'systemml' in x and 'logit' in x, data_la_sparse.columns)
            # data_native_sparse.loc[:, sysml] = data_la_sparse.loc[:,sysml]
            data_native_sparse.columns = map(
                lambda x: x.replace('systemml', 'systemml_driver_native'),
                data_native_sparse.columns)

            ix = sysnames.index('systemml')
            sysnames[ix] = 'systemml_driver_native'
            data_native_sparse['plotvar'] = np.log2(data_native_sparse['nodes'])
            ticks = data_native_sparse['plotvar']
            labels = data_native_sparse['nodes']
            make_plot(op, data_native_sparse, sysnames,
                      depvar_name='plotvar',
                      stub='_adclick_native_sparse',
                      logy=True,
                      xlab='Nodes',
                      figsize=(7,5),
                      text_pch=24,
                      axis_pch=24,
                      title_pref=title_stub,
                      ylab='',
                      xticks=(ticks,labels),
                      legend=False)

        # make_plot(op, data_la, sysnames,
        #       depvar_name='nodes',
        #       stub='_adclick_la',
        #       logy=True,
        #       title_stub=' (LA)',
        #       xlab='Nodes',
        #       ylab=yaxis_label,
        #       legend=legend_flag)
        ix = ix + 1

        if legend_flag is True:
            legend_flag = False
            yaxis_label = None


def ml_dense_plots():
    in_dir = '../external/dense_la_algos'
    filenames = os.listdir(in_dir)
    cond = lambda x: ('.log' not in x) and ('tall' in x) and ('8' in x)
    scale_paths = map(lambda x: os.path.join(in_dir, x), filter(cond, filenames))

    cond = lambda x: ('.log' not in x) and ('tall' in x) and ('8' not in x)
    nodes_paths = map(lambda x: os.path.join(in_dir, x), filter(cond, filenames))

    yaxis_label = 'Seconds'
    legend_flag = True
    rows = [2500000,5000000,10000000,20000000]

    stubs = ['A','B','C','D']
    ix = 0
    for op in ['reg','logit','gnmf','robust']:
        average_iters = op in ['logit','gnmf']
        title_stub = '({}) '.format(stubs[ix])
        ix = ix+1
        scale_data, sysnames = merge_data(scale_paths, op,
                                          merge_var='rows',
                                          average_iters=average_iters,
                                          dtypes=np.float64,
                                          insert_ix=rows)
    
        toplot = scale_data.copy()
        toplot['log_rows'] = np.log10(scale_data['rows'])
        ticks = toplot['log_rows']
        labels = toplot['rows']*1e-6
        make_plot(op, toplot, sysnames,
                  depvar_name='log_rows',
                  stub='_tall_la_rows',
                  logy=True,
                  title_pref=title_stub,
                  xticks=(ticks, labels),
                  xlab='Million Rows',
                  ylab=yaxis_label,
                  legend=True)

        if legend_flag is True:
            legend_flag = False
            yaxis_label = None

def ml_sparse_plots():
    in_dir = '../external/sparse_la_algos'
    filenames = os.listdir(in_dir)
    cond = lambda x: ('.log' not in x) and ('tall' in x) and ('8' in x)
    scale_paths = map(lambda x: os.path.join(in_dir, x), filter(cond, filenames))

    yaxis_label = 'Seconds'
    legend_flag = True
    for stub, op in zip(['A','B'],['gnmf','robust']):
        title_stub = '({}) '.format(stub)
        average_iters = op in ['logit','gnmf']
        data, sysnames = merge_data(scale_paths, op,
                                    merge_var='sr',
                                    dtypes='O')

        data['sr'] = '0.' + data['sr']
        data = data.astype(np.float64)
        data['sr'] = 100*data['sr']
        ticks=data['sr']
        labels = data['sr']
        make_plot(op, data, sysnames,
                  depvar_name='sr',
                  stub='_sparse_la',
                  logy=True,
                  logx=True,
                  title_pref=title_stub,
                  xticks=(ticks, labels),
                  xlab='Percentage nonzero values',
                  ylab=yaxis_label,
                  legend=True)

        if legend_flag is True:
            legend_flag = False
            yaxis_label = None

def pipelines_plots():
    in_dir = '../external/pipelines/'
    filenames = os.listdir(in_dir)
    filenames = filter(lambda x: '.log' not in x, filenames)
    paths = map(lambda x: os.path.join(in_dir, x), filenames)
    data, sysnames = merge_data(paths, 'pipelines',
                                merge_var='rows',
                                dtypes=np.float64)

    labels = ['{}'.format(x) for x in data['rows']*1e-6]
    data['rows'] = np.log10(data['rows'])
    ticks = data['rows']
    make_plot('pipelines', data, sysnames,
              depvar_name='rows',
              stub='_dist',
              logy=True,
              logx=False,
              title_pref='(B) ',
              xticks=(ticks,labels),
              xlab='Million Rows',
              ylab='Log Seconds',
              legend=True)

if __name__=='__main__':
    main()
