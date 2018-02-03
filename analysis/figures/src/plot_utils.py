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
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rc, rcParams
from matplotlib.font_manager import FontProperties

plt.style.use('seaborn-muted')
rc('font', weight='bold')

OPLABELS = {
    "NORM":   "MAT.2: NORM",
    "MVM":    "MAT.4: MVM",
    "ADD":    "MAT.5: ADD",
    "GMM":    "MAT.6: GMM",
    "TSM":    "MAT.3: GRM",
    "TRANS":  "MAT.1: TRANS",
    "reg":    "ALG.1: OLS",
    "logit":  "ALG.2: LR",
    "gnmf":   "ALG.3: NMF",
    "robust": "ALG.4: HRSE",
    "pca":    "PCA",
    "SVD":    "PIPE.2: SVD",
    "pipelines": "PIPE.2: MMC"
}

def make_plot(op, data, sysnames,
              depvar_name='',
              stub='',
              figsize=(7,4),
              legend_only=False,
              errbars=True,
              legend=False,
              ticks='in',
              xticks=None,
              logx=False,
              logy=False,
              title_pref='',
              text_pch=18,
              axis_pch=20,
              xlab=None,
              ylab=None,
              lab_placement=0.5,
              title_stub=''):

    if legend_only:
        legend=True

    with open('plot_styles.json', 'rb') as fh:
        meta = json.load(fh)

    handle = plt.figure(figsize=figsize)

    ax = handle.add_subplot(111)
    if logx:
        ax.set_xscale('log')
    if logy:
        ax.set_yscale('log')
    plt.tick_params(labelsize=text_pch)
    if xticks is not None:
        plt.xticks(*xticks)

    for sys in sorted(sysnames):
        colnames = [depvar_name] + filter(lambda x: sys in x, data.columns)
        sys_data = data.ix[:,colnames]
        _make_subplot(sys_data, depvar_name, meta[sys], errbars, legend_only)

    if legend_only:
        handles, labels = ax.get_legend_handles_labels()
        plt.axes().get_xaxis().set_visible(False)
        plt.axes().get_yaxis().set_visible(False)
        plt.box(False)
        lgd = plt.legend(handles, labels, fancybox=True,
                 framealpha=1, frameon=False, loc=7, fontsize='large')
        plt.savefig(
            '../output/{}{}_legend.png'.format(op, stub), bbox_inches='tight')
        plt.close()
        return

    alignment = {'horizontalalignment': 'center',
                 'verticalalignment': 'top'}
    font = FontProperties().copy()
    font.set_weight('bold')
    font.set_size(text_pch)
    plt.text(lab_placement, .98, title_pref + OPLABELS[op] + title_stub,
             transform=ax.transAxes,
             fontproperties=font, **alignment)

    ax = handle.gca()
    ticks = ticks if legend is False else 'out'
    handles, labels = ax.get_legend_handles_labels()
    if xlab is not None:
        plt.xlabel(xlab, fontsize=axis_pch, fontweight='bold')
    if ylab is not None:
        plt.ylabel(ylab, fontsize=axis_pch, fontweight='bold')
    if legend:
        lgd = plt.legend(handles, labels, fancybox=True,
                         framealpha=.5, loc='best')

    plt.savefig('../output/{}{}.png'.format(op, stub), bbox_inches='tight')
    plt.close()

def get_plot_bounds(data):
    min_col = filter(lambda x: 'min' in x, data.columns)
    max_col = filter(lambda x: 'max' in x, data.columns)
    lb = data[min_col].values
    ub = data[max_col].values

    return (lb.min(), ub.max())

def _make_subplot(data, depvar_name, sysmeta, errbars=True, noplot=False):
    median_col = filter(lambda x: 'median' in x, data.columns)
    min_col = filter(lambda x: 'min' in x, data.columns)
    max_col = filter(lambda x: 'max' in x, data.columns)
    if noplot:
        plt.plot([],[],linewidth=1.5,
                 marker=sysmeta['pty'],
                 dashes=sysmeta['lty'],
                 color=sysmeta['color'],
                 label=sysmeta['pretty-name'])
    else:
        plt.plot(data[depvar_name],
                 data[median_col],
                 linewidth=1.5,
                 markersize=12,
                 marker=sysmeta['pty'],
                 dashes=sysmeta['lty'],
                 color=sysmeta['color'],
                 label=sysmeta['pretty-name'])
    if errbars:
        errmin = np.abs(data[median_col].values.ravel() -
                        data[min_col].values.ravel())
        errmax = np.abs(data[median_col].values.ravel() -
                         data[max_col].values.ravel())
        plt.errorbar(data[depvar_name].values.ravel(),
                     data[median_col].values.ravel(),
                     yerr=[errmin,errmax],
                     fmt=sysmeta['pty'],
                     markersize=12,
                     color=sysmeta['color'],
                     capsize=4,
                     label='')

def merge_data(paths,
               op_name,
               merge_var,
               dtypes=None,
               average_iters=False,
               exclude_from_avg=None,
               filter_text_not=None,
               filter_text_yes=None,
               insert_ix=None):

    if exclude_from_avg is None:
        exclude_from_avg = []
    if not type(exclude_from_avg) is list:
        exclude_from_avg = [exclude_from_avg]
    relevant_paths = sorted(filter(lambda x: op_name in x, paths))
    if filter_text_not is not None:
        relevant_paths = filter(
            lambda x: filter_text_not not in x, relevant_paths)
    if filter_text_yes is not None:
        relevant_paths = filter(
            lambda x: filter_text_yes in x, relevant_paths)

    all_data, sysnames = aggregate(relevant_paths, dtypes)

    base = None
    for data, sysname in zip(all_data, sysnames):
        select_vars = ['time{}'.format(x) for x in range(2,6)]
        if merge_var in data.columns:
            select_vars.insert(0, merge_var)

        data = data.ix[:,select_vars]
        if (merge_var not in data.columns) and (insert_ix is not None):
            data[merge_var] = insert_ix[:data.shape[0]]
        data = data.set_index(merge_var)

        time_vars = filter(lambda x: 'time' in x, data.columns)
        if average_iters and (sysname not in exclude_from_avg):
            data.ix[:,time_vars] = data.ix[:,time_vars] / 3.0
        varname = 'median_{}'.format(sysname)
        data.ix[:,varname] = data.ix[:,time_vars].median(axis=1)

        varname = 'min_{}'.format(sysname)
        data.ix[:,varname] = data.ix[:,time_vars].min(axis=1)

        varname = 'max_{}'.format(sysname)
        data.ix[:,varname] = data.ix[:,time_vars].max(axis=1)
        data = data.drop(time_vars, axis=1)

        if base is None:
            base = data
        else:
            base = base.join(data, how='outer', rsuffix='')

    base = base.reset_index()
    return base, sysnames

def aggregate(paths, dtypes=None):
    data = {}
    for p in paths:
        sysname = os.path.basename(p).split('_')[0]
        tmp = pd.read_csv(p, dtype=dtypes)
        if (sysname not in data):
            data[sysname] = tmp
        else:
            data[sysname] = data[sysname].append(tmp)
    return (data.values(), data.keys())
