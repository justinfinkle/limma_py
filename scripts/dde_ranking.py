from pydiffexp import analyze
import sys, ast
from pydiffexp import DEAnalysis, DEPlot
import pandas as pd
import numpy as np
import networkx as nx
from pydiffexp.gnw.sim_explorer import tsv_to_dg, degree_info, make_perturbations, to_gephi
import matplotlib.pyplot as plt


def get_data(path, c, n_timeseries, reps, perturbation_labels, t=None):
    df = pd.read_csv(path, sep='\t')
    df['condition'] = c
    times = sorted(list(set(df['Time'].values)))

    # For safety
    if not n_timeseries.is_integer():
        raise ValueError('Number of time points for each replicate is not the same')

    p_rep_list = np.array(list(range(reps)) * int(n_timeseries))
    ts_p_index = np.ceil((df.index.values + 1) / len(times)).astype(int) - 1
    ts_rep_list = p_rep_list[ts_p_index]
    ts_p_list = perturbation_labels[ts_p_index]

    df['perturb'] = ts_p_list
    df['rep'] = ts_rep_list

    idx = pd.IndexSlice
    full_data = df.set_index(['condition', 'rep', 'perturb', 'Time']).sort_index()

    if t is None:
        t = full_data.index.levels[full_data.index.names.index('Time')].values

    return full_data.loc[idx[:, :, :, t], :].copy()


if __name__ == '__main__':

    t = [0, 15, 30, 60, 120, 240, 480]
    condition = ['wt', 'ko']
    base_dir = '../data/insilico/strongly_connected/'
    net_name = 'Yeast-100'
    ko_gene = 'YMR016C'
    reps = 3
    perturb = 'high_pos'

    # Organize perturbations
    perturbations = pd.read_csv("{}labeled_perturbations.csv".format(base_dir), index_col=0)
    p_labels = perturbations.index.values
    n_timeseries = len(p_labels)/reps

    df, dg = tsv_to_dg("{}Yeast-100.tsv".format(base_dir))

    data = pd.DataFrame()
    for c in condition:
        ts_file = '{bd}/{c}_sim/{nn}_{c}_dream4_timeseries.tsv'.format(bd=base_dir, nn=net_name, c=c)
        data = pd.concat([data, get_data(ts_file, c, n_timeseries, reps, p_labels)])
    idx = pd.IndexSlice
    full_dea = DEAnalysis(data.sort_index().loc[idx[:, :, perturb, :], :].T, time='Time', replicate='rep',
                          reference_labels=['condition', 'Time'], log2=False)
    # Censored object
    dea = DEAnalysis(data.sort_index().loc[idx[:, :, perturb, t], :].T, time='Time', replicate='rep',
                     reference_labels=['condition', 'Time'], log2=False)

    dea.fit_contrasts()
    der = dea.results['ko-wt']
    scores = der.score_clustering()

    # Remove clusters that have no dynamic DE (i.e. all 1, -1, 0)
    interesting = scores.loc[scores.Cluster.apply(ast.literal_eval).apply(set).apply(len) > 1]
    print(interesting)
    dep = DEPlot()

    for i in interesting.index.values[:10]:
        dep.tsplot(dea.data.loc[i], legend=False, subgroup='Time')

    plt.show()
