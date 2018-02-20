from pydiffexp import analyze
import sys
from pydiffexp import DEAnalysis, DEPlot
import pandas as pd
import numpy as np
import networkx as nx
from pydiffexp.gnw.sim_explorer import tsv_to_dg, degree_info, make_perturbations, to_gephi
import matplotlib.pyplot as plt

t = [0, 15, 30, 60, 120, 240, 480]
condition = ['wt', 'ko']
base_dir = '../data/insilico/strongly_connected/'
net_name = 'Yeast-100'
ko_gene = 'YMR016C'

# Organize perturbations
perturbations = pd.read_csv("{}labeled_perturbations.csv".format(base_dir), index_col=0)
p_labels = perturbations.index.values

df, dg = tsv_to_dg("{}Yeast-100.tsv".format(base_dir))
for perturb in p_labels:
    clean_data = pd.DataFrame()
    for c in condition:
        # Get data in the correct form
        base_path = '{}/{}_sim/'.format(base_dir, c)
        data_path = '{}{}_{}_dream4_timeseries.tsv'.format(base_path, net_name, c)
        p_path = '{}{}_{}_dream4_timeseries_perturbations.tsv'.format(base_path, net_name, c)
        df = pd.read_csv(data_path, sep='\t')
        df['condition'] = c
        p_df = pd.read_csv(p_path, sep='\t')
        times = sorted(list(set(df['Time'].values)))
        n_timeseries = len(df)/len(times)

        # For safety
        if not n_timeseries.is_integer():
            raise ValueError('Number of time points for each replicate is not the same')

        assert n_timeseries == len(p_df)

        p_rep_list = np.array([1, 2, 3] * 8)
        ts_p_index = np.ceil((df.index.values+1)/len(times)).astype(int)-1
        ts_rep_list = p_rep_list[ts_p_index]
        ts_p_list = p_labels[ts_p_index]

        df['perturb'] = ts_p_list
        df['rep'] = ts_rep_list

        idx = pd.IndexSlice
        full_data = df.set_index(['condition', 'rep', 'perturb', 'Time']).sort_index()

        if t is None:
            t = full_data.index.levels[full_data.index.names.index('Time')].values

        # Censored
        clean_data = pd.concat([clean_data, full_data.loc[idx[:, :, :, t], :]])

    dea = DEAnalysis(clean_data.sort_index().loc[idx[:, :, perturb, :], :].T,
                         time='Time', replicate='rep', reference_labels=['condition', 'Time'], log2=False)
    dea.fit_contrasts()
    der = dea.results['ko-wt']
    if perturb == 'high_pos':
        print(perturb, der.cluster_count.sort_values("Count", ascending=False))
        genes = (der.discrete_clusters[der.discrete_clusters.Cluster == '(0, 0, 0, 0, 1, 1, 1)'])
        print(genes)
        dep = DEPlot(dea)
        for gene in genes.index:
            print(nx.has_path(dg, ko_gene, gene))
            # sys.exit()
            # gene = 'YCL066W'
            dep.tsplot(dea.data.loc[gene], legend=False, subgroup='Time')
            plt.tight_layout()
        plt.show()
        sys.exit()
    elif perturb == 'steady':
        dep = DEPlot(dea)
        dep.tsplot(dea.data.loc['YFR053C'], legend=False, subgroup='Time')
        plt.tight_layout()
    else:
        continue

    print(sum(der.cluster_count.Count))
