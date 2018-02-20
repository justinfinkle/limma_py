import sys
import pandas as pd
import numpy as np
from pydiffexp.utils.io import read_dea_pickle
from pydiffexp import DEPlot, DEAnalysis, DEResults
from scipy import stats
import pydiffexp.utils.fisher_test as ft
from get_motif_stats import threshold
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from time import time
from sklearn.metrics.pairwise import pairwise_distances
import string

"""
This module is meant to facilitate the analysis of gene trajectories
"""


def get_group_dict():
    """
    Define super groups
    :return:
    """
    global group_a, group_b, group_c, group_d, group_e, group_f, group_g, group_h

    # Genes that become overexpressed and remain that way
    group_a = np.array([[0, 0, 0, 0, 1],
                        [0, 0, 0, 1, 1],
                        [0, 0, 1, 1, 1],
                        [0, 1, 1, 1, 1]])

    # Genes that become underexpressed and remain that way
    group_b = -1 * group_a

    # Genes that start overexpressed but converge
    group_c = 1 - group_a

    # Genes that start underexpressed but converge
    group_d = -1 * group_c

    # Genes that diverge by becoming overexpressed but reconverge
    group_e = np.array([[0, 0, 0, 1, 0],
                        [0, 0, 1, 1, 0],
                        [0, 1, 1, 1, 0],
                        [0, 1, 1, 0, 0],
                        [0, 1, 0, 0, 0]])

    # Genes that diverge by becoming underexpressed but reconverge
    group_f = -1 * group_e

    # Genes that start off overexpressed, converge, then become overexpressed again
    group_g = 1 - group_e

    # Genes that start off underexpressed, converge, then become underexpressed again
    group_h = -1 * group_g

    group_dict = {letter: eval('group_{}'.format(letter)) for letter in string.ascii_lowercase[:8]}

    return group_dict


def get_scores(grouped_df, lfc_df):
    scores = pd.DataFrame(grouped_df.apply(group_scores, lfc_df))
    scores = scores.reset_index().sort_values(['Cluster', 'score'], ascending=[False, False])
    return scores


def group_scores(clusters, gene_info):
    # Get the scores for each trajectory based on the assumed cluster
    expected = np.array([int(s) for s in clusters.name.strip('())').split(',')])
    clus_lfc = gene_info.loc[clusters.index]
    penalty = -np.abs(np.sign(clus_lfc).values - expected)
    scores = np.abs(clus_lfc).values * penalty + np.abs(clus_lfc).values * (penalty == 0)   # type: np.ndarray

    nonzero = max(1, sum(np.abs(expected)))

    # Normalize by length
    scores = scores/nonzero

    return pd.Series(data=np.sum(scores, axis=1), index=clusters.index, name='score')


def pairwise_corr(x, y, axis=0):
    """
    Efficient vector calculation of features
    Adapted from https://stackoverflow.com/questions/33650188/efficient-pairwise-correlation-for-two-matrices-of-features

    :param x: DataFrame
    :param y: DataFrame, must share one dimension of x
    :param axis: int; 0 or 1. Axis along which to correlate
    :return:
    """

    if axis == 1:
        x = x.T
        y = y.T

    n_rows = y.shape[0]
    sx = x.sum(0)
    sy = y.sum(0)
    p1 = n_rows * np.dot(y.T, x)
    p2 = pd.DataFrame(sx).dot(pd.DataFrame(sy).T).T
    p3 = n_rows * ((y ** 2).sum(0)) - (sy ** 2)
    p4 = n_rows * ((x ** 2).sum(0)) - (sx ** 2)

    return (p1 - p2) / np.sqrt(pd.DataFrame(p4).dot(pd.DataFrame(p3).T).T)


def to_idx_cluster(x):
    """
    Make list-like into indexable string
    :param x: list-like
    :return:
    """

    return str(tuple(x))


if __name__ == '__main__':
    pd.set_option('display.width', 2000)
    sim_data = pd.read_csv('../gnw_networks/all_simulation_stats.tsv', sep='\t', index_col=[0,1,2], header=[0,1])
    pcorr = pd.read_hdf('gene_motif_correlation.hdf', 'mydata', mode='r')
    dea = read_dea_pickle("../../differential_expression/results/sprouty_pickle.pkl")
    dep = DEPlot(dea)
    der = dea.results['KO-WT']
    # gnw_meta = pd.read_pickle('../gnw_networks/simulation_info.pkl')
    spry_clusters = pd.read_pickle('../../differential_expression/code/gene_grouping_05_005.pkl')
    gene_dict = pd.read_pickle('../../archive/clustering/tf_enrichment/gene_association_dict.pickle')
    print(pcorr.loc['CTGF'].sort_values())
    sys.exit()

    # Use a dual filter
    # First genes that are significantly DE at any one or more timepoint using the F test
    de_data = der.top_table(p=0.05)

    # Filter out a few more by doing individual tests and using a global correction
    d = der.discrete
    ind_test = d[~(d == 0).all(axis=1)]

    filtered_genes = set(de_data.index.values).intersection(set(ind_test.index.values))

    flat_list = [item for sublist in spry_clusters.apply(lambda x: x.index.values.tolist()).values.flatten() for item in
                 sublist]

    filtered_data = np.log2(dea.data.loc[filtered_genes])       # type: pd.DataFrame

    # Group genes by condition and time. Calculate the mean zscore of each condition trajectory
    gene_mean = filtered_data.groupby(level=['condition', 'time'], axis=1).mean()
    gene_mean_grouped = gene_mean.groupby(level='condition', axis=1)
    mean_z = gene_mean_grouped.transform(stats.zscore, ddof=1).fillna(0)
    # mean_lfc = gene_mean_grouped.apply(lambda x: x.sub(x.values[:, 0], axis=0))

    # Correlate zscored means for each gene with each node in every simulation
    sim_means = sim_data.loc[:, ['ko_mean', 'wt_mean']]
    sim_mean_z = sim_means.groupby(level='stat', axis=1).transform(stats.zscore, ddof=1).fillna(0)

    print('Computing pairwise')
    pcorr = pairwise_corr(sim_mean_z, mean_z, axis=1)
    pcorr.to_hdf('gene_motif_correlation.hdf', 'mydata', mode='w')


    weighted_lfc = (1 - der.p_value) * der.continuous.loc[der.p_value.index, der.p_value.columns]
    grouped = der.cluster_discrete(der.decide_tests(p_value=0.05)).groupby('Cluster')

    spry_scores = get_scores(grouped, weighted_lfc)
    spry_scores.columns = ['Cluster', 'gene', 'score']
    spry_scores.set_index(['gene'], inplace=True)
    spry_scores = spry_scores[(spry_scores.Cluster != to_idx_cluster([1, 1, 1, 1, 1])) &
                              (spry_scores.Cluster != to_idx_cluster([-1, -1, -1, -1, -1])) &
                              (spry_scores.Cluster != to_idx_cluster([0, 0, 0, 0, 0]))]
    print(spry_scores[spry_scores.Cluster == '(0, 0, 0, 1, 1)'])
    sys.exit()


    gnw_lfc = pd.read_pickle('../gnw_networks/net_lfc.pkl')
    gnw_p = pd.read_pickle('../gnw_networks/net_p.pkl')

    gnw_weighted_lfc = (1 - gnw_p) * gnw_lfc.loc[gnw_p.index]
    gnw_discrete = threshold(gnw_lfc, gnw_p, lfc_thresh=0)
    gnw_discrete.index.names = ['node', 'group', 'net']
    gnw_discrete = gnw_discrete.swaplevel(i=1, j=2)
    gnw_grouped = der.cluster_discrete(gnw_discrete).groupby('Cluster')
    gnw_scores = get_scores(gnw_grouped, gnw_weighted_lfc)
    gnw_scores.set_index(['node', 'group', 'net'], inplace=True)


    gnw_meta.drop(gnw_grouped.get_group('(0, 0, 0, 0, 0)').index, axis=0, inplace=True)

    print('Computing pairwise')
    print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    # pdist = pairwise_distances(gnw_mean, gene_mean, n_jobs=1, metric=np.corrcoef)
    pdist = np.corrcoef(gnw_meta, meta)
    print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    # pd.to_pickle(pdist, 'correlation.pkl')
    combined_index = gnw_meta.index.values.tolist() + gene_mean.index.values.tolist()
    pdist = pd.DataFrame(pdist, index=combined_index, columns=combined_index)
    idx = pd.IndexSlice
    print(pdist.loc[('CDKN1A')].sort_values(ascending=False))
    print(pdist.loc[idx['y', '896', 1], 'CDKN1A'])
    print(stats.pearsonr(gnw_mean.loc[idx['y', '896', 1]], gene_mean.loc['CDKN1A']))

    group_dict = get_group_dict()
    group_df = pd.DataFrame()
    for g, val in group_dict.items():
        df = pd.DataFrame(val, columns=gnw_discrete.columns)
        df['group'] = g
        group_df = pd.concat([group_df, df])
    group_df.set_index('group', inplace=True)
    group_df['Cluster'] = der.cluster_discrete(group_df).Cluster
    cg_dict = dict(zip(group_df.Cluster, group_df.index))
    cluster_df = der.cluster_discrete(gnw_discrete)
    cluster_df['group'] = [cg_dict[c] if c in cg_dict.keys() else 'NA' for c in cluster_df.Cluster]
    gene_group = cluster_df[cluster_df.group != 'NA'].sort_values('Cluster').groupby('group')

    # Add the metadata
    # m_add = pd.DataFrame([gnw_meta.loc[int(n)].values for g, p, n in gnw_scores.index],
    #                      index=gnw_scores.index, columns=gnw_meta.columns)
    # gnw_scores = pd.concat([gnw_scores, m_add], axis=1)
    gnw_scores = gnw_scores[(gnw_scores.Cluster != to_idx_cluster([1, 1, 1, 1, 1])) &
                            (gnw_scores.Cluster != to_idx_cluster([-1, -1, -1, -1, -1])) &
                            (gnw_scores.Cluster != to_idx_cluster([0, 0, 0, 0, 0]))]

    gnw_scores = gnw_scores[gnw_scores.score.values > 0]
    gnw_scores.sort_values(['Cluster', 'score'], inplace=True)
    # hm_data = gnw_scores.loc['y', [c for c in gnw_scores.columns if ('in' in c) or ('->' in c)]]
    # print(hm_data)
    # print('plotting')
    # sns.heatmap(hm_data, xticklabels=False, yticklabels=False)
    # plt.show()
    # sys.exit()
    spry_scores['AvgExp'] = der.continuous.loc[spry_scores.index, 'AveExpr']
    gene = 'angptl4'.upper()
    print(spry_scores.loc[gene])
    sys.exit()

    c = [1, 0, 0, 1, 1]


    print(gnw_scores[gnw_scores.Cluster == to_idx_cluster(c)].sort_values('score', ascending=False))
    print(spry_scores[spry_scores.Cluster == to_idx_cluster(c)].sort_values('AvgExp', ascending=False))

    # for gene in spry_scores[spry_scores.Cluster == to_idx_cluster(c)].sort_values('score', ascending=False).index:
    #     dep.tsplot(dea.data.loc[gene], legend=False)
    #     plt.tight_layout()
    # plt.show()




    data = dea.data.loc[gene]

    # Norm the data for display
    data = data/np.max(np.abs(data.values))
    dep.tsplot(data, legend=False)
    plt.ylabel('Expression (% of Max)')
    plt.tight_layout()
    plt.show()
