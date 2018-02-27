import sys

import matplotlib.pyplot as plt
import pandas as pd
from pydiffexp import DEPlot
from pydiffexp.gnw import GnwSimResults, get_graph, draw_results

if __name__ == "__main__":
    dea = pd.read_pickle('intermediate_data/strongly_connected_dea.pkl')
    genes = ['YKL071W']
    dep = DEPlot()
    for g in genes:
        dep.tsplot(dea.data.loc[g], legend=False, subgroup='Time')

    network = 582
    data_dir = '../data/motif_library/gnw_networks/{}/'.format(network)
    network_structure = "{}{}_goldstandard_signed.tsv".format(data_dir, network)
    p = .75
    t = [0, 15, 30, 60, 120, 240, 480]

    wt_gsr = GnwSimResults(data_dir, network, 'wt', sim_suffix='dream4_timeseries.tsv',
                           perturb_suffix="dream4_timeseries_perturbations.tsv")
    ko_gsr = GnwSimResults(data_dir, network, 'ko', sim_suffix='dream4_timeseries.tsv',
                           perturb_suffix="dream4_timeseries_perturbations.tsv")

    data = pd.concat([wt_gsr.data, ko_gsr.data]).T
    dg = get_graph(network_structure)
    draw_results(data, p, times=t, g=dg)
    plt.tight_layout()
    plt.show()
    sys.exit()
