import sys

import matplotlib.pyplot as plt
import pandas as pd
from pydiffexp.gnw import GnwSimResults, get_graph, draw_results

if __name__ == "__main__":
    network = 1454
    data_dir = '../data/motif_library/gnw_networks/{}/'.format(network)
    network_structure = "{}{}_goldstandard_signed.tsv".format(data_dir, network)
    p = .25
    t = [0, 15, 30, 60, 120, 240, 480]

    wt_gsr = GnwSimResults(data_dir, network, 'wt', sim_suffix='dream4_timeseries.tsv',
                           perturb_suffix="dream4_timeseries_perturbations.tsv")
    ko_gsr = GnwSimResults(data_dir, network, 'ko', sim_suffix='dream4_timeseries.tsv',
                           perturb_suffix="dream4_timeseries_perturbations.tsv")

    data = pd.concat([wt_gsr.data, ko_gsr.data]).T
    dg = get_graph(network_structure)
    draw_results(data, p, times=t, g=dg, dpi=10)
    plt.tight_layout()
    plt.show()
    sys.exit()
