import pandas as pd
from pydiffexp.gnw.results import GnwNetResults

if __name__ == '__main__':
    pd.set_option('display.width', 2000)
    data_dir = "../data/motif_library/gnw_networks/"

    # Make a list of all the subdirectories
    gnr = GnwNetResults(data_dir)
    t = [0, 15, 30, 60, 120, 240, 480]
    all_stats = gnr.compile_results(censor_times=t)
    all_stats.to_csv('../intermediate_data/sim_stats_censoredtimes.tsv', sep='\t')