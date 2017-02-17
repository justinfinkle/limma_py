import sys
import pandas as pd
from pydiffexp.utils.io import read_dea_pickle
import numpy as np
from pydiffexp.utils.utils import column_unique
from pydiffexp import DiffExpPlot
import matplotlib.pyplot as plt

pd.set_option('display.width', 1000)

# Load the DEAnalysis object with fits and data
dea = read_dea_pickle("./sprouty_pickle.pkl")

# Initialize a plotting object
dep = DiffExpPlot(dea)
dep.tsplot(dea.data.loc['SERPINB2'])
plt.tight_layout()
plt.show()
sys.exit()

# Volcano Plot
x = dea.results['KO-WT'].top_table(coef=1, use_fstat=False)
# dep.volcano_plot(x, top_n=5, show_labels=True)

# Time Series Plot
x = dea.data.loc['CISH']

# dep.tsplot(x)

x = dea.results['KO_ts-WT_ts']
y = dea.results['KO-WT']

gene = 'CXCL1'
# print(x.discrete_clusters[(x.discrete_clusters['Cluster'] != '(0, 0, 0, 0)') & (y.discrete['KO_0-WT_0'] == 0)])
print(x.continuous.head())
print(dea.results['(KO-WT)_ts'].continuous.head())

p_data = dea.data.loc[gene]

dep.tsplot(p_data)
plt.show()
