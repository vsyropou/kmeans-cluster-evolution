
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt

from glob import glob

import numpy as np
import pandas as  pd
import seaborn as sns

import os
import json

from kmeans_cluster_evolution.plotting import SeabornPairGridWrapper, kMeansClusterEvolutionOnPairGrid
from kmeans_cluster_evolution.utilities import parse_evolution_files

# generate toy data
means = [0, 1]#, 2]
cov = [[1,0.1],[0.1,1]] # [[1,0.5, 0.2],[0.5,1,0.2], [0.2,0.5,1] ]
ftrs = ['Feature_1', 'Feature_2']
data = pd.DataFrame(np.random.multivariate_normal(mean=means,cov=cov,size=500),
                    columns = ftrs)


# configuration
cluster_color = { 0: 'purple', 1: 'blue', 2: 'green'}
cluster_cmaps = { 0: 'Purples', 1: 'Blues', 2: 'Greens'}
kmclusterName = 'kmCluster'
pGdiagHistBins = 15

train_data = data[ftrs]

# run kmeans
km = KMeans(n_clusters=len(ftrs),
            verbose = True,
            n_jobs = 1,
            algorithm='full'
).fit(train_data)


data[kmclusterName] = km.labels_
plot_data = data[ftrs+[kmclusterName]]


# read all json files
evolution_files = map(lambda jn: json.load(open(jn,'r')), glob('*.json'))


# parse cluster evolution data
cluster_centers_evolution = parse_evolution_files(evolution_files, km, train_data)


# plot
wrapperPairGridArgs = [plot_data, plt.scatter, sns.kdeplot, sns.kdeplot]

wrapperPairGridKwargs = {'pairGridKwargs' : dict(vars = ftrs,
                                                 hue = kmclusterName,
                                                 diag_sharey = False,
                                                 palette = cluster_color,
                                                 hue_kws = {'cmap' : cluster_cmaps }),
                         'mapLowerKwargs' : dict(s = 1),
                         'mapUpperKwargs' : dict(zorder = 0),
                         # 'mapDiagKwargs'  : dict(bins = pGdiagHistBins,
                         #                         # label = None,
                         #                         # hist_kws = {},
                         #                         # kde_kws = {'lw':1}
#                          )
                         }

plt.ion()

plot = SeabornPairGridWrapper(*wrapperPairGridArgs,
                              **wrapperPairGridKwargs)


kMeansClusterEvolutionOnPairGrid(plot.axes,
                                 cluster_centers_evolution,
                                 clustColors = cluster_color,
                                 saveall=True)


# clean up
os.system('rm -f *.json')
