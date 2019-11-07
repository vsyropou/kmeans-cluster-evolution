
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
means = [-2, 2]#, 2]
cov = [[1, 0.3],
       [0.3,  1]] # [[1,0.5, 0.2],[0.5,1,0.2], [0.2,0.5,1] ]
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
evolution_files = [json.load(open(jn,'r')) for jn in glob('*.json')][-1]


# # parse cluster evolution data
# cluster_centers_evolution = parse_evolution_files(evolution_files, km, train_data)

plt.ion()
centroids = []
for it_num, info in enumerate(evolution_files):

    inertia = info['inertia']

    data_means = plot_data[ftrs].mean(axis=0).values
    centroids  += [ [c + data_means for c in info['centers']] ]

    plot_data['kmCluster'] = info['labels']


    wrapperPairGridArgs = [plot_data, plt.scatter, sns.kdeplot, sns.kdeplot]

    wrapperPairGridKwargs = {'pairGridKwargs' : dict(vars = ftrs,
                                                     hue = kmclusterName,
                                                     diag_sharey = False,
                                                     palette = cluster_color,
                                                     hue_kws = {'cmap' : cluster_cmaps}),
                             'mapLowerKwargs' : dict(s = 1),
                             'mapUpperKwargs' : dict(zorder = 0)}

    plot = SeabornPairGridWrapper(*wrapperPairGridArgs,
                                  **wrapperPairGridKwargs)

    kMeansClusterEvolutionOnPairGrid(plot.axes,
                                     centroids,
                                     clustColors = cluster_color,
                                     saveall=False)


    plot.fig.suptitle('Iteration %s ; Inertia=%.2f'%(it_num,inertia))

    plot.fig.savefig('cluster_evolution_iter_%s.pdf'%it_num)
# clean up
os.system('rm -f *.json')
