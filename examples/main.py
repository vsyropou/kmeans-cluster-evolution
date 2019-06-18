
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
from scipy import sparse
from glob import glob

import numpy as np
import pandas as  pd
import seaborn as sns
import json

from plotting import SeabornPairGridWrapper, kMeansClusterEvolutionOnPairGrid
from kmeans_cluster_evolution import kmeans_single_lloyd_decorator

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



# parse cluster evolution data
#  read all json files
evolution_files = map(lambda jn: json.load(open(jn,'r')), glob('*.json'))
#  choose the best iteration (same logic as standard kmeans)
evolution_data = list(filter(lambda jstr: any([ jsn_ith_iter['inertia'] == km.inertia_ for jsn_ith_iter in jstr]), evolution_files))[0]


# append data means to centroinds coordiantes, same as standard k means
_ar = lambda x: np.array(x)
data_means = train_data.mean(axis=0)

if not sparse.issparse(data):

    cluster_centers_evolution = []
    for i_th_centroids in [dct['centers'] for dct in evolution_data]:
        cluster_centers_evolution += [ list(map(lambda cntrd: cntrd + _ar(data_means), _ar(i_th_centroids)))]

else:
    cluster_centers_evolution = [dct['centers'] for dct in evolution_data]
        


# check if last iteration is the same as the km result
assert all(evolution_data[-1]['labels'] == km.labels_), 'labels'
assert evolution_data[-1]['inertia'] == km.inertia_, 'inertia'
assert all((_ar(cluster_centers_evolution[-1]) - km.cluster_centers_ <= 1e-5).flatten()), 'centers'


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

for idx, centroids in enumerate(cluster_centers_evolution):
    kMeansClusterEvolutionOnPairGrid(plot.axes,
                                     [centroids],
                                     clustColors = cluster_color)
    plot.savefig('%s.pdf'%idx)
