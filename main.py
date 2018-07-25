
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt

import pandas as  pd
import seaborn as sns

from plotting import SeabornPairGridWrapper

# read data
data = pd.read_json('dwh-user-features.json', orient='columns')

ftrs = ['AssessmentsProgressed','AveragePassScore']

# configuration
cluster_color = { 0: 'purple', 1: 'blue'}
cluster_cmaps = { 0: 'Purples', 1: 'Blues'}
kmclusterName = 'kmCluster'
pGdiagHistBins = 15

# run kmeans
km = KMeans(n_clusters=2,
            verbose = True,
            n_jobs = 1,
            algorithm='full'
).fit(data[ftrs])


data[kmclusterName] = km.labels_
plot_data = data[ftrs+[kmclusterName]]





import json
evolution_data = json.load(open('km_cluster_evolution_inertia=%s.json'%km.inertia_, 'r'))

# best_iteration_key = min(map(lambda e: e['inertia'],evolution_data))
last_iteration_key = km.inertia_

# pick up the first one with the best inertia


assert False
cluster_centers_evolution = evolution_data[km.inertia]['centers']
data_means = data[ftrs].mean(axis=0)

cluster_centers_evolution += data_means

assert False

# plot
wrapperPairGridArgs = [plot_data, plt.scatter, sns.kdeplot, sns.distplot]

wrapperPairGridKwargs = {'pairGridKwargs' : dict(vars = ftrs,
                                                 hue = kmclusterName,
                                                 diag_sharey = False,
                                                 palette = cluster_color,
                                                 hue_kws = {'cmap' : cluster_cmaps }),
                         'mapLowerKwargs' : dict(s = 10),
                         'mapUpperKwargs' : dict(zorder = 0),
                         # 'mapDiagKwargs'  : dict(bins = pGdiagHistBins,
                         #                         hist_kws = {},
                         #                         kde_kws = {'lw':1} )
                         }

plt.ion()
plot = SeabornPairGridWrapper(*wrapperPairGridArgs, **wrapperPairGridKwargs)
