
try:
    from sklearn.cluster import k_means_

except ImportError as err:
    msg =  'Falied to improt "sklearn.cluster". '
    msg += 'Make sure sklearn is installed properly. '
    msg += 'Maybe try "pip install sklearn". '
    msg += 'Them re install "kmeans-cluster-evolution" package.'

    raise ImportError('%s: %s'%(msg,repr(err)))
    
assert hasattr(k_means_, '_kmeans_single_lloyd'), '"k_means_" class has no attribute "_kmeans_single_lloyd". This exception shold not happen; It seems that the file k_means_.pyfrom sklearn/cluster is renamed or moved. Check official "sklearn" package structure to figure out happend.'

assert hasattr(k_means_._kmeans_single_lloyd, 'calls'): '"_kmeans_single_lloyd" algorithm was not properly decorated; re-install "kmeans_cluster_evolution" package'

