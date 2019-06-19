
import numpy as np

from scipy import sparse

def parse_evolution_files(files, km_instance, train_data):


    #  kmeans performs several batches of iteration and chooses the
    # one that converged to the lowest innertia. So we do the same
    # in order to chose the best set of cluster evoltuion 

    convered_inertia = km_instance.inertia_

    # the right file is the one that contains the converged inertia 
    filter_func = lambda jstr: any([ ith_iter['inertia'] == convered_inertia for ith_iter in jstr])
    
    evolution_data = list(filter(filter_func, files))[0]

    #  kmeans centers the data around zero in case the feature matrix
    # is not sparse, so we also have to do the same to the cluste centroids

    if not sparse.issparse(train_data):
        # TODO: This is a bit ugly, i must admit

        _ar = lambda x: np.array(x)
        data_means = train_data.mean(axis=0)

        cluster_centers_evolution = []
        for i_th_centroids in [dct['centers'] for dct in evolution_data]:
            cluster_centers_evolution += [ list(map(lambda cntrd: cntrd + _ar(data_means), _ar(i_th_centroids)))]

    else:
        cluster_centers_evolution = [dct['centers'] for dct in evolution_data]


    #  lastly, make super sure that we picked up the right file by checking
    # if last iteration results are the same as the km result

    msg_ = ' Please submit a bug request, or re-run sometimes it magically helps'
    msg  = lambda scope: 'We picked up the wrong cluster evolution file, %s are wrong; %s'%(scope,msg_)
    
    assert all(evolution_data[-1]['labels'] == km_instance.labels_), msg('labels')
    assert evolution_data[-1]['inertia'] == km_instance.inertia_, msg('inertia')
    assert all((_ar(cluster_centers_evolution[-1]) - km_instance.cluster_centers_ <= 1e-5).flatten()), msg('centers')

    return cluster_centers_evolution
