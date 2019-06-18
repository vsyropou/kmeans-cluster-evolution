from setuptools import setup

setup(name='kmeans_cluster_evolution',
      version='0.1',
      description='Decorate and patch the standard "._kmeans_single_lloyd" to access clusrter evolution after each iteration.',
      url='https://github.com/vsyropou/kmeans-cluster-evolution.git',
      author='vasilis syropoulos',
      author_email='vsyropou5@gmail.com',
      packages=['kmeans_cluster_evolution'],
      zip_safe=False)


try:
    from sklearn import cluster

except ImportError as err:
    msg =  'Falied to improt "sklearn.cluster". '
    msg += 'Make sure sklearn is installed properly. '
    msg += 'Maybe try "pip install sklearn".'

    raise ImportError('%s: %s'%(msg,repr(err)))

assert hasattr(cluster, 'k_means_'), 'This error is not suposed to happen. It seems that the modeule "sklearn.clsuter" does not have a "k_means_" class. Check official "sklearn" package structure to figure out happend.'

assert hasattr(cluster.k_means_, '_kmeans_single_lloyd'), '"k_means_" class has no attribute "_kmeans_single_lloyd". Cannot enable cluster evolution. It seems that the file is renamed or moved. Check official "sklearn" package structure to figure out happend.'


# patch
if not hasattr(cluster.k_means_._kmeans_single_lloyd, 'calls'):

    import os
    
    kmeans_path = cluster.__file__.replace('__init__.py', 'k_means_.py')

    patch_path = 'kmeans_cluster_evolution/patch_sklearn_k_means.patch'

    assert os.path.exists(kmeans_path), 'Cannot locate "k_means_.py" file. The file is typically located in "site-packages/sklearn/cluster" %s. To manualy specify the path edit: "%s"'%(cluster.__file__, os.getcwd())
    # if it has not been pached
    if not os.path.exists('%s.orig'%kmeans_path):

        # validate that it will work
        cmd = "patch -l --dry-run %s %s/kmeans_cluster_evolution/patch_sklearn_k_means.patch | grep -i failed | wc -l"%(kmeans_path,os.getcwd())

        try:
            rsp_raw = os.popen(cmd).read().strip()
        except Exception as err:
            msg = 'Cannot validate patching; Command failed "%s" with output'%cmd
            raise SystemExit('%s: %s'(msg,repr(err)))

        try:
            rsp = int(rsp_raw)
        except Exception as err:
            msg = 'Cannot validate patching; Failed to cast response from command: %s, with output'%cmd
            raise SystemExit('%s: %s'(msg,repr(err)))

        assert rsp == 0, 'Patch will fail since "%s" returns %s'%(cmd,rsp)

        # then patch
        cmd = "patch -l %s %s/kmeans_cluster_evolution/patch_sklearn_k_means.patch"%(kmeans_path,os.getcwd())

        try:
            rsp_raw = os.popen(cmd).read()
        except Exception as err:
            msg = 'This exception should not have happened. Patching probably failed.'
            raise SystemExit('%s: %s'(msg,repr(err)))
