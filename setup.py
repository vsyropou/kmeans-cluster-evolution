from setuptools import setup

setup(name='kmeans_cluster_evolution',
      version='0.1',
      description='Decorate and patch the standard "._kmeans_single_lloyd"'\
                  ' to access clusrter evolution after each iteration.',
      url='https://github.com/vsyropou/kmeans-cluster-evolution.git',
      author='vasilis syropoulos',
      author_email='vsyropou5@gmail.com',
      packages=['kmeans_cluster_evolution'],
      zip_safe=False)
