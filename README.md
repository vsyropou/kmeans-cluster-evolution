The code in this repository patches and decorates the standard scikit-learn kmeans_single_lloyd clustering algorithm such that you get back the evolution of the cluster centroids at each iteration. One could use this information to visualize the convergence of the kmeans algorithm. I suspect that this feature is not officialy supported in order to save memory in cases of large datasets. Given that you will need to patch the standard scikit-learn package in order to avoid messign it up, it is recomended that you setup a seperate environment where you can run apply the patch.

In case you have applyied the patch, you could go on an run an exaple by running the main.py file in the repository.

The script will generate test data, run kmeans and plot the evolution of clusters on top of seaborn pairgrid facet.

Cluster centroids evolution information is dumped as a json file. Each json file corresponds to a seperate kmeans initilization. Scikit-learn chooses the best iteration that has the smallest inertia. The same criterion is applied in the main.py scipt as well. Keep in mind that only the single threaded kmneas, n_jobs=1, was tested. 
