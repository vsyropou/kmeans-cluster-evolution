The code in this repository patches and decorates the standard scikit-learn kmeans_single_lloyd clustering algorithm such that you get back the evolution of the cluster centroids at each iteration. One could use this information to visualize the convergence of the kmeans algorithm. I suspect that this feature is not officialy supported in order to save memory in cases of large datasets. Given that you will need to patch the standard scikit-learn package in order to avoid messign it up, it is recomended that you setup a seperate environment where you can run apply the patch.

To apply the patch run: patch <original_file_path> <patch_file>, where <orignal_file_path> is the location of the k_means_.py fileof the standard scikit-learn package. The file is located under the site-packages directory, site-packages/sklearn/cluster/k_means_.py. If you are using anaconda that could be something like ~/anaconda3/envs/<envirnment-name>/lib/python3.6/site-packages/sklearn/cluster/k_means_.py.

Having succesfully applied the patch, you could go on and run an example by running the main.py file of this repository.

The script will generate test data, run kmeans and plot the evolution of clusters on top of a seaborn PairGrid.

Cluster centroids evolution information is dumped as a json file. Each json file corresponds to a seperate kmeans initilization. Scikit-learn chooses the best iteration that has the smallest inertia. The same criterion is applied in the main.py scipt as well. Keep in mind that only the single threaded kmneas, n_jobs=1, was tested. 
