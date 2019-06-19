
Isn't annoying that you cannot see how the cluster positions evolve through each iteration of the kmeans algorithm in the scikit-learn package? Well, for those who cannot stand black boxes when it especially comes to the topic of algorithm coenvergence, here is a handy, but a bit nasty hack.

In order to get the cluster evolution:

- Clone this repo
- Optional: Make a python virtual environment using the requirements file in the cloned repo, as you will be patching the standard scikit-learn module. If you are using virtualenvwrapper then this is just a line

```bash
mkvirtualenv -r requirements.txt <env-name>
```
  
- Navigate to the sklearn library directory. It should be something like: "lib/python3.6/site-packages/sklearn/cluster"  Tip: use `which python` so that you get a hint on the location.

- You are in the correct directory if you can see the k_means_.py file, which is the one we woudl liek to patch.
- Apply the patch  by typing 

```bash
patch -p9 -b k_means_.py <  <this-repo-path>/kmeans-cluster-evolution/patch_sklearn_k_means.patch
```

- Return to the repo directory and install the package

```bash
pip install .
```

Run an example:
```bash
python -i <this-repo-path>/examples/main.py
```

The examples script will generate test data, run kmeans and plot the evolution of clusters on top of a seaborn PairGrid.

This patching will probably reduce hte performance of the algorithm and it cannot run in parallel mode of k_means. It might serve you in case you want to check convergece or visulaize on small scale kmenas runs before the big one.

![Alt Text](https://raw.githubusercontent.com/vsyropou/kmeans-cluster-evolution/master/examples/evolution.png)
