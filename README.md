
Isn't annoying that you cannot see how the cluster positions evolve through each iteration of the kmeans algorithm in the scikit-learn package? 
Would't it be nice if sklearn provided some handles so that you can visualize the convergence of kmeans cluster like the image below?

![all](https://github.com/vsyropou/kmeans-cluster-evolution/assets/7230298/e6b9b90c-3d4f-4b2a-891e-6b874756f7ae)

Well, for those who cannot stand black boxes when it especially comes to the topic of algorithm coenvergence, here is a handy, but a bit sneaky hack.

## Hoe to Run
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

This patching will probably reduce the performance of the algorithm and it cannot run in parallel mode of k_means. It might serve you in case you want to check convergece or visulaize on small scale kmenas runs before the big one

