
from functools import wraps
from sklearn.cluster import k_means_


def call_count(proxy):

    @wraps(proxy)
    def decorated_proxy(*args,**kwargs):
        decorated_proxy.calls += 1
        return proxy(*args, **kwargs)

    decorated_proxy.calls = 0
    return decorated_proxy

# apply decorators
k_means_._kmeans_single_lloyd = call_count(k_means_._kmeans_single_lloyd)

