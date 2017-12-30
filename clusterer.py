from sklearn.cluster import KMeans
import random

class PartitionGenerator:


    def __init__(self, X, Y):
        # Determine number of clusters
        # Determine random state
        self.__model = KMeans( n_clusters = 20,
                               init="k-means++",
                               n_init=10,
                               max_iter=300,
                               tol=1e-4,
                               precompute_distances="auto",
                               verbose=0,
                               random_state=None,
                               copy_x=True )
        self.__model.fit(X)


    def get_labels(self):
        return self.__model.labels_


class BaggingPlanGenerator:

    def __init__(self):
        pass

    def generate(self, cols, p_count, max_features, min_features):
        # create a sampling plan for bagging
        bagging_plan = []
        for p in range(p_count):
            num_features_this_sample = random.randint(min_features, max_features)
            features_this_sample = []
            for i in range(num_features_this_sample):
                features_this_sample.append(random.choice(cols))
            bagging_plan.append(features_this_sample)
        return bagging_plan


