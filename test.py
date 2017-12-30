import dataset as ds
import clusterer as cl
import random
from dateutil import parser
from datetime import datetime

# data_file_path = 'datasets/nsl-kdd/KDDTrain+.csv'
# data_file_path = 'datasets/unsw-nb15/UNSW-NB15_1.csv'

# TODO: NSLKDD must have NaN values somewhere, since it breaks clustering
# dataset = ds.NSLKDD('datasets/nsl-kdd/KDDTrain+.csv')
dataset = ds.UNSW('datasets/unsw-nb15/UNSW-NB15_1.csv')
df = dataset.get_df()
ds.analyze_df(df)

# create a sampling plan for bagging
p_count = 5  # number of partitions to generate
max_features = 30
min_features = 5
bagging_plan = []
cols = dataset.get_feature_cols()
for p in range(p_count):
    num_features_this_sample = random.randint(min_features, max_features)
    features_this_sample = []
    for i in range(num_features_this_sample):
        features_this_sample.append(random.choice(cols))
    bagging_plan.append(features_this_sample)


# given df, generate clustering solutions and append result to df
cluster_result = cl.PartitionGenerator(dataset.get_X(), dataset.get_Y())
labels = cluster_result.get_labels()
print "Cluster result list has %s records." % len(labels)
df['zzz'] = labels

ds.analyze_df(df)

# given df and the variety of cluster ids appended to the end, evaluate results to find anomalies

# given results of clustering ensemble analysis, determine probability that a host is under attack

#### Other stuff
# for date in fv['Stime'].unique():
#     print datetime.fromtimestamp(long(date))

# for attack in fv['attack_cat'].unique():
#    print attack

# print fv.corr()

# columns = list(fv.columns.values)
# columns.sort()
# for col in columns:
#     print col, fv[col].dtype, len(fv[col].unique())

# array = fv.values
# X = array[:,0:41]
# Y = array[:,41]
# model = KMeans(n_clusters=20, random_state=0).fit(X)
# print model.labels_
# model = ExtraTreesClassifier()
# model.fit(X,Y)
# print model.feature_importances_

# print df.head()

