from dataset import UNSW
from dataset import NSLKDD
from clusterer import PartitionGenerator
from collections import Counter
from clusterer import Evaluator
import pandas as pd
import numpy as np

# Constants
num_partitions = 100
# distance = 1.5

print("Loading dataset...")
# dataset = UNSW('datasets/unsw-nb15/UNSW-NB15_1.csv')
dataset = NSLKDD('datasets/derived/kdd_u2r_r2l.csv')

# get a copy of the original dataset without preprocessing
print("Getting dataframe of original dataset...")
df = dataset.get_df()

# preprocess the dataset for clustering
print("Preprocessing the data for clustering")
dataset.preprocess()

print("Instantiating PartitionGenerator...")
pg = PartitionGenerator(dataset, num_partitions)

# append clustering ensemble labels to original dataset
i = 0
for partition in pg.get_labels():
    label = "P%s" % i
    df[label] = partition[1]
    i += 1

# Change value of label to normal vs. anomaly
df.loc[df.label <> "normal", ["label"]] = 'anomaly'

# print df

# df.to_csv("experiment1_out.csv", index=False)

# Create a pivot table to analyze each partition, store results in results[] dataframe
i = 0
result_df = pd.DataFrame()
for partition in pg.get_labels():
    partition_label = "P%s" % i
    pivot_df = df.pivot_table(index=partition_label, columns='label', values='protocol_type', aggfunc=len)
    pivot_df = pivot_df.fillna({'anomaly': 0})
    pivot_df = pivot_df.fillna({'normal': 0})
    normal_mean = pivot_df['normal'].mean()
    normal_stdev = pivot_df['normal'].std()
    threshold = normal_mean + 2*normal_stdev
    pivot_df['pct_normal'] = pivot_df['normal']/(pivot_df['anomaly']+pivot_df['normal'])
    filter_df = pivot_df.loc[pivot_df['normal'] > threshold]
    for index, row in filter_df.iterrows():
        result_df = result_df.append(pd.DataFrame({'normal': [row['normal']],
                          'anomaly': [row['anomaly']],
                          'pct_normal': [row['pct_normal']]}), ignore_index=True)
    # select values that are X stdev above mean
    # print resulting row along with label
    # print new_df
    i += 1
print "Saving results..."
result_df.to_csv('experiment1_results.csv')


# evaluate labels to see if they are anomalies and append to dataframe
# ev = Evaluator(pg)

# ev.print_eval_object()

# final_df.to_csv("sample_out.csv")

