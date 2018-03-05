from dataset import UNSW
from dataset import NSLKDD
from clusterer import PartitionGenerator
from collections import Counter
from clusterer import Evaluator
import pandas as pd

# Constants
num_partitions = 10
# distance = 1.5

print("Loading dataset...")
# dataset = UNSW('datasets/unsw-nb15/UNSW-NB15_1.csv')
dataset = NSLKDD('datasets/nsl-kdd/KDDTrain+.csv')

# get a copy of the original dataset without preprocessing
print("Getting dataframe of original dataset...")
df = dataset.get_df()

# preprocess the dataset for clustering
print("Preprocessing the data for clustering")
dataset.preprocess()

print("Instantiating PartitionGenerator...")
pg = PartitionGenerator(dataset, num_partitions)

# results of all partitions are in this list
# print("Getting labels...")
# partitions = pg.get_labels()

# append clustering ensemble labels to original dataset
i = 0
for partition in pg.get_labels():
    label = "P%s" % i
    df[label] = partition[1]
    i += 1

# evaluate labels to see if they are anomalies and append to dataframe
ev = Evaluator(pg)

ev.print_eval_object()

# final_df.to_csv("sample_out.csv")

