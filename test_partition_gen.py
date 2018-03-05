from dataset import UNSW
from dataset import NSLKDD
from clusterer import PartitionGenerator
from collections import Counter
from clusterer import Evaluator
import pandas as pd

# Constants
num_partitions = 1000
distance = 1.5

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
    df[label] = partition
    i += 1

# evaluate labels to see if they are anomalies and append to dataframe
ev = Evaluator(pg)
i = 0
for partition in ev.get_anoms(distance):
    label = "A%s" % i
    df[label] = partition
    i +=1

prob_df = df.loc[:, "A0"::]
prob_df["prob"] = prob_df.sum(axis=1)/num_partitions
prob_df = prob_df["prob"]

label_df = df["label"]
final_df = pd.concat([label_df,prob_df], axis=1)
final_df.to_csv("sample_out.csv")


# print("Evaluating partitions...")
# pg.eval_partitions()
# ev = Evaluator(pg)
# ev.print_eval_object()

# df = dataset.get_df()


# i = 0
# for cluster_labels in partitions:
#     col_label = "P%s" % (i)
#     counter = Counter()
#     for label in cluster_labels:
#         counter[label] += 1
#     df[col_label] = cluster_labels
#     i += 1
#     print col_label, counter

# print "Writing csv..."
# df.to_csv('datasets/test_output.csv')
