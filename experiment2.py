import pandas as pd
from collections import Counter

dataset_file_name = "partitions_nslkdd_100.csv"
num_partitions = 1000
num_stdev = 2

# Load the dataset with pre-generated partitions from experiment 1
print "Loading dataset..."
df = pd.read_csv(dataset_file_name, low_memory=False)

# For each partition, find which partitions are anomalies
print "Updating partitions 0 = normal, 1 = anomaly..."
for i in range(0, num_partitions):
    partition_label = "P%s" % i
    this_part = df[partition_label].value_counts()
    threshold = this_part.mean() + num_stdev*this_part.std()
    for index, value in this_part[this_part > threshold].iteritems():
        # if greater than the threshold value, then must be normal - update to -1 as a marker
        df.loc[df[partition_label] == index, partition_label] = -1
    # if it was not marked as normal, then mark it as an anomaly (0)
    df.loc[df[partition_label] <> -1, partition_label] = 0
    # if it was marked as normal, then update the marker to a 1 to reflect normal
    df.loc[df[partition_label] == -1, partition_label] = 1

# For each event, E, calculate P(E)
print "Calculating P(E) for each record..."
df['P_E'] = 0.0
for i in range(0, num_partitions):
    col_label = "P%s" % i
    df['P_E'] += df[col_label]
    # p_e += df.ix[index, col_label]
# df.ix[index, 'P_E'] = p_e / num_partitions
# df['P_E'] = df['P_E']/num_partitions

print "Saving records to CSV for analysis..."
df = df[['label','P_E']]
df.to_csv("results/experiment_2/experiment2_p_e_out.csv")

# Append P(E) to each record

# Evaluate accuracy based on original labels

# For host, calculate P(A) given P(E)

# Evaluate the accuracy based on the original labels

