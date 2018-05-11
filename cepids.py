import sys
from dataset import UNSW
from dataset import NSLKDD
from clusterer import PartitionGenerator
import numpy as np
import pandas as pd
import math

def main():

    # Settings for experiments

    # experiment 1
    dataset_file = 'datasets/unsw-nb15/UNSW-NB15_1.csv'
    dataset_class = 'UNSW'
    num_partitions = 100
    min_feature_ratio = 0.25
    max_feature_ratio = 0.75
    min_clusters=40
    max_clusters=100

    # experiment 2
    #    if input_partition_file is set, then experiment 1 will not be executed, but the partition file
    #    will be loaded with existing partitions
    input_partition_file = 'experiment1_partitions_unsw_100p_01.csv'
    num_stdev = 2

    # experiment 3
    sample_size_exponent = 1.0/4    # note: 1.0/4 is the same as the fourth root
    active_learning_output_file = 'final_output.csv'

    # End of settings for experiments

    # Instantiate the class
    ids = CEPIDS()

    # Run the first experiment
    if input_partition_file is None:
        df = ids.experiment1(dataset_file=dataset_file, dataset_class=dataset_class,
                             num_partitions=num_partitions, min_feature_ratio=min_feature_ratio,
                             max_feature_ratio=max_feature_ratio, min_clusters=min_clusters,
                             max_clusters=max_clusters)
    else:
        # Shortcut if partitions have already been run, just load them from a file
        df = pd.read_csv(input_partition_file, encoding='latin-1', low_memory=False)

    if dataset_class == 'UNSW':

        # Run the second experiment but only for the UNSW dataset
        df = ids.experiment2(df, num_partitions=num_partitions, num_stdev=num_stdev)

        # Run the third experiment but only for the UNSW dataset
    if dataset_class == 'UNSW':
        df = ids.experiment3(df, sample_size_exponent=sample_size_exponent)

        # Write the ouput as file with less columns for offline analysis
        eval_df = df[['srcip','dstip','P_E','P_A_SRC','P_A_DST','label']]
        eval_df.to_csv(active_learning_output_file, index=False, encoding='latin-1')


# The purpose of this class is to tie the experiments together into
# a usable prototype.
class CEPIDS:

    def __init__(self):
        pass

    def experiment1(self, dataset_class, dataset_file, num_partitions,
                    min_feature_ratio, max_feature_ratio, min_clusters, max_clusters, output_file=None):
        print "Running Experiment 1..."
        df = self.generate_partitions(dataset_class=dataset_class,
                                      dataset_file = dataset_file,
                                      num_partitions = num_partitions,
                                      min_feature_ratio=min_feature_ratio,
                                      max_feature_ratio=max_feature_ratio,
                                      min_clusters=min_clusters,
                                      max_clusters=max_clusters,
                                      partition_output_file = output_file)
        return df

    def experiment2(self, df, num_partitions, num_stdev, output_file = None):
        print "Running Experiment 2..."
        df = self.calculate_p_e(df, num_partitions=num_partitions,
                               num_stdev=num_stdev)
        # Calculate P(A) for the srcip, since it was found to be more distinguishable
        # found that combining srcip and dstip in this calculation was better than just srcip
        p_a_srcip_df = self.calculate_p_a(df, host_fields=['srcip','dstip'], return_field='srcip')
        # Now calcualte P(A) for the dstip
        p_a_dstip_df = self.calculate_p_a(df, host_fields=['dstip'], return_field='dstip')
        # Create a column for P(A)
        df['P_A_SRC'] = 0
        # Loop through all of the srcip addresses and update P(E) using P(A) for the srcip
        for index, row in p_a_srcip_df.iterrows():
            df.loc[df['srcip'] == row['srcip'], 'P_A_SRC'] = row['P_A']
        # Append P(A) for dstip to dataframe
        df['P_A_DST'] = 0
        for index, row in p_a_dstip_df.iterrows():
            df.loc[df['dstip'] == row['dstip'], 'P_A_DST'] = row['P_A']
        if output_file is not None:
            df.to_csv(output_file, index=False)
        return df

    def experiment3(self, df, sample_size_exponent, output_file=None):
        print "Running Experiment 3..."
        # Regenerate p_a_srcip_df by grouping by srcip and returning the maximum (they are all the same)
        p_a_srcip_df = df.groupby('srcip')['P_A_SRC'].max().reset_index()
        for index, row in p_a_srcip_df.loc[p_a_srcip_df['P_A_SRC'] >= 0.8].iterrows():
            # grab a sample of data with number sqrt of N
            events_df = df.loc[df['srcip'] == row['srcip']]
            # sample_size = int(round(math.sqrt(len(events_df))))
            sample_size = int(round(len(events_df) ** (sample_size_exponent)))
            print "Performing active learning for %s (%s of %s records)..." % (row['srcip'], sample_size, len(events_df))
            events_df = events_df.sample(sample_size)
            # ask the oracle by looking up the label for this sample
            if len(events_df.loc[events_df['label'] == 'anomaly']) > 0:
                df.loc[df['srcip'] == row['srcip'], 'P_A_SRC'] = 1.0
            else:
                df.loc[df['srcip'] == row['srcip'], 'P_A_SRC'] = 0
        # Regenerate p_a_dstip_df by grouping by dstip and returning the maximum (they are all the same)
        p_a_dstip_df = df.groupby('dstip')['P_A_DST'].max().reset_index()
        for index, row in p_a_dstip_df.loc[p_a_dstip_df['P_A_DST'] >= 0.8].iterrows():
            events_df = df.loc[df['dstip'] == row['dstip']]
            # sample_size = int(round(math.sqrt(len(events_df))))
            sample_size = int(round(len(events_df) ** (sample_size_exponent)))
            print "Performing active learning for %s (%s of %s records)..." % (row['dstip'], sample_size, len(events_df))
            events_df = events_df.sample(sample_size)
            # ask the oracle by looking up the label for this sample
            if len(events_df.loc[events_df['label'] == 'anomaly']) > 0:
                df.loc[df['dstip'] == row['dstip'], 'P_A_DST'] = 1.0
            else:
                df.loc[df['dstip'] == row['dstip'], 'P_A_DST'] = 0
            if output_file is not None:
                df.to_csv(output_file, index=False)
        return df

    def load_dataset(self, dataset_class, dataset_file):
        print "Loading the dataset..."
        # This class will either load UNSW or NSLKDD
        # uses globals to load class based on variable name
        dataset_obj = globals()[dataset_class]
        dataset = dataset_obj(dataset_file)
        return dataset

    def generate_partitions(self, dataset_class, dataset_file, num_partitions,
                            min_feature_ratio, max_feature_ratio,
                            min_clusters, max_clusters,
                            partition_output_file = None):
        print "Generating the partitions..."
        # Generate the partitions. Optionally save as csv file.
        # load the dataset from file
        dataset = self.load_dataset(dataset_class, dataset_file)
        # get a clean copy of the original dataset without preprocessing
        df = dataset.get_df()
        # preprocess the dataset
        dataset.preprocess()
        # generate partitions
        pg = PartitionGenerator(dataset=dataset, num_partitions=num_partitions,
                                min_feature_ratio=min_feature_ratio, max_feature_ratio=max_feature_ratio,
                                min_clusters=min_clusters, max_clusters=max_clusters)
        # append clustering results to original dataset as dataframe
        # label each partition P0 to PN
        i = 0
        for partition in pg.get_labels():
            label = "P%s" % i
            df[label] = partition[1]
            i += 1
        # change dataset labels to reflect normal vs. anomaly
        if dataset_class == 'UNSW':
            df['label'] = np.where(df['Label'] == 0, 'normal', 'anomaly')
        else:
            df.loc[df.label <> "normal", ["label"]] = 'anomaly'
        # create a consistent label that can be used for counting since
        # datasets have different column names
        df['counter'] = df['label']
        # if an output filename was provided, save it to csv
        if partition_output_file is not None:
            print "Saving partition output file..."
            df.to_csv(partition_output_file)
        return df

    def calculate_p_e(self, df, num_partitions, num_stdev, output_file=None):
        # Given a dataframe with partitions, calculate P(E) for each event

        # Fore each partition, calculate which clusters should be anomalies
        # Normal records will be 1 and anomalies 0 so that normal records
        # can be more easily tallied as votes for normal
        # Results are saved directly into the df Px labels
        print "Updating partitions to 1 = normal and 0 = anomaly..."
        for i in range(0, num_partitions):
            partition_label = 'P%s' % i
            this_part = df[partition_label].value_counts()
            threshold = this_part.mean() + num_stdev*this_part.std()
            for index, value in this_part[this_part > threshold].iteritems():
                # if greater than the threshold, then update to -21 as marker
                df.loc[df[partition_label] == index, partition_label] = -1
            # if not marked as normal, then update to anomaly (0)
            df.loc[df[partition_label] <> -1, partition_label] = 0
            # if marked as normal, then update to normal (1)
            df.loc[df[partition_label] == -1, partition_label] = 1

        # For each event (E), calculate P(E)
        print "Calculating P(E) for each event..."
        # Create columns to hold number of votes and P(E)
        if 'P_E' in df.columns:
            # Column exists, so don't reset it
            pass
        else:
            df['P_E'] = 0.50
        if 'votes' in df.columns:
            # Column exists, so don't reset it
            pass
        else:
            df['votes'] = 0
        # Now loop through each partition to tally the votes
        for i in range(0, num_partitions):
            col_label = 'P%s' % i
            df['votes'] += df[col_label]
        # find maximum votes for events
        mid_point = df['votes'].max()/2
        # Above the halfway point of votes, P(E) = 0
        df.loc[df['votes'] >= mid_point, 'P_E'] = 0.0
        # Below the halfway point, create gradient of P(E)
        df.loc[df['votes'] < mid_point, 'P_E'] = (1 - df['votes']/mid_point)/2
        # If an output file name was provided, save as CSV
        if output_file is not None:
            print "Saving output file..."
            df[['label','votes','P_E']].to_csv(output_file, index=False)
        # Everything is calculated, now return results in df
        return df

    def calculate_p_a(self, df, host_fields, return_field):
        # Using the dataframe with P(E) calculated, calculate P(A)
        # host_field can be either 'srcip' or 'dstip'
        # This will only work with the UNSW dataset
        print "Calculating P(A) for %s..." % host_fields
        p_a_df = df.groupby(host_fields)['P_E'].mean().reset_index()
        p_a_df['P_E'] -= p_a_df['P_E'].min()
        p_a_df['P_E'] /= p_a_df['P_E'].max()

        # Now sum the probabilities for each src_ip
        # sum_p_a = p_a_df['P_E'].sum()
        p_a_df = p_a_df.groupby([return_field])['P_E'].sum().reset_index()
        p_a_df['P_E'] -= p_a_df['P_E'].min()
        p_a_df['P_E'] /= p_a_df['P_E'].max()
        p_a_df = p_a_df.sort_values(by=['P_E'], ascending=False)

        # Observations - the srcip is more descriptive of an attacker than the dstip
        # P(Asrcip) may be able to predict P(E|A)

        # rename the column names to make sense, since P_E was only used for calculation
        p_a_df.columns = [return_field, 'P_A']

        return p_a_df

        pass

if __name__ == "__main__":
    main()