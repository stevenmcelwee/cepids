import sys
from dataset import UNSW
from dataset import NSLKDD
from clusterer import PartitionGenerator
import numpy as np
import pandas as pd
import math

def main():
    # Algorithm
    # 1. Load and preprocess dataset
    # 2. Generate the partitions
    # 3. Calculate P(E)
    # 4. Calculate P(A) for srcip
    # 5. Update P(E) using P(A) for srcip
    # 6. Calculate P(A) for dstip

    # sample size for active learning is the power to which to raise N ( 1.0/2 is the square root)
    sample_size_for_active_learning = 1.0/4
    dataset_class = 'UNSW'
    dataset_file = 'datasets/derived/kdd_u2r_r2l.csv'
    num_partitions = 100
    num_stdev = 2
    partition_output_file = 'cepids_partitions.csv'
    partition_input_file = 'experiment1_partitions_unsw_100p_01.csv'
    # partition_input_file = 'partitions_nslkdd_1000p.csv'
    p_e_output_file = 'p_e_nslkdd_1000.csv'
    ids = CEPIDS()
    df = ids.generate_partitions(dataset_class=dataset_class,
                                 dataset_file = dataset_file,
                                 num_partitions = num_partitions,
                                 partition_input_file = partition_input_file)
    df = ids.calculate_p_e(df, num_partitions=num_partitions,
                           num_stdev=num_stdev, output_file=p_e_output_file)
    # Calculate P(A) for the srcip, since it was found to be more distinguishable
    # found that combining srcip and dstip in this calculation was better than just srcip
    p_a_srcip_df = ids.calculate_p_a(df, host_fields=['srcip','dstip'], return_field='srcip')
    # Update P(E) based on P(A) for srcip
    df = ids.update_p_e(df, p_a_srcip_df)
    # Now calcualte P(A) for the dstip
    # Found that this only worked with dstip by itself. Combined with srcip was highly inaccurate
    p_a_dstip_df = ids.calculate_p_a(df, host_fields=['dstip'], return_field='dstip')
    # Append P(A) for dstip to dataframe
    df['P_A_DST'] = 0
    for index, row in p_a_dstip_df.iterrows():
        df.loc[df['dstip'] == row['dstip'], 'P_A_DST'] = row['P_A']

    # Perform active learning component for experiment 3
    # Use df, which is the complete dataframe, including all data and results and update accordinlgly
    # Design consideration - all that is needed is one high probability event to confirm that P(A) is correct
    # observation cubed root is consistent, fifth root is not always consistent
    for index, row in p_a_srcip_df.loc[p_a_srcip_df['P_A'] >= 0.8].iterrows():
        # grab a sample of data with number sqrt of N
        events_df = df.loc[df['srcip'] == row['srcip']]
        # sample_size = int(round(math.sqrt(len(events_df))))
        sample_size = int(round(len(events_df) ** (sample_size_for_active_learning)))
        print "Performing active learning for %s (%s of %s records)..." % (row['srcip'], sample_size, len(events_df))
        events_df = events_df.sample(sample_size)
        # ask the oracle by looking up the label for this sample
        if len(events_df.loc[events_df['label'] == 'anomaly']) > 0:
            print 'x'
            df.loc[df['srcip'] == row['srcip'], 'P_A_SRC'] = 1.0
        else:
            df.loc[df['srcip'] == row['srcip'], 'P_A_SRC'] = 0
    for index, row in p_a_dstip_df.loc[p_a_dstip_df['P_A'] >= 0.8].iterrows():
        events_df = df.loc[df['dstip'] == row['dstip']]
        # sample_size = int(round(math.sqrt(len(events_df))))
        sample_size = int(round(len(events_df) ** (sample_size_for_active_learning)))
        print "Performing active learning for %s (%s of %s records)..." % (row['dstip'], sample_size, len(events_df))
        events_df = events_df.sample(sample_size)
        # ask the oracle by looking up the label for this sample
        if len(events_df.loc[events_df['label'] == 'anomaly']) > 0:
            print 'x'
            df.loc[df['dstip'] == row['dstip'], 'P_A_DST'] = 1.0
        else:
            df.loc[df['dstip'] == row['dstip'], 'P_A_DST'] = 0


    eval_df = df[['srcip','dstip','P_E','P_A_SRC','P_E_UP','P_A_DST','label']]
    eval_df.to_csv('results/experiment3_4rt_09.csv', index=False)

    # print p_a_srcip_df
    # p_a_dstip_df.to_csv('p_a_src_dst')


# The purpose of this class is to tie the experiments together into
# a usable prototype.
class CEPIDS:

    def __init__(self):
        pass

    def load_dataset(self, dataset_class, dataset_file):
        print "Loading the dataset..."
        # This class will either load UNSW or NSLKDD
        # uses globals to load class based on variable name
        dataset_obj = globals()[dataset_class]
        dataset = dataset_obj(dataset_file)
        return dataset

    def generate_partitions(self, dataset_class, dataset_file, num_partitions,
                            partition_input_file = None,
                            partition_output_file = None):
        if partition_input_file is not None:
            print "Loading existing partition file..."
            df = pd.read_csv(partition_input_file, low_memory=False)
        else:
            print "Generating the partitions..."
            # Generate the partitions. Optionally save as csv file.
            # load the dataset from file
            dataset = self.load_dataset(dataset_class, dataset_file)
            # get a clean copy of the original dataset without preprocessing
            df = dataset.get_df()
            # preprocess the dataset
            dataset.preprocess()
            # generate partitions
            pg = PartitionGenerator(dataset, num_partitions)
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
            df[['label','votes','P_E']].to_csv(output_file, index=False)
        # Everything is calculated, now return results in df
        return df

    def update_p_e(self, df, p_a_srcip):
        print "Updating P(E) using P(A) for srcip..."
        # Create a column for P(A)
        df['P_A_SRC'] = 0
        # Loop through all of the srcip addresses and update P(E) using P(A) for the srcip
        for index, row in p_a_srcip.iterrows():
            df.loc[df['srcip'] == row['srcip'], 'P_A_SRC'] = row['P_A']
        df['P_E_UP'] = df['P_A_SRC']*df['P_E']
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
        sum_p_a = p_a_df['P_E'].sum()
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