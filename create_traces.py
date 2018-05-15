from dataset import NSLKDD
import pandas as pd


# Create derived KDD dataset
destination_file = "datasets/derived/kdd_u2r_r2l.csv"

# Load the dataset
dataset = NSLKDD("datasets/nsl-kdd/KDDTrain+.csv")

# Get a DataFrame of the dataset
df = dataset.get_df()

# Labels to include in the derived dataset
labels = ['normal', 'warezclient', 'guess_passwd', 'buffer_overflow', 'warezmaster', 'land',
          'imap', 'rootkit', 'loadmodule', 'ftp_write', 'multihop', 'phf', 'perl', 'spy']

# Filter dataframe to only include labels and save as a CSV with no index and no column headings
df.loc[df['label'].isin(labels)].to_csv(destination_file, index=False, header=False)



