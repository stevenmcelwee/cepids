from dataset import NSLKDD
from dataset import UNSW
import pandas as pd

destination_path = "datasets/derived/"

kdd_traces = [["kdd_normal.csv","normal"],
              ["kdd_buffer_overflow.csv","normal","buffer_overflow"],
              ["kdd_loadmodule.csv","normal","loadmodule"],
              ["kdd_perl.csv","normal","perl"],
              ["kdd_rootkit.csv","normal","rootkit"],
              ["kdd_ftp_write.csv","normal","ftp_write"],
              ["kdd_guess_password.csv","normal","guess_passwd"],
              ["kdd_imap.csv","normal","imap"]
              ]

unsw_traces = [["unsw_normal.csv",[0,""]],
               ["unsw_exploit.csv",[0,""], [1,"Exploits"]],
               ["unsw_shellcode.csv",[0,""],[1,"Shellcode"]]
               ]

# Create KDD dataset
dataset = NSLKDD("datasets/nsl-kdd/KDDTrain+.csv")
df = dataset.get_df()

# Create a dataset that only has normal records, the U2R records, and the R2L records
file_name = "%skdd_u2r_r2l.csv" % destination_path
labels = ['normal', 'warezclient', 'guess_passwd', 'buffer_overflow', 'warezmaster', 'land',
          'imap', 'rootkit', 'loadmodule', 'ftp_write', 'multihop', 'phf', 'perl', 'spy']
print "%s..." % file_name
df.loc[df['label'].isin(labels)].to_csv(file_name, index=False, header=False)


# dataset = UNSW("datasets/unsw-nb15/UNSW-NB15_1.csv")
# df = dataset.get_df()

# for definition in unsw_traces:
#    file_name = "%s%s" % (destination_path, definition[0])
#    labels = definition[1:]
#    print "%s... " % file_name
#    if len(labels) == 1:
#        # only the normal records
#        df.loc[df['Label'] == 0].to_csv(file_name)
#    else:
#        df.loc[df['Label'] == 0].append(df.loc[df['attack_cat'] == labels[1][1]]).to_csv(file_name)


