import pandas as pd
import random

df_orig = pd.DataFrame({'A': [1,2,3,4,5,6,7,8,9,10],
                        'B': [10,9,8,7,6,5,4,3,2,1]})

df_row_count = len(df_orig.index)
print df_row_count

rows = random.sample(df_orig.index, df_row_count/2)

df_sample = df_orig.ix[rows]

df_orig = df_orig.drop(rows)

print df_sample

print df_orig
