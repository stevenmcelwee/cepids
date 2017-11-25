import dataset as ds
# from sklearn.ensemble import ExtraTreesClassifier
import clusteringensemble as ce
from dateutil import parser
from datetime import datetime

data_file_path = 'datasets/nsl-kdd/KDDTrain+.csv'
kdd = ds.NSLKDD(data_file_path)
fv = kdd.get_df()

# data_file_path = 'datasets/unsw-nb15/UNSW-NB15_1.csv'
# unsw = ds.UNSW(data_file_path)
# fv = unsw.get_df()

# for date in fv['Stime'].unique():
#     print datetime.fromtimestamp(long(date))

# for attack in fv['attack_cat'].unique():
#    print attack

# print fv.corr()

# columns = list(fv.columns.values)
# columns.sort()
# for col in columns:
#     print col, fv[col].dtype, len(fv[col].unique())

# see which features are most important
# array = fv.values
# X = array[:,0:42]
# Y = array[:,42]
# model = ExtraTreesClassifier()
# model.fit(X,Y)
# print model.feature_importances_

print fv.head()

