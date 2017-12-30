import dataset
import random

cols = ['srcip', 'sport', 'dstip', 'dsport', 'proto', 'state', 'dur',
        'sbytes', 'dbytes', 'sttl', 'dttl', 'sloss', 'dloss', 'service',
        'Sload', 'Dload', 'Spkts', 'Dpkts', 'swin', 'dwin', 'stcpb', 'dtcpb',
        'smeansz', 'dmeansz', 'trans_depth', 'res_bdy_len', 'Sjit', 'Djit',
        'Sintpkt', 'Dintpkt', 'tcprtt', 'synack', 'ackdat',
        'is_sm_ips_ports', 'ct_state_ttl', 'ct_flw_http_mthd', 'is_ftp_login',
        'ct_ftp_cmd', 'ct_srv_src', 'ct_srv_dst', 'ct_dst_ltm', 'ct_src_ ltm',
        'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm' ]


# create a sampling plan for bagging
p_count = 5  # number of partitions to generate
max_features = 6
min_features = 2
bagging_plan = []
# cols = dataset.get_feature_cols()
for p in range(p_count):
    num_features_this_sample = random.randint(min_features, max_features)
    features_this_sample = []
    for i in range(num_features_this_sample):
        features_this_sample.append(random.choice(cols))
    bagging_plan.append(features_this_sample)

print bagging_plan
