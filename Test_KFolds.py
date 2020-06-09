import os
import re
from sklearn.model_selection import KFold
from itertools import cycle

ad_files = os.listdir("/home/dthomas/AD/2D_MultiModal/OASIS3/AD/")
cn_files = os.listdir("/home/dthomas/AD/2D_MultiModal/OASIS3/CN/")

sub_id_ad = []
sub_id_cn = []
for file in ad_files:
    sub_id = re.search('(OAS\\d*)', file).group(1)
    if sub_id not in sub_id_ad:
        sub_id_ad.append(sub_id)

for file in cn_files:
    sub_id = re.search('(OAS\\d*)', file).group(1)
    if sub_id not in sub_id_cn:
        sub_id_cn.append(sub_id)

slices_s = [60, 61, 62]
slices_a = [80, 81, 82]

kf = KFold(n_splits=15)
kf_ad_sub_id = kf.split(sub_id_ad)
kf = KFold(n_splits=50)
kf_cn_sub_id = kf.split(sub_id_cn)

train_indexes_ad = []
test_indexes_ad = []

for train_index_ad, test_index_ad in kf_ad_sub_id:
    train_indexes_ad.append(list(train_index_ad))
    test_indexes_ad.append(list(test_index_ad))

train_index_iterator = cycle(train_indexes_ad)

print("Length of ad_folds: ",len(train_indexes_ad))
for train_index, test_index in kf_cn_sub_id:
    print(len(train_index))
    print(len(test_index))
    print(next(train_index_iterator))
