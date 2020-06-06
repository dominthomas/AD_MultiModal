import random
import os
import re

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

random.seed(1)

random.shuffle(sub_id_ad)
random.shuffle(sub_id_cn)

os.chdir("/home/dthomas/AD/2D_MultiModal/OASIS3/AD/")
ad_sub_train = sub_id_ad[0:164]
ad_sub_test = sub_id_ad[165:177]

ad_sub_train_files = []
ad_sub_test_files = []

for file in ad_files:
    file_sub_id = re.search('(OAS\\d*)', file).group(1)
    if file_sub_id in ad_sub_train:
        ad_sub_train_files.append(file)
    elif file_sub_id in ad_sub_test:
        ad_sub_test_files.append(file)

os.chdir("/home/dthomas/AD/2D_MultiModal/OASIS3/CN")
cn_sub_train = sub_id_cn[0:574]
cn_sub_test = sub_id_cn[575:587]

cn_sub_train_files = []
cn_sub_test_files = []

for file in cn_files:
    file_sub_id = re.search('(OAS\\d*)', file).group(1)
    if file_sub_id in cn_sub_train:
        cn_sub_train_files.append(file)
    elif file_sub_id in cn_sub_test:
        cn_sub_test_files.append(file)


os.chdir("/home/dthomas/")

with open('ad_sub_train_files', 'w') as filehandle:
    filehandle.writelines("%s\n" % file for file in ad_sub_train_files)

with open('ad_sub_test_files', 'w') as filehandle:
    filehandle.writelines("%s\n" % file for file in ad_sub_test_files)

with open('cn_sub_train_files', 'w') as filehandle:
    filehandle.writelines("%s\n" % file for file in cn_sub_train_files)

with open('cn_sub_test_files', 'w') as filehandle:
    filehandle.writelines("%s\n" % file for file in cn_sub_test_files)

