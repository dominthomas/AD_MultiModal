import os

os.chdir('/home/dthomas/')
ad_sub_train_files = []
with open('ad_sub_train_files', 'r') as filehandle:
    ad_sub_train_files = [current_file.rstrip() for current_file in filehandle.readlines()]

print(ad_sub_train_files)
