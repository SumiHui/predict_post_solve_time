# -*- coding: utf-8 -*-
# @File    : bug_fixed_prediction/merge_data_include_timestamp.py
# @Info    : @ TSMC-SIGGRAPH, 2019/5/22
# @Desc    :
# -.-.. - ... -- -.-. .-.. .- -... .---.   -.-- ..- .-.. --- -. --.   ..-. .- -.


import csv
import os
import re

import numpy as np

save_dir = "dataset"
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
ref_timestamp = np.datetime64('2013-01-01T00:00:00')


dr = re.compile(r'<[^>]+>|&[^;]+;', re.S)

# raw_keys = ['Id', 'AcceptedAnswerId', 'CreationDate', 'Score',  'Body', 'Title', 'Tags']

Id_list = []
CreationDate_list = []
with open("dataset/stackoverflow_2013_answer.csv", 'r', encoding='UTF-8') as csv_file:
    reader = csv.reader(csv_file)
    _ = next(reader)
    for row in reader:
        Id_list.append(row[0])
        CreationDate_list.append(row[1])

merge_file = open("dataset/stackoverflow_2013.csv", "w", newline='', encoding='UTF-8')
posts_file = "dataset/stackoverflow_2013_posts.csv"
with open(posts_file, 'r', encoding='UTF-8') as csv_file:
    reader = csv.reader(csv_file)
    csv_header_list = next(reader)
    print(csv_header_list)
    merge_writer = csv.writer(merge_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    merge_writer.writerow(csv_header_list)
    for row in reader:
        # tags = [tag.lower() for tag in row[-2].rstrip(">").lstrip("<").split("><")]
        # if "java" not in tags:
        #     continue
        try:
            idx = Id_list.index(row[1])
        except ValueError:
            continue
        row[2] = round(float(CreationDate_list[idx]) - float(row[2]), 3)
        merge_writer.writerow(row)

merge_file.close()
