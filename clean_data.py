# -*- coding: utf-8 -*-
# @File    : bug_fixed_prediction/clean_data.py
# @Info    : @ TSMC-SIGGRAPH, 2019/5/10
# @Desc    : 
# -.-.. - ... -- -.-. .-.. .- -... .---.   -.-- ..- .-.. --- -. --.   ..-. .- -. 


import csv
import os
import re

import numpy as np

import time

time.strftime("%w",)


save_dir = "dataset"
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
ref_timestamp = np.datetime64('2013-01-01T00:00:00')

dr = re.compile(r'<[^>]+>|&[^;]+;', re.S)

# raw_keys = ['Id', 'AcceptedAnswerId', 'CreationDate', 'Score', 'ViewCount', 'Body', 'Title', 'Tags', 'AnswerCount',
#             'CommentCount', 'FavoriteCount', 'ParentId', 'ClosedDate', 'LastActivityDate', 'LastEditDate']

# discard the blank lines between each row
posts_file = open("dataset/stackoverflow_2013_posts.csv", "w", newline='', encoding='UTF-8')
raw_file = "/dataset/2013posts3.csv"
with open(raw_file, 'r', encoding='UTF-8') as csv_file:
    reader = csv.reader(csv_file)
    csv_header_list = next(reader)
    print(csv_header_list)
    save_writer = csv.writer(posts_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    csv_header_list.remove('ViewCount')
    csv_header_list = csv_header_list[:-7]
    csv_header_list.append("Timerate")
    csv_header_list.append("weekday")

    save_writer.writerow(csv_header_list)
    for row in reader:
        if row[1] == "":
            continue  # AcceptedAnswerId is Null
        row.pop(4)
        row = row[:-7]
        row[4] = dr.sub('', row[4])  # remove html label from Body

        hours, minutes, seconds = row[2].split('T')[-1].split(":")
        timerate = round((3600 * int(hours) + 60 * int(minutes) + float(seconds)) / 86400, 6)
        row.append(timerate)

        weekday = int(time.strftime("%w", time.strptime(row[2].split('T')[0].replace('-', ''), "%Y%m%d")))
        row.append(round(weekday / 7, 6))

        # convert np.datetime64 to relative time
        row[2] = str((np.datetime64(row[2]) - ref_timestamp) / np.timedelta64(1, 's'))

        save_writer.writerow(row)

posts_file.close()

# # answer_keyes=['Id', 'CreationDate', 'Score', 'ViewCount', 'Body', 'Title', 'Tags', 'AnswerCount', 'CommentCount',
# #             'FavoriteCount', 'ParentId', 'LastActivityDate']
# raw_file = "/dataset/2013answer.csv"
# answer_file = open("dataset/stackoverflow_2013_answer.csv", "w", newline='', encoding='UTF-8')
# with open(raw_file, 'r', encoding='UTF-8') as csv_file:
#     reader = csv.reader(csv_file)
#     csv_header_list = next(reader)
#     print(csv_header_list)
#     save_writer = csv.writer(answer_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
#     csv_header_list = csv_header_list[:2]
#
#     save_writer.writerow(csv_header_list)
#     for row in reader:
#         row = row[:2]
#         # convert np.datetime64 to relative time
#         row[1] = str((np.datetime64(row[1]) - ref_timestamp) / np.timedelta64(1, 's'))
#         save_writer.writerow(row)
#
# answer_file.close()
