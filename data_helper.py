# -*- coding: utf-8 -*-
# @File    : defect_classifier/data_helper.py
# @Info    : @ TSMC-SIGGRAPH, 2019/3/11
# @Desc    : 
# -.-.. - ... -- -.-. .-.. .- -... .---.   -.-- ..- .-.. --- -. --.   ..-. .- -. 


import random

import h5py
import numpy as np


def get_train_batch(filepath="dataset/train.hdf5", batch_size=20):
    dataset = h5py.File(filepath, "r")
    body_ndarray = dataset['body'].value
    tags_ndarray = dataset['tags'].value
    rate_ndarray = dataset['rate'].value
    time_ndarray = dataset['time'].value
    title_ndarray = dataset['title'].value
    week_ndarray = dataset['week'].value

    # delete over 4 days
    body_ndarray = body_ndarray[time_ndarray <= 4e5]
    tags_ndarray = tags_ndarray[time_ndarray <= 4e5]
    rate_ndarray = rate_ndarray[time_ndarray <= 4e5]
    title_ndarray = title_ndarray[time_ndarray <= 4e5]
    week_ndarray = week_ndarray[time_ndarray <= 4e5]
    time_ndarray = time_ndarray[time_ndarray <= 4e5]  # delete samples that time over 4e5, attention: must put on this

    n_batches = int(len(time_ndarray) / batch_size)
    print("[get_batches] batch_size: {}, n_batches: {}".format(batch_size, n_batches))

    # discarding not divisible part
    body_ndarray = body_ndarray[:batch_size * n_batches, ...]
    tags_ndarray = tags_ndarray[:batch_size * n_batches, ...]
    rate_ndarray = rate_ndarray[:batch_size * n_batches, ...]
    time_ndarray = time_ndarray[:batch_size * n_batches, ...]
    week_ndarray = week_ndarray[:batch_size * n_batches, ...]
    title_ndarray = title_ndarray[:batch_size * n_batches, ...]

    # shuffle data, `fancy indexing`, the info/label data is correctly connected to each set of features
    indexes = np.arange(time_ndarray.shape[0])
    random.shuffle(indexes)
    body_ndarray = np.take(body_ndarray, indexes, axis=0)
    tags_ndarray = np.take(tags_ndarray, indexes, axis=0)
    rate_ndarray = np.take(rate_ndarray, indexes, axis=0)
    time_ndarray = np.take(time_ndarray, indexes, axis=0)
    week_ndarray = np.take(week_ndarray, indexes, axis=0)
    title_ndarray = np.take(title_ndarray, indexes, axis=0)

    # time_ndarray[time_ndarray > 4e5] = 4e5        # clip time

    # time_max = np.max(time_ndarray)
    # time_min = np.min(time_ndarray)   # default min=0
    time_max = np.asarray([4e5])

    # time_ndarray = (time_ndarray - time_min) / (time_max - time_min)
    time_ndarray = time_ndarray / time_max    # (time_ndarray - 0) / (time_max - 0)
    print("time var:{}".format(np.var(time_ndarray)))
    # 6334731.664 0.0 100629.65152401557 211610680767.4005
    # print(np.max(time_ndarray),np.min(time_ndarray), np.mean(time_ndarray), np.var(time_ndarray))
    #
    # from matplotlib import pyplot as plt
    # x = np.arange(time_ndarray.shape[0])
    # fig = plt.figure()
    # ax1 = fig.add_subplot(111)
    # ax1.set_title('Scatter Plot')
    # plt.xlabel('X')
    # plt.ylabel('Y')
    # ax1.scatter(x, time_ndarray, c='r', marker='o')
    # plt.legend('x1')
    # plt.show()

    for n in range(0, time_ndarray.shape[0], batch_size):
        batch_body = body_ndarray[n:n + batch_size, ...]
        batch_tags = tags_ndarray[n:n + batch_size, ...]
        batch_time = time_ndarray[n:n + batch_size]
        batch_title = title_ndarray[n:n + batch_size, ...]
        batch_rate = rate_ndarray[n:n + batch_size]
        batch_week = week_ndarray[n:n + batch_size]
        yield batch_body, batch_tags, batch_title, batch_time, batch_rate, batch_week


if __name__ == "__main__":
    # test demo
    for batch_body, batch_tags, batch_title, batch_time, batch_rate, batch_week in get_train_batch("dataset/test.hdf5", batch_size=20):
        print(batch_body.shape)
        print(batch_tags.shape)
        print(batch_title.shape)
        print(batch_time.shape)
        print(batch_rate)
        print(batch_week)
        break
