import os

import h5py
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import re


def load_metadata(file_path='dataset/stackoverflow_2013.csv', split_rate=0.8):
    """
    :param file_path: full path of csv file
    :param split_rate: Split the data into training and testing using split_rate/(1-split_rate) split
    :return:
    """
    stackoverflow_data = pd.read_csv(file_path)
    num_rows = stackoverflow_data.__len__()
    boundary = int(split_rate * num_rows)
    print("[load_metadata] process {} rows metadata in `{}`, where {} for train".format(num_rows, file_path, boundary))
    train_data, test_data = stackoverflow_data[:boundary], stackoverflow_data[boundary:]
    return train_data, test_data


def plot_distribution(tags_tl, title="test-top10",figsize=20):
    data = [t[1] for t in tags_tl]
    labels = [t[0] for t in tags_tl]

    plt.figure(figsize=(figsize, figsize))
    plt.bar(range(len(data)), data, tick_label=labels)
    plt.xticks(rotation=70)
    idx = list(range(len(labels)))
    for x, y in zip(idx, data):
        plt.text(x, y + 1, y)

    plt.title(title)

    plt.savefig("{}-tag.png".format(title))


if __name__ == '__main__':
    dr = re.compile(r'<(\w+)>', re.S)

    # Id,AcceptedAnswerId,CreationDate,Score,Body,Title,Tags, Timerate
    train_df, test_df = load_metadata('dataset/stackoverflow_2013.csv', 0.9)  # 9:1

    train_tags = list(train_df["Tags"])
    test_tags = list(test_df["Tags"])

    train_hashmap = {}
    for train_tag in train_tags:
        for tag in [tag.lower() for tag in train_tag.rstrip(">").lstrip("<").split("><")]:
            if tag in train_hashmap:
                train_hashmap[tag] += 1
            else:
                train_hashmap[tag] = 1
    train_tl = sorted(train_hashmap.items(), key=lambda d: d[1], reverse=True)

    plot_distribution(train_tl[:10], "train-top10", 9)
    plot_distribution(train_tl[:20], "train-top20", 15)
    plot_distribution(train_tl[:50], "train-top50", 21)

    del train_hashmap, train_tags, train_df, train_tl

    test_hashmap = {}
    for test_tag in test_tags:
        for tag in [tag.lower() for tag in test_tag.rstrip(">").lstrip("<").split("><")]:
            if tag in test_hashmap:
                test_hashmap[tag] += 1
            else:
                test_hashmap[tag] = 1
    test_tl = sorted(test_hashmap.items(), key=lambda d: d[1], reverse=True)

    plot_distribution(test_tl[:10], "test-top10",10)
    plot_distribution(test_tl[:20], "test-top20",15)
    plot_distribution(test_tl[:50], "test-top50",20)
