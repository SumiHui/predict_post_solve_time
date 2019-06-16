# -*- coding: utf-8 -*-
# @File    : bug_fixed_prediction/build_hdf5_dataset.py
# @Info    : @ TSMC-SIGGRAPH, 2019/5/15
# @Desc    : 
# -.-.. - ... -- -.-. .-.. .- -... .---.   -.-- ..- .-.. --- -. --.   ..-. .- -. 

import os

import h5py
import numpy as np
import pandas as pd
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
from configuration import cfg


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


def cut_sentence(text):
    result = []
    for sentence in text:
        result.append(word_tokenize(sentence))
    return result


def get_embedding_vec(text, vector_size=10, checkpoint="ckpt/stackoverflow_doc2vec_model"):
    sentence = cut_sentence(text)
    documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(sentence)]

    if not os.path.exists(checkpoint):
        model = Doc2Vec(documents, vector_size=vector_size, window=3, min_count=1, workers=4)
        model.train(documents, epochs=20, total_examples=model.corpus_count)
        model.save(checkpoint)
    else:
        model = Doc2Vec.load(checkpoint)

    vec = [model.infer_vector(doc_words=tagdoc.words) for tagdoc in documents]
    ndarray = np.asarray(vec)
    return ndarray


if __name__ == '__main__':
    # Id,AcceptedAnswerId,CreationDate,Score,Body,Title,Tags, Timerate
    train_df, test_df = load_metadata('dataset/stackoverflow_2013.csv', 0.9)  # 9:1

    train_time = list(train_df["CreationDate"])
    train_body = list(train_df["Body"])
    train_title = list(train_df["Title"])
    train_tags = list(train_df["Tags"])
    train_rate = list(train_df["Timerate"])
    train_week = list(train_df["weekday"])

    test_time = list(test_df["CreationDate"])
    test_body = list(test_df["Body"])
    test_title = list(test_df["Title"])
    test_tags = list(test_df["Tags"])
    test_rate = list(test_df["Timerate"])
    test_week = list(test_df["weekday"])

    body = train_body + test_body
    title = train_title + test_title
    tags = train_tags + test_tags

    _ = get_embedding_vec(body, cfg.body_vec_size, "ckpt/so_body_doc2vec_model")
    _ = get_embedding_vec(title, cfg.title_vec_size, "ckpt/so_title_doc2vec_model")
    _ = get_embedding_vec(tags, cfg.tags_vec_size, "ckpt/so_tags_doc2vec_model")

    del body, title, tags

    body_ndarray = get_embedding_vec(train_body, cfg.body_vec_size, "ckpt/so_body_doc2vec_model")
    title_ndarray = get_embedding_vec(train_title, cfg.title_vec_size, "ckpt/so_title_doc2vec_model")
    tags_ndarray = get_embedding_vec(train_tags, cfg.tags_vec_size, "ckpt/so_tags_doc2vec_model")
    time_ndarray = np.asarray(train_time)
    rate_ndarray = np.asarray(train_rate)
    week_ndarray = np.asarray(train_week)

    f = h5py.File("dataset/train.hdf5", "w")
    _ = f.create_dataset("body", data=body_ndarray)
    _ = f.create_dataset("title", data=title_ndarray)
    _ = f.create_dataset("tags", data=tags_ndarray)
    _ = f.create_dataset("time", data=time_ndarray)
    _ = f.create_dataset("rate", data=rate_ndarray)
    _ = f.create_dataset("week", data=week_ndarray)

    f.close()

    del train_time, train_body, train_title, train_tags, train_rate
    del body_ndarray, title_ndarray, tags_ndarray, time_ndarray, rate_ndarray

    # todo: 测试集的构建使用训练集的doc2vec模型，并减除训练集的最小值来归一化
    body_ndarray = get_embedding_vec(test_body, cfg.body_vec_size, "ckpt/so_body_doc2vec_model")
    title_ndarray = get_embedding_vec(test_title, cfg.title_vec_size, "ckpt/so_title_doc2vec_model")
    tags_ndarray = get_embedding_vec(test_tags, cfg.tags_vec_size, "ckpt/so_tags_doc2vec_model")
    time_ndarray = np.asarray(test_time)
    rate_ndarray = np.asarray(test_rate)
    week_ndarray = np.asarray(test_week)

    f = h5py.File("dataset/test.hdf5", "w")
    _ = f.create_dataset("body", data=body_ndarray)
    _ = f.create_dataset("title", data=title_ndarray)
    _ = f.create_dataset("tags", data=tags_ndarray)
    _ = f.create_dataset("time", data=time_ndarray)
    _ = f.create_dataset("rate", data=rate_ndarray)
    _ = f.create_dataset("week", data=week_ndarray)

    f.close()
