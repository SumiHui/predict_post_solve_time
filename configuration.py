# -*- coding: utf-8 -*-
# @Info    : @ TSMC-SIGGRAPH, 2018/12/15
# @Desc    :
# -.-.. - ... -- -.-. .-.. .- -... .---.   -.-- ..- .-.. --- -. --.   ..-. .- -.

import tensorflow as tf


class ModelConfig(object):
    """Wrapper class for training hyper parameters."""

    def __init__(self):
        """Sets the default model hyper parameters."""
        # dropout
        # self.keep_prob = 0.5
        self.keep_prob = 0.8

        # Number of examples per epoch of training dataset.
        self.num_examples_per_epoch = 49200
        self.num_steps_per_epoch = 2460
        self.num_steps_per_decay = 49200  # (49200/batch_size)*20
        # # [load_metadata] process 5080 rows metadata in `dataset/stackoverflow_2013_java.csv`, where 4064 for train
        # self.num_examples_per_epoch = 3908  # there are 203.2 steps/epoch, [java, clip:4064, del: 3908]
        # self.num_steps_per_epoch = 195  # [java, clip:203, del: 195]
        # self.num_steps_per_decay = 1950  # (samples/batch_size)*10

        # Learning rate for the initial phase of training.
        self.initial_learning_rate = 0.01  # 0.01
        self.learning_rate_decay_factor = 0.1
        self.num_epochs_per_decay = 20.0

        # Output checkpoint and summary data directory.
        self.ckpt_dir = 'checkpoint'

        # How many model checkpoints to keep.
        self.max_checkpoints_to_keep = 1

        # also denotes the number steps per summary op
        self.num_steps_per_display = 200

        # batch
        self.batch_size = 20
        self.epoch = 60
        self.buffer_size = 1000

        # feature length, 用于训练的特征长度需要修改（譬如删除某一个特征）
        self.body_vec_size = 50
        self.title_vec_size = 20
        self.tags_vec_size = 5
        self.rate_vec_size = 1
        self.week_vec_size = 1
        # self.feature_len = self.body_vec_size + self.title_vec_size + self.tags_vec_size + self.rate_vec_size + self.week_vec_size
        self.feature_len = self.body_vec_size + self.title_vec_size + self.tags_vec_size
        # self.feature_len = self.body_vec_size + self.title_vec_size + self.rate_vec_size + self.week_vec_size
        # self.feature_len = self.title_vec_size + self.tags_vec_size + self.rate_vec_size + self.week_vec_size

        # clip grad
        self.grad_clip = 10

        # the output layer activation func
        self.hidden_act = tf.nn.relu    # default tf.nn.relu
        self.output_act = tf.nn.sigmoid  # tf.nn.relu


cfg = ModelConfig()

###############################
#    data set parameters      #
###############################
# 随数据集更换而容易发生改动的外部参数
FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string("data_basedir", "dataset",
                       "Train/Test data base directory. Default is `./dataset`.")
