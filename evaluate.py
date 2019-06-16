# -*- coding: utf-8 -*-
# @File    : defect_classifier/evaluate.py
# @Info    : @ TSMC-SIGGRAPH, 2019/3/12
# @Desc    : 
# -.-.. - ... -- -.-. .-.. .- -... .---.   -.-- ..- .-.. --- -. --.   ..-. .- -. 


import os

import numpy as np
import tensorflow as tf

from configuration import cfg
from data_helper import get_train_batch
from fixed_time_predictor import TimePredictor

os.environ["CUDA_VISIBLE_DEVICES"] = '0'


def path_exists(*directorylist):
    """make directory to save run log etc."""
    for directory in directorylist:
        if not os.path.exists(directory):
            os.makedirs(directory)
    return directorylist


def main(_):
    # build model
    defect_model = TimePredictor(mode="eval")
    defect_model.build()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver(max_to_keep=cfg.max_checkpoints_to_keep)

        if os.path.exists(os.path.join(cfg.ckpt_dir, "checkpoint")):
            model_file = tf.train.latest_checkpoint(cfg.ckpt_dir)
            saver.restore(sess, model_file)
            # saver.restore(sess, "model.ckpt")
        else:
            _ = path_exists(cfg.ckpt_dir)

        # testing loop
        mse = []
        for batch_body, batch_tags, batch_title, batch_time, batch_rate, batch_week in get_train_batch("dataset/test.hdf5",
                                                                                           cfg.batch_size):
            step = tf.train.global_step(sess, defect_model.global_step)
            batch_rate = np.expand_dims(batch_rate, -1)  # note: batch_tags is point to time_rate feature col
            batch_week = np.expand_dims(batch_week, -1)
            # train_batches = np.concatenate([batch_body, batch_title, batch_tags, batch_rate, batch_week], -1)
            # train_batches = np.concatenate([batch_body, batch_title, batch_tags], -1)
            # train_batches = np.concatenate([batch_body, batch_tags, batch_rate, batch_week], -1)
            train_batches = np.concatenate([batch_tags, batch_title, batch_rate, batch_week], -1)
            label_batches = np.expand_dims(batch_time, -1)
            feed_dict = {defect_model.input_data: train_batches, defect_model.labels: label_batches}

            loss = sess.run(defect_model.loss, feed_dict=feed_dict)
            print("[steps {}] loss: {}".format(step, loss))
            mse.append(loss)

        print(" ------ mean loss: {} ------ ".format(np.mean(np.array(mse))))


if __name__ == '__main__':
    tf.app.run()
