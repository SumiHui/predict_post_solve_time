# -*- coding: utf-8 -*-
# @File    : bug_classifier/train.py
# @Info    : @ TSMC-SIGGRAPH, 2018/12/16
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
    defect_model = TimePredictor(mode="train")
    defect_model.build()

    summary_op = tf.summary.merge_all()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver(max_to_keep=cfg.max_checkpoints_to_keep)

        if os.path.exists(os.path.join(cfg.ckpt_dir, "checkpoint")):
            model_file = tf.train.latest_checkpoint(cfg.ckpt_dir)
            saver.restore(sess, model_file)
        else:
            _ = path_exists(cfg.ckpt_dir)

        summary_writer = tf.summary.FileWriter(cfg.ckpt_dir, sess.graph)

        # training loop
        for epoch in range(cfg.epoch):
            # iterate the whole dataset n epochs
            print("iterate the whole dataset {} epochs".format(cfg.epoch))
            # batch_time 是标签，batch_rate, batch_week是一起出现或删除的
            for batch_body, batch_tags, batch_title, batch_time, batch_rate, batch_week in get_train_batch("dataset/train.hdf5",
                                                                                               cfg.batch_size):
                step = tf.train.global_step(sess, defect_model.global_step)
                batch_rate = np.expand_dims(batch_rate, -1)
                batch_week = np.expand_dims(batch_week, -1)
                # train_batches = np.concatenate([batch_body, batch_title, batch_tags, batch_rate, batch_week], -1)
                train_batches = np.concatenate([batch_body, batch_title, batch_tags], -1)
                # train_batches = np.concatenate([batch_body, batch_tags, batch_rate, batch_week], -1)
                # train_batches = np.concatenate([batch_title, batch_tags, batch_rate, batch_week], -1)
                label_batches = np.expand_dims(batch_time, -1)
                feed_dict = {defect_model.input_data: train_batches, defect_model.labels: label_batches}
                if step % cfg.num_steps_per_display == 0:
                    _, loss, summary = sess.run([defect_model.train_op, defect_model.loss, summary_op],
                                                feed_dict=feed_dict)
                    print("[epoch {}, steps {}] loss: {}".format(epoch, step, loss))
                    summary_writer.add_summary(summary, step)
                else:
                    sess.run(defect_model.train_op, feed_dict=feed_dict)
            saver.save(sess, os.path.join(cfg.ckpt_dir, 'model.epoch'), epoch)
        saver.save(sess, os.path.join(cfg.ckpt_dir, 'model.final-%d' % cfg.epoch))
        print(" ------ Training process finish! Arriving at the end of data ------ ")


if __name__ == '__main__':
    tf.app.run()
