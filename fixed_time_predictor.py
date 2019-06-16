# -*- coding: utf-8 -*-
# @File    : bug_fixed_prediction/fixed_time_predictor.py
# @Info    : @ TSMC-SIGGRAPH, 2019/5/16
# @Desc    : 
# -.-.. - ... -- -.-. .-.. .- -... .---.   -.-- ..- .-.. --- -. --.   ..-. .- -. 

import tensorflow as tf
from tensorflow.python.ops import clip_ops
from configuration import cfg


class TimePredictor(object):
    def __init__(self, mode):
        assert mode in ["train", "eval", "inference"]
        self.mode = mode
        self.initializer = tf.initializers.variance_scaling(scale=1.0, mode="fan_in")

        self.input_data = None
        self.labels = None

        self.prediction = None

        # loss and optimizer
        self.loss = None
        self.train_op = None

        # Global step tensor.
        self.global_step = None

    def dropout_prob(self):
        if self.mode == "train":
            return cfg.keep_prob
        else:
            return 1.0

    def build_inputs(self):
        self.labels = tf.placeholder(tf.int32, shape=(None, 1), name='labels')
        self.input_data = tf.placeholder(tf.float32, shape=(None, cfg.feature_len), name='post_features')

    def build_model(self):
        with tf.variable_scope("predictor", initializer=self.initializer):
            # note: self.input_data shape is (batch, time_steps, feature_len), in which, feature dim is `b,ir,or`
            print("[build_embedding_layer] input_data shape {}".format(self.input_data.get_shape()))
            # input_embedding = tf.layers.dense(inputs=self.input_data, units=128, activation=cfg.hidden_act)
            #
            # input_dropout = tf.layers.dropout(input_embedding, rate=self.dropout_prob())
            # layer1 = tf.layers.dense(input_dropout, 256, activation=cfg.hidden_act)
            # layer1_dropout = tf.layers.dropout(layer1, rate=self.dropout_prob())
            # layer2 = tf.layers.dense(layer1_dropout, 512, activation=cfg.hidden_act)
            # layer2_dropout = tf.layers.dropout(layer2, rate=self.dropout_prob())
            # layer3 = tf.layers.dense(layer2_dropout, 128, activation=cfg.hidden_act)
            # layer3_dropout = tf.layers.dropout(layer3, rate=self.dropout_prob())

            input_embedding = tf.layers.dense(inputs=self.input_data, units=100, activation=cfg.hidden_act)

            input_dropout = tf.layers.dropout(input_embedding, rate=self.dropout_prob())
            layer1 = tf.layers.dense(input_dropout, 200, activation=cfg.hidden_act)
            layer1_dropout = tf.layers.dropout(layer1, rate=self.dropout_prob())
            layer2 = tf.layers.dense(layer1_dropout, 100, activation=cfg.hidden_act)
            layer3_dropout = tf.layers.dropout(layer2, rate=self.dropout_prob())

            # softmax layer return prob distribution
            self.prediction = tf.layers.dense(layer3_dropout, 1, activation=cfg.output_act, name="predictions")

    # loss_layer
    def build_loss(self):
        loss_var = 0.001 * tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
        self.loss = tf.losses.mean_squared_error(self.labels, self.prediction) + loss_var

        # Add summaries.
        tf.summary.scalar("losses/total", self.loss)
        tf.summary.scalar("losses/L2", loss_var)

    def setup_global_step(self):
        """Sets up the global step tensor."""
        global_step = tf.Variable(initial_value=0, trainable=False, name="global_step",
                                  collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES])
        self.global_step = global_step

    #   optimizer
    def build_optimizer(self):
        # lr = tf.train.exponential_decay(cfg.initial_learning_rate,
        #                                 self.global_step,
        #                                 cfg.num_steps_per_decay,
        #                                 cfg.learning_rate_decay_factor,
        #                                 staircase=True)
        # tf.summary.scalar("learning_rate", lr)
        #
        # # using clipping gradients
        # tvars = tf.trainable_variables()
        #
        # grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), cfg.grad_clip)
        # # optimizer = tf.train.AdamOptimizer(cfg.initial_learning_rate)
        # optimizer = tf.train.MomentumOptimizer(cfg.initial_learning_rate, momentum=0.9)
        # self.train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step)
        #
        # # Compute gradients.
        # gradients = optimizer.compute_gradients(loss=self.loss, var_list=tvars)
        # tf.summary.scalar("global_norm/gradient_norm", clip_ops.global_norm(list(zip(*gradients))[0]))

        optimizer = tf.train.AdamOptimizer(cfg.initial_learning_rate)
        # optimizer = tf.train.MomentumOptimizer(cfg.initial_learning_rate, momentum=0.9)
        self.train_op = optimizer.minimize(self.loss, self.global_step)

    def build(self):
        """Creates all ops for training and evaluation."""
        self.build_inputs()
        self.build_model()

        self.setup_global_step()

        self.build_loss()
        self.build_optimizer()
