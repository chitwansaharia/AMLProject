#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf

import sys
import os
from models import LSTMModel
from data import data_iter
from config import config
import pdb

flags = tf.flags
logging = tf.logging

flags.DEFINE_string("save_path", None,
                    "base save path for the experiment")
# flags.DEFINE_string("log_path",None,"Log Directory path")


FLAGS = flags.FLAGS

def main(_):
    model_config = config.config().base_model_config

    print("Config :")
    print(model_config)
    print("\n")

    save_path = os.path.join(FLAGS.save_path)
    # log_path = os.path.join(FLAGS.log_path)

    with tf.Graph().as_default():
        model = LSTMModel.LSTMModel(model_config)
        best_saver = tf.train.Saver()

        with tf.Session() as session:

            session.run(tf.global_variables_initializer())

            if model_config.load_mode == "best":
                best_saver.restore(
                    sess=session,
                    save_path=os.path.join(save_path, "best_model.ckpt"))


            i, patience = 0, 0
            best_valid_metric = 1e10

            while patience < model_config.patience:
                i += 1

                iterator_train = data_iter.SSIterator(model_config,mode = "train")
                iterator_valid = data_iter.SSIterator(model_config,mode = "valid")
                
                print("\nEpoch: %d" % (i))
                model.run_epoch(session, reader = iterator_train, is_training=True, verbose=True)

                valid_loss = model.run_epoch(session, reader = iterator_valid, verbose=True)

                if valid_loss < best_valid_metric:
                    best_valid_metric = valid_loss

                    print("\nsaving best model...")
                    best_saver.save(sess=session, save_path=os.path.join(save_path, "best_model.ckpt"))
                    patience = 0
                else:
                    patience += 1
                    print("\nLosing patience...")

if __name__ == "__main__":
    tf.app.run()
