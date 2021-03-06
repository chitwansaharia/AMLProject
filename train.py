#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf

import sys
import os
from models import LSTMModel
from models import LSTMModelwithTF

from data import data_iter
from config import config
import pdb
import csv 

flags = tf.flags
logging = tf.logging

flags.DEFINE_string("save_path", None,
                    "base save path for the experiment")
flags.DEFINE_string("mode","train","mode in which model needs to run")
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
        model = LSTMModelwithTF.LSTMModel(model_config)
        best_saver = tf.train.Saver()

        with tf.Session() as session:

            session.run(tf.global_variables_initializer())

            if FLAGS.mode == "test":
                iterator_test = data_iter.SSIterator(model_config,mode = "test")
                best_saver.restore(
                    sess=session,
                    save_path=os.path.join(save_path, "best_model.ckpt"))
                final_outputs = model.test(session, reader = iterator_test)
                with open("output.csv", "wb") as f:
                    writer = csv.writer(f)
                    writer.writerows(final_outputs)


            else:
                if model_config.load_mode == "best":
                     best_saver.restore(
                        sess=session,
                        save_path=os.path.join(save_path, "best_model.ckpt"))

                i, patience = 0, 0
                best_valid_metric = 1e10

                    
                while patience < model_config.patience:
                    
                    i += 1
                    print("\nEpoch: %d" % (i))

                    iterator_train = data_iter.SSIterator(model_config,mode = "train")
                    iterator_valid = data_iter.SSIterator(model_config,mode = "valid")
                    print("Calculating Normal Loss")

                    
                    train_loss = model.run_epoch(session, reader = iterator_train, is_training=True, verbose=True)
                    print(train_loss[0]/train_loss[1])

                    valid_loss = model.run_epoch(session, reader = iterator_valid, verbose=True)
                    print(valid_loss[0]/valid_loss[1])

                    print("Calculating RMS")
                    valid_rms = model.calc_rms(session, reader = iterator_valid)
                    print(valid_rms)

                    if valid_loss[0] < best_valid_metric:
                        best_valid_metric = valid_loss[0]

                        print("\nsaving best model...")
                        best_saver.save(sess=session, save_path=os.path.join(save_path, "best_model.ckpt"))
                        patience = 0
                    else:
                        patience += 1
                        print("\nLosing patience...")


if __name__ == "__main__":
    tf.app.run()
