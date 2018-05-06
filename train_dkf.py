#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf

import sys
import os
from DKF import DKF
from data import data_iter
from config import config
import pdb
import csv 


flags = tf.flags
logging = tf.logging

flags.DEFINE_string("save_path", "./",
                    "base save path for the experiment")
flags.DEFINE_string("mode","train","mode in which model needs to run")
# flags.DEFINE_string("log_path",None,"Log Directory path")


FLAGS = flags.FLAGS

def main(_):
    model_config = config.config().dkf_model_config

    print("Configuration DKF:")
    
    import pprint
    pprint.PrettyPrinter().pprint(model_config)

    dkf_config = {
        "batch_size" : model_config.batch_size,
        "time_len" : model_config.max_time_steps,

        "x_size" : model_config.output_size,
        "u_size" : model_config.input_size,
        "z_size" : model_config.latent_state_size,

        "num_hidden_layers" : model_config.num_hidden_layers,
        "num_hidden_units" : model_config.num_hidden_units,
        "keep_prob" : model_config.keep_prob,
    
        "z1_prior_mean" : tf.zeros(shape=(model_config.latent_state_size)),
        "z1_prior_covar" : tf.eye(model_config.latent_state_size),

        "n_samples_term_1" : model_config.nsamples_e1,
        "n_samples_term_3" : model_config.nsamples_e3,
        "lsm_time" : model_config.lsm_time,

        "learning_rate" : model_config.learning_rate,
        "logfolder": model_config.logfolder
    }

    save_path = os.path.join(FLAGS.save_path)
    # log_path = os.path.join(FLAGS.log_path)

    model = DKF(config=dkf_config, device="gpu")
    best_saver = tf.train.Saver()

    with tf.Session() as session:

        session.run(tf.global_variables_initializer())

        if FLAGS.mode == "test":
            iterator_test = data_iter.SSIterator(model_config,mode = "test")
            best_saver.restore(
                sess=session,
                save_path=os.path.join(save_path, "best_model_dkf.ckpt"))
            final_outputs = model.test(session, reader = iterator_test)
            with open("output.csv", "wb") as f:
                writer = csv.writer(f)
                writer.writerows(final_outputs)


        else:
            if model_config.load_mode == "best":
                 best_saver.restore(
                    sess=session,
                    save_path=os.path.join(save_path, "best_model_dkf.ckpt"))

            i, patience = 0, 0
            best_valid_metric = 1e10

                
            while patience < model_config.patience:
                i += 1

                iterator_train = data_iter.SSIterator(model_config, mode = "train")
                iterator_valid = data_iter.SSIterator(model_config, mode = "valid")

                
                print("\nEpoch: %d" % (i))
                model.run_epoch(session, reader=iterator_train, validate=False, verbose=True)

                valid_loss = model.run_epoch(session, reader=iterator_valid, validate=True, verbose=True)

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
