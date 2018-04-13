from config import config
from data import data_iter
from models import LSTMModel
import tensorflow as tf 


tf.reset_default_graph()

new_config = config.config().base_model_config
reader = data_iter.SSIterator(new_config)

model = LSTMModel.LSTMModel(config = new_config)

with tf.Session() as sess:
	model.run_epoch(sess,reader,True,True)