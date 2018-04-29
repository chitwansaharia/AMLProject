from __future__ import division
from __future__ import print_function

import time
import numpy as np
import tensorflow as tf
import pdb
from tools import my_lib


class LSTMModel(object):

	def __init__(self, config = None, scope_name=None, device='gpu'):
		self.config = config
		self.scope = scope_name or "LSTMModel"
		self.refresh = 0

		self.create_placeholders()
		self.global_step = \
			tf.contrib.framework.get_or_create_global_step()

		self.metrics = {}
		if device == 'gpu':
			tf.device('/gpu:0')
		else:
			tf.device('/cpu:0')

		with tf.variable_scope(self.scope):
			self.build_model()
			self.compute_loss_and_metrics()
			self.compute_gradients_and_train_op()

	def create_placeholders(self):
		input_size = self.config.input_size
		self.max_time_steps = self.config.max_time_steps
		self.batch_size = self.config.batch_size
		
		# input_data placeholders
		self.inputs = tf.placeholder(
			tf.float32, shape=[self.batch_size,self.max_time_steps,input_size], name="inputs")
		self.targets = tf.placeholder(
			tf.float32, shape=[self.batch_size,self.max_time_steps,2], name="targets")
		self.mask = tf.placeholder(
			tf.float32,shape=[self.batch_size,self.max_time_steps],name = "mask")

		self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")
		self.phase_train = tf.placeholder(tf.bool, name="phase_train")

	def process_inputs(self,inputs):
		output = [tf.nn.embedding_lookup(self.store_type_embedding,tf.cast(inputs[:,:,0],dtype= tf.int32))]
		output.append(tf.nn.embedding_lookup(self.assortment_type_embedding,tf.cast(inputs[:,:,1],dtype= tf.int32)))
		temp = tf.reshape(inputs[:,:,2:3],[-1,1])
		output.append(tf.reshape(tf.add(tf.matmul(temp, self.comp_dist_weights),
								  self.comp_dist_bias), [self.batch_size, self.max_time_steps, -1]))
		output.append(tf.nn.embedding_lookup(self.competition_open_since_month_embedding,tf.cast(inputs[:,:,3],dtype= tf.int32)))
		temp = tf.reshape(inputs[:, :, 4:5], [-1, 1])
		output.append(tf.reshape(tf.add(tf.matmul(temp, self.comp_open_since_year_weights),
								  self.comp_open_since_year_bias), [self.batch_size, self.max_time_steps, -1]))
		output.append(tf.nn.embedding_lookup(self.promo2_embedding,tf.cast(inputs[:,:,5],dtype= tf.int32)))
		temp = tf.reshape(inputs[:, :, 6:7], [-1, 1])
		output.append(tf.reshape(tf.add(tf.matmul(temp, self.promo2_week_weights),
								  self.promo2_week_bias), [self.batch_size, self.max_time_steps, -1]))
		output.append(tf.nn.embedding_lookup(self.promo2_since_year_embedding,tf.cast(inputs[:,:,7],dtype= tf.int32)))
		temp = tf.reshape(inputs[:,:,8:20],[-1,12])
		output.append(tf.reshape(tf.add(tf.matmul(temp,self.promo_interval_weights),self.promo_interval_bias),[self.batch_size,self.max_time_steps,-1]))
		output.append(tf.nn.embedding_lookup(self.day_of_week_embedding,tf.cast(inputs[:,:,20],dtype= tf.int32)))
		output.append(tf.nn.embedding_lookup(self.year_embedding,tf.cast(inputs[:,:,21],dtype= tf.int32)))
		output.append(tf.nn.embedding_lookup(self.month_embedding,tf.cast(inputs[:,:,22],dtype= tf.int32)))
		output.append(tf.nn.embedding_lookup(self.day_embedding,tf.cast(inputs[:,:,23],dtype= tf.int32)))
		output.append(tf.nn.embedding_lookup(self.open_embedding,tf.cast(inputs[:,:,24],dtype= tf.int32)))
		output.append(tf.nn.embedding_lookup(self.promo_embedding,tf.cast(inputs[:,:,25],dtype= tf.int32)))
		output.append(tf.nn.embedding_lookup(self.state_holiday_embedding,tf.cast(inputs[:,:,26],dtype= tf.int32)))
		output.append(tf.nn.embedding_lookup(self.school_holiday_embedding,tf.cast(inputs[:,:,27],dtype= tf.int32)))
		return tf.concat(output,axis = 2)


	def build_model(self):
		config = self.config
		lstm_units = config.lstm_units
		input_size = config.input_size
		num_hidden_layers = config.num_hidden_layers

		rand_uni_initializer = \
			tf.random_uniform_initializer(
				-self.config.init_scale,self.config.init_scale)


		# Embeddings

		with tf.variable_scope("embeddings"):
			self.store_type_embedding = tf.get_variable("store_type_embedding",[4,100])
			self.assortment_type_embedding = tf.get_variable("assortment_type_embedding",[3,100])
			self.competition_open_since_month_embedding = tf.get_variable("competition_open_since_month_embedding",[13,200])
			self.promo2_embedding = tf.get_variable("promo2_embedding",[2,70])
			self.promo2_since_year_embedding= tf.get_variable("promo2_since_year_embedding",[8,100])
			self.promo_interval_weights = tf.get_variable("promo_interval_weights",[12,200],dtype=tf.float32)
			self.promo_interval_bias = tf.get_variable("promo_interval_bias",[200],dtype=tf.float32)
			
			self.comp_dist_weights = tf.get_variable("comp_dist_weights",[1,100],dtype=tf.float32)
			self.comp_dist_bias = tf.get_variable("comp_dist_bias",[100],dtype=tf.float32)

			self.comp_open_since_year_weights = tf.get_variable(
				"compe_open_since_year_weights", [1, 100], dtype=tf.float32)
			self.comp_open_since_year_bias = tf.get_variable(
				"compe_open_since_year_bias", [100], dtype=tf.float32)

			self.promo2_week_weights = tf.get_variable(
				"promo2_week_weights", [1, 100], dtype=tf.float32)
			self.promo2_week_bias = tf.get_variable(
				"promo2_week_bias", [100], dtype=tf.float32)
			
			self.day_of_week_embedding = tf.get_variable("day_of_week_embedding",[7,200])
			self.year_embedding = tf.get_variable("year_embedding",[3,100])
			self.month_embedding = tf.get_variable("month_embedding",[12,200])
			self.day_embedding = tf.get_variable("day_embedding",[31,200])
			self.open_embedding = tf.get_variable("open_embedding",[2,70])
			self.promo_embedding = tf.get_variable("promo_embedding",[2,70])
			self.state_holiday_embedding = tf.get_variable("state_holiday_embedding",[4,100])
			self.school_holiday_embedding = tf.get_variable("school_holiday_embedding",[2,70])


		processed_inputs = self.process_inputs(self.inputs)

		def rnn_cell():
			return tf.contrib.rnn.DropoutWrapper(
				tf.contrib.rnn.BasicLSTMCell(num_units=lstm_units),
				output_keep_prob=self.keep_prob,
				variational_recurrent=True,
				dtype=tf.float32)

		cells = tf.contrib.rnn.MultiRNNCell([rnn_cell() for _ in range(num_hidden_layers)])



		state = self.initial_state =  cells.zero_state(self.batch_size, dtype=tf.float32)

		
		# outputs,state = tf.nn.dynamic_rnn(cells,processed_inputs,initial_state=rnn_initial_state,dtype=tf.float32)
		
		outputs = []
		with tf.variable_scope("lstm", initializer=rand_uni_initializer):
			for time_step in range(self.max_time_steps):
				if time_step > 0:
					tf.get_variable_scope().reuse_variables()
				(cell_output, state) = cells(processed_inputs[:,time_step,:], state)
				outputs.append(cell_output)

		self.metrics["final_state"] = state

		full_conn_layers = [tf.reshape(tf.concat(axis=1, values=outputs), [-1, lstm_units])]
		with tf.variable_scope("output_layer"):
			self.model_outputs = tf.contrib.layers.fully_connected(
					inputs=full_conn_layers[-1],
					num_outputs=2,
					activation_fn=None,
					weights_initializer=rand_uni_initializer,
					biases_initializer=rand_uni_initializer,
					trainable=True)


	def compute_loss_and_metrics(self):
		temp = tf.multiply(tf.reshape(self.model_outputs,[self.batch_size,self.max_time_steps,-1]),tf.expand_dims(self.mask,-1))

		self.metrics["entropy_loss"] = tf.divide(tf.nn.l2_loss(tf.reshape(temp,[-1,2]) - tf.reshape(self.targets,[-1,2])),tf.reduce_sum(self.mask))


	def compute_gradients_and_train_op(self):
		tvars = self.tvars = my_lib.get_scope_var(self.scope)
		my_lib.get_num_params(tvars)
		grads = tf.gradients(self.metrics["entropy_loss"], tvars)
		grads, _ = tf.clip_by_global_norm(grads, self.config.max_grad_norm)

		self.metrics["grad_sum"] = tf.add_n([tf.reduce_sum(g) for g in grads])
		learning_rate = tf.train.exponential_decay(self.config.learning_rate, self.global_step,
										   10000, 0.96, staircase=True)
		optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
		self.train_op = optimizer.apply_gradients(
			zip(grads, tvars),
			global_step=self.global_step)

	def model_vars(self):
		return self.tvars

	def init_feed_dict(self):
		return {self.phase_train.name: True}


	def run_epoch(self, session,reader, is_training=False, verbose=False):
		start_time = time.time()
		epoch_metrics = {}
		keep_prob = self.config.keep_prob
		fetches = {
			"entropy_loss": self.metrics["entropy_loss"],
			"grad_sum": self.metrics["grad_sum"],
			"final_state": self.metrics["final_state"]
		}
		if is_training:
			if verbose:
				print("\nTraining...")
			fetches["train_op"] = self.train_op
			keep_prob = self.config.keep_prob
			phase_train = True
		else:
			phase_train = False
			if verbose:
				print("\nEvaluating...")


		state = session.run(self.initial_state)


		i, total_loss, grad_sum, data_processed = 0, 0.0, 0.0, 0.0

		reader.start()
		total_data_points = 916859

		i = 0
		batch = reader.next()
		while batch != None:        
			feed_dict = {}
			feed_dict[self.targets.name] = batch["outputs"]
			feed_dict[self.inputs.name] = batch["inputs"]
			feed_dict[self.keep_prob.name] = keep_prob
			feed_dict[self.phase_train.name] = phase_train
			feed_dict[self.mask.name] = batch["mask"]
			
			feed_dict[self.initial_state] = state

			vals = session.run(fetches, feed_dict)

			if batch["refresh"] == 1:
				state = session.run(self.initial_state)

			else:
				state = vals["final_state"] 


			total_loss += vals["entropy_loss"]

			grad_sum += vals["grad_sum"]

			data_processed += np.sum(batch["mask"])



			i += 1
			
			if verbose:
				print(
					"% Iter Done :", round(data_processed*100/total_data_points, 1),
					"Loss :", round(vals["entropy_loss"],3),
					"Gradient :", round(vals["grad_sum"],3))

			batch = reader.next()

		return total_loss

	def test(self,session,reader):
		keep_prob = self.config.keep_prob
		final_outputs = []
		fetches = {
			"outputs" : self.model_outputs,
			"final_state" : self.metrics["final_state"]
		}
		phase_train = False
		state = session.run(self.initial_state)

		reader.start()

		i = 0
		batch = reader.next()
		while batch != None:        
			feed_dict = {}
			feed_dict[self.inputs.name] = batch["inputs"]
			feed_dict[self.keep_prob.name] = keep_prob
			feed_dict[self.phase_train.name] = phase_train
			
			
			feed_dict[self.initial_state] = state

			vals = session.run(fetches, feed_dict)

			if batch["refresh"] == 1:
				state = session.run(self.initial_state)

			else:
				state = vals["final_state"] 
			reshape_outputs = np.reshape(vals["outputs"],[self.batch_size,self.max_time_steps,-1])

			for i in range(len(batch["stores"])):
				for j in range(self.max_time_steps):
					if batch['mask'][i][j] == 1.0:
						temp = [batch["stores"][i]]
						temp.extend(list(batch["inputs"][i][j][20:]))
						temp.extend([reshape_outputs[i][j][0]])
						final_outputs.append(temp)
			batch = reader.next()

		return final_outputs

