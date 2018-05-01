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
		self.output_size = self.config.output_size
		self.forced_size = self.config.forced_size
		self.lstm_units = self.config.lstm_units
		
		# input_data placeholders
		self.inputs = tf.placeholder(
			tf.float32, shape=[self.batch_size,self.max_time_steps,input_size], name="inputs")
		self.targets = tf.placeholder(
			tf.float32, shape=[self.batch_size,self.max_time_steps,self.output_size], name="targets")
		self.mask = tf.placeholder(
			tf.float32,shape=[self.batch_size,self.max_time_steps],name = "mask")

		self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")
		self.phase_train = tf.placeholder(tf.bool, name="phase_train")
		self.phase_test = tf.placeholder(tf.bool, name="phase_test")
		self.sample_weight = tf.placeholder(tf.float32,shape=[], name="sample_weight")


		self.initial_force = tf.placeholder(tf.float32,shape=[self.batch_size,self.output_size],name="initial_force")


	def process_inputs(self,inputs):
		output = [tf.nn.embedding_lookup(self.store_type_embedding,tf.cast(inputs[:,:,0],dtype= tf.int32))]
		output.append(tf.nn.embedding_lookup(self.assortment_type_embedding,tf.cast(inputs[:,:,1],dtype= tf.int32)))
		temp = tf.reshape(inputs[:,:,2:3],[-1,1])
		output.append(tf.reshape(tf.add(tf.matmul(temp, self.comp_dist_weights),
								  self.comp_dist_bias), [self.batch_size, self.max_time_steps, -1]))
		output.append(tf.nn.embedding_lookup(self.competition_open_since_month_embedding,tf.cast(inputs[:,:,3],dtype= tf.int32)))
		output.append(tf.nn.embedding_lookup(self.comp_year_embedding,tf.cast(inputs[:,:,4],dtype= tf.int32)))

		output.append(tf.nn.embedding_lookup(self.promo2_embedding,tf.cast(inputs[:,:,5],dtype= tf.int32)))
		output.append(tf.nn.embedding_lookup(self.promo2_week_embedding,
                                       tf.cast(inputs[:, :, 6], dtype=tf.int32)))

		output.append(tf.nn.embedding_lookup(self.promo2_since_year_embedding,tf.cast(inputs[:,:,7],dtype= tf.int32)))
		output.append(tf.nn.embedding_lookup(
			self.promo_interval_embedding, tf.cast(inputs[:, :, 8], dtype=tf.int32)))
		output.append(tf.nn.embedding_lookup(self.day_of_week_embedding,tf.cast(inputs[:,:,9],dtype= tf.int32)))
		output.append(tf.nn.embedding_lookup(self.year_embedding,tf.cast(inputs[:,:,10],dtype= tf.int32)))
		output.append(tf.nn.embedding_lookup(self.month_embedding,tf.cast(inputs[:,:,11],dtype= tf.int32)))
		output.append(tf.nn.embedding_lookup(self.day_embedding,tf.cast(inputs[:,:,12],dtype= tf.int32)))
		output.append(tf.nn.embedding_lookup(self.open_embedding,tf.cast(inputs[:,:,13],dtype= tf.int32)))
		output.append(tf.nn.embedding_lookup(self.promo_embedding,tf.cast(inputs[:,:,14],dtype= tf.int32)))
		output.append(tf.nn.embedding_lookup(self.state_holiday_embedding,tf.cast(inputs[:,:,15],dtype= tf.int32)))
		output.append(tf.nn.embedding_lookup(self.school_holiday_embedding,tf.cast(inputs[:,:,16],dtype= tf.int32)))
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
			self.store_type_embedding = tf.get_variable("store_type_embedding",[4,4])
			self.assortment_type_embedding = tf.get_variable("assortment_type_embedding",[3,3])
			self.competition_open_since_month_embedding = tf.get_variable("competition_open_since_month_embedding",[13,13])
			self.promo2_embedding = tf.get_variable("promo2_embedding",[2,2])
			self.promo2_since_year_embedding= tf.get_variable("promo2_since_year_embedding",[8,7])
			self.promo_interval_embedding = tf.get_variable("promo_interval_embedding",[4,4],dtype=tf.float32)
			
			self.comp_dist_weights = tf.get_variable("comp_dist_weights",[1,2],dtype=tf.float32)
			self.comp_dist_bias = tf.get_variable("comp_dist_bias",[2],dtype=tf.float32)

			self.comp_year_embedding =  tf.get_variable("comp_year_embedding",[29,20],	dtype=tf.float32)
			self.promo2_week_embedding = tf.get_variable("promo2_week_embedding",[14,10])
			
			self.day_of_week_embedding = tf.get_variable("day_of_week_embedding",[7,7])
			self.year_embedding = tf.get_variable("year_embedding",[3,3])
			self.month_embedding = tf.get_variable("month_embedding",[12,12])
			self.day_embedding = tf.get_variable("day_embedding",[31,20])
			self.open_embedding = tf.get_variable("open_embedding",[2,2])
			self.promo_embedding = tf.get_variable("promo_embedding",[2,2])
			self.state_holiday_embedding = tf.get_variable("state_holiday_embedding",[4,4])
			self.school_holiday_embedding = tf.get_variable("school_holiday_embedding",[2,2])


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
		forced_input = self.initial_force		
		outputs = []
		with tf.variable_scope("lstm", initializer=rand_uni_initializer):
			for time_step in range(self.max_time_steps):
				if time_step > 0:
					tf.get_variable_scope().reuse_variables()

					forced_input = tf.cond(self.phase_test,
											lambda: output,
											lambda: tf.scalar_mul(self.sample_weight,self.targets[:,time_step-1,:]) + tf.scalar_mul(tf.constant(1,dtype=tf.float32)-self.sample_weight,output))

				forced_output = tf.contrib.layers.fully_connected(
					inputs=forced_input,
					num_outputs=self.forced_size,
					activation_fn=None,
					weights_initializer=rand_uni_initializer,
					biases_initializer=rand_uni_initializer,
					trainable=True,
					reuse=tf.AUTO_REUSE,
					scope="forced_output_layer")

				present_inputs = tf.concat([processed_inputs[:,time_step,:],forced_output],axis=1)
				
				(cell_output, state) = cells(present_inputs, state)
				
				output = tf.contrib.layers.fully_connected(
					inputs=cell_output,
					num_outputs=self.output_size,
					activation_fn=None,
					weights_initializer=rand_uni_initializer,
					biases_initializer=rand_uni_initializer,
					trainable=True,
					reuse=tf.AUTO_REUSE,
					scope="output_layer")

				outputs.append(output)
		self.final_output = output

		self.metrics["final_state"] = state

		# full_conn_layers = [tf.reshape(tf.concat(axis=1, values=outputs), [-1, lstm_units])]
		# with tf.variable_scope("output_layer"):
		# 	self.model_outputs = tf.contrib.layers.fully_connected(
		# 			inputs=full_conn_layers[-1],
		# 			num_outputs=self.output_size,
		# 			activation_fn=None,
		# 			weights_initializer=rand_uni_initializer,
		# 			biases_initializer=rand_uni_initializer,
		# 			trainable=True)
		self.model_outputs = tf.stack(outputs,axis=1)

	def compute_loss_and_metrics(self):
		temp = tf.multiply(tf.reshape(self.model_outputs,[self.batch_size,self.max_time_steps,-1]),tf.expand_dims(self.mask,-1))

		self.metrics["entropy_loss"] = tf.divide(tf.nn.l2_loss(tf.reshape(temp,[-1,self.output_size]) - tf.reshape(self.targets,[-1,self.output_size])),tf.reduce_sum(self.mask)) 

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
		keep_prob = 1
		phase_test = False
		fetches = {
			"entropy_loss": self.metrics["entropy_loss"],
			"grad_sum": self.metrics["grad_sum"],
			"final_state": self.metrics["final_state"],
			"final_output": self.final_output
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
			phase_test = True


		state = session.run(self.initial_state)


		i, total_loss, grad_sum, data_processed = 0, 0.0, 0.0, 0.0

		reader.start()
		total_data_points = 916859

		i = 0
		batch = reader.next()
		feed_to_initial = np.zeros((self.batch_size,self.output_size))
		while batch != None:        
			feed_dict = {}
			feed_dict[self.targets.name] = batch["outputs"]
			feed_dict[self.inputs.name] = batch["inputs"]
			feed_dict[self.keep_prob.name] = keep_prob
			feed_dict[self.phase_train.name] = phase_train
			feed_dict[self.mask.name] = batch["mask"]
			feed_dict[self.initial_force.name] = feed_to_initial
			feed_dict[self.phase_test.name] = phase_test
			feed_dict[self.sample_weight.name] = 0.4


			
			feed_dict[self.initial_state] = state

			vals = session.run(fetches, feed_dict)

			if batch["refresh"] == 1:
				state = session.run(self.initial_state)
				feed_to_initial = np.zeros((self.batch_size,self.output_size))

			else:
				state = vals["final_state"] 
				feed_to_initial = batch["outputs"][:,-1,:]


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

		return (total_loss,i)

	def test(self,session,reader):
		keep_prob = 1
		final_outputs = []
		fetches = {
			"outputs" : self.model_outputs,
			"final_state" : self.metrics["final_state"],
			"final_output" : self.final_output
			
		}
		phase_train = False
		state = session.run(self.initial_state)
		reader.start()

		i = 0
		batch = reader.next()
		feed_to_initial = batch["init_feed"]
		while batch != None:        
			feed_dict = {}
			feed_dict[self.inputs.name] = batch["inputs"]
			feed_dict[self.keep_prob.name] = keep_prob
			feed_dict[self.phase_train.name] = phase_train
			feed_dict[self.initial_force] = feed_to_initial
			feed_dict[self.phase_test] = True
			feed_dict[self.sample_weight.name] = 0.5
			feed_dict[self.targets.name] = np.zeros((self.batch_size,self.max_time_steps,self.output_size))			
			feed_dict[self.initial_state] = state


			vals = session.run(fetches, feed_dict)

			if batch["refresh"] == 1:
				state = session.run(self.initial_state)
			else:
				state = vals["final_state"] 
				feed_to_initial = vals["final_output"]




			reshape_outputs = np.reshape(vals["outputs"],[self.batch_size,self.max_time_steps,-1])

			for i in range(len(batch["stores"])):
				for j in range(self.max_time_steps):
					if batch['mask'][i][j] == 1.0:
						temp = [batch["stores"][i]]
						temp.extend(list(batch["inputs"][i][j][9:]))
						temp.extend([reshape_outputs[i][j][0]])
						final_outputs.append(temp)
			if batch["refresh"] == 1:
				batch = reader.next()
				if batch != None:
					feed_to_initial = batch["init_feed"]
			else:
				batch = reader.next()

		return final_outputs

	def calc_rms(self,session,reader):
		keep_prob = 1
		final_outputs = []
		fetches = {
			"outputs" : self.model_outputs,
			"final_state" : self.metrics["final_state"],
			"final_output" : self.final_output
		}
		phase_train = False
		state = session.run(self.initial_state)
		################################################
		import pandas as pd
		train = pd.read_csv('data/train.csv')
		mean = train['Sales'].mean()
		std = train['Sales'].std()
		################################################
		reader.start()

		i = 0
		batch = reader.next()
		final_val,final_count = 0.0,0.0
		feed_to_initial = np.zeros((self.batch_size,self.output_size))
		while batch != None:        
			feed_dict = {}
			feed_dict[self.phase_test] = True
			feed_dict[self.inputs.name] = batch["inputs"]
			feed_dict[self.keep_prob.name] = keep_prob
			feed_dict[self.phase_train.name] = phase_train
			feed_dict[self.initial_force] = feed_to_initial	
			feed_dict[self.targets.name] = np.zeros((self.batch_size,self.max_time_steps,self.output_size))
			feed_dict[self.sample_weight.name] = 0.5
			
			
			feed_dict[self.initial_state] = state

			vals = session.run(fetches, feed_dict)

			if batch["refresh"] == 1:
				state = session.run(self.initial_state)
				feed_to_initial = np.zeros((self.batch_size,self.output_size))

			else:
				state = vals["final_state"] 
				feed_to_initial = vals["final_output"]

			reshape_outputs = np.reshape(vals["outputs"],[self.batch_size,self.max_time_steps,-1])

			for i in range(self.batch_size):
				for j in range(self.max_time_steps):
					if batch['mask'][i][j] == 1.0:
						if batch["outputs"][i][j][0]*std+mean > 1:
							final_val += ((reshape_outputs[i][j][0]*std - batch["outputs"][i][j][0]*std)/(batch["outputs"][i][j][0]*std+mean))**2
							final_count += 1



			batch = reader.next()

		return np.sqrt(final_val/final_count)


