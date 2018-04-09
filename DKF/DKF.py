from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import numpy as np
import tensorflow as tf
import pdb

batch_size = 20
max_time_steps = 12
num_hidden_layers = 3
x_size = 1000
u_size = 1000

z_size = 100

n_samples_term_1 = 1000

z_distr_params_size = z_size + z_size * z_size
x_distr_params_size = x_size + x_size * x_size


class DKF(object):

	def __init__(self, config = None, scope_name=None, device='gpu'):
		self.config = config
		self.scope = scope_name or "DKF"

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
		# batch_size = self.config.batch_size
		# max_time_steps = self.config.max_time_steps
		# input_size = self.config.input_size

		# get input dimensions

		# batch size x time steps x input feature
		self.x = tf.placeholder(dtype=tf.float32, shape=(batch_size, max_time_steps, x_size))
		# batch size x time steps x input action
		self.u = tf.placeholder(dtype=tf.float32, shape=(batch_size, max_time_steps, u_size))

	def recognition_model(x, u):

		def rnn_cell():
			return tf.contrib.rnn.DropoutWrapper(
				tf.contrib.rnn.BasicLSTMCell(num_units=num_hidden_units),
				output_keep_prob=self.keep_prob,
				variational_recurrent=True,
				dtype=tf.float32)

		# batch size x time steps x (input + action)
		processed_inputs = tf.concat([x, u] axis=2)

		cells = tf.contrib.rnn.MultiRNNCell([rnn_cell() for _ in range(num_hidden_layers)])
		rnn_initial_state = cells.zero_state(batch_size, dtype=tf.float32)
		
		state = rnn_initial_state
		with tf.variable_scope("recognition_model/recur", initializer=rand_uni_initializer):
			outputs = []
			for time_step in range(max_time_steps):
				if time_step > 0:
					tf.get_variable_scope().reuse_variables()
				cell_output,hs = cells(processed_inputs[:,time_step,:], state)
				outputs.append(cell_output)

		# batch size x time steps x output dim
		outputs=np.concat(outputs, axis=0)

		with tf.variable_scope("recognition_model/feed"):
			w = tf.get_variable("weight", shape=(x_size + u_size, z_distr_params_size))
			b = tf.get_variable("bias", shape=(z_distr_params_size))

		out = tf.matmul( outputs, w ) + b
		return out

	def custom_gaussian_sampler(param):

		mean = param[,,:z_size]
		covariance = tf.exp(tf.reshape(param[,,z_size:], shape=(batch_size, max_time_steps, z_size, z_size)))

		ds = tf.contrib.distributions

		mvg = ds.MultivariateNormalFullCovariance(
				loc=mean,
				covariance_matrix=covariance
			)

		samples = mvg.sample(sample_shape=(n_samples_term_1))
		return samples

	def transition_model(z, u):
		pass

	def generation_model(z):

		# check order of sample, z_size
		z1 = tf.transpose(z, perm=(0, 1, 3, 2))

		with tf.variable_scope("generation_model/feed"):
			w = tf.get_variable("weight", shape=(z_size, x_distr_params_size))
			b = tf.get_variable("bias", shape=(x_distr_params_size))

		out = tf.matmul(z, w) + b
		return out

	def pdf_value_multivariate(param, arg):

		mean = param[,,,:x_size]
		covariance = tf.exp(tf.reshape(param[,,,x_size:], shape=(batch_size, max_time_steps, None, x_size, x_size)))

		ds = tf.contrib.distributions

		mvg = ds.MultivariateNormalFullCovariance(
				loc=mean,
				covariance_matrix=covariance
			)

		values = mvg.prob(arg)
		return values

	def build_model(self):
		# config = self.config
		# batch_size = config.batch_size
		# lstm_units = config.lstm_units
		# input_size = config.input_size
		# num_hidden_layers = config.num_hidden_layers

		processed_inputs = self.inputs

		z1_prior = mean, sigma
		z_transition = mean, sigma

		# batch size x time steps x z_distr_params_size ((mean, log of variance))
		z_param = self.recognition_model(x, u)

		# batch size x time steps x N samples x z_size
		samples_z = custom_gaussian_sampler( z_param )

		# batch size x time steps x N x x_distr_params_size (mean, log of variance)
		x_param = self.generation_model( samples_z )

		# error term 1 
		# batch size x time steps x N
		out1 = tf.log( pdf_value_multivariate ( x_param, x ) )

		# batch size 
		error_term1 = reduce_sum( out1, axes=[1, 2] )


		# batch size x 1 (t = 0) x 2 (mean, log variance)
		z_param0 = z_param[:, 0,:]
		# batch size x time steps-1 (t = 1:max) x 2 (mean, log variance)
		z_param1_t = z_param[:, +1:,:]
		# batch size x time steps-1 (t = 0:max-1) x 2 (mean, log variance)
		z_param0_t_1 = z_param[:, :-1,:]



		# batch size
		error_term2 = kl( z_param0, z1_prior )

		# batch size x time steps-1 x N
		samples_z1 = sampler( z_param1 )


		# TODO
		# batch size X time steps-1 x N
		error_term3 = ?

		pass

	def compute_loss_and_metrics(self):
		pass

	def compute_gradients_and_train_op(self):
		tvars = self.tvars = my_lib.get_scope_var(self.scope)
		my_lib.get_num_params(tvars)
		grads = tf.gradients(self.metrics["entropy_loss"], tvars)
		grads, _ = tf.clip_by_global_norm(grads, self.config.max_grad_norm)

		self.metrics["grad_sum"] = tf.add_n([tf.reduce_sum(g) for g in grads])

		optimizer = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate)
		self.train_op=optimizer.apply_gradients(
			zip(grads, tvars),
			global_step=self.global_step)

	def model_vars(self):
		return self.tvars

	def init_feed_dict(self):
		return {self.phase_train.name: True}


	def run_epoch(self, session,reader, is_training=False, verbose=False):
		start_time = time.time()
		epoch_metrics = {}
		keep_prob = 1.0
		fetches = {
			"loss": self.metrics["loss"],
			"grad_sum": self.metrics["grad_sum"]

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



		i, total_loss, grad_sum = 0, 0.0, 0.0

		reader.start()

		i = 0
		print("Reading till the Batch Number")
		batch = reader.next()
		while batch != None:

			
			feed_dict = {}
			feed_dict[self.outputs.name] = batch["outputs"]
			feed_dict[self.inputs.name] = batch["inputs"]
			feed_dict[self.keep_prob.name] = keep_prob
			feed_dict[self.phase_train.name] = phase_train
			feed_dict[self.refresh] = batch["refresh"]
			feed_dict[self.mask.name] = batch["mask"]
			

			vals = session.run(fetches, feed_dict)
			total_loss += vals["entropy_loss"]
			grad_sum += vals["grad_sum"]
			total_entries += np.sum(batch["mask"])
			i += 1
			
			if verbose:
				print(
					"% Iter Done :", round(i, 0),
					"loss :", round((total_loss/total_entries), 3), \
					"Gradient :", round(vals["grad_sum"],3))
			batch = reader.next()


		epoch_metrics["loss"] = round(np.exp(total_loss / total_words), 3)
		return epoch_metrics

if __name__ == "__main__":
	