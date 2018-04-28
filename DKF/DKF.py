from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import numpy as np
import tensorflow as tf
import pdb

tf.set_random_seed(0)
np.random.seed(0)

batch_size = 2
max_time_steps = 5
learning_rate = 0.000002

x_size = 8
u_size = 2

num_hidden_units = x_size + u_size
num_hidden_layers = 3

z_size = 10

n_samples_term_1 = 10
n_samples_term_3 = 10

keep_prob = 0.5

z_distr_params_size = z_size + z_size * z_size
x_distr_params_size = x_size + x_size * x_size

latent_state_max_time = 100

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

	def recognition_model(self, x, u):

		batch_len = int(x.shape[0])
		time_len = int(x.shape[1])

		def rnn_cell():
			return tf.contrib.rnn.DropoutWrapper(
				tf.contrib.rnn.BasicLSTMCell(num_units=num_hidden_units),
				output_keep_prob=keep_prob,
				variational_recurrent=True,
				dtype=tf.float32)


		# batch size x time steps x (input + action)
		processed_inputs = tf.concat([x, u], axis=2)

		cells = tf.contrib.rnn.MultiRNNCell([rnn_cell() for _ in range(num_hidden_layers)])
		rnn_initial_state = cells.zero_state(batch_len, dtype=tf.float32)
		
		state = rnn_initial_state
		with tf.variable_scope("recognition_model/recur"):
			outputs = []
			for time_step in range(max_time_steps):
				if time_step > 0:
					tf.get_variable_scope().reuse_variables()
				cell_output,hs = cells(processed_inputs[:,time_step,:], state)
				outputs.append(cell_output)

		# batch size x time steps x output dim (num_hidden_units)
		outputs=tf.stack(outputs, axis=1)

		outputs_r = tf.reshape(outputs, shape=(-1, num_hidden_units))

		with tf.variable_scope("recognition_model/feed"):
			w = tf.get_variable("weight", shape=(x_size + u_size, z_distr_params_size))
			b = tf.get_variable("bias", shape=(z_distr_params_size))

		out = tf.matmul( outputs_r, w ) + b
		out = tf.reshape(out, shape=(-1, time_len, z_distr_params_size))

		mean = out[:,:,:z_size]
		cov1 = tf.reshape(out[:,:,z_size:], shape=(-1, time_len, z_size, z_size))
		cov2 = tf.transpose(cov1, perm=(0, 1, 3, 2))

		cov1_shaped = tf.reshape( cov1, shape=(-1, z_size, z_size) )
		cov2_shaped = tf.reshape( cov2, shape=(-1, z_size, z_size) )

		covariance_shaped = tf.matmul(cov1_shaped, cov2_shaped)

		# le = int(cov1.shape[0])
		# covariance_shaped = tf.eye(z_size, batch_shape=[le, time_len])

		covariance = tf.reshape( covariance_shaped, shape=(-1, time_len, z_size, z_size) )

		return mean, covariance

	def custom_gaussian_sampler(self, mean, covariance, n_samples):

		dims = [ int(x) for x in covariance.shape ]
		out = 1
		for i in dims:
			out= i * out
		covariance = tf.Print(covariance, [covariance], summarize=out)

		ds = tf.contrib.distributions

		mvg = ds.MultivariateNormalFullCovariance(
				loc=mean,
				covariance_matrix=covariance
			)

		samples = mvg.sample(sample_shape=(n_samples))
		samples = tf.transpose(samples, perm=(1, 0, 2))

		return samples

	def generation_model(self, z):

		with tf.variable_scope("generation_model/feed"):
			w = tf.get_variable("weight", shape=(z_size, x_distr_params_size))
			b = tf.get_variable("bias", shape=(x_distr_params_size))

		out = tf.matmul(z, w) + b

		mean = out[:,:x_size]
		cov1 = tf.reshape(out[:,x_size:], shape=(-1, x_size, x_size))
		cov2 = tf.transpose(cov1, perm=(0, 2, 1))
		covariance = tf.matmul(cov1, cov2)

		# le = int(z.shape[0])
		# covariance = tf.eye(num_rows=x_size, batch_shape=[le])

		return mean, covariance

	def pdf_value_multivariate(self, mean, covariance, arg):

		ds = tf.contrib.distributions

		mvg = ds.MultivariateNormalFullCovariance(
				loc=mean,
				covariance_matrix=covariance,
				validate_args=True
			)

		arg_r = tf.reshape(arg, shape=(-1, 1, x_size))

		values = mvg.prob(arg_r)

		return values

	@staticmethod
	def kl(mean1, covar1, mean2, covar2):
		ds = tf.contrib.distributions
		mvg1 = ds.MultivariateNormalFullCovariance(
				loc=mean1,
				covariance_matrix=covar1,
				validate_args=True
			)

		mvg2 = ds.MultivariateNormalFullCovariance(
				loc=mean2,
				covariance_matrix=covar2,
				validate_args=True
			)

		diverg = tf.distributions.kl_divergence(mvg1, mvg2)
		return diverg

	def custom_kl_e2(self, mean1, covar1, mean2, covar2, batch_size):
		# mean2 and covar2 need broadcasting to batch_size
		mean1 = tf.reshape(mean1, shape=(-1, z_size))
		covar1 = tf.reshape(covar1, shape=(-1, z_size, z_size))

		mean2 = tf.reshape(mean2, shape=(-1, z_size)) + tf.zeros(shape=(batch_size, z_size))
		covar2 = tf.reshape(covar2, shape=(-1, z_size, z_size)) + tf.zeros(shape=(batch_size, z_size, z_size))

		return DKF.kl(mean1, covar1, mean2, covar2)

	def transition_model(self, z, u):
		# batch_size x (z, u)

		in1 = tf.concat([z, u], axis=1)

		with tf.variable_scope("transition_model/layer1"):
			w = tf.get_variable("weight", shape=(z_size + u_size, z_distr_params_size))
			b = tf.get_variable("bias", shape=(z_distr_params_size))

		out = tf.matmul(in1, w) + b

		mean = out[:,:z_size]
		cov1 = tf.reshape(out[:,z_size:], shape=(-1, z_size, z_size))
		cov2 = tf.transpose(cov1, perm=(0, 2, 1))
		covariance = tf.matmul(cov1, cov2)

		# le = int(in1.shape[0])
		# covariance = tf.eye(num_rows=z_size, batch_shape=[le])

		return mean, covariance

	def custom_kl_e3(self, mean1, covar1, mean2, covar2, n_samples, batch_len, time_len):
		# batch size x time steps-1 x samples = kl of batch size x time steps -1 x z_distr_size and batch size * time steps -1 * N x z_distr_size

		# broadcast mean1, covar1 to samples
		mean1 = tf.reshape( mean1, shape=(-1, 1, z_size) ) + tf.zeros( shape=(batch_len * time_len, n_samples, z_size) )
		covar1 = tf.reshape( covar1, shape=(-1, 1, z_size, z_size) ) + tf.zeros( shape=(batch_len * time_len, n_samples, z_size, z_size) )
		# and reshape
		mean1 = tf.reshape( mean1, shape=(-1, z_size) )
		covar1 = tf.reshape( covar1, shape=(-1, z_size, z_size) )

		mean2 = tf.reshape( mean2, shape=(-1, z_size) )
		covar2 = tf.reshape( covar2, shape=(-1, z_size, z_size) )

		# batch_len x time len x samples
		prob_values = tf.reshape(DKF.kl(mean1, covar1, mean2, covar2), shape=(batch_len, time_len, n_samples))

		return prob_values

	def build_model(self):
		# config = self.config
		# batch_size = config.batch_size
		# lstm_units = config.lstm_units
		# input_size = config.input_size
		# num_hidden_layers = config.num_hidden_layers

		# processed_inputs = self.inputs

		z1_prior_mean = tf.zeros(shape=(z_size))
		z1_prior_covar = tf.eye(z_size)
		# z_transition = mean, sigma

		# batch size x time steps x z_distr_params_size ((mean, log of variance))
		z_param_mean, z_param_covar = self.recognition_model(self.x, self.u)

		# reshaping into batch size * timesteps x z_distr_params_size for generality
		z_param_mean_shaped = tf.reshape( z_param_mean, shape=(-1, z_size) )
		z_param_covar_shaped = tf.reshape( z_param_covar, shape=(-1, z_size, z_size) )
		# batch size * time steps x N samples x z_size
		samples_z = self.custom_gaussian_sampler( z_param_mean_shaped, z_param_covar_shaped, n_samples_term_1)
		# shaping samples into batch size * time steps * N x z_size
		samples_z_shaped = tf.reshape( samples_z, shape=(-1, z_size) )

		# batch size * time steps * N x x_distr_params_size (mean, log of variance)
		x_param_mean, x_param_covar = self.generation_model( samples_z_shaped )

		# x_param = batch size * time steps * N x x_distr_params_size
		# shaped_x_param_e1 = batch size * time steps x N x x_size
		shaped_x_param_mean_e1 = tf.reshape( x_param_mean, shape=(-1, n_samples_term_1, x_size) )
		shaped_x_param_covar_e1 = tf.reshape( x_param_covar, shape=(-1, n_samples_term_1, x_size, x_size) )
		# shaped_x_e1 = batch size * time steps x x_size
		shaped_x_e1 = tf.reshape( self.x, shape=(-1, x_size) )
		# error term 1 
		# batch size * time steps x N
		out1 = tf.log( self.pdf_value_multivariate ( shaped_x_param_mean_e1, shaped_x_param_covar_e1, shaped_x_e1 ) )

		expectation_out1 = tf.reduce_mean(out1, axis=[1])
		# batch size 
		error_term1 = tf.reduce_sum(tf.reshape( expectation_out1, shape=(-1, max_time_steps) ), axis=[1])

		# batch size x 1 (t = 0) x 2 (mean, log variance)
		z_param_mean_0, z_param_covar_0 = z_param_mean[:,0,:], z_param_covar[:,0,:,:]
		# batch size x time steps-1 (t = 1:max) x 2 (mean, log variance)
		z_param_mean_1_t, z_param_covar_1_t = z_param_mean[:,1:,:], z_param_covar[:,1:,:,:]
		# batch size x time steps-1 (t = 0:max-1) x 2 (mean, log variance)
		z_param_mean_0_t_1, z_param_covar_0_t_1 = z_param_mean[:,:-1,:], z_param_covar[:,:-1,:,:]

		# batch size
		error_term2 = self.custom_kl_e2( z_param_mean_0, z_param_covar_0, z1_prior_mean, z1_prior_covar, batch_size)


		# reshaping into batch size * timesteps-1 x z_distr_params_size for generality
		z_param_mean_0_t_1_shaped = tf.reshape( z_param_mean_0_t_1, shape=(-1, z_size) )
		z_param_covar_0_t_1_shaped = tf.reshape( z_param_covar_0_t_1, shape=(-1, z_size, z_size) )
		# batch size * time steps-1 x N x z_size
		samples_z1 = self.custom_gaussian_sampler( z_param_mean_0_t_1_shaped, z_param_covar_0_t_1_shaped, n_samples_term_3)
		actions_0_t_1 = tf.reshape(self.u[:, :-1, :], shape=(-1, 1, u_size))
		# broadcast_actions matrix
		actions_0_t_1_broadcasted = actions_0_t_1 + tf.zeros( shape=( batch_size*(max_time_steps-1), n_samples_term_3, u_size) )

		# reshaped for generality
		actions_0_t_1_broadcasted_shaped = tf.reshape( actions_0_t_1_broadcasted, shape=(-1, u_size) )
		samples_z1_shaped = tf.reshape( samples_z1, shape=(-1, z_size) )

		# batch size * time steps -1 * N x ( mean, covar )
		z_param_trans_mean, z_param_trans_covar = self.transition_model( samples_z1_shaped, actions_0_t_1_broadcasted_shaped )

		# batch size x time steps-1 x samples = kl of batch size x time steps -1 x z_distr_size and batch size * time steps -1 * N x z_distr_size
		kl_samples_e3 = self.custom_kl_e3( z_param_mean_1_t, z_param_covar_1_t, z_param_trans_mean, z_param_trans_covar, n_samples_term_3, batch_size, max_time_steps-1 )

		out_e3 = tf.reduce_mean( kl_samples_e3, axis=[2] )

		# batch size 
		error_term3 = tf.reduce_sum( out_e3, axis=[1] )

		# self.metrics["cost"] = tf.reduce_mean(-error_term3)
		self.metrics["cost"] = tf.reduce_mean(error_term1-error_term2-error_term3)

		# # #
		x = self.x[:, :latent_state_max_time, :]
		u = self.u[:, :latent_state_max_time, :]
		ls = self.acquire_latent_state(x, u)

		u_r = self.u[:, latent_state_max_time:, :]

		x_p = self.predict_x(ls, u_r)
		x_r = self.x[:, latent_state_max_time:, :]

		self.metrics["validation_error"] = tf.reduce_sum( tf.reduce_mean( tf.square(x_r-x_p) , axis = 1 ) , axis=0)

		# # #
		self.metrics["prediction"] = x_p

	def acquire_latent_state(self, x, u):

		# as input
		# batch size x time_steps x x_size
		# batch size x time_steps x u_size
		# batch size x time_steps x mean, var

		distr_z_mean, distr_z_covar = self.recognition_model(x, u)

		# consider last output and reshape
		distr_z_mean, distr_z_covar = tf.reshape(distr_z_mean[:,-1,:], shape=(-1, z_size)), tf.reshape(distr_z_covar[:,-1,:,:], shape=(-1, z_size, z_size))

		# batch size x z_size reshaped
		return tf.reshape(self.custom_gaussian_sampler( distr_z_mean, distr_z_covar, 1), shape=(-1, z_size))

	def predict_x(self, z, u):
		# batch size x z_size
		# batch size x time_len x u_size

		time_len = int(u.shape[1])

		curr_z = z

		x = []
		for i in range(time_len):
			# transition
			curr_u = tf.reshape( u[:,i,:], shape=(-1, u_size) )
			mean_l, covar_l = self.transition_model( curr_z, curr_u )

			# sample
			curr_z = tf.reshape( self.custom_gaussian_sampler(mean_l, covar_l, 1), shape=(-1, z_size) )

			# generate distr
			mean_x, covar_x = self.generation_model( curr_z )

			# sample x
			x_t = tf.reshape( self.custom_gaussian_sampler(mean_x, covar_x, 1), shape=(-1, x_size) )

			x.append( x_t )

		return x

	def compute_gradients_and_train_op(self):
		tvars = self.tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope)

		grads = tf.gradients(self.metrics["cost"], tvars)

		optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
		self.train_op=optimizer.apply_gradients(
			zip(grads, tvars),
			global_step=self.global_step)

	def run_epoch(self, session, validate=True, verbose=False):

		epoch_metrics = {}
		fetches = {
			"cost": self.metrics["cost"]
		}

		if verbose:
			print("\nTraining...")
		fetches["train_op"] = self.train_op

		i = 0
		# use batch iterator here
		batch = {
			"x" : np.random.randn( batch_size, max_time_steps, x_size ), 
			"u" : np.random.randn( batch_size, max_time_steps, u_size )
		}

		while batch != None:
			
			feed_dict = {}
			feed_dict[self.x.name] = batch["x"]
			feed_dict[self.u.name] = batch["u"]

			vals = session.run(fetches, feed_dict)

			i += 1
			if verbose:
				print(
					"% Iter Done :", round(i, 0),
					"loss :", round(vals["cost"]),
				)
				print ("<~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~>")

		if validate:
			if verbose:
				print("\nValidating...")
			fetches["validation_error"] = self.metrics["validation_error"]
			# load from validation data set
			batch = {
				"x": np.random.randn(batch_size, max_time_steps, x_size),
				"u": np.random.randn(batch_size, max_time_steps, u_size)
			}

			vals = session.run(fetches, feed_dict)

			print("MSE: %f" % vals["validaton_error"])

		epoch_metrics["loss"] = vals["cost"]
		epoch_metrics["validation_error"] = vals["validation_error"]
		return epoch_metrics

	def run_test(self, session, verbose=False):

		fetches = {}

		if verbose:
			print("\nPredicting...")
		# load from test set
		batch = {
			"x": np.random.randn(batch_size, max_time_steps, x_size),
			"u": np.random.randn(batch_size, max_time_steps, u_size)
		}
		fetches["prediction"] = self.metrics["prediction"]

		feed_dict = {}
		feed_dict[self.x.name] = batch["x"]
		feed_dict[self.u.name] = batch["u"]

		vals = session.run(fetches, feed_dict)

		print("prediction: %f" % vals["prediction"])		

if __name__ == "__main__":

	dkf = DKF(device="cpu")
	dkf.build_model()

	session = tf.Session()

	init = tf.global_variables_initializer()
	session.run(init)

	dkf.run_epoch(session, validate=True, verbose=True)
	dkf.run_test(session, verbose=True)
