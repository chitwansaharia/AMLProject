from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

import pdb

SEED = 0

tf.set_random_seed(SEED)
np.random.seed(SEED)
ds = tf.contrib.distributions

class DKF(object):


	def __init__(self, config = None, scope_name=None, device='gpu'):
		self.variable_print = []

		################################################
		import pandas as pd
		train = pd.read_csv('data/train.csv')
		self.mean = train['Sales'].mean()
		self.std = train['Sales'].std()
		################################################

		self.config = config
		self.config["x_distr_params_size"] = self.config["x_size"] + self.config["x_size"] * self.config["x_size"]
		self.config["z_distr_params_size"] = self.config["z_size"] + self.config["z_size"] * self.config["z_size"]

		self.scope = scope_name or "DKF"

		self.create_placeholders()
		self.global_step = \
			tf.train.get_or_create_global_step()

		self.metrics = {}
		if device == 'gpu':
			tf.device('/gpu:0')
		else:
			tf.device('/cpu:0')

		initializer = tf.random_normal_initializer()

		with tf.variable_scope(self.scope, initializer=initializer):
			self.build_model()
		with tf.variable_scope("optimizer"):
			self.compute_gradients_and_train_op()

	def variable_print_append(self, var):
		# import ipdb; ipdb.set_trace()
		self.variable_print.append(var)

	@staticmethod
	def variable_summaries(var):
		"""Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
		with tf.name_scope('summaries'):
			mean = tf.reduce_mean(var)
			tf.summary.scalar('mean', mean)
			with tf.name_scope('stddev'):
				stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
				tf.summary.scalar('stddev', stddev)
			tf.summary.scalar('max', tf.reduce_max(var))
			tf.summary.scalar('min', tf.reduce_min(var))
			tf.summary.tensor_summary('val', var)


	def create_placeholders(self):
		# get input dimensions
		batch_size = self.config["batch_size"]
		time_len = self.config["time_len"]

		input_size = self.config["u_size"]
		output_size = self.config["x_size"]
		hidden_size = self.config["z_size"]

		# batch size x time steps x input feature
		self.x = tf.placeholder(dtype=tf.float32,
				 shape=(batch_size,
				 time_len,
				 output_size)
				 )
		# # batch size x time steps x input action
		self.u = tf.placeholder(dtype=tf.float32,
				 shape=(batch_size,
				 time_len,
				 input_size)
				 )
		# # batch size x time steps
		self.mask = tf.placeholder(dtype=tf.float32,
				 shape=(batch_size,
				 time_len)
				 )

		# # batch size x z_size
		self.initial_latent_mean = tf.placeholder(dtype=tf.float32,
				 shape=(batch_size,
				 hidden_size)
				 )
		self.initial_latent_covar = tf.placeholder(dtype=tf.float32,
				 shape=(batch_size,
				 hidden_size,
				 hidden_size)
				 )

	def create_embeddings(self):

		with tf.variable_scope("embeddings"):
			self.store_type_embedding = tf.get_variable("store_type_embedding", [4, 4])
			self.assortment_type_embedding = tf.get_variable(
				"assortment_type_embedding", [3, 3])
			self.competition_open_since_month_embedding = tf.get_variable(
				"competition_open_since_month_embedding", [13, 13])
			self.promo2_embedding = tf.get_variable("promo2_embedding", [2, 2])
			self.promo2_since_year_embedding = tf.get_variable(
				"promo2_since_year_embedding", [8, 7])
			self.promo_interval_embedding = tf.get_variable(
				"promo_interval_embedding", [4, 4], dtype=tf.float32)

			self.comp_dist_weights = tf.get_variable(
				"comp_dist_weights", [1, 2], dtype=tf.float32)
			self.comp_dist_bias = tf.get_variable(
				"comp_dist_bias", [2], dtype=tf.float32)

			self.comp_year_embedding = tf.get_variable(
				"comp_year_embedding", [29, 20],	dtype=tf.float32)
			self.promo2_week_embedding = tf.get_variable(
				"promo2_week_embedding", [14, 10])

			self.day_of_week_embedding = tf.get_variable(
				"day_of_week_embedding", [7, 7])
			self.year_embedding = tf.get_variable("year_embedding", [3, 3])
			self.month_embedding = tf.get_variable("month_embedding", [12, 12])
			self.day_embedding = tf.get_variable("day_embedding", [31, 20])
			self.open_embedding = tf.get_variable("open_embedding", [2, 2])
			self.promo_embedding = tf.get_variable("promo_embedding", [2, 2])
			self.state_holiday_embedding = tf.get_variable(
				"state_holiday_embedding", [4, 4])
			self.school_holiday_embedding = tf.get_variable(
				"school_holiday_embedding", [2, 2])

	def process_data(self, inputs):

		batch_size = self.config["batch_size"]
		time_len = self.config["time_len"]

		output = [tf.nn.embedding_lookup(
			self.store_type_embedding, tf.cast(inputs[:, :, 0], dtype=tf.int32))]
		output.append(tf.nn.embedding_lookup(
			self.assortment_type_embedding, tf.cast(inputs[:, :, 1], dtype=tf.int32)))
		temp = tf.reshape(inputs[:, :, 2:3], [-1, 1])
		# self.variable_print_append(tf.gradient(self.comp_dist_weights))
		# self.variable_print_append(tf.gradient(self.comp_dist_bias))
		# self.variable_print_append(self.comp_dist_weights)
		# self.variable_print_append(self.comp_dist_bias)
		# self.variable_print_append(temp)

		output.append(tf.reshape(tf.add(tf.matmul(temp, self.comp_dist_weights),
                                  self.comp_dist_bias), [batch_size, time_len, -1]))
		output.append(tf.nn.embedding_lookup(
			self.competition_open_since_month_embedding, tf.cast(inputs[:, :, 3], dtype=tf.int32)))
		output.append(tf.nn.embedding_lookup(self.comp_year_embedding,
                                       tf.cast(inputs[:, :, 4], dtype=tf.int32)))

		output.append(tf.nn.embedding_lookup(self.promo2_embedding,
                                       tf.cast(inputs[:, :, 5], dtype=tf.int32)))
		output.append(tf.nn.embedding_lookup(self.promo2_week_embedding,
                                       tf.cast(inputs[:, :, 6], dtype=tf.int32)))

		output.append(tf.nn.embedding_lookup(
			self.promo2_since_year_embedding, tf.cast(inputs[:, :, 7], dtype=tf.int32)))
		output.append(tf.nn.embedding_lookup(
			self.promo_interval_embedding, tf.cast(inputs[:, :, 8], dtype=tf.int32)))
		output.append(tf.nn.embedding_lookup(self.day_of_week_embedding,
                                       tf.cast(inputs[:, :, 9], dtype=tf.int32)))
		output.append(tf.nn.embedding_lookup(self.year_embedding,
                                       tf.cast(inputs[:, :, 10], dtype=tf.int32)))
		output.append(tf.nn.embedding_lookup(self.month_embedding,
                                       tf.cast(inputs[:, :, 11], dtype=tf.int32)))
		output.append(tf.nn.embedding_lookup(self.day_embedding,
                                       tf.cast(inputs[:, :, 12], dtype=tf.int32)))
		output.append(tf.nn.embedding_lookup(self.open_embedding,
                                       tf.cast(inputs[:, :, 13], dtype=tf.int32)))
		output.append(tf.nn.embedding_lookup(self.promo_embedding,
                                       tf.cast(inputs[:, :, 14], dtype=tf.int32)))
		output.append(tf.nn.embedding_lookup(
			self.state_holiday_embedding, tf.cast(inputs[:, :, 15], dtype=tf.int32)))
		output.append(tf.nn.embedding_lookup(
			self.school_holiday_embedding, tf.cast(inputs[:, :, 16], dtype=tf.int32)))

		u = tf.concat(output, axis=2)

		self.config["u_embedding_size"] = int(u.shape[-1])
		return u

	def recognition_model(self, x, u):

		batch_len = int(x.shape[0])
		time_len = int(x.shape[1])

		z_size = self.config["z_size"]
		u_size = int(u.shape[-1])
		z_distr_params_size = self.config["z_distr_params_size"]

		def rnn_cell():
			return tf.contrib.rnn.DropoutWrapper(
				tf.contrib.rnn.BasicLSTMCell(num_units=self.config["num_hidden_units"]),
				output_keep_prob=self.config["keep_prob"],
				variational_recurrent=True,
				dtype=tf.float32)

		# batch size x time steps x (input + action)

		input_concat = tf.concat([x, u], axis=2)
		# self.variable_print_append(u)


		cells = tf.contrib.rnn.MultiRNNCell([rnn_cell() for _ in range(self.config["num_hidden_layers"])])
		state = self.initial_rnn = cells.zero_state(batch_len, dtype=tf.float32)

		with tf.variable_scope("recognition_model/rnn"):

			outputs = []
			for time_step in range(time_len):
				if time_step > 0:
					tf.get_variable_scope().reuse_variables()
				cell_output,state = cells(input_concat[:,time_step,:], state)
				outputs.append(cell_output)

			self.metrics["final_rnn_state"] = state

		# batch size x time steps x output dim (num_hidden_units)
		outputs=tf.stack(outputs, axis=1)
		

		outputs_r = tf.reshape(outputs, shape=(-1, self.config["num_hidden_units"]))
		# self.variable_print_append(outputs_r)

		with tf.variable_scope("recognition_model/feed"):
			w = tf.get_variable("weight", shape=(self.config["num_hidden_units"], z_distr_params_size))
			b = tf.get_variable("bias", shape=(z_distr_params_size))

		out = tf.matmul( outputs_r, w ) + b
		out = tf.reshape(out, shape=(-1, time_len, z_distr_params_size))

		mean = out[:,:,:z_size]
		cov1 = tf.reshape(out[:,:,z_size:], shape=(-1, time_len, z_size, z_size))
		cov2 = tf.transpose(cov1, perm=(0, 1, 3, 2))

		# le = int(cov1.shape[0])
		# covariance_shaped = tf.eye(z_size, batch_shape=[le, time_len])
		
		covariance = tf.matmul(cov1, cov2)

		return mean, cov1

	def custom_gaussian_sampler(self, mean, covariance, n_samples, covar_is_diag=False):

		# dims = [ int(x) for x in covariance.shape ]
		# out = 1
		# for i in dims:
		# 	out= i * out
		# covariance = tf.Print(covariance, [covariance], summarize=out)

		mvg = DKF.get_mvg(mean, covariance, covar_is_diag=covar_is_diag)

		samples = mvg.sample(sample_shape=(n_samples))
		samples = tf.transpose(samples, perm=(1, 0, 2))

		return samples

	def generation_model(self, z):

		x_size = self.config["x_size"]
		z_size = self.config["z_size"]
		x_distr_params_size = self.config["x_distr_params_size"]

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

		return mean, cov1

	@staticmethod
	def get_mvg(mean, covariance, stri=None, covar_is_diag=False):

		dims = [int(x) for x in covariance.shape]
		dimt = 1
		for i in dims:
			dimt *= i
		
		# covariance = tf.Print( covariance, ["Covariance " + str(stri), covariance])
		
		# with tf.control_dependencies([tf.assert_equal(covariance, tf.transpose(covariance, perm=(0, 2, 1)))]):

			# return ds.MultivariateNormalFullCovariance(
			# 		loc=mean,
			# 		covariance_matrix=covariance
			# 		# validate_args=True,
			# 	)

		if not covar_is_diag:
			return ds.MultivariateNormalTriL(
				loc=mean,
				scale_tril=covariance
				# validate_args=True,
			)
		else:
			return ds.MultivariateNormalDiag(
				loc=mean,
				scale_diag=covariance
				# validate_args=True,
			)	
		
	def pdf_value_multivariate_custom(self, mean, covariance, arg):

		# broadcast arg and return prob value
		arg_size = int(arg.shape[-1])
		comp_size = int(arg.shape[-2])

		arg_r = tf.reshape(arg, shape=(-1, arg_size))
		mean_r = tf.reshape(mean, shape=(-1, arg_size))
		covariance_r = tf.reshape(covariance, shape=(-1, arg_size, arg_size))

		mvg = DKF.get_mvg(mean_r, covariance_r)

		values = mvg.prob(arg_r)
		values = tf.reshape(values, shape=(-1, comp_size))
		return values

	def kl_divergence(self, a, b, name=None):
		"""Batched KL divergence `KL(a || b)` for multivariate Normals.

		With `X`, `Y` both multivariate Normals in `R^k` with means `mu_a`, `mu_b` and
		covariance `C_a`, `C_b` respectively,

		```
		KL(a || b) = 0.5 * ( L - k + T + Q ),
		L := Log[Det(C_b)] - Log[Det(C_a)]
		T := trace(C_b^{-1} C_a),
		Q := (mu_b - mu_a)^T C_b^{-1} (mu_b - mu_a),
		```

		This `Op` computes the trace by solving `C_b^{-1} C_a`. Although efficient
		methods for solving systems with `C_b` may be available, a dense version of
		(the square root of) `C_a` is used, so performance is `O(B s k**2)` where `B`
		is the batch size, and `s` is the cost of solving `C_b x = y` for vectors `x`
		and `y`.

		Args:
			a: Instance of `MultivariateNormalLinearOperator`.
			b: Instance of `MultivariateNormalLinearOperator`.
			name: (optional) name to use for created ops. Default "kl_mvn".

		Returns:
			Batchwise `KL(a || b)`.
		"""

		def squared_frobenius_norm(x):
			"""Helper to make KL calculation slightly more readable."""
			return tf.reduce_sum(tf.square(x), axis=[-2, -1])
		
		def log_abs_determinant(scale):
			return tf.reduce_sum(
					tf.log(tf.abs(scale._diag) + tf.constant(1e-8)), reduction_indices=[-1])

		with tf.name_scope(name, "kl_mvn", values=[a.loc, b.loc] +
							a.scale.graph_parents + b.scale.graph_parents):
			# Calculation is based on:
			# http://stats.stackexchange.com/questions/60680/kl-divergence-between-two-multivariate-gaussians
			# and,
			# https://en.wikipedia.org/wiki/Matrix_norm#Frobenius_norm
			# i.e.,
			#   If Ca = AA', Cb = BB', then
			#   tr[inv(Cb) Ca] = tr[inv(B)' inv(B) A A']
			#                  = tr[inv(B) A A' inv(B)']
			#                  = tr[(inv(B) A) (inv(B) A)']
			#                  = sum_{ij} (inv(B) A)_{ij}**2
			#                  = ||inv(B) A||_F**2
			# where ||.||_F is the Frobenius norm and the second equality follows from
			# the cyclic permutation property.
			b_inv_a = b.scale.solve(a.scale.to_dense())
			
			# self.variable_print_append(b_inv_a)

			kl_div = (log_abs_determinant(b.scale)
					- log_abs_determinant(a.scale)
					+ 0.5 * (
						- tf.cast(a.scale.domain_dimension_tensor(), a.dtype)
						+ squared_frobenius_norm(b_inv_a)
						+ squared_frobenius_norm(b.scale.solve(
							(b.mean() - a.mean())[..., tf.newaxis]))))
			kl_div.set_shape(tf.broadcast_static_shape(
				a.batch_shape, b.batch_shape))

			# self.variable_print_append(kl_div)
			
			return kl_div

	def kl(self, mean1, covar1, mean2, covar2, covar2_is_diag=False):

		mvg1 = DKF.get_mvg(mean1, covar1)

		mvg2 = DKF.get_mvg(mean2, covar2, covar_is_diag=covar2_is_diag)

		# diverg = tf.distributions.kl_divergence(mvg1, mvg2)
		diverg = self.kl_divergence(mvg1, mvg2)

		# diverg = tf.Print(diverg, ["KL", diverg])
		return diverg

	def custom_kl_e2(self, mean1, covar1, mean2, covar2, batch_size):
		# mean2 and covar2 need broadcasting to batch_size
		
		z_size = self.config["z_size"]

		mean1 = tf.reshape(mean1, shape=(-1, z_size))
		covar1 = tf.reshape(covar1, shape=(-1, z_size, z_size))

		mean2 = tf.reshape(mean2, shape=(-1, z_size)) + np.zeros(shape=(batch_size, z_size))
		covar2 = tf.reshape(covar2, shape=(-1, z_size, z_size)) + np.zeros(shape=(batch_size, z_size, z_size))

		return self.kl(mean1, covar1, mean2, covar2)

	def transition_model(self, z, u):
		# batch_size x (z, u)

		u_size = int(u.shape[-1])
		z_size = self.config["z_size"]
		z_distr_params_size = self.config["z_distr_params_size"]

		in1 = tf.concat([z, u], axis=1)

		with tf.variable_scope("transition_model/layer1"):
			w = tf.get_variable("weight", shape=(z_size + u_size, z_size + z_size))
			b = tf.get_variable("bias", shape=(z_size + z_size))

		out = tf.matmul(in1, w) + b

		mean = out[:,:z_size]
		cov1 = tf.reshape(out[:,z_size:], shape=(-1, z_size))
		# cov2 = tf.transpose(cov1, perm=(0, 2, 1))
		# covariance = tf.matmul(cov1, cov2)

		# le = int(in1.shape[0])
		# covariance = tf.eye(num_rows=z_size, batch_shape=[le])

		return mean, cov1

	def custom_kl_e3(self, mean1, covar1, mean2, covar2, n_samples, batch_len, time_len):
		# batch size x time steps-1 x samples = kl of batch size x time steps -1 x z_distr_size and batch size * time steps -1 * N x z_distr_size
		# covar 2 is not cholesky matrix but the covariance matrix itself

		z_size = self.config["z_size"]

		# broadcast mean1, covar1 to samples
		mean1 = tf.reshape( mean1, shape=(-1, 1, z_size) ) + np.zeros( shape=(batch_len * time_len, n_samples, z_size) )
		covar1 = tf.reshape( covar1, shape=(-1, 1, z_size, z_size) ) + np.zeros( shape=(batch_len * time_len, n_samples, z_size, z_size) )
		# and reshape
		mean1 = tf.reshape( mean1, shape=(-1, z_size) )
		covar1 = tf.reshape( covar1, shape=(-1, z_size, z_size) )
		# self.variable_print_append(mean1)
		# self.variable_print_append(covar1)

		# COVAR2 IS DIAGONAL!
		mean2 = tf.reshape( mean2, shape=(-1, z_size) )
		covar2 = tf.reshape( covar2, shape=(-1, z_size) )
		# self.variable_print_append(mean2)
		# self.variable_print_append(covar2)


		# batch_len x time len x samples
		prob_values = tf.reshape(self.kl(mean1, covar1, mean2, covar2, covar2_is_diag=True), shape=(batch_len, time_len, n_samples))
		# self.variable_print_append(prob_values)

		return prob_values

	def build_model(self):
		# config = self.config
		# batch_size = config.batch_size
		# lstm_units = config.lstm_units
		# input_size = config.input_size
		# num_hidden_layers = config.num_hidden_layers

		batch_size = self.config["batch_size"]
		time_len = self.config["time_len"]

		x_size = self.config["x_size"]
		z_size = self.config["z_size"]

		n_samples_term_1 = self.config["n_samples_term_1"]
		n_samples_term_3 = self.config["n_samples_term_3"]

		# create identity and broadcast
		self.batch_cov_identity = tf.eye(z_size, batch_shape=[batch_size])


		z1_prior_mean = self.initial_latent_mean
		z1_prior_covar = self.initial_latent_covar

		# build embeddings
		self.create_embeddings()
		# process inputs
		embed_u = self.process_data(self.u)
		embed_u_size = self.config["u_embedding_size"]

		# batch size x time steps x z_distr_params_size ((mean, log of variance))
		z_param_mean, z_param_covar = self.recognition_model(self.x, embed_u)

		self.metrics["final_latent_mean"] = tf.reshape(z_param_mean[:, -1, :], shape=(batch_size, z_size))
		self.metrics["final_latent_covar"] = tf.reshape(z_param_covar[:, -1, :, :], shape=(batch_size, z_size, z_size))

		# reshaping into batch size * timesteps x z_distr_params_size for generality
		z_param_mean_shaped = tf.reshape( z_param_mean, shape=(-1, z_size) )
		z_param_covar_shaped = tf.reshape( z_param_covar, shape=(-1, z_size, z_size) )
		
		# self.variable_print_append(z_param_mean_shaped)
		# self.variable_print_append(z_param_covar_shaped)

		# z_param_mean_shaped = tf.Print(z_param_mean_shaped, [z_param_mean_shaped], summarize=batch_size * time_len * int(z_param_mean_shaped.shape[1]))
		# z_param_covar_shaped = tf.Print(z_param_covar_shaped, [z_param_covar_shaped], summarize=batch_size * time_len * int(z_param_covar_shaped.shape[1]) * int(z_param_covar_shaped.shape[2]))

		# batch size * time steps x N samples x z_size
		samples_z = self.custom_gaussian_sampler( z_param_mean_shaped, z_param_covar_shaped, n_samples_term_1)
		# shaping samples into batch size * time steps * N x z_size
		samples_z_shaped = tf.reshape( samples_z, shape=(-1, z_size) )

		# self.variable_print_append(samples_z_shaped)

		# batch size * time steps * N x x_distr_params_size (mean, log of variance)
		x_param_mean, x_param_covar = self.generation_model( samples_z_shaped )
		


		# x_param = batch size * time steps * N x x_distr_params_size
		# shaped_x_param_e1 = batch size * time steps x N x x_size
		shaped_x_param_mean_e1 = tf.reshape( x_param_mean, shape=(-1, n_samples_term_1, x_size) )
		shaped_x_param_covar_e1 = tf.reshape( x_param_covar, shape=(-1, n_samples_term_1, x_size, x_size) )

		# self.variable_print_append(shaped_x_param_mean_e1)
		# self.variable_print_append(shaped_x_param_covar_e1)

		# shaped_x_e1 = batch size * time steps x x_size
		shaped_x_e1 = tf.reshape(self.x, shape=(-1, x_size))
		shaped_x_broadcasted_e1 = tf.reshape(shaped_x_e1, shape=(-1, 1, x_size)) + np.zeros(
			shape=(batch_size*time_len, n_samples_term_1, x_size))
		# error term 1 
		# batch size * time steps x N
		out1 = tf.log( self.pdf_value_multivariate_custom ( shaped_x_param_mean_e1, shaped_x_param_covar_e1, shaped_x_broadcasted_e1 ) + tf.constant(1e-8))

		# self.variable_print_append(out1)

		expectation_out1 = tf.reduce_mean(out1, axis=[1])
		# batch size 
		error_term1 = tf.reduce_sum( tf.multiply(self.mask, tf.reshape( expectation_out1, shape=(-1, time_len) )), axis=[1])

		# batch size * 1 (t = 0) x 2 (mean, log variance)
		z_param_mean_0, z_param_covar_0 = z_param_mean[:,0,:], z_param_covar[:,0,:,:]
		# self.variable_print_append(z_param_mean_0)
		# # self.variable_print_append(z_param_covar_0)


		# batch size x time steps-1 (t = 1:max) x 2 (mean, log variance)
		z_param_mean_1_t, z_param_covar_1_t = z_param_mean[:,1:,:], z_param_covar[:,1:,:,:]
		# batch size x time steps-1 (t = 0:max-1) x 2 (mean, log variance)
		z_param_mean_0_t_1, z_param_covar_0_t_1 = z_param_mean[:,:-1,:], z_param_covar[:,:-1,:,:]


		# batch size
		error_term2 = self.custom_kl_e2( z_param_mean_0, z_param_covar_0, z1_prior_mean, z1_prior_covar, batch_size)

		# self.variable_print_append(error_term2)

		# reshaping into batch size * timesteps-1 x z_distr_params_size for generality
		z_param_mean_0_t_1_shaped = tf.reshape( z_param_mean_0_t_1, shape=(-1, z_size) )
		z_param_covar_0_t_1_shaped = tf.reshape( z_param_covar_0_t_1, shape=(-1, z_size, z_size) )

		# self.variable_print_append(z_param_mean_0_t_1_shaped)
		# self.variable_print_append(z_param_covar_0_t_1_shaped)

		# batch size * time steps-1 x N x z_size
		samples_z1 = self.custom_gaussian_sampler( z_param_mean_0_t_1_shaped, z_param_covar_0_t_1_shaped, n_samples_term_3)
		actions_0_t_1 = tf.reshape(embed_u[:, :-1, :], shape=(-1, 1, embed_u_size))
		# broadcast_actions matrix
		actions_0_t_1_broadcasted = actions_0_t_1 + np.zeros( shape=( batch_size*(time_len-1), n_samples_term_3, embed_u_size) )



		# reshaped for generality
		actions_0_t_1_broadcasted_shaped = tf.reshape( actions_0_t_1_broadcasted, shape=(-1, embed_u_size) )
		samples_z1_shaped = tf.reshape( samples_z1, shape=(-1, z_size) )

		# self.variable_print_append(actions_0_t_1_broadcasted_shaped)
		# self.variable_print_append(samples_z1_shaped)

		# batch size * time steps -1 * N x ( mean, covar )
		z_param_trans_mean, z_param_trans_covar = self.transition_model( samples_z1_shaped, actions_0_t_1_broadcasted_shaped )

		# self.variable_print_append(z_param_trans_mean)
		# self.variable_print_append(z_param_trans_covar)

		# batch size x time steps-1 x samples = kl of batch size x time steps -1 x z_distr_size and batch size * time steps -1 * N x z_distr_size
		kl_samples_e3 = self.custom_kl_e3( z_param_mean_1_t, z_param_covar_1_t, z_param_trans_mean, z_param_trans_covar, n_samples_term_3, batch_size, time_len-1 )

		# self.variable_print_append(kl_samples_e3)

		out_e3 = tf.multiply( self.mask[:, :-1], tf.reduce_mean( kl_samples_e3, axis=[2] ))

		# self.variable_print_append(out_e3)

		# batch size 
		error_term3 = tf.reduce_sum( out_e3, axis=[1] )

		# self.metrics["loss"] = tf.reduce_mean(error_term3)
		self.metrics["loss"] = -tf.reduce_mean(error_term1-error_term2-error_term3)

		tf.get_variable_scope().reuse_variables()

		# # #

		u_e_curr = embed_u
		ls = self.acquire_latent_state()

		x_p = self.predict_x(ls, u_e_curr)
		self.metrics["prediction"] = x_p

		self.merged_summary = tf.summary.merge_all()

	def acquire_latent_state(self):

		# as input
		# sample from prior meam, covariance

		z_size = self.config["z_size"]
		distr_z_mean, distr_z_covar = self.initial_latent_mean, self.initial_latent_covar

		# batch size x z_size reshaped
		return tf.reshape(self.custom_gaussian_sampler( distr_z_mean, distr_z_covar, 1), shape=(-1, z_size))

	def predict_x(self, z, embed_u):
		# batch size x z_size
		# batch size x time_len x u_size

		time_len = int(embed_u.shape[1])
		x_size = self.config["x_size"]
		embed_u_size = self.config["u_embedding_size"]
		z_size = self.config["z_size"]

		curr_z = z

		x = []
		for i in range(time_len):
			# transition
			curr_u = tf.reshape( embed_u[:,i,:], shape=(-1, embed_u_size) )
			mean_l, covar_l = self.transition_model( curr_z, curr_u )

			# sample
			curr_z = tf.reshape( self.custom_gaussian_sampler(mean_l, covar_l, 1, covar_is_diag=True), shape=(-1, z_size) )

			# generate distr
			mean_x, covar_x = self.generation_model( curr_z )

			# sample x
			x_t = tf.reshape( self.custom_gaussian_sampler(mean_x, covar_x, 1), shape=(-1, x_size) )

			x.append( x_t )	

		x = tf.stack(x, axis=1)

		return x

	def compute_gradients_and_train_op(self):
		tvars = self.tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope)

		# import pprint
		# pprint.PrettyPrinter().pprint(tvars)

		self.grads = tf.gradients(self.metrics["loss"], self.tvars)
		# self.grads1 = tf.gradients(self.metrics["loss"], self.comp_dist_bias)
		# self.variable_print_append(grads)
		# self.variable_print_append(grads1)


		# self.grads = grads
		# import ipdb; ipdb.set_trace()	
		# optimizer = tf.train.AdamOptimizer(learning_rate=self.config["learning_rate"])
		# self.train_op=optimizer.apply_gradients(
		# 	zip(grads, tvars),
		# 	global_step=self.global_step
		# )

		# self.grads = grads
		# self.grads[0] = tf.Print(grads[0], ["Grads"])
		# self.grads[0] = tf.Print(grads[0], grads)

		optimizer = tf.train.AdamOptimizer(
				learning_rate=self.config["learning_rate"]
			)

		# grads = optimizer.compute_gradients(self.metrics["loss"])
		# capped_grads = [(tf.clip_by_value(grad, -self.config['max_grad_norm'], self.config['max_grad_norm']), var) for grad, var in grads]

		capped_grads, _ = tf.clip_by_global_norm(self.grads, self.config['max_grad_norm'])
		self.train_op = optimizer.apply_gradients(zip(capped_grads, tvars), global_step=self.global_step)

		# self.train_op = optimizer.minimize(
				# loss=self.metrics["loss"],
				# global_step=self.global_step
			# )
		self.check_op = tf.add_check_numerics_ops()

	def run_epoch(self, session, reader, verbose=False):

		np.set_printoptions(threshold='nan')

		epoch_metrics = {}
		fetches = {
			"loss": self.metrics["loss"],
			"train_op" : self.train_op,
			"final_rnn_state" : self.metrics["final_rnn_state"],
			"final_latent_mean" : self.metrics["final_latent_mean"],
			"final_latent_covar" : self.metrics["final_latent_covar"]
		}

		print("\nTraining...")

		i = 0
		reader.start()

		batch = reader.next()
		feed_initial_rnn = session.run(self.initial_rnn)
		feed_initial_latent_mean = np.zeros((self.config["batch_size"], self.config["z_size"]))
		feed_initial_latent_covar = session.run(self.batch_cov_identity)

		while batch != None:

			feed_dict = {}
			feed_dict[self.u.name] = batch["inputs"]
			feed_dict[self.x.name] = batch["outputs"]
			feed_dict[self.mask.name] = batch["mask"]
			feed_dict[self.initial_latent_mean.name] = feed_initial_latent_mean
			feed_dict[self.initial_latent_covar.name] = feed_initial_latent_covar

			feed_dict[self.initial_rnn] = feed_initial_rnn

			vals = session.run(fetches, feed_dict)

			if verbose:
				print(
					"% Iter Done :", round(i, 0),
					"loss :", round(vals["loss"]),
				)

				print ("<~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~>")

			if batch["refresh"] == 1:
				feed_initial_rnn = session.run(self.initial_rnn)
				feed_initial_latent_mean = np.zeros((self.config["batch_size"], self.config["z_size"]))
				feed_initial_latent_covar = session.run(self.batch_cov_identity)
			else:
				feed_initial_rnn = vals["final_rnn_state"]
				feed_initial_latent_mean = vals["final_latent_mean"]
				feed_initial_latent_covar = vals["final_latent_covar"]

			i += 1
			batch = reader.next()

		epoch_metrics["loss"] = vals["loss"]

		return epoch_metrics

	def run_test(self, session, reader, calc_error=False, verbose=False):

		print("\nPredicting...")

		fetches = {
			"outputs" : self.metrics["prediction"],
			"final_rnn_state" : self.metrics["final_rnn_state"],
			"final_latent_mean" : self.metrics["final_latent_mean"],
			"final_latent_covar" : self.metrics["final_latent_covar"]
		}
		reader.start()

		# for rms calculation
		final_val,final_count = 0.0,0.0

		# load from test set
		batch = reader.next()
		feed_initial_rnn = session.run(self.initial_rnn)
		feed_initial_latent_mean = np.zeros((self.config["batch_size"], self.config["z_size"]))
		feed_initial_latent_covar = session.run(self.batch_cov_identity)

		while batch != None:

			feed_dict = {}
			feed_dict[self.u.name] = batch["inputs"]
			feed_dict[self.x.name] = batch["outputs"]
			feed_dict[self.mask.name] = batch["mask"]
			feed_dict[self.initial_rnn] = feed_initial_rnn
			feed_dict[self.initial_latent_mean.name] = feed_initial_latent_mean
			feed_dict[self.initial_latent_covar.name] = feed_initial_latent_covar

			vals = session.run(fetches, feed_dict)


			if batch["refresh"] == 1:
				feed_initial_rnn = session.run(self.initial_rnn)
				feed_initial_latent_mean = np.zeros((self.config["batch_size"], self.config["z_size"]))
				feed_initial_latent_covar = session.run(self.batch_cov_identity)
			else:
				feed_initial_rnn = vals["final_rnn_state"]
				feed_initial_latent_mean = vals["final_latent_mean"]
				feed_initial_latent_covar = vals["final_latent_covar"]


			# print("OUTPUTS store 1:", vals["outputs"])

			reshape_outputs = np.reshape(vals["outputs"],[self.config["batch_size"],self.config["time_len"],-1])

			for i in range(self.config["batch_size"]):
				for j in range(self.config["time_len"]):
					if batch['mask'][i][j] == 1.0:
						if batch["outputs"][i][j][0]*self.std+self.mean > 1:
							final_val += ((reshape_outputs[i][j][0]*self.std - batch["outputs"][i][j][0]*self.std)/(batch["outputs"][i][j][0]*self.std+self.mean))**2
							final_count += 1

			batch = reader.next()


		return np.sqrt(final_val/final_count)

