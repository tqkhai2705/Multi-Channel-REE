import tensorflow as tf
import numpy as np
import Configs

ACTIVATION = tf.tanh
# COMMON_INIT = tf.truncated_normal_initializer(mean=0.,stddev=0.1)
COMMON_INIT = tf.keras.initializers.he_normal() #MSRA initializer: used in the paper
# COMMON_INIT = tf.contrib.layers.xavier_initializer()
ZERO_INIT = tf.zeros_initializer()
CHANNEL_DIM = -1 if Configs.FORMAT=='NHWC' else 1

def Conv(a, name, output_chan, k_size=3, stride=1, dr=1, 
			activate=None, weight_init=COMMON_INIT, bias_init=None):
	""" This function automatically provides filter and bias.
		So it is a little convenient
		Note: Activation can be done later
	"""
	conv = tf.contrib.layers.conv2d(a, num_outputs=output_chan, 
					kernel_size=k_size, stride=stride,
					activation_fn=activate, reuse=tf.AUTO_REUSE,
					weights_initializer=weight_init, 
					biases_initializer=bias_init,
					data_format=Configs.FORMAT, rate=dr, scope=name)
	return conv

def GradientHighwayUnit(Xt, GHU_prev, num_chan=128, k_size=5, name='GHU'):
	with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
		Xt_conv 	   = Conv(Xt, name+'/Xt_conv', num_chan*2, k_size, bias_init=COMMON_INIT)
		GHU_prev_conv  = Conv(GHU_prev, name+'/GHU_prev_conv', num_chan*2, k_size, bias_init=None)
		Pt_x, St_x 	   = tf.split(Xt_conv, 2, -1)
		Pt_ghu, St_ghu = tf.split(GHU_prev_conv, 2, -1)
		
		Pt = tf.tanh(Pt_x + Pt_ghu)
		St = tf.sigmoid(St_x + St_ghu)
		GHU = St*Pt + (1 - St)*GHU_prev
		return GHU
	
def Gen_Output0(fmaps):
	decon = Deconv(fmaps, 'out_deconv', [Configs.HEIGHT,Configs.WIDTH], new_chan=Configs.CHANNEL*4, 
						k_size=3, activate=tf.nn.leaky_relu, bias_init=COMMON_INIT)
	conv1 = Conv(decon, name='out_conv1', output_chan=Configs.CHANNEL*2, 
						k_size=3, activate=tf.nn.leaky_relu, bias_init=COMMON_INIT)
	conv2 = Conv(conv1, name='out_conv2', output_chan=Configs.CHANNEL, 
						k_size=1, activate=None, bias_init=None)
	### NOTE: we do not clip the output here (in training) but clipping in validating and testing later
	return conv2

def Gen_Output2(fmaps):
	#fmaps: [64,64,64]
	reduced_fmaps =	Conv(fmaps, name='out_conv', output_chan=64, 
					k_size=3, stride=1, 
					activate=tf.nn.leaky_relu, bias_init=COMMON_INIT)
	fmap_list = tf.split(reduced_fmaps, Configs.CHANNEL, -1) # Each map item is of size [51,51,16]
	output_channel_list = []
	for i, fmap in enumerate(fmap_list):
		decon = Deconv(fmap, 'out_deconv%d'%i, [Configs.HEIGHT,Configs.WIDTH], new_chan=8, 
						k_size=3, activate=tf.nn.leaky_relu, bias_init=COMMON_INIT)
		conv1 = Conv(decon, name='out_conv%d_1'%i, output_chan=4, 
						k_size=3, activate=tf.nn.leaky_relu, bias_init=COMMON_INIT)
		conv2 = Conv(conv1, name='out_conv%d_2'%i, output_chan=1, 
						k_size=1, activate=None, bias_init=None)
		output_channel_list.append(conv2)
	### NOTE: we do not clip the output here (in training) but clipping in validating and testing later	
	return tf.concat(output_channel_list,axis=-1)
	
def Gen_Output1(fmaps):
	#fmaps: [51,51,64]
	fmap_list = tf.split(fmaps, Configs.CHANNEL, -1)
	output_channel_list = []
	for i, fmap in enumerate(fmap_list):
		decon = Deconv(fmap, 'out_deconv%d'%i, [Configs.HEIGHT,Configs.WIDTH], new_chan=8, 
						k_size=3, activate=tf.nn.leaky_relu, bias_init=COMMON_INIT)
		conv1 = Conv(decon, name='out_conv%d_1'%i, output_chan=4, 
						k_size=3, activate=tf.nn.leaky_relu, bias_init=COMMON_INIT)
		conv2 = Conv(conv1, name='out_conv%d_2'%i, output_chan=1, 
						k_size=1, activate=None, bias_init=None)
		output_channel_list.append(conv2)
	### NOTE: we do not clip the output here (in training) but clipping in validating and testing later	
	return tf.concat(output_channel_list,axis=-1)
	
def Inp_Conv(img):
	fmaps = Conv(img, name='pre_conv', output_chan=32, 
						k_size=3, stride=2, 
						activate=tf.nn.leaky_relu, bias_init=COMMON_INIT)
	return fmaps
	
def Deconv1(a, name, new_size, new_chan, k_size=3, stride=2,
				activate=None, weight_init=COMMON_INIT, bias_init=None):
	deconv = tf.contrib.layers.conv2d_transpose(a, num_outputs=new_chan, 
					kernel_size=k_size, stride=stride,
					activation_fn=activate, reuse=tf.AUTO_REUSE,
					weights_initializer=weight_init, biases_initializer=bias_init,
					data_format=Configs.FORMAT, scope=name)
	return deconv
	
def Deconv(a, name, new_size, new_chan, k_size=3, stride=2,
				activate=None, weight_init=COMMON_INIT, bias_init=None):
	if Configs.FORMAT == 'NHWC':
		a_resized = tf.image.resize_nearest_neighbor(a, size=new_size)
	else:
		a_transpose = tf.transpose(a, [0,2,3,1]) # Convert from NCHW --> NHWC (for the resize-operation)
		a_transpose_resized = tf.image.resize_nearest_neighbor(a_transpose, size=new_size)
		a_resized = tf.transpose(a_transpose_resized, [0,3,1,2]) 	# Convert back to NCHW
	
	deconv = tf.contrib.layers.conv2d(a_resized, num_outputs=new_chan, 
					kernel_size=k_size, 						# No need to have stride here
					activation_fn=activate, reuse=tf.AUTO_REUSE,
					weights_initializer=weight_init, 
					biases_initializer=bias_init,
					data_format=Configs.FORMAT, scope=name)
	return deconv
	
def LayerNorm(a, name):
	norm_axis = 1 if CHANNEL_DIM == -1 else 2
	return tf.contrib.layers.layer_norm(a, activation_fn=None,
								begin_norm_axis=norm_axis, begin_params_axis=CHANNEL_DIM,
								scope=name+'/l_norm', reuse=tf.AUTO_REUSE)
def BatchNorm(a, name, training=True):
	return tf.contrib.layers.batch_norm(a, is_training=training, data_format=Configs.FORMAT,
										scope=name+'/b_norm', reuse=tf.AUTO_REUSE)

class Bilinear_Sampler:
	""" This function is used for making the 'warping' operation
		Adapted from: https://github.com/iwyoo/tf-bilinear_sampler 
	"""
	
	def get_grid_array(N, H, W):
		N_i = tf.range(N)
		H_i = tf.range(1, H+1)
		W_i = tf.range(1, W+1)
		n, h, w, = tf.meshgrid(N_i, H_i, W_i, indexing='ij')
		n = tf.expand_dims(n, axis=3) # [N, H, W, 1]
		h = tf.expand_dims(h, axis=3) # [N, H, W, 1]
		w = tf.expand_dims(w, axis=3) # [N, H, W, 1]
		n = tf.cast(n, tf.float32) # [N, H, W, 1]
		h = tf.cast(h, tf.float32) # [N, H, W, 1]
		w = tf.cast(w, tf.float32) # [N, H, W, 1]
		return n, h, w
	  
	def bilinear_sampler(input, v, grid_components, dim):
		"""	Args:
			x - Input tensor [N, H, W, C]
			v - Vector flow tensor [N, H, W, 2], tf.float32
			grid_components - returns (n,h,w) of get_grid_array function
			dim - dimension of the feature map (H,W)
		"""
		x = tf.pad(input, ((0,0), (1,1), (1,1), (0,0)), mode='CONSTANT')
		vy, vx = tf.split(v, 2, axis=3)
		n, h, w = grid_components
		H,W = dim

		vx0 = tf.floor(vx)
		vy0 = tf.floor(vy)
		vx1 = vx0 + 1
		vy1 = vy0 + 1 # [N, H, W, 1]

		H_1 = tf.cast(H+1, tf.float32)
		W_1 = tf.cast(W+1, tf.float32)
		iy0 = tf.clip_by_value(vy0 + h, 0., H_1)
		iy1 = tf.clip_by_value(vy1 + h, 0., H_1)
		ix0 = tf.clip_by_value(vx0 + w, 0., W_1)
		ix1 = tf.clip_by_value(vx1 + w, 0., W_1)

		i00 = tf.concat([n, iy0, ix0], 3)
		i01 = tf.concat([n, iy1, ix0], 3)
		i10 = tf.concat([n, iy0, ix1], 3)
		i11 = tf.concat([n, iy1, ix1], 3) # [N, H, W, 3]
		i00 = tf.cast(i00, tf.int32)
		i01 = tf.cast(i01, tf.int32)
		i10 = tf.cast(i10, tf.int32)
		i11 = tf.cast(i11, tf.int32)

		x00 = tf.gather_nd(x, i00)
		x01 = tf.gather_nd(x, i01)
		x10 = tf.gather_nd(x, i10)
		x11 = tf.gather_nd(x, i11)
		w00 = tf.cast((vx1 - vx) * (vy1 - vy), tf.float32)
		w01 = tf.cast((vx1 - vx) * (vy - vy0), tf.float32)
		w10 = tf.cast((vx - vx0) * (vy1 - vy), tf.float32)
		w11 = tf.cast((vx - vx0) * (vy - vy0), tf.float32)
		output = tf.add_n([w00*x00, w01*x01, w10*x10, w11*x11])

		return output
