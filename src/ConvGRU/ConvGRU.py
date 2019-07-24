import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__))+'/../')

import tensorflow as tf
import numpy as np
import Configs, Ops

Conv 	= Ops.Conv
Deconv 	= Ops.Deconv
	
class ConvGRUCell(tf.contrib.rnn.RNNCell):
	""" Some Abbrev.:
		- dim: Dimension of both input and state
				(So here the input must have the same shape with state. 
				Please change this to meet your needs)
		- dr:  Dilation Rate
		- output type: 'no', 'same', 'down', 'up'
	"""
	def __init__(self, name, dim,  
					inp_ker=[3,3], inp_dr=1,
					state_chan=1, state_ker=[3,3], state_dr=1, tied_bias=False,
					output_chan=None, out_ker=None, out_type='same', act_function=Ops.ACTIVATION, output_dim=None):
		self._name			= name
		self._dim			= dim
		self._inp_ker		= inp_ker
		self._out_ker		= out_ker
		self._inp_dr		= inp_dr
		self._state_chan	= state_chan
		self._state_ker		= state_ker
		self._state_dr		= state_dr
		self._tied_bias		= tied_bias
		self._output_chan	= output_chan #if output_chan else state_chan
		self._output_type	= out_type.lower() 
		self._state_shape	= dim + [state_chan] if Configs.FORMAT == 'NHWC' else [state_chan] + dim
		self._act			= act_function
		self._output_dim	= output_dim
		
		""" Bias Weights (If not to normalize) """
		if tied_bias:
			self._cnn_bias = None
			with tf.variable_scope(self._name, reuse=tf.AUTO_REUSE):
				# self._bz = tf.get_variable('bz', self._state_shape, initializer=tf.constant_initializer(-2))
				self._bz = tf.get_variable('bz', self._state_shape, initializer=Ops.COMMON_INIT)
				self._br = tf.get_variable('br', self._state_shape, initializer=Ops.COMMON_INIT)
				self._bh = tf.get_variable('bh', self._state_shape, initializer=Ops.COMMON_INIT)
		else:
			self._cnn_bias = Ops.COMMON_INIT
	
	@property
	def state_size(self): return np.sum(self._state_shape)
	@property
	def output_size(self): return self._state_shape
	
	def ini_states(self, batch_s): return tf.zeros([batch_s] + self._state_shape)
		
	def __call__(self, Xt, H_prev, training=False):
		_input   = Xt is not None ### By default, each cell has input but in some cases has no input
		
		H_prev_conv = Conv(H_prev, self._name+'/H_prev_conv', 
						self._state_chan*3, self._state_ker, dr=self._state_dr, 
						bias_init = self._cnn_bias)
		Rt, Zt, Ht_hh = tf.split(H_prev_conv, 3, -1)
		
		if _input:
			Xt_conv = Conv(Xt, self._name+'/Xt_conv', 
						self._state_chan*3, self._inp_ker, bias_init=None)
			Rt_i, Zt_i, Ht_i = tf.split(Xt_conv, 3, -1)
			Rt += Rt_i
			Zt += Zt_i
			
		""" Reset Gate """
		if self._tied_bias: Rt += self._br
		Rt_act = tf.sigmoid(Rt) # Activate it before the next step
		
		""" Update Gate """
		if self._tied_bias: Zt += self._bz
		Zt_act = tf.sigmoid(Zt) # Activate it before the next step
		
		""" Candidate State """
		Ht_ = Rt_act*Ht_hh
		if _input: Ht_ += Ht_i
		if self._tied_bias: Ht_ += self._bh
		Ht_can = self._act(Ht_)
		
		""" New State """
		Ht = (1 - Zt_act)*H_prev + Zt_act*Ht_can
		
		""" Block Output """
		if 	 self._output_type == 'no':   Yt = None
		elif self._output_type == 'same': # Keep output with the same shape
			# if self._output_chan and self._output_chan != self._state_chan:
				# Yt = Conv(Ht, self._name+'/Yt', self._output_chan, self._out_ker, stride=1, 
						# bias_init=Ops.COMMON_INIT, activate=self._act)
			# else: Yt = Ht
			Yt = Ht
		elif self._output_type == 'down': # Downsampling
			Yt = Conv(Ht, self._name+'/Yt', self._output_chan, self._out_ker, stride=2, 
						bias_init=Ops.COMMON_INIT, activate=self._act) 	
		elif self._output_type == 'up':   # Upsampling
			if self._output_dim: out_dim = self._output_dim
			else: 				 out_dim = [d*2 for d in self._dim]
			Yt = Deconv(Ht, self._name+'/Yt', out_dim, self._output_chan, self._out_ker,
						bias_init=Ops.COMMON_INIT, activate=self._act)
				
		return (Yt, Ht)

