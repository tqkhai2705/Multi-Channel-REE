import tensorflow as tf
import numpy as np
from Utils import Metrics

b_MAE_MSE = True
def Make_Mask(a):
	mask40 = tf.cast(tf.greater(a, 0.5881), tf.float32)*8
	mask20 = tf.cast(tf.greater(a, 0.3528), tf.float32)*4
	mask10 = tf.cast(tf.greater(a, 0.2352), tf.float32)*2
	mask5  = tf.cast(tf.greater(a, 0.1764), tf.float32)*1
	mask   = tf.reduce_max([mask40, mask20, mask10, mask5, tf.ones_like(a)], axis=0)
	return mask
	
def CalculateStepLosses(step_target, step_output, metric='MSE'):
	if b_MAE_MSE:
		m = Make_Mask(step_target)
		mse_tensor = tf.multiply(m, (step_target - step_output)**2)
		mae_tensor = tf.multiply(m, tf.abs(step_target - step_output))
	else:
		mse_tensor = (step_target - step_output)**2
		mae_tensor = tf.abs(step_target - step_output)
	mse = tf.reduce_mean(mse_tensor)
	mae = 0.1*tf.reduce_mean(mae_tensor)
	
	ms_ssim = 0.02*(1. - Metrics.MS_SSIM(step_target, step_output, True))
	ssim 	= 0.02*(1. - Metrics.SSIM(step_target, step_output, True))
	
	if metric == 'MSE': return mse
	if metric == 'MAE': return mae
	if metric == 'SSIM': return ssim
	if metric == 'MS_SSIM': return ms_ssim
	if metric == 'MSE+MAE': return 0.5*(mse + mae)
	if metric == 'MSE+SSIM': return 0.5*(mse + ssim)
	if metric == 'MSE+MS_SSIM': return 0.5*(mse + ms_ssim)
	if metric == 'MAE+MS_SSIM': return 0.5*(mae + ms_ssim)
	if metric == 'MSE+MAE+SSIM': return (mse + mae + ssim)/3.
	if metric == 'MSE+MAE+MS_SSIM': return (mse + mae + ms_ssim)/3.
	
def CalculateStepLosses_Sum(step_target_4c, step_output_4c, metric='MSE'):
	sum_loss = 0
	for i in range(4):
		sum_loss += CalculateStepLosses(step_target_4c[:,:,:,i:i+1], step_output_4c[:,:,:,i:i+1], metric)
	return sum_loss

def CalculateLosses(targets, outputs, out_len, metric='MSE'):
	losses = [CalculateStepLosses(targets[i], outputs[i], metric) for i in range(out_len)]
	return tf.reduce_mean(losses)

def Get_Gradients(targets, outputs, out_len, params, bptt=False, metric='MSE', balance=False):
	global b_MAE_MSE; b_MAE_MSE = balance
	"""==================== LOSSES CUMULATION ===================="""
	losses = [CalculateStepLosses(targets[i], outputs[i], metric) for i in range(out_len)]
	cost = tf.reduce_mean(losses)
	
	"""========== GRADIENTS ACCUMULATION for BPTT-TRAINING =========="""
	def BPTT_gradients():
		print(' - Back-Propagation Through Time (Full-BPTT)')		
		#gradients = None
		for step in range(out_len):
			print(' -Timestep:',step+1)
			step_grads = tf.gradients(losses[step], params)
			if step==0: gradients = step_grads
			else: ### Accumulating gradients among timesteps ###
				for i in range(len(params)):
					if gradients[i] is None:    gradients[i]  = step_grads[i]
					elif step_grads[i] != None: gradients[i] += step_grads[i]
					
		# gradients = [grad for grad in gradients if grad is not None]
		f = 1./out_len; gradients = [grad*f if grad != None else None for grad in gradients ]
		return gradients
	
	if bptt: grads = BPTT_gradients()
	else: 	 grads = tf.gradients(cost, params)
	return grads, cost

def Build_Training(targets, outputs, out_len, learning_rate, batch_size, 
					max_norm=1.0, bptt=False, summarize=False):
	"""==================== LOSSES CUMULATION ===================="""
	losses = [CalculateStepLosses(targets[i], outputs[i]) for i in range(out_len)]
	mean_loss = tf.reduce_mean(losses)
	trainable_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='encoder|decoder')
	n_params = len(trainable_params)
	print('Total number of trainable-params:', n_params)
	
	"""========== GRADIENTS ACCUMULATION for BPTT-TRAINING =========="""
	def BPTT_gradients():
		print(' - Back-Propagation Through Time (Full-BPTT)')		
		#gradients = None
		for step in range(out_len):
			print(' -Timestep:',step+1)
			step_grads = tf.gradients(losses[step], trainable_params)
			if step==0: gradients = step_grads
			else: ### Accumulating gradients among timesteps ###
				for i in range(n_params):
					if gradients[i] != None and step_grads[i] != None: 
						gradients[i] += step_grads[i]		
		return gradients
	
	"""==================== TRAINING OPTIMIZER ===================="""
	#global_step = tf.Variable(0, trainable=False, dtype=tf.int32)
	if bptt: grads = BPTT_gradients()
	else: 	 grads = tf.gradients(mean_loss, trainable_params)	
	
	f = 1/batch_size #*out_len
	grads = [grad*f if grad != None else None for grad in grads]
	
	if summarize:
		grad_histograms = []
		grad_norms = []
		for i in range(n_params):
			if grads[i] is None: continue
			var_name = trainable_params[i].name[:-2]
			grad_histograms.append(tf.summary.histogram('%s-grad'%(var_name), grads[i]))
			grad_norms.append(tf.norm(grads[i]))
		tf.summary.merge(grad_histograms)
		tf.summary.scalar('grad-norms/max', tf.reduce_max(grad_norms))
		tf.summary.scalar('grad-norms/mean', tf.reduce_mean(grad_norms))
		tf.summary.scalar('grad-norms/min', tf.reduce_min(grad_norms))
	
	if max_norm != None: grads = [tf.clip_by_norm(grad, max_norm) for grad in grads]
	grads_and_params = zip(grads, trainable_params)
	
	with tf.variable_scope('train_optimizer', reuse=tf.AUTO_REUSE):
		# optimizer = tf.train.AdamOptimizer(beta1=0.9, learning_rate=learning_rate)
		optimizer = tf.train.GradientDescentOptimizer()
		train_op  = optimizer.apply_gradients(grads_and_params, global_step=None)
	return train_op, mean_loss

def Numpy_Augment(batch_img_seq):
	new_batch = []
	n = batch_img_seq.shape[0]
	np.random.seed()
	random_transform_choices = np.random.randint(2, size=n) #[0,1] distribution
	random_reverse_choices = np.random.randint(2, size=n) #[0,1] distribution
	random_illum_choices = np.random.randint(2, size=n) #[0,1] distribution
	
	for i_seq in range(n):
		img_seq = batch_img_seq[i_seq]#[15, 101, 101, 4]
		if random_transform_choices[i_seq]:
			randtype = np.random.randint(7)
			if randtype == 0: img_seq = np.flip(img_seq, 1) #flip_up_down
			elif randtype == 1: img_seq = np.flip(img_seq, 2) #flip_left_right
			elif randtype == 2: img_seq = np.flip(np.flip(img_seq, 1),2) #flip_ud_lr = rotate(1,2) two times
			elif randtype == 3: img_seq = np.rot90(img_seq, 1, (1,2)) #rotate counter-clockwise
			elif randtype == 4: img_seq = np.rot90(img_seq, 1, (2,1)) #rotate clockwise
			elif randtype == 5: img_seq = np.flip(np.rot90(img_seq, 1, (1,2)), 1) #flip across the 1st diagonal
			elif randtype == 6: img_seq = np.flip(np.rot90(img_seq, 1, (1,2)), 2) #flip across the 2nd diagonal
		if random_reverse_choices[i_seq]: img_seq = np.flip(img_seq, 0) #reverse sequence order
		if random_illum_choices[i_seq]: img_seq = np.uint8(np.float64(img_seq)*np.random.uniform(0.8,1.2))
		new_batch.append(img_seq)
	return np.asarray(new_batch)

def Train_Data(dat, n_samples, max_range):
	np.random.seed()
	indices = np.random.permutation(n_samples)[:max_range]
	#print('  - Random indices:', indices)
	return dat[indices]
