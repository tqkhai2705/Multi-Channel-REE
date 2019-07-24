import tensorflow as tf

def MSE(target, output):
	mse = tf.losses.mean_squared_error(target, output)
	return mse

def MAE(target, output):
	mae = tf.losses.absolute_difference(target, output, reduction=tf.losses.Reduction.MEAN)
	return mae
	
def PSNR(target, output, NHWC=True):
	if NHWC: psnr = tf.image.psnr(target, output, max_val=1)
	else: psnr = tf.image.psnr(tf.transpose(target,[0,2,3,1]), tf.transpose(output,[0,2,3,1]), max_val=1)
	return tf.reduce_mean(psnr)
	
def SSIM(target, output, NHWC=True):
	if NHWC: ssim = tf.image.ssim(target, output, max_val=1)
	else: ssim = tf.image.ssim(tf.transpose(target,[0,2,3,1]), tf.transpose(output,[0,2,3,1]), max_val=1)
	return tf.reduce_mean(ssim)
	
def MS_SSIM(target, output, NHWC=True):
	if NHWC: Yt, Yo = target, output
	else: 	 Yt, Yo = tf.transpose(target,[0,2,3,1]), tf.transpose(output,[0,2,3,1])
	# ms_ssim = tf.image.ssim_multiscale(Yt, Yo, max_val=1, power_factors=[0.25,0.25,0.25,0.25])
	ms_ssim = tf.image.ssim_multiscale(Yt, Yo, max_val=1, power_factors=[0.3,0.3,0.2,0.2])
	return tf.reduce_mean(ms_ssim)

def abs_diff(Y1,Y2): 
	return tf.losses.absolute_difference(Y1, Y2, reduction=tf.losses.Reduction.NONE)
def GDL(target, output, alpha=2, NHWC=True):
	"""
		Gradient Difference Loss: Mathieu et al. (2016)
		Note:
		1) This function returns the mean GDL of a batch (for convenience)
		2) We can change to tf.image.image_gradient() later (for absolute_difference)
		3) Only ONE CHANNEL is allowed in this version
	"""
	if NHWC: Yt, Yo = target[:,:,:,0], output[:,:,:,0]
	else: 	 Yt, Yo = target[:,0,:,:], output[:,0,:,:]
	
	### Calculate component gradients ###
	### (1) Vertical direction ###
	ver_diff = abs_diff(abs_diff(Yt[:,1:,:], Yt[:,:-1,:]), abs_diff(Yo[:,1:,:], Yo[:,:-1,:]))
	### (2) Horizontal direction ###
	hor_diff = abs_diff(abs_diff(Yt[:,:,:-1], Yt[:,:,1:]), abs_diff(Yo[:,:,:-1], Yo[:,:,1:]))
	
	### Calculate sum of each image in the batch ###
	grad_diff_ver = tf.reduce_mean(tf.pow(ver_diff, alpha), axis=[1,2])
	grad_diff_hor = tf.reduce_mean(tf.pow(hor_diff, alpha), axis=[1,2])
	# grad_diff_ver = tf.pow(ver_diff[:,:,:-1], alpha)
	# grad_diff_hor = tf.pow(hor_diff[:,1:,:], alpha)
	
	batch_gdl = grad_diff_ver + grad_diff_hor
	return tf.reduce_mean(batch_gdl)

def GradSharp(target, output, NHWC=True):
	if NHWC: Yt, Yo = target[:,:,:,0], output[:,:,:,0]
	else: 	 Yt, Yo = target[:,0,:,:], output[:,0,:,:]
	
	Yt_grad_i = abs_diff(Yt[:,1:,1:], Yt[:,:-1,1:])
	Yt_grad_j = abs_diff(Yt[:,1:,1:], Yt[:,1:,:-1])
	Yt_grad   = Yt_grad_i + Yt_grad_j
	
	Yo_grad_i = abs_diff(Yo[:,1:,1:], Yo[:,:-1,1:])
	Yo_grad_j = abs_diff(Yo[:,1:,1:], Yo[:,1:,:-1])
	Yo_grad   = Yo_grad_i + Yo_grad_j
	
	### We use max_val = 1
	# max_val = 1.
	# temp = max_val*max_val/tf.reduce_mean(abs_diff(Yt_grad, Yo_grad))
	temp = 1./tf.reduce_mean(abs_diff(Yt_grad, Yo_grad))
	gradient_sharpness = 10.*tf.log(temp)/tf.log(10.)
	return gradient_sharpness
	
	
def PearsonCorr(target, output):
	delta_target = target-tf.reduce_mean(target)
	delta_output = output-tf.reduce_mean(output)
	numerator  = tf.reduce_sum(delta_target*delta_output)
	term1 = tf.reduce_sum(tf.square(delta_target))
	term2 = tf.reduce_sum(tf.square(delta_output))
	denominator = tf.sqrt(term1*term2)
	return numerator/denominator

if __name__ == '__main__':
	a = tf.constant([[[0.0, 0.1, 0.2, 0.3, 0.4],
					  [0.4, 0.3, 0.3, 0.9, 0.2]],
					 [[0.8, 0.1, 0.2, 0.3, 0.1],
					  [0.2, 0.5, 0.3, 0.1, 0.4]]])
	b = tf.random.uniform(tf.shape(a))
	# a = tf.random.normal([4,101,101,4], mean=0.3, stddev=0.3)
	mask40 = tf.cast(tf.greater(a, 0.5882), tf.float32)*16.0
	mask20 = tf.cast(tf.greater(a, 0.3529), tf.float32)*8.0
	mask10 = tf.cast(tf.greater(a, 0.2353), tf.float32)*4.0
	mask5  = tf.cast(tf.greater(a, 0.1765), tf.float32)*2.0
	maskall  = tf.reduce_max([mask40, mask20, mask10, mask5, tf.ones_like(a)], axis=0)
	new_a = tf.multiply(a, maskall)
	
	F = tf.cast(tf.random_uniform([], minval=0, maxval=2, dtype=tf.int32), tf.float32)
	c = F*a + (1 - F)*b
	sess = tf.Session()
	m = sess.run(maskall)
	b = sess.run(new_a)
	print(m[1])
	print(b[1])
	# for i in range(10):
		# x = sess.run(c[0])
		# print(x)
	
