import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__))+'/../')
from datetime import datetime
import tensorflow as tf
import numpy as np
import Configs
import TrainOps
from Utils import DataLoader, Logging

from ConvGRU.Model_3L import Model

train_cfg = Configs.TrainingConfiguration
valid_batch_size = train_cfg.VALIDATION_BATCH_SIZE
AUGMENTATION = False#True
BALANCE 	 = False#True
AUG_TEXT = 'AUG' if AUGMENTATION else 'NoAUG'
BAL_TEXT = 'BAL' if BALANCE else 'NoBAL'
# train_cfg.BATCH_SIZE = 2
""" Data folders """
data_dir		= Configs.DATA_ROOT+train_cfg.DATA_PACKAGE
train_data_file = data_dir+'/train.npy'
print('DATA-FOLDER: ', data_dir)

METRIC='MSE'
# METRIC='MSE+MAE+SSIM'
""" Folders for storing the training process """
model_dir = Model.MODEL_DIR + '-%s/'%(METRIC)
train_dir = model_dir + 'STEP%d-B%d-%s-%s'%(Configs.OUTPUT_SEQ_LEN, train_cfg.BATCH_SIZE, AUG_TEXT, BAL_TEXT)
log_dir    		= train_dir + '/logs'
summary_dir  	= train_dir + '/summary/%s-%s'%(Configs.CURRENT_DAY, Configs.CURRENT_TIME)
save_dir   		= train_dir + '/checkpoint'
save_dir_best 	= train_dir + '/best-model'

log_training   = log_dir+'/log-training-%s.txt'%(Configs.CURRENT_DAY)
log_validating = log_dir+'/log-validating-%s.txt'%(Configs.CURRENT_DAY)
print('MODEL-FOLDER:', train_dir)

def MAKE_FOLDERS_FOR_SAVING():
	if not os.path.isdir(model_dir): os.mkdir(model_dir)
	if not os.path.isdir(train_dir): os.mkdir(train_dir)
	if not os.path.isdir(log_dir) and train_cfg.LOGGING:  os.mkdir(log_dir)
	if not os.path.isdir(train_dir + '/summary') and train_cfg.SUMMARY:  os.mkdir(train_dir + '/summary')
	if not os.path.isdir(summary_dir) and train_cfg.SUMMARY:  os.mkdir(summary_dir)
	if not os.path.isdir(save_dir) and train_cfg.SAVING: os.mkdir(save_dir)
	if not os.path.isdir(save_dir_best) and train_cfg.SAVING: os.mkdir(save_dir_best)
	#if not os.path.isdir(validation_dir) and train_cfg.SAVING: os.mkdir(validation_dir)

def Denormed_MSE(targets, outputs):
	mse_list = []
	for i in range(Configs.OUTPUT_SEQ_LEN):
		mse_list.append(tf.losses.mean_squared_error(targets[i]*Configs.MAX_VALUE, 
													 outputs[i]*Configs.MAX_VALUE))
	return tf.reduce_mean(mse_list)

print('*GRAPH-BUILDING for TRAINING: STARTING')
graph = tf.Graph()
with graph.as_default():
	lr = tf.placeholder(tf.float32, shape=[], name="learning_rate")
	expand_dim_axis = -1 if Configs.FORMAT == 'NHWC' else 2
	img_seq = tf.placeholder(tf.float32, [None,Configs.SEQ_LEN,Configs.HEIGHT,Configs.WIDTH,Configs.CHANNEL], name='input_sequence')
	images_list = tf.unstack(img_seq/Configs.MAX_VALUE, axis=1)
	inputs  = images_list[:Configs.INPUT_SEQ_LEN]
	targets = images_list[Configs.INPUT_SEQ_LEN:]

	outputs = Model.EncodeAndDecode(inputs, out_len=Configs.OUTPUT_SEQ_LEN, 
					batch_size=train_cfg.BATCH_SIZE, training=True)
	train_mse = Denormed_MSE(targets, outputs)
	
	trainable_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='seq2seq_model')
	n_params = len(trainable_params)
	total_params = np.sum([np.prod(v.get_shape().as_list()) for v in trainable_params])
	params_text = ' --> Trainable-params: %d; Total no. of params: %d'%(n_params, total_params)
	print(params_text)
	# exit()
	grads, train_cost = TrainOps.Get_Gradients(targets, outputs, out_len=Configs.OUTPUT_SEQ_LEN,
								params=trainable_params, bptt=train_cfg.BPTT, metric=METRIC, balance=BALANCE)
	
	grads, __ = tf.clip_by_global_norm(grads, 2.5*train_cfg.BATCH_SIZE)
	
	update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
	with tf.control_dependencies(update_ops):
		optimizer  = tf.train.AdamOptimizer(beta1=train_cfg.MOMENTUM, learning_rate=lr)
		# optimizer = tf.train.GradientDescentOptimizer()
		train_step = optimizer.apply_gradients(zip(grads, trainable_params))
	
	model_saver = tf.train.Saver(trainable_params, max_to_keep=100)
	checkpoint_saver = tf.train.Saver()
	
	""" For validation """
	valid_outputs  = Model.EncodeAndDecode(inputs, out_len=Configs.OUTPUT_SEQ_LEN, 
						batch_size=valid_batch_size, training=False)
	valid_pred_seq = tf.clip_by_value(tf.squeeze(tf.stack(valid_outputs,axis=1)), 0., 1.)
	# valid_pred_seq = tf.nn.relu(tf.squeeze(tf.stack(valid_outputs,axis=1)))
	valid_mse = Denormed_MSE(targets, valid_outputs)
	valid_loss = TrainOps.CalculateLosses(targets, valid_outputs, Configs.OUTPUT_SEQ_LEN, metric=METRIC)
print('*GRAPH-BUILDING: FINISHED')

def Validating(sess, valid_data, n_valid, draw_result=False):
	if train_cfg.VERBOSE: print('		+Validating. Please wait...')
	valid_err = 0
	valid_cost = 0
	best, worst = 1.0, 0.0
	if AUGMENTATION: valid_data = TrainOps.Numpy_Augment(valid_data)
	for i in range(0, n_valid, valid_batch_size):
		valid_batch = valid_data[i:i+valid_batch_size]
		batch_loss, batch_mse, pred_batch = sess.run([valid_loss,valid_mse, valid_pred_seq], feed_dict={img_seq: valid_batch})
		valid_err += batch_mse
		valid_cost += batch_loss
		
		if best > batch_loss:  best_batch=(valid_batch, pred_batch)
		if worst < batch_loss: worst_batch=(valid_batch, pred_batch)
	
	valid_err /= (n_valid//valid_batch_size)
	valid_cost /= (n_valid//valid_batch_size)
	if train_cfg.VERBOSE: print('		+Validation: Cost: %f; MSE: %f'%(valid_cost, valid_err))
	
	return valid_cost, valid_err
	
def Train():
	train_data, valid_data = DataLoader.Load_Divided_Data(train_data_file)
	n_train, n_valid = train_data.shape[0], valid_data.shape[0]	
	print('*DATA-LOADING: FINISHED')	
	
	""" Prepare folders and files for saving information of the training process """
	MAKE_FOLDERS_FOR_SAVING()
	if train_cfg.LOGGING:
		model_description='LR=%f, 1-epoch=%d, batch=%d, n-train=%d, n-validation=%d'%\
					(train_cfg.LEARNING_RATE,train_cfg.N_ITER,train_cfg.BATCH_SIZE, n_train, n_valid)
		Logging.CREATE_LOG(log_training, Model.MODEL_DESCRIPTION,
					msg=' Training: '+model_description+'\n'+params_text)
		Logging.CREATE_LOG(log_validating, Model.MODEL_DESCRIPTION,
					msg=' Validating: '+model_description+'\n'+params_text)
	if train_cfg.SUMMARY:
		train_sum_writer = tf.summary.FileWriter(summary_dir + '/train')
		valid_sum_writer = tf.summary.FileWriter(summary_dir + '/validation')
		mse_summary = tf.Summary(value=[tf.Summary.Value(tag='MSE', simple_value=None)])		
	
	""" Prepare hyper-parameters """
	min_valid		= 10.0
	train_lr		= train_cfg.LEARNING_RATE
	""" ======================================================================= """
	sess_cfg = tf.ConfigProto()
	sess_cfg = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
	# sess_cfg.gpu_options.allow_growth = True
	# sess_cfg.gpu_options.per_process_gpu_memory_fraction = Configs.GPU_MEM_FRACTION
	
	with tf.Session(graph=graph, config=sess_cfg) as sess:
		tf.global_variables_initializer().run()		
		# checkpoint_saver.restore(sess, save_dir+'/training-checkpoint')		
		print('*PREPARATION is finished. TRAINING is starting now...')
		
		start_time = datetime.now()
		for iEpoch in range(0,train_cfg.N_EPOCHS):
			if (iEpoch+1)%train_cfg.DECAY_STEP == 0: 
				if train_lr > 1e-12: train_lr = train_lr*train_cfg.DECAY_RATE
			""" Training Epoch starts """
			iter_loss, iter_mse = 0, 0
			epoch_data = TrainOps.Train_Data(train_data, n_train, n_train)
			if AUGMENTATION: epoch_data = TrainOps.Numpy_Augment(epoch_data)
			for iBatch in range(0, n_train, train_cfg.BATCH_SIZE):
				train_batch = epoch_data[iBatch:iBatch+train_cfg.BATCH_SIZE]
				if train_cfg.SUMMARY:
					merged_summary 	 = tf.summary.merge_all()
					__, loss, err, summary = sess.run([train_step, train_cost, train_mse, merged_summary], 
											feed_dict={img_seq: train_batch, lr: train_lr})
				else:
					__, loss, err = sess.run([train_step, train_cost, train_mse], 
											feed_dict={img_seq: train_batch, lr: train_lr})
				iter_mse += err; iter_loss += loss
				# print('Successful Training Iter:', iEpoch+1, loss, err)
			iter_mse /= train_cfg.N_ITER; iter_loss /= train_cfg.N_ITER			
			""" Epoch finishs """
			if iEpoch == 0 or (iEpoch+1)%train_cfg.LOG_STEP == 0: 
				text = ' -Epoch %5d: Loss=%f; MSE=%f\n'%(iEpoch+1, iter_loss, iter_mse) + \
						'		Time passed: %s'%(datetime.now() - start_time)
				if train_cfg.VERBOSE: print(text)
				if train_cfg.LOGGING: Logging.WRITE_LOG(log_training, msg=text)
				if train_cfg.SAVING: checkpoint_saver.save(sess, save_dir+'/training-checkpoint')
				if train_cfg.SUMMARY:
					mse_summary.value[0].simple_value = iter_mse
					train_sum_writer.add_summary(mse_summary, iEpoch+1)
					train_sum_writer.add_summary(summary, iEpoch+1)
			
			""" Validation starts """
			if iEpoch == 0 or (iEpoch+1)%train_cfg.VALIDATE_STEP == 0:
				valid_cost, valid_mse = Validating(sess, valid_data, n_valid)				
				text = ' -Epoch %5d: Cost=%f; MSE=%f'%(iEpoch+1,valid_cost,valid_mse)
				if train_cfg.LOGGING: Logging.WRITE_LOG(log_validating, msg=text)
				if train_cfg.SAVING:  model_saver.save(sess, save_dir_best+'/model', global_step=(iEpoch+1))
			""" Validation finishs """
				
			
		end_time = datetime.now()
		if train_cfg.SUMMARY:
			train_sum_writer.close()
			valid_sum_writer.close()
	
	print('	- TOTAL TRAINING TIME:', end_time-start_time)
	""" ======================================================================= """
if __name__ == '__main__':
	a=1
	# Train()
