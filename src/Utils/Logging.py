def CREATE_LOG(file_path, model_des, msg=None):
	from datetime import datetime
	with open(file_path, 'at') as f:
		f.write('===================================\n')
		f.write('#Model: '+model_des + '\n')
		f.write('#Start time: %s\n'%(datetime.now()))
		if msg is not None: f.write(msg+'\n')
		f.write('===================================\n')
def WRITE_LOG(file_path, msg):
	with open(file_path, 'at') as f:
		f.write(msg+'\n')

