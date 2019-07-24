""" NOTE: This version works with 1 channels """
import os
import datetime
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'

now=datetime.datetime.now()
CURRENT_DAY  = now.strftime('%m%d')
CURRENT_TIME = now.strftime('%H%M')

BASE_DIR = os.path.dirname(os.path.realpath(__file__))
DATA_ROOT = BASE_DIR+'/../DATA'

WIDTH			= 128
HEIGHT			= 128
CHANNEL			= 4

SEQ_LEN 	   	= 15
INPUT_SEQ_LEN  	= 7
OUTPUT_SEQ_LEN	= SEQ_LEN - INPUT_SEQ_LEN
MAX_VALUE	   	= 255
FORMAT			= 'NHWC'
# FORMAT			= 'NCHW'
GPU_MEM_FRACTION = 0.49

class TrainingConfiguration:
	DATA_PACKAGE	= '/size-101/all-channels'
	BATCH_SIZE 		= 4
	BPTT	 = True
	N_EPOCHS = 200
	N_ITER	 = 8000//BATCH_SIZE
	LOG_STEP = 1
	VALIDATE_STEP = 5
	VALIDATION_BATCH_SIZE = BATCH_SIZE
	VERBOSE  = False#True
	LOGGING  = True
	SAVING	 = True
	SUMMARY  = False
	
	LEARNING_RATE	= 1e-4
	MOMENTUM		= 0.50
	DECAY_RATE		= 0.70
	DECAY_STEP		= 5
