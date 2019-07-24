import os, sys
from .ConvGRU import ConvGRUCell
import Ops
import tensorflow as tf

INPUT_K  = 3
OUTPUT_K = 3
# STATE_CHANS = [64,128,128]
STATE_CHANS = [64,32,16]
# STATE_CHANS = [32,64,64]

class Model:
	MODEL_DIR = os.path.dirname(os.path.abspath(__file__))+'/Model_3L'
	MODEL_DESCRIPTION = 'Our Setting; ConvGRUCell: 3 Layers'
	
	def EncodeAndDecode(inputs, out_len, batch_size, training=True, targets=None):

		"""==================== ENCODER Cells ===================="""
		with tf.variable_scope('seq2seq_model', reuse=tf.AUTO_REUSE):
			e_cell1 = ConvGRUCell('e_cell1', inp_ker=INPUT_K,
								dim=[64,64], state_chan=STATE_CHANS[0], state_ker=5, state_dr=1, 
								output_chan=STATE_CHANS[1], out_ker=OUTPUT_K, out_type='down')
			e_cell2 = ConvGRUCell('e_cell2', inp_ker=INPUT_K,
								dim=[32,32], state_chan=STATE_CHANS[1], state_ker=5, state_dr=1, 
								output_chan=STATE_CHANS[2], out_ker=OUTPUT_K, out_type='down')
			e_cell3 = ConvGRUCell('e_cell3', inp_ker=INPUT_K,
								dim=[16,16], state_chan=STATE_CHANS[2], state_ker=3, state_dr=1, 
								out_type='no')
			
			""" ENCODING Operations """
			h1,h2,h3 = e_cell1.ini_states(batch_size), e_cell2.ini_states(batch_size),e_cell3.ini_states(batch_size)
			for inp in inputs:
				inp_ = Ops.Inp_Conv(inp)
				e1_o, h1 = e_cell1(inp_, h1)
				e2_o, h2 = e_cell2(e1_o, h2)
				__,   h3 = e_cell3(e2_o, h3)
		
			d_cell3 = ConvGRUCell('d_cell3', inp_ker=INPUT_K,
								dim=[16,16], state_chan=STATE_CHANS[2], state_ker=3, state_dr=1, 
								output_chan=STATE_CHANS[1], out_ker=OUTPUT_K, out_type='up')
			d_cell2 = ConvGRUCell('d_cell2', inp_ker=INPUT_K,
								dim=[32,32], state_chan=STATE_CHANS[1], state_ker=5, state_dr=1, 
								output_chan=STATE_CHANS[0], out_ker=OUTPUT_K, out_type='up')
			d_cell1 = ConvGRUCell('d_cell1', inp_ker=INPUT_K,
								dim=[64,64], state_chan=STATE_CHANS[0], state_ker=5, state_dr=1, 
								out_type='same')
			
			""" DECODING Operations """
			outputs = []
			for i in range(out_len):
				d3_o, h3 = d_cell3(None, h3)
				d2_o, h2 = d_cell2(d3_o, h2)
				d1_o, h1 = d_cell1(d2_o, h1)				
				
				outputs.append(Ops.Gen_Output2(d1_o))
		
		return outputs
