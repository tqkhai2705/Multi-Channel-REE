import cv2
import numpy as np

v_line = np.ones((101,2,4), dtype=np.uint8)*255
h_line = np.ones((2,1545), dtype=np.uint8)*255

def Seq2Img(data_seq):
	#[15, 101, 101, 4]
	img_seq = np.hstack([np.hstack([data_seq[_], v_line]) for _ in range(15)])
	img_seq = np.vstack([np.vstack([img_seq[:,:,_], h_line]) for _ in range(4)])
	return img_seq

def Save_BatchSeq2Img(batch_seq, name_prefix):
	# import matplotlib.pyplot as plt
	import matplotlib
	matplotlib.use('Agg')
	import matplotlib.pyplot as plt
	plt.switch_backend('agg')
	for i in range(batch_seq.shape[0]):
		img_seq = Seq2Img(batch_seq[i])
		plt.figure(figsize=img_seq.shape, dpi=5)
		img_seq_color = plt.imshow(img_seq, cmap='nipy_spectral')
		# cv2.imwrite('%s-%d.png'%(name_prefix, i), img_seq_color.get_array())
		# plt.tight_layout()
		plt.xticks([]), plt.yticks([])
		plt.savefig('%s-%d.png'%(name_prefix, i), dpi=5, bbox_inches='tight', aspect='auto')
		plt.close('all')

def Save_IO_Batch(in_batch, out_batch, batch_size=4, name_prefix='test'):
	in_len = in_batch.shape[1] - out_batch.shape[1]
	for i in range(batch_size):
		in_img_seq = Seq2Img(in_batch[i])
		out_seq = np.concatenate([np.ones((in_len,101,101,4))*255, out_batch[i]])
		out_img_seq = Seq2Img(out_seq)
		img_seq = np.vstack([in_img_seq, out_img_seq])
		# print(img_seq.shape)
		cv2.imwrite('%s-%d.png'%(name_prefix, i), img_seq)

def Save_IO_Batch2(in_batch, out_batch, batch_size=4, name_prefix='test'):
	# import matplotlib.pyplot as plt
	import matplotlib
	matplotlib.use('Agg')
	import matplotlib.pyplot as plt
	plt.switch_backend('agg')
	in_len = in_batch.shape[1] - out_batch.shape[1]
	for i in range(batch_size):
		in_img_seq = Seq2Img(in_batch[i])
		out_seq = np.concatenate([np.ones((in_len,101,101,4))*255, out_batch[i]])
		out_img_seq = Seq2Img(out_seq)
		img_seq = np.vstack([in_img_seq, out_img_seq])
		
		plt.figure(figsize=img_seq.shape, dpi=5)
		img_seq_color = plt.imshow(img_seq, cmap='nipy_spectral')
		plt.xticks([]), plt.yticks([])
		plt.savefig('%s-%d.png'%(name_prefix, i), dpi=5, bbox_inches='tight', aspect='auto')
		plt.close('all')

def CSI(ground_truth, prediction, dBZ_threshold=10):
	gt_dBZ = ground_truth*70. - 10.
	pr_dBZ = prediction*70. - 10.
	# print(np.min(dBZ_map), np.max(dBZ_map))
	
	gt_mask = gt_dBZ > dBZ_threshold
	pr_mask = pr_dBZ > dBZ_threshold
	
	TP = np.sum(np.logical_and(gt_mask == True, pr_mask == True))
	FP = np.sum(np.logical_and(gt_mask == False, pr_mask == True))
	FN = np.sum(np.logical_and(gt_mask == True, pr_mask == False))
	TN = np.sum(np.logical_and(gt_mask == False, pr_mask == False))
	csi_val = TP/(TP+FN+FP)
	print(TP, FP, FN, TN)
	print(csi_val)

def ConversionTest(ground_truth):
	gt_dBZ = ground_truth*95./255. - 10.
	print(np.min(gt_dBZ), np.max(gt_dBZ))
	
	gt_mask = gt_dBZ > 40
	occur_no = np.count_nonzero(gt_mask)
	p = occur_no/gt_dBZ.size
	
	print('No. of occurences:', occur_no)
	print('Portion:', p*100)
	exit()

check_list = [63, 814,1213,1368,1579,1868,2057,2213,2617,2695,2774,3074,3081,3517]

def CheckTestData(test_dat):
	high_intensity_list = []
	for i in range(4000):
		# if np.max(test_dat[i,:,:,:,-1]) < 250: continue
		if i not in check_list: continue
		print(i)
		high_intensity_list.append(test_dat[i])
	if len(high_intensity_list) > 0:
		Save_BatchSeq2Img(np.asarray(high_intensity_list), "check-fluctuation/img")
	
if __name__ == '__main__':
	from DataLoader import Load_TestAB
	data_dir = '../../DATA/size-101/all-channels'
	test_data = Load_TestAB(data_dir)
	test_data[test_data>=250] = 0
	CheckTestData(test_data[:,:,:,:,:]); exit()
	
	# test_batch = test_data[:2000,:,:,:,-1]
	# test_batch = test_data[2000:,:,:,:,-1]
	test_batch = test_data[:,:,:,:,-1]
	ConversionTest(test_batch)
	
	gt = test_batch[:,5:,:,:,-1]
	pr = np.concatenate([test_batch[:,4:5,:,:,-1] for _ in range(10)], axis=1)
	CSI(gt/255., pr/255., 40)
	
