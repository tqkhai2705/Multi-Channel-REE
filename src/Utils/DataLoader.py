import numpy as np
import os, sys

def Load_Sample_Set(file_path):
	#Default is Training data with shape: [10000,15,101,101]
	print(' *** LOADING FILE: '+file_path)
	dat = np.load(file_path)
	n_samples = dat.shape[0]
	print('	--> Finished. No. of samples: %d'%(n_samples))
	return dat

def Load_Divided_Data(file_path):
	print(' *** LOADING FILE: '+file_path)
	dat = np.load(file_path)
	n = dat.shape[0]
	print(' 	==> Total samples: ', n)
	n1 = np.int(0.8*n)
	train_data = dat[:n1]
	valid_data = dat[n1:]
	return train_data, valid_data

def Load_TestAB(data_dir):
	dat1 = Load_Sample_Set(os.path.join(data_dir, 'testA.npy'))
	dat2 = Load_Sample_Set(os.path.join(data_dir, 'testB.npy'))
	all_data = np.concatenate([dat1, dat2], 0)
	print('All sample sets are loaded: shape = ', all_data.shape)
	return all_data
	
def Load_Mixed_Data(data_dir, training=True):
	sample_sets = ['train', 'testA', 'testB']
	all_data = []
	
	""" Load 3 samples sets """
	for sample_set in sample_sets:
		dat = Load_Sample_Set(os.path.join(data_dir, sample_set+'.npy'))
		all_data.append(dat)
	all_data = np.concatenate(all_data, 0)
	print('All sample sets are loaded: shape = ', all_data.shape)
	
	""" Load or prepare random indices --> for mixing data 
		Indices are stored and loaded so that the models will be trained, 
		validated and tested on the same corresponding samples
	"""
	index_file = os.path.join(data_dir, 'mixed_data_random_indices.txt')
	if os.path.isfile(index_file): 
		indices = np.loadtxt(index_file, dtype=np.int)
	else:
		np.random.seed()
		indices = np.random.permutation(all_data.shape[0])
		np.savetxt(index_file, indices, fmt='%d')
	
	""" Pick the corresponding sets """
	train_data = all_data[indices[0:10000]]
	valid_data = all_data[indices[10000:12000]]
	test_data = all_data[indices[12000:]]
	# return [train_data, valid_data, test_data]
	
	if training: return train_data, valid_data
	else: return test_data
		
def Next_Train_Batch(dat, n_samples, batch_size):
	np.random.seed()
	indices = np.random.permutation(n_samples)[:batch_size]
	#print('  - Random indices:', indices)
	return dat[indices]

def Train_Data(dat, n_samples, range):
	np.random.seed()
	indices = np.random.permutation(n_samples)[:range]
	#print('  - Random indices:', indices)
	return dat[indices]

def Data_Histogram(img_data):
		hist,bins = np.histogram(img_data.ravel(),256,[0,255], density=None)
		hist=hist[1:] # Remove the 0-value which dominates the data
		# return hist
		return hist/np.max(hist)

def Draw_Data_Histogram(data_dir):
	# This function loads 3 sets of data and draw
	sample_sets = ['train', 'testA', 'testB']	
	colors = ['r', 'g', 'b']
	styles = ['-', '--', ':']
	plt.figure(figsize=(5,3), dpi=72)
	plt.subplots_adjust(left=0.060, right=0.995, bottom=0.140, top=0.995)
	for i, sample_set in enumerate(sample_sets):
		dat = Load_Sample_Set(data_dir+sample_set+'.npy')
		print(dat.shape)
		img = dat#[2]
		hist = Data_Histogram(img)
		xs = np.nonzero(hist)[0]
		plt.plot(xs, hist[hist>0], color=colors[i], linestyle=styles[i], label=sample_set)
	plt.xlim([1,256])
	plt.xlabel('Pixel values', fontsize=12)
	plt.legend(fontsize=12)
	# plt.grid(True)
	plt.show()

def Draw_Data_Histogram2(sample_sets):
	# This function receive 3 sets of data and draw
	colors = ['r', 'g', 'b']
	styles = ['-', '--', ':']
	labels = ['train', 'validation', 'test']
	plt.figure(figsize=(5,3), dpi=72)
	plt.subplots_adjust(left=0.060, right=0.995, bottom=0.140, top=0.995)
	for i, sample_set in enumerate(sample_sets):
		print(sample_set.shape)
		hist = Data_Histogram(sample_set)
		xs = np.nonzero(hist)[0]
		plt.plot(xs, hist[hist>0], color=colors[i], linestyle=styles[i], label=labels[i])
	plt.xlim([1,256])
	plt.xlabel('Pixel values', fontsize=12)
	plt.legend(fontsize=12)
	# plt.grid(True)
	plt.show()

def Find_KNN(dat, folder):	
	from sklearn.neighbors import NearestNeighbors
	from sklearn.metrics import mean_squared_error as mse
	from skimage.measure import compare_ssim as ssim
	from datetime import datetime
	
	def Draw_Images(imgs, save_name=None, toshow=True):
		h,w = 5,15
		fig, axes = plt.subplots(h,1, figsize=(w, h+.2), dpi=72, facecolor='white',edgecolor='black',frameon=True)
		for i in range(h): axes[i].imshow(imgs[i], cmap='gray')
		
		plt.tight_layout(); plt.setp(axes, xticks=[], yticks=[])
		plt.subplots_adjust(wspace=0.01, hspace=0.01, left=0.01, right=0.99, bottom=0.01, top=0.99)
		if save_name: plt.savefig(save_name, bbox_inches='tight',dpi=72, aspect='auto')
		if toshow: plt.show()
		plt.close('all')
	
	def LoadSavedFiles(metric='mse'):
		distances = np.loadtxt(os.path.join(folder,'distances-%s.txt'%(metric)), dtype=np.float)
		indices = np.loadtxt(os.path.join(folder,'indices-%s.txt'%(metric)), dtype=np.int)
		return distances, indices
		
	def KNN_2D(imgs, k, metric='mse'):
		n = imgs.shape[0]
		temp_distances = []
		distances, indices = [],[]
		start_time = datetime.now()
		
		if metric is 'mse': reverse_order = False
		def Imgs_Distance(img_i, img_j):
			if metric is 'mse': return mse(img_i, img_j)
			if metric is 'ssim': return ssim(img_i, img_j, win_size=5, data_range=1.)
			
		for i in range(n):
			i_distances = []
			for j in range(n):
				if i == j: dis_ij = 1.
				if i > j:  dis_ij = temp_distances[j][i]
				else: dis_ij = Imgs_Distance(imgs[i], imgs[j])
				i_distances.append(dis_ij)			
			temp_distances.append(i_distances)
			# indices.append(i_indices)
			if (i+1)%10 == 0: print('	*Iter: %d; Time passed: %s'%(i+1, datetime.now() - start_time))
		
		for i_distances in temp_distances:
			# The indices are from 1 to k because the distance to itself is not counted
			i_indices = np.asarray(sorted(range(len(i_distances)), reverse=reverse_order, key=lambda t: i_distances[t])[1:k+1])
			i_distances = np.asarray(i_distances)[i_indices]
			indices.append(i_indices); distances.append(i_distances)
			# print(i_distances)
		
		if not os.path.isdir(folder): os.mkdir(folder)
		np.savetxt(os.path.join(folder,'distances-%s.txt'%(metric)), distances, fmt='%f')
		np.savetxt(os.path.join(folder,'indices-%s.txt'%(metric)), indices, fmt='%d')
		return distances, indices
		
	X = dat#[0:1000]
	X = np.reshape(X, [-1, 15*101, 101])
	X = np.transpose(X, [0,2,1])
	
	metric = 'mse'
	# distances, indices = KNN_2D(X/255., 10, metric=metric)
	distances, indices = LoadSavedFiles(metric=metric)
	# exit()
	# fX = np.reshape(X, [-1, 15*101*101])/255.
	# nbrs = NearestNeighbors(n_neighbors=5, algorithm='brute', metric='euclidean').fit(fX)
	# distances, indices = nbrs.kneighbors(fX)
	visited_indices = [0]
	for iTest,d in enumerate(distances):
		if iTest in visited_indices: continue
		if d[0] < 0.001:
			print(d)
			visited_indices.extend(indices[iTest])
			Draw_Images(X[indices[iTest]])

if __name__ == '__main__':
	import matplotlib.pyplot as plt
	import matplotlib
	matplotlib.rcParams['mathtext.fontset'] = 'cm'
	matplotlib.rcParams['font.family'] = 'STIXGeneral'
	
	data_dir = '../../DATA/size-101/channel-4/'
	# Draw_Data_Histogram(data_dir)
	
	# train_dat, valid_dat = Load_Mixed_Data(data_dir)
	sample_set = 'train'
	dat = Load_Sample_Set(data_dir+sample_set+'.npy')
	Find_KNN(dat, os.path.join(data_dir, 'kNN-%s-set'%sample_set))
	
