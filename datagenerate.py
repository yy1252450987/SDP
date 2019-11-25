import numpy as np
import keras


class DataGenerator(keras.utils.Sequence):
	def __init__(self, data_dir, file_prefix, list_IDs, labels, batch_size=8, dim=(1001, 1024, 3), n_classes=1, shuffle=True, *args, **kwargs):
		
		self.data_dir = data_dir
		self.file_prefix = file_prefix
		self.list_IDs = list_IDs
		self.labels = labels
		self.batch_size = batch_size
		self.dim = dim
		self.n_classes = n_classes
		self.shuffle = shuffle
		self.on_epoch_end()

	def __len__(self):

		return int(np.floor(len(self.list_IDs) / self.batch_size))

	def __getitem__(self, index):

		indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]
		list_IDs_temp = [self.list_IDs[k] for k in indices]
		X, Y = self.__data_generation(list_IDs_temp)
		return X, Y

	def on_epoch_end(self):

		self.indices = np.arange(len(self.list_IDs))
		np.random.shuffle(self.indices)

	def __data_generation(self, list_IDs_temp):

		X_mut = np.empty((self.batch_size, *self.dim))
		X_wild = np.empty((self.batch_size, *self.dim))
		Y = np.empty((self.batch_size, self.n_classes), dtype=int)
		mut_file_prefix = self.file_prefix.replace('*', 'mut')
		wild_file_prefix = self.file_prefix.replace('*', 'wild')
		for i, ID in enumerate(list_IDs_temp):
			if len(self.dim) == 2:
				temp_wild = np.mean(np.load(self.data_dir+wild_file_prefix+'.%s.npy' % ID).transpose(1, 2, 0), axis=2)
				X_wild[i, ] = np.pad(temp_wild, ((0,self.dim[0]-temp_wild.shape[0]),(0,0)) ,'constant')
				temp_mut = np.mean(np.load(self.data_dir+mut_file_prefix+'.%s.npy' % ID).transpose(1, 2, 0), axis=2)
				X_mut[i, ] = np.pad(temp_mut, ((0,self.dim[0]-temp_mut.shape[0]),(0,0)) ,'constant')
				Y[i, ] = self.labels[ID]
			if len(self.dim) == 3:
				temp_wild = np.load(self.data_dir+wild_file_prefix+'.%s.npy' % ID).transpose(1, 2, 0)
				X_wild[i, ] = np.pad(temp_wild, ((0,self.dim[0]-temp_wild.shape[0]),(0,0),(0,0)) ,'constant')
				temp_mut = np.load(self.data_dir+mut_file_prefix+'.%s.npy' % ID).transpose(1, 2, 0)
				X_mut[i, ] = np.pad(temp_mut, ((0,self.dim[0]-temp_mut.shape[0]),(0,0),(0,0)) ,'constant')
				Y[i, ] = self.labels[ID]

		return [X_wild, X_mut], Y

class TestDataGenerator():
	def __init__(self, data_dir, file_prefix, list_IDs, batch_size=8, dim=(1001, 1024, 3), n_classes=1):
		
		self.data_dir = data_dir
		self.file_prefix = file_prefix
		self.list_IDs = list_IDs
		self.batch_size = batch_size
		self.dim = dim
		self.n_classes = n_classes

	def load_batch_data(self, start):
		mut_file_prefix = self.file_prefix.replace('*', 'mut')
		wild_file_prefix = self.file_prefix.replace('*', 'wild')
		X_wild, X_mut = [], []		
		for i, ID in enumerate(self.list_IDs[start:start+self.batch_size]):
			if len(self.dim) == 2:
				temp_wild = np.mean(np.load(self.data_dir+wild_file_prefix+'.%s.npy' % ID).transpose(1, 2, 0), axis=2)
				X_wild[i, ] = np.pad(temp_wild, ((0,self.dim[0]-temp_wild.shape[0]),(0,0)) ,'constant')
				temp_mut = np.mean(np.load(self.data_dir+mut_file_prefix+'.%s.npy' % ID).transpose(1, 2, 0), axis=2)
				X_mut[i, ] = np.pad(temp_mut, ((0,self.dim[0]-temp_mut.shape[0]),(0,0)) ,'constant')
				Y[i, ] = self.labels[ID]
			if len(self.dim) == 3:
				temp_wild = np.load(self.data_dir+wild_file_prefix+'.%s.npy' % ID).transpose(1, 2, 0)
				X_wild[i, ] = np.pad(temp_wild, ((0,self.dim[0]-temp_wild.shape[0]),(0,0),(0,0)) ,'constant')
				temp_mut = np.load(self.data_dir+mut_file_prefix+'.%s.npy' % ID).transpose(1, 2, 0)
				X_mut[i, ] = np.pad(temp_mut, ((0,self.dim[0]-temp_mut.shape[0]),(0,0),(0,0)) ,'constant')
				Y[i, ] = self.labels[ID]
		return [np.array(X_wild), np.array(X_mut)]

	def load_whole_data(self):
		mut_file_prefix = self.file_prefix.replace('*', 'mut')
		wild_file_prefix = self.file_prefix.replace('*', 'wild')
		X_wild, X_mut = [], []		
		for i, ID in enumerate(self.list_IDs):
			if len(self.dim) == 2:
				temp_wild = np.mean(np.load(self.data_dir+wild_file_prefix+'.%s.npy' % ID).transpose(1, 2, 0), axis=2)
				X_wild.append(np.pad(temp_wild, ((0,self.dim[0]-temp_wild.shape[0]),(0,0)) ,'constant'))
				temp_mut = np.mean(np.load(self.data_dir+mut_file_prefix+'.%s.npy' % ID).transpose(1, 2, 0), axis=2)
				X_mut.append(np.pad(temp_mut, ((0,self.dim[0]-temp_mut.shape[0]),(0,0)) ,'constant'))
			if len(self.dim) == 3:
				temp_wild = np.load(self.data_dir+wild_file_prefix+'.%s.npy' % ID).transpose(1, 2, 0)
				X_wild.append(np.pad(temp_wild, ((0,self.dim[0]-temp_wild.shape[0]),(0,0),(0,0)) ,'constant'))
				temp_mut = np.load(self.data_dir+mut_file_prefix+'.%s.npy' % ID).transpose(1, 2, 0)
				X_mut.append(np.pad(temp_mut, ((0,self.dim[0]-temp_mut.shape[0]),(0,0),(0,0)) ,'constant'))
		return [np.array(X_wild), np.array(X_mut)]



