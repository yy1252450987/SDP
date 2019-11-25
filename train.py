import numpy as np
np.random.seed(1)
import random
random.seed(1)

#from utils import *
from networks import *
from datagenerate import DataGenerator, TestDataGenerator

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn import metrics

import os, gc


def model_train(data_dir, file_prefix, data_df, base_network='cnn1d'):
	if base_network == 'cnn2d':
		input_shape = (1001, 1024, 3)
	else:
		input_shape = (1001, 1024)

	model = ModelCNNNetwork(input_shape=input_shape, base_network=base_network)

	bs = 32

	test_size = 0.2
	all_idx = data_df.index.values
	test_idx = np.random.choice(all_idx, int(len(all_idx)*test_size))
	train_idx = list(set(all_idx)-set(test_idx))
	train_df = data_df.ix[train_idx, :].sample(frac=1)
	test_df = data_df.ix[test_idx, :].sample(frac=1)
	train_generator = DataGenerator(data_dir, file_prefix, list_IDs=train_df.index.values, labels=train_df['label'], dim=input_shape)
	test_generator = TestDataGenerator(data_dir, file_prefix, list_IDs=test_df.index.values, dim=input_shape)
	model.fit_generator(generator=train_generator, steps_per_epoch=5, shuffle=True, verbose=1, use_multiprocessing=True, workers=8, epochs=1)
	
	y_pred = model.predict(test_generator.load_whole_data())
	#y_pred = []
	#for i in range(0, test_df.shape[0], 16):
	#	print(i)
	#	y_pred_ = model.predict(test_generator.load_batch_data(i))
	#	y_pred = np.concatenate((y_pred, y_pred_.ravel()), axis=0)
	y_test = test_df['label']
	auc = roc_auc_score(y_test, y_pred)
	print(auc)

	# x_tr_a = np.array(x_tr)[:, 0]
	# x_tr_b = np.array(x_tr)[:, 1]
	# x_va_a = np.array(x_va)[:, 0]
	# x_va_b = np.array(x_va)[:, 1]
	# x_test_a = np.array(x_test)[:, 0]
	# x_test_b = np.array(x_test)[:, 1]
	# y_test = np.array(y_test)


	# if base_network == 'cnn2d':
	# 	x_tr_a = x_tr_a.reshape(x_tr_a.shape[0],x_tr_a.shape[1],x_tr_a.shape[2],1)
	# 	x_tr_b = x_tr_b.reshape(x_tr_b.shape[0],x_tr_b.shape[1],x_tr_b.shape[2],1)
	# 	x_va_a = x_va_a.reshape(x_va_a.shape[0],x_va_a.shape[1],x_va_a.shape[2],1)
	# 	x_va_b = x_va_b.reshape(x_va_b.shape[0],x_va_b.shape[1],x_va_a.shape[2],1)
	# 	x_test_a = x_test_a.reshape(x_test_a.shape[0],x_test_a.shape[1],x_test_a.shape[2],1)
	# 	x_test_b = x_test_b.reshape(x_test_b.shape[0],x_test_b.shape[1],x_test_b.shape[2],1)
