import numpy as np
np.random.seed(1)

from utils import *

import random
random.seed(1)

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn import metrics

import os, gc
# os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

# # Create a session for running Ops on the Graph.
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# sess = tf.Session(config=config)
# K.set_session(sess)



def create_cnn1d(input_shape):

	seq = Sequential()
	seq.add(Conv1D(64, 3, activation='relu', kernel_initializer='glorot_uniform', input_shape=input_shape))
	seq.add(BatchNormalization(epsilon=1e-06, mode=0, axis=1, momentum=0.9))
	#seq.add(Conv1D(128, 3, activation='relu', kernel_initializer='glorot_uniform'))
	seq.add(MaxPooling1D(5))
	seq.add(Dropout(0.2))

	seq.add(Conv1D(64, 3, activation='relu', kernel_initializer='glorot_uniform'))
	seq.add(BatchNormalization(epsilon=1e-06, mode=0, axis=1, momentum=0.9))
	seq.add(MaxPooling1D(5))
	seq.add(Dropout(0.2))

	seq.add(Conv1D(64, 3, activation='relu', kernel_initializer='glorot_uniform'))
	seq.add(BatchNormalization(epsilon=1e-06, mode=0, axis=1, momentum=0.9))
	seq.add(MaxPooling1D(5))
	seq.add(Dropout(0.2))

	seq.add(Flatten())
	seq.add(Dense(128, kernel_regularizer=l2(0.0005), activation='relu', kernel_initializer='glorot_uniform'))
	seq.add(Dropout(0.5))
	seq.add(Dense(32, kernel_regularizer=l2(0.0005), activation='relu', kernel_initializer='glorot_uniform'))

	return seq


def create_cnn2d(input_shape):

	seq = Sequential()
	seq.add(Conv2D(64, (3,3), activation='relu', kernel_initializer='glorot_uniform', input_shape=input_shape))
	#seq.add(BatchNormalization(epsilon=1e-06, mode=0, axis=1, momentum=0.9))
	seq.add(MaxPooling2D((2,2)))
	seq.add(Dropout(0.5))

	seq.add(Conv2D(64, (3,3), activation='relu', kernel_initializer='glorot_uniform'))
	#seq.add(BatchNormalization(epsilon=1e-06, mode=0, axis=1, momentum=0.9))
	seq.add(MaxPooling2D((2,2)))
	seq.add(Dropout(0.5))

	seq.add(Conv2D(64, (3,3), activation='relu', kernel_initializer='glorot_uniform'))
	#seq.add(BatchNormalization(epsilon=1e-06, mode=0, axis=1, momentum=0.9))
	seq.add(MaxPooling2D((2,2)))
	seq.add(Dropout(0.5))

	seq.add(Flatten())
	seq.add(Dense(128, kernel_regularizer=l2(0.0005), activation='relu', kernel_initializer='glorot_uniform'))
	seq.add(Dropout(0.5))
	seq.add(Dense(32, kernel_regularizer=l2(0.0005), activation='relu', kernel_initializer='glorot_uniform'))
	# seq.add(Dropout(0.5))
	# seq.add(Dense(32, kernel_regularizer=l2(0.0005), activation='relu', kernel_initializer='glorot_uniform')) # softmax changed to relu

	return seq


def create_lstm(input_shape):

	seq = Sequential()

	seq.add(LSTM(units=512, input_shape=input_shape, activation='relu', kernel_initializer='glorot_uniform'))
	seq.add(Dropout(0.5))
	#seq.add(LSTM(units=256, activation='relu', kernel_initializer='glorot_uniform'))
	#seq.add(LSTM(units=128, activation='relu', kernel_initializer='glorot_uniform'))
	#seq.add(LSTM(units=32, activation='relu', kernel_initializer='glorot_uniform', dropout=0.5, return_sequences=True))

	seq.add(Flatten())
	seq.add(Dense(64, kernel_regularizer=l2(0.0005), activation='relu', kernel_initializer='glorot_uniform'))
	seq.add(Dropout(0.5))
	seq.add(Dense(32, kernel_regularizer=l2(0.0005), activation='relu', kernel_initializer='glorot_uniform'))

	return seq


def create_bilstm(input_shape):

	seq = Sequential()

	seq.add(Bidirectional(LSTM(units=256, input_shape=input_shape, activation='relu', kernel_initializer='glorot_uniform', return_sequences=True)))
	seq.add(Dropout(0.5))
	seq.add(Bidirectional(LSTM(units=512, activation='relu', kernel_initializer='glorot_uniform', return_sequences=True)))
	seq.add(Dropout(0.5))
	seq.add(Flatten())
	seq.add(Dense(128, kernel_regularizer=l2(0.0005), activation='relu', kernel_initializer='glorot_uniform'))
	seq.add(Dropout(0.5))
	seq.add(Dense(32, kernel_regularizer=l2(0.0005), activation='softmax', kernel_initializer='glorot_uniform'))

	return seq


def train_siamese(x_data, y_data):
	input_shape = (201, 1024)

	input_a = Input(shape=(input_shape))
	input_b = Input(shape=(input_shape))
	a = Bidirectional(LSTM(units=256, input_shape=input_shape, activation='relu', kernel_initializer='glorot_uniform', return_sequences=True))(input_a)
	#a = Dropout(0.0)(a)
	a = Bidirectional(LSTM(units=512, activation='relu', kernel_initializer='glorot_uniform', return_sequences=True))(a)
	#a = Dropout(0.0)(a)
	a = Flatten()(a)
	a = Dense(128, kernel_regularizer=l2(0.0005), activation='relu', kernel_initializer='glorot_uniform')(a)
	a = Dropout(0.5)(a)
	a = Dense(32, kernel_regularizer=l2(0.0005), activation='softmax', kernel_initializer='glorot_uniform')(a)

	b = Bidirectional(LSTM(units=256, input_shape=input_shape, activation='relu', kernel_initializer='glorot_uniform', return_sequences=True))(input_b)
	#b = Dropout(0.5)(b)
	b = Bidirectional(LSTM(units=512, activation='relu', kernel_initializer='glorot_uniform', return_sequences=True))(b)
	#b = Dropout(0.5)(b)
	b = Flatten()(b)
	b = Dense(128, kernel_regularizer=l2(0.0005), activation='relu', kernel_initializer='glorot_uniform')(b)
	b = Dropout(0.5)(b)
	b = Dense(32, kernel_regularizer=l2(0.0005), activation='softmax', kernel_initializer='glorot_uniform')(b)
	distance = Lambda(euclidean_distance, output_shape=None)([a, b])

	#x = concatenate([a, b])
	#x = BatchNormalization()(x)
	#x = Dropout(0.5)(x)
	#x = Dense(16, activation='relu')(x)
	#x = BatchNormalization()(x)	
	#x = Dropout(0.5)(x)
	#output = Dense(1, activation='sigmoid')(x)


	model = Model(inputs=[input_a, input_b], outputs=distance)

	rms = RMSprop(lr=1e-4, rho=0.9, epsilon=1e-08)
	adadelta = Adadelta()
	adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.99, epsilon=1e-08, decay=0.9)
	model.compile(loss=contrastive_loss, optimizer=adam, metrics=['acc'])

	batch_size = 32

	x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.1, random_state=11, shuffle=True)
	print(np.sum(y_train)/y_train.shape[0], np.sum(y_test)/y_test.shape[0])
	del x_data, y_data
	gc.collect()
	x_tr, x_va, y_tr, y_va = train_test_split(x_train, y_train, test_size=0.1, random_state=11, shuffle=True)
	del x_train, y_train
	gc.collect()
	model.fit(x=[np.array(x_tr)[:,0], np.array(x_tr)[:,1]], y=y_tr, shuffle=True, batch_size=128, epochs=2, verbose=1,\
		validation_data=([np.array(x_va)[:,0], np.array(x_va)[:,1]], y_va))
	y_pred = model.predict([np.array(x_test)[:,0], np.array(x_test)[:,1]])
	auc = roc_auc_score(y_test, y_pred)
	print(auc)



def model_train_onbatch(data_file, seqvec_model_dir):
	input_shape = (201, 1024)

	base_network = create_cnn2d(input_shape)

	input_a = Input(shape=(input_shape))
	input_b = Input(shape=(input_shape))

	processed_a = base_network(input_a)
	processed_b = base_network(input_b)
	distance = Lambda(euclidean_distance, output_shape=None)([processed_a, processed_b])
	model = Model(inputs=[input_a, input_b], outputs=distance)

	rms = RMSprop(lr=1e-4, rho=0.9, epsilon=1e-08)
	adadelta = Adadelta()
	model.compile(loss=contrastive_loss, optimizer=rms)

	model.fit_generator(LoadDataGenerator(data_file, seqvec_model_dir, 8),steps_per_epoch=1, epochs=10)




def model_train(x_data, y_data):
	input_shape = (201, 1024)

	#base_network = create_cnn2d(input_shape)
	base_network = create_bilstm(input_shape)


	input_a = Input(shape=(input_shape))
	input_b = Input(shape=(input_shape))

	#print(input_a, input_b)
	processed_a = base_network(input_a)
	processed_b = base_network(input_b)
	#print(processed_a, processed_b)
	distance = Lambda(euclidean_distance, output_shape=None)([processed_a, processed_b])
	#print(distance)
	model = Model(inputs=[input_a, input_b], outputs=distance)

	rms = RMSprop(lr=1e-4, rho=0.9, epsilon=1e-08)
	adadelta = Adadelta()
	adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.99, epsilon=1e-08, decay=0.9)
	model.compile(loss=contrastive_loss, optimizer=adam)

	batch_size = 32

	x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.1, random_state=11, shuffle=True)
	print(np.sum(y_train)/y_train.shape[0], np.sum(y_test)/y_test.shape[0])
	del x_data, y_data
	gc.collect()
	x_tr, x_va, y_tr, y_va = train_test_split(x_train, y_train, test_size=0.1, random_state=11, shuffle=True)
	del x_train, y_train
	gc.collect()
	model.fit(x=[np.array(x_tr)[:,0], np.array(x_tr)[:,1]], y=y_tr, shuffle=True, batch_size=128, epochs=2, verbose=1,\
		validation_data=([np.array(x_va)[:,0], np.array(x_va)[:,1]], y_va))
	y_pred = model.predict([np.array(x_test)[:,0], np.array(x_test)[:,1]])
	auc = roc_auc_score(y_test, y_pred)
	print(auc)
