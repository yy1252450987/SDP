import numpy as np
np.random.seed(1)
import tensorflow as tf

#from utils import *

from keras.layers.core import Lambda
from keras.layers import Dense, Dropout, Input, Flatten, Conv1D,concatenate, MaxPooling1D,Conv2D, MaxPooling2D, CuDNNLSTM, LSTM, Bidirectional
from keras import backend as K
from keras.models import Sequential, Model
from keras.optimizers import SGD, RMSprop, Adadelta, Adam
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2

import random
random.seed(1)

def cnn1d(input_shape, conv_nums=[64, 64, 64], filter_size=[3,3,3], dropout_rate=0.5, pooling_size=[3,3,3], dense_nums=[64,16], BN=True):

    if len(conv_nums) != len(filter_size) or len(conv_nums)!= len(pooling_size) or len(filter_size) != len(pooling_size) :
        return None

    seq = Sequential()

    for i in range(len(conv_nums)):
        if i == 0:
            seq.add(Conv1D(conv_nums[i], filter_size[i], activation='relu', kernel_initializer='glorot_uniform', input_shape=input_shape))
        else:
            seq.add(Conv1D(conv_nums[i], filter_size[i], activation='relu', kernel_initializer='glorot_uniform'))
        if BN == True:
            seq.add(BatchNormalization())
        seq.add(MaxPooling1D(pooling_size[i]))
        seq.add(Dropout(dropout_rate))

    seq.add(Flatten())
    for i in range(len(dense_nums)):
        if i != len(dense_nums)-1:
            seq.add(Dense(dense_nums[i], kernel_regularizer=l2(0.0005), activation='relu', kernel_initializer='glorot_uniform'))
            seq.add(Dropout(dropout_rate))
        else:
            seq.add(Dense(dense_nums[i], kernel_regularizer=l2(0.0005), activation='softmax', kernel_initializer='glorot_uniform'))


    return seq


def cnn2d(input_shape, conv_nums=[64, 64, 64], filter_size=[3,3,3], dropout_rate=0.5, pooling_size=[3,3,3], dense_nums=[64,16], BN=True):

    if len(conv_nums) != len(filter_size) or len(conv_nums)!= len(pooling_size) or len(filter_size) != len(pooling_size) :
        return None

    seq = Sequential()

    for i in range(len(conv_nums)):
        if i == 0:
            seq.add(Conv2D(conv_nums[i], (filter_size[i], filter_size[i]), activation='relu', kernel_initializer='glorot_uniform', input_shape=input_shape))
        else:
            seq.add(Conv2D(conv_nums[i], (filter_size[i], filter_size[i]), activation='relu', kernel_initializer='glorot_uniform'))
        if BN == True:
            seq.add(BatchNormalization())
        seq.add(MaxPooling2D((pooling_size[i],pooling_size[i])))
        seq.add(Dropout(dropout_rate))

    seq.add(Flatten())
    for i in range(len(dense_nums)):
        if i != len(dense_nums)-1:
            seq.add(Dense(dense_nums[i], kernel_regularizer=l2(0.0005), activation='relu', kernel_initializer='glorot_uniform'))
            seq.add(Dropout(dropout_rate))
        else:
            seq.add(Dense(dense_nums[i], kernel_regularizer=l2(0.0005), activation='softmax', kernel_initializer='glorot_uniform'))

    return seq

def bilstm(input_shape, lstm_nums=[64, 64], dropout_rate=0.5, dense_nums=[64,16], BN=True):

    seq = Sequential()

    for i in range(len(lstm_nums)):
        if i == 0:
            seq.add(Bidirectional(LSTM(units=lstm_nums[i], input_shape=input_shape, activation='relu', kernel_initializer='glorot_uniform', return_sequences=True)))
            seq.add(Dropout(dropout_rate))
        else:
            seq.add(Bidirectional(LSTM(units=lstm_nums[i], activation='relu', kernel_initializer='glorot_uniform', return_sequences=True)))
            seq.add(Dropout(dropout_rate))

    seq.add(Flatten())
    for i in range(len(dense_nums)):
        if i != len(dense_nums)-1:
            seq.add(Dense(dense_nums[i], kernel_regularizer=l2(0.0005), activation='relu', kernel_initializer='glorot_uniform'))
            seq.add(Dropout(dropout_rate))
        else:
            seq.add(Dense(dense_nums[i], kernel_regularizer=l2(0.0005), activation='softmax', kernel_initializer='glorot_uniform'))

    return seq


def cnn1d_bilstm(input_shape, conv_nums=[64, 64, 64], filter_size=[3,3,3], lstm_nums=[64], conv_dense_nums=[128, 64], dropout_rate=0.5, lstm_dense_nums=[32, 16], pooling_size=[3,3,3],  BN=True):

    seq = Sequential()
    for i in range(len(conv_nums)):
        if i == 0:
            seq.add(Conv1D(conv_nums[i], filter_size[i], activation='relu', kernel_initializer='glorot_uniform', input_shape=input_shape))
        else:
            seq.add(Conv1D(conv_nums[i], filter_size[i], activation='relu', kernel_initializer='glorot_uniform'))
        if BN == True:
            seq.add(BatchNormalization())
        seq.add(MaxPooling1D(pooling_size[i]))
        seq.add(Dropout(dropout_rate))

    #seq.add(Flatten())
    for i in range(len(conv_dense_nums)):
        seq.add(Dense(conv_dense_nums[i], kernel_regularizer=l2(0.0005), activation='relu', kernel_initializer='glorot_uniform'))
        if i != len(conv_dense_nums)-1:
            seq.add(Dropout(dropout_rate))
    
    for i in range(len(lstm_nums)):
        seq.add(Bidirectional(LSTM(units=lstm_nums[i], activation='relu', kernel_initializer='glorot_uniform', return_sequences=True)))
        seq.add(Dropout(dropout_rate))
    
    for i in range(len(lstm_dense_nums)):
        if i != len(lstm_dense_nums)-1:
            seq.add(Dense(lstm_dense_nums[i], kernel_regularizer=l2(0.0005), activation='relu', kernel_initializer='glorot_uniform'))
            seq.add(Dropout(dropout_rate))
        else:
            seq.add(Dense(lstm_dense_nums[i], kernel_regularizer=l2(0.0005), activation='softmax', kernel_initializer='glorot_uniform'))

    return 0


def euclidean_distance(vec):
    vec1, vec2 = vec
    d = K.sqrt(K.sum(K.square(vec1-vec2), axis=1, keepdims=True))

    return d

def cosine_distance(vec):
    vec1, vec2 = vec
    d = 1-K.sum(vec1*vec2, axis=1, keepdims=True)/K.sqrt(K.sum(K.square(vec1), axis=1, keepdims=True))/K.sqrt(K.sum(K.square(vec2), axis=1, keepdims=True))
    return d

def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))
  



def ModelCNNNetwork(input_shape, base_network='cnn1d', dropout_rate=0.5, learning_rate=0.0001):
    if base_network == 'cnn1d':
        network = cnn1d(input_shape, conv_nums=[64, 64, 64], filter_size=[3,3,3], dropout_rate=0.5, pooling_size=[3,3,3], dense_nums=[32,16])
    if base_network == 'cnn2d':
        network = cnn2d(input_shape,conv_nums=[32,32], filter_size=[3,3], dropout_rate=0.5, pooling_size=[3,3], dense_nums=[32,16])
    if base_network == 'bilstm':
        network = bilstm(input_shape, lstm_nums=[256, 512], dropout_rate=0.5, dense_nums=[64,16], BN=True)
    if base_network == 'cnn1d_bilstm':
        network = cnn1d_bilstm(input_shape, conv_nums=[64, 64, 64], filter_size=[3,3,3], lstm_nums=[64], conv_dense_nums=[128, 64], dropout_rate=0.5, lstm_dense_nums=[32, 16], pooling_size=[3,3,3])
    if base_network == None:
        return None

    input_a = Input(shape=(input_shape))
    input_b = Input(shape=(input_shape))

    processed_a = network(input_a)
    processed_b = network(input_b)
    
    distance = Lambda(euclidean_distance, output_shape=None)([processed_a, processed_b])
    model = Model(inputs=[input_a, input_b], outputs=distance)

    adam = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.99, epsilon=1e-08, decay=0.9)
    model.compile(loss=contrastive_loss, optimizer=adam, metrics=['acc'])

    return model
