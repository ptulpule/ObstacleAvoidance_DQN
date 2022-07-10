#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  7 14:10:28 2022

@author: tulpule.3
"""
import tensorflow as tf
from keras.models import Sequential
from keras.layers import BatchNormalization
from keras.layers import Dense
import numpy as np
from MyPlots import plot_DNN
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint

class DNNQ:
    def __init__(self,state_size,nHidden,action_size,optimizer,max_epsilon,experience_replay):
        self.state_size = state_size
        self.nHidden = nHidden
        self.action_size = action_size        
        self.epsilon = max_epsilon
        self.experience_reply = experience_replay
        self.optimizer = optimizer
        
        self.primary_network = self.build_networs()
        self.primary_network.compile(loss='mse', optimizer= optimizer)
        
        self.target_network = self.build_networs()
        self.target_network.compile(loss='mse', optimizer= optimizer)
        
        self.eps_decay = 0.99991
        self.learning_rate = 0.5
        
    def build_networs(self):
        #initializer = tf.keras.initializers.HeUniform(0.1)
        model = Sequential()
        
        model.add(Dense(self.state_size,activation='linear',kernel_initializer='he_uniform'))
        #model.add(BatchNormalization())
        model.add(Dense(self.nHidden, activation='relu', kernel_initializer='he_uniform'))
        #model.add(Dense(self.nHidden, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(self.action_size, activation='linear'))
        return model
    
    def initial_train(self,S,Q,batch_size):
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1,patience=250)
        checkpoint = ModelCheckpoint('BestModel.hdf5', monitor='val_loss', verbose=2, save_best_only=True, mode='min')
        opt = tf.keras.optimizers.Adam(learning_rate=0.01)
        self.primary_network.compile(optimizer=opt, loss='mean_squared_error')
        #history = self.model.fit(states, V, validation_split=0.2, epochs=1000,batch_size=10)
        model = self.primary_network
        
        history = model.fit(S, Q, validation_split=0.2, epochs=6000,batch_size=batch_size,callbacks=[es,checkpoint],verbose=0)
        
        model.load_weights('BestModel.hdf5')
        self.primary_network = model
        self.target_network = model
        
        return history
    
    def update_epsilon(self,min_epsilon):
        if self.epsilon < min_epsilon:
            self.epsilon = min_epsilon
        else: 
            self.epsilon = self.epsilon*self.eps_decay
            
    def update_buffer(self,S,A,R,S_next,terminated):
        self.experience_reply.store(S,A,R,S_next,terminated)
        
    def evaluate(self,state):
        if np.random.rand()<self.epsilon:
            return np.random.randint(0,self.action_size)
        else:
            state_tf = tf.convert_to_tensor(state)
            state_tf = tf.expand_dims(state_tf , 0)
            q_values = self.primary_network(state_tf)
            return np.argmax(q_values)
        
    
    def update_model(self):
        for t, e in zip(self.target_network.trainable_variables, 
                    self.primary_network.trainable_variables): t.assign(t * (1 - self.learning_rate) + e * self.learning_rate)
        
        
    def train(self,batch_size):
        if len(self.experience_reply.buffer)<batch_size:
            return 0
        
        batch = self.experience_reply.get_batch(batch_size)
        S,A,R,S_next,terminated = self.experience_reply.get_arrays_from_batch(batch)
        
        Q_A = self.primary_network(S).numpy()
        #Q = Q_A[0:,A]
        
        Q_next = self.target_network(S_next).numpy()
        A_opt = np.argmax(Q_next,axis = 1)
        Q_target = np.zeros(batch_size)
        Q_update = Q_A
        
        for idx in range(batch_size):
            
            if terminated[idx]:
                Q_update[idx, A[idx]] = R[idx]
            else:
                Q_update[idx,A[idx]] = R[idx] + Q_next[idx,A_opt[idx]]
        
        
        #loss = self.primary_network.fit(S,Q_update,batch_size=batch_size,epochs=100,verbose=0)
        loss = self.primary_network.train_on_batch(S,Q_update)
        print(loss)
        
        #plot_DNN(self.primary_network)
        
        #print(loss.history['loss'][99])
        #self.update_model()
        
        return loss
        #return loss.history['loss'][99]
        
        
        