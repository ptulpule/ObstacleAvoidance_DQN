#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  7 18:32:59 2022

@author: tulpule.3
"""
from collections import deque
import random 
import numpy as np

class ExperienceReplayBuffer:
    def __init__(self,max_len):
        self.buffer = deque(maxlen=max_len)
        
        
    def store(self,S,A,R,S_next,terminated):
        self.buffer.append((S,A,R,S_next,terminated))
        
    def get_batch(self,batch_size):
        if batch_size >len(self.buffer):
            batch = self.buffer
        else:
            batch = random.sample(self.buffer,batch_size)
            
        return batch
    
    def get_arrays_from_batch(self,batch):
        states = np.array([x[0] for x in batch])
        actions = np.array([x[1] for x in batch])
        rewards = np.array([x[2] for x in batch])
        next_states = np.array([x[3] for x in batch])
        terminated = np.array([x[4] for x in batch])
        return states, actions, rewards, next_states, terminated
        