#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  7 14:06:38 2022

@author: tulpule.3
"""
import numpy as np
import matplotlib.pyplot as plt


class AgentTrainer():
    def __init__(self,agent,environment):
        self.agent = agent
        self.environment =  environment
        
    def take_action(self, action):
        S_next,R,terminated = self.enviromnet.step(action)
        return S_next,R, terminated
    
    def initialize(self,m):
        nA = len(self.environment.ActionSpace)
        nS = self.environment.nStates
        
        S = np.zeros([m,nS])
        Q = np.zeros([m,nA])
        
        
        for n in range(m):
            S[n] = self.environment.reset()
            print(n)
            for idx,action in enumerate(self.environment.ActionSpace):
                #print(idx)
                self.environment.Tidx = 0
                _,R,_ = self.environment.step(idx)
                Q[n,idx]=R
        
        #plt.scatter(S[0:,4],S[0:,3],c=Q[0:,4])
        self.agent.initial_train(S,Q,batch_size=1000)
            
            
    
    def train(self,num_episodes,BATCH_SIZE):
        
        t_steps = 0
        
        for episode in range(num_episodes):
            
            S = self.environment.reset()
            
            terminated = False
            
            while not terminated:
                
                action= self.agent.evaluate(S)
                
                S_next,R,terminated = self.environment.step(action)
                
                self.agent.update_buffer(S,action,R,S_next,terminated)
                
                loss = self.agent.train(BATCH_SIZE)
                
                S = S_next
                
                
                t_steps+=1                
                
                if episode>num_episodes*0.1:
                    self.agent.update_epsilon(0.1)
                    self.agent.update_model()
                
                #if -loss <-10:
                #    return
                
            if loss !=0:
                print(self.agent.epsilon) 
                
            
                