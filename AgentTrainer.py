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
        #Constructor
        self.agent = agent
        self.environment =  environment
        
    def take_action(self, action):
        # Take action and get next state and reward
        S_next,R,terminated = self.enviromnet.step(action)
        return S_next,R, terminated
    
    def initialize(self,m):
        # Use this function to initilize the Q function using one-step simulations
        nA = len(self.environment.ActionSpace)
        nS = self.environment.nStates
        
        S = np.zeros([m,nS])
        Q = np.zeros([m,nA])
        
        # Simulate all state-space points at all possible control commands and get reward. This will generate one-step Q function. 
        for n in range(m):
            S[n] = self.environment.reset()
            print(n)
            for idx,action in enumerate(self.environment.ActionSpace):
                self.environment.Tidx = 0
                _,R,_ = self.environment.step(idx)
                Q[n,idx]=R
        
        
        self.agent.initial_train(S,Q,batch_size=1000)
            
            
    
    def train(self,num_episodes,BATCH_SIZE):
        # Train Q DNN
        t_steps = 0
        
        # Run episodes
        for episode in range(num_episodes):
            
            # Reset environment to start
            S = self.environment.reset()
            
            terminated = False
            
            # Keep simulating until terminated (due to constraints simulation always terminates)
            while not terminated:
                # Get optimal action for current state
                action= self.agent.evaluate(S)
                
                # Simulate one-step
                S_next,R,terminated = self.environment.step(action)
                
                # Update memory buffer with SARS
                self.agent.update_buffer(S,action,R,S_next,terminated)
                
                # Train on batch
                loss = self.agent.train(BATCH_SIZE)
                
                S = S_next
                
                
                t_steps+=1                
                
                
                if episode>num_episodes*0.1:
                    self.agent.update_epsilon(0.1)
                    self.agent.update_model()
                
                # Used to keep track of the loss while working
                if loss!=0:
                    print(loss)
                
            
                