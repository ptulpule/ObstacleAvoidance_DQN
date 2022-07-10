#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  7 20:23:59 2022

@author: tulpule.3
"""

from ExperienceReplayBuffer import ExperienceReplayBuffer
from environment import env
from AgentTrainer import AgentTrainer
from DNNQ import DNNQ
from tensorflow.keras.optimizers import Adam
from MyPlots import plot_DNN
from keras.models import load_model

experience_replay = ExperienceReplayBuffer(5000) 
environment = env(50)
agent = DNNQ(environment.nStates,2000,9,'Adam',1,experience_replay)

trainer = AgentTrainer(agent, environment)

#trainer.initialize(10000)
#Q_net = trainer.agent.primary_network;
#Q_net.save('OptimalControl/Q_net'+'.hdf5')

#Q_net = load_model('OptimalControl/Q_net'+'.hdf5');

#trainer.agent.primary_network = Q_net
#trainer.agent.target_network = Q_net
trainer.train(num_episodes= 5000,BATCH_SIZE=1000)

Q_net = trainer.agent.primary_network;
Q_net.save('OptimalControl/Q_net_trained'+'.hdf5')
plot_DNN(Q_net)


