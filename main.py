#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  7 20:23:59 2022

@author: tulpule.3

Main script to obtain optimal control policy for obstacle avoidance.
last update: May 13 2022
"""

from ExperienceReplayBuffer import ExperienceReplayBuffer
from environment import env
from AgentTrainer import AgentTrainer
from DNNQ import DNNQ
from MyPlots import plot_DNN



# Initialize a rondom memory buffer
experience_replay = ExperienceReplayBuffer(5000) 

# Generate environment
environment = env(50)

# Initialize agent (DQN)
agent = DNNQ(environment.nStates,2000,9,'Adam',1,experience_replay)

# Initialize agent trainer for the enviromnet
trainer = AgentTrainer(agent, environment)

# Start training
trainer.train(num_episodes= 5000,BATCH_SIZE=1000)

Q_net = trainer.agent.primary_network;
Q_net.save('OptimalControl/Q_net_trained'+'.hdf5')

# Visualize results
plot_DNN(Q_net)


