# Optimal Obstacle Avoidance Using Deep Q Network
This python based code is used to find optimal trajectory and steering control for obstacle avoidance maneuver.

## Background
Autonomous (or semi-autonomous) vehicles should avoid obstacles along the way without any intervention from the driver. After the obstacle is detected using perception sensors, a controller should plan a trajectory around the obstacle and compute steering (and braking/acceleration) control commands. The planned trajectory and the control inputs should be physically feasible. The feasiblity (defined using vehicle stability) is limited by the vehicle dynamics and surrounding conditions like the tire-road friction, wind speeds etc. To avoid issues due to modeling and environmental uncertainties, most often MPC type optimal control algorithm is used to recursively compute the optimal control inputs and trjactory. However due to computational commplexity of solvers, linear vehicle dynamics models are used. 
The other option is to use machine learning approaches to learn optimal control under various uncertainties. This code is an example of how to use machine learning to compute optimal trajectory and optimal control input for obstacle avoidance maneuver.

## Fundamental Principles
The fundamental priciple is that of deep Q learning - which is most commonly used in reinforcement learning methods. The code uses monte carlo based approach with Deep Neural Network (DNN) to approximate the Q function and optimaml control policy. The monte-carlo simulation results consisting of current state, current action, reward and next state (SARS) are stored in a buffer. A batch is extracted at each step to train the Q network.  
The reward at each simulation step is defined in terms of yaw rate to avoid high yaw rates. There is a reward for the vehicle to reach any position after the obstacle. If the vehicle hits the obstacle, the simulation is terminated and there is no terminal reward.

The novelty of this code lies in the fact that a nonlinear lateral vehicle dynamics model is used to capture the instabilities. A 6 state (2DoF) vehicle dynamics model is used. The states are - side-slip angle, yaw angle, yaw rate, lateral speed, global or road-centric longitudinal position and global lateral position.

## Code structure
There are two main parts of the code - 1) The environment and 2) the DQN. The DQN in-tern has two main classes - the DQN agent class itelf and an agent traner.   

### Environment
The environment.py defines vehicle dynamics model, allows simulation by one step, defines reward and checks for termination conditions. It uses RK method for one-step simulation of the vehicle dynamics model and returns the reward. 

### DQN
The DNNQ class defines the Deep Neural Network required to approximate the Q function and optimal policy. The DQN structure is used from keras (tensor flow). A simple sequential NN is used with one hidden layer in this example. 

### Agene trainer
AgentTrainer class is used to implement deep Q training algorithm. 

### Utilities
ExperienceReplayBuffer class is used to store state, action, reward and next state time histories obtained from the episodes. This class helps generate batches of data for training.

To visualize the results, a Plot_DNN functionality is provided. In this, the state-space is discretized first using meshgrid. At every grid point, the DNN is evalueted. Finally, the values obtained from the DNN are plotted against the grid point. 
The Test_environment.py is used to simulate the environment and to compare its correctness against Simulink continuous time model (not included).

### main
The main script is used as a wrapper function for the agent, environment and trainer.

## Installation
The VD_RL.yml file lists all the dependencies. If you are using anaconda, use the yml to create a virtual environment and run main_VI.py from the venv.
