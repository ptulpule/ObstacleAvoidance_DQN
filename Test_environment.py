
import numpy as np
from environment import env
import matplotlib.pyplot as plt

dt = 0.1
Tend = 5
nT = int(Tend/dt+1)
T = np.linspace(0,Tend,int(Tend/dt+1))
U = np.sin(T)*0.005


env = env(nT)

while not env.terminated:
    idx=env.Tidx
    state, state_,reward=env.step(U[idx])



plt.plot(env.state_memory[0:,4],env.state_memory[0:,3])
plt.show