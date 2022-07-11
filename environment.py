

import numpy as np

class env:
    def __init__(self,nT):
        # Initialize environment model
        
        # Set simulation parameters
        self.nT = nT   # Maximum expected timespan of simulatoin
        self.dT = 0.1  # Independent variable discretization
        
        # Define vehicle model parameters
        nStates = 5
        nInputs = 1;
        self.nStates = nStates
        self.nInputs = nInputs
        self.ActionSpace  =  np.linspace(-0.1, 0.1,9)

        # State upper and lower bounds [Should be slightly bigger than constraints]
        self.S_l = np.array([-0.42, -0.5, -0.03, -0.1, 0]);  # Lowr bound 
        self.S_u = np.array([0.42, 0.5, 0.03, 7.4, 110]);    # Upper bound
        self.V  = 20
        self.Lr = 1.6 #Distance from the center of gravity of the vehicle (CG) to the rear axle
        self.Cf = 1.9e5 # Cornering Stiffness of Front Tires x2
        self.Cr = 3.3e5 # Cornering Stiffness of Rear Tires x2
        self.m  = 1575
        self.Lf = 1.2
        self.J  = 2875
        self.obs1 = [80 ,2, 2,4]         #X,Y,W,H - Obstacle
        self.obs2 = [50,-0.5,100,1]      #X,Y,W,H - right road edge
        self.obs3 = [50,8.2-0.5, 100, 1] #X,Y,W,H - left road edge
        self.reset()
        
    def reset(self):
        rng = np.random.default_rng()
        # Initialize memory buffers
        self.state_memory   = np.zeros([self.nT+1,self.nStates])  # State history
        self.state_memory[0] = rng.uniform(low = self.S_l, high=self.S_u)
        self.action_memory  = np.zeros([self.nT+1,self.nInputs]) # Action history
        self.reward_memory  = np.zeros([self.nT+1,1])       # Rewards
        
        # Set initial conditions
        self.Tidx = 0
        self.terminated = bool(0) 
        return self.state_memory[0]
 
    def VD(self,State):
        # The vehicle dynamics model
        idx = self.Tidx
        action = self.action_memory[idx]
        

        # Vehicle dynamics model
        # Refer: https://saemobilus.sae.org/content/2022-01-0070/
        a4 = -(self.Cr+self.Cf)/(self.m*self.V)
        a5 = -1+(self.Cr*self.Lr-self.Cf*self.Lf)/(self.m*pow(self.V,2))
        a1 = (self.Cr*self.Lr-self.Cf*self.Lf)/self.J
        a2 = -((self.Cf*pow(self.Lf,2))+(self.Cr*pow(self.Lr,2)))/(self.J*self.V)
        a6 = self.Cf/(self.m*self.V)
        a3 = self.Cf*self.Lf/self.J
        
        S_next = np.zeros([1,5])
        S_next[0,0]   =   a4*State[0] + a5*State[1] + a6*action # Slip angle
        S_next[0,1]   =   a1*State[0] + a2*State[1] + a3*action #r
        S_next[0,2]   =   State[1] #phi
        S_next[0,3]   =   self.V*np.sin( State[2] + State[0]) # Y
        S_next[0,4]   =   self.V*np.cos( State[2] + State[0]) # X
        
        
        return S_next
    
    def Termination(self,S_):
        # Termination condition
        idx = self.Tidx   # Time index
        
        # Terminate if too long simulatio or if vehicle exits road
        if idx == self.nT or S_[4]>110: 
            self.terminated=bool(1)
        
    def step(self,action):
        
        U = self.ActionSpace[action]
        idx = self.Tidx   # Time index
        self.action_memory[idx] = U
        
        S = self.state_memory[idx]  # Current state
        
        # Simulate by 1 step
        S_=self.VD_step()
        
        # Reward at state S' 
        reward,term = self.reward_function(S_)
        idx+=1  # Increase time index by 1
        
        
        
        self.terminated = term
            
        # Store in the state memory          
        self.state_memory[idx] =S_
        self.Tidx = idx # Set time index
        self.reward_memory[idx]=reward # Set reward
        
        # Check if simulation should stop
        self.Termination(S_)
        
        return S_, reward, self.terminated
    
    def VD_step(self):
        # RK method to simulate dynamical system by 1 time step:
        dt = self.dT
        idx = self.Tidx
        S = self.state_memory[idx]
        
        
        k1 = dt*self.VD(S)
        k2 = dt*self.VD(S+0.5*k1[0])
        k3 = dt*self.VD(S+0.5*k2[0])
        k4 = dt*self.VD(S+k3[0])
        S_ = S+k1[0]/6+k2[0]/3+k3[0]/3+k4[0]/6
        return S_
    
    # Define reward
    def reward_function(self,S_):
        y = S_[3]
        x = S_[4]
        r = S_[1]
        
        
        
        Reward = -10*pow(r*10,2)
        Terminated = bool(0)
        
        #Obs1 : 
            
        if x>self.obs1[0]-self.obs1[2]/2 and x <self.obs1[0]+self.obs1[2]/2 and y >=self.obs1[1]-self.obs1[3]/2 and y<=self.obs1[1]+self.obs1[3]/2:
            Terminated  = bool(1)
        
        # Right road edge
        if x>self.obs2[0]-self.obs2[2]/2 and x <self.obs2[0]+self.obs2[2]/2 and y >=self.obs2[1]-self.obs2[3]/2 and y<=self.obs2[1]+self.obs2[3]/2:
            Terminated = bool(1)
        
        # Left road edge
        if x>self.obs3[0]-self.obs3[2]/2 and x <self.obs3[0]+self.obs3[2]/2 and y >=self.obs3[1]-self.obs3[3]/2 and y<=self.obs3[1]+self.obs3[3]/2:
            Terminated = bool(1)
        
        # End of road segment (successful obstacle avoidance) 
        if x > 105:
            Terminated = bool(1)
            Reward = 1000
        return Reward, Terminated
