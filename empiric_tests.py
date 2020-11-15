from qEnvironments import quantum_sparse_sampling as qSS
import numpy as np

#########################################################################################
###### GENERATE THE DISTRIBUTIONS ACCORDING TO A VARYING SAMPLE SIZE FOR DIFF MDPS ######
#########################################################################################

sample_size = [100,500,1000,2000,5000,7000,10000,15000]

#########################################################################################
######################################### MDP 1 #########################################
#########################################################################################

states=4
actions=2
transition_kernel= np.zeros((states,actions),dtype=object)
steps = 1 

sigma = 2.58
epsilon = 0.01
k = round(((sigma**2)/(8*(epsilon**2)))*(np.log2(actions))*(np.sqrt(16*(epsilon**2) + 1) + 1))

#probality, sprime, reward
transition_kernel[0][0] = [(0.9,1,10),(0.1,2,50)]
transition_kernel[0][1] = [(1,3,20)] 
for s in range(1,states):
    for a in range(actions):
        transition_kernel[s][a] = [(1,s,0)]

for ss in sample_size:
    qGW = qSS(states=states,actions=actions,tKernel=transition_kernel,gamma=0.9,env="grid")
    qGW.step(initState=0,horizon=steps)
    qGW.solve(shots=ss,algorithm="Exponential",outpath="/home/andre/Desktop/quantum/QRL/empiric_data/",filename="mdp1_"+str(ss)+"pi_over_4"+".png")

qGW = qSS(states=states,actions=actions,tKernel=transition_kernel,gamma=0.9,env="grid")
qGW.step(initState=0,horizon=steps)
qGW.solve(shots=k,algorithm="Exponential",outpath="/home/andre/Desktop/quantum/QRL/empiric_data/",filename="mdp1_optimal_"+str(k)+"pi_over_4"+".png")

#########################################################################################
######################################### MDP 2 #########################################
#########################################################################################

states=4
actions=2
transition_kernel= np.zeros((states,actions),dtype=object)
steps = 2

sigma = 2.58
epsilon = 0.01
k = round(((sigma**2)/(8*(epsilon**2)))*(np.log2(actions))*(np.sqrt(16*(epsilon**2) + 1) + 1))

#probality, sprime, reward
transition_kernel[0][0] = [(0.9,1,10),(0.1,2,50)]
transition_kernel[0][1] = [(1,3,20)] 
transition_kernel[1][0] = [(0.9,0,50),(0.1,1,10)]
transition_kernel[1][1] = [(1,2,20)] 
transition_kernel[2][0] = [(1,2,20)] 
transition_kernel[2][1] = [(1,2,20)] 
transition_kernel[3][0] = [(1,3,10)] 
transition_kernel[3][1] = [(1,3,10)] 

for ss in sample_size:
    qGW = qSS(states=states,actions=actions,tKernel=transition_kernel,gamma=0.9,env="grid")
    qGW.step(initState=0,horizon=steps)
    qGW.solve(shots=ss,algorithm="Exponential",outpath="/home/andre/Desktop/quantum/QRL/empiric_data/",filename="mdp2_"+str(ss)+"pi_over_4"+".png")

qGW = qSS(states=states,actions=actions,tKernel=transition_kernel,gamma=0.9,env="grid")
qGW.step(initState=0,horizon=steps)
qGW.solve(shots=k,algorithm="Exponential",outpath="/home/andre/Desktop/quantum/QRL/empiric_data/",filename="mdp2_optimal_"+str(k)+"pi_over_4"+".png")

#########################################################################################
######################################### MDP 3 #########################################
#########################################################################################

states = 2
actions = 2
steps = 2

sigma = 2.58
epsilon = 0.01
k = round(((sigma**2)/(8*(epsilon**2)))*(np.log2(actions))*(np.sqrt(16*(epsilon**2) + 1) + 1))

transition_kernel= np.zeros((states,actions),dtype=object)


#probality, sprime, reward
transition_kernel[0][0] = [(1,0,0)]
transition_kernel[0][1] = [(0.8,0,0),(0.2,1,2)]
transition_kernel[1][0] = [(1,1,2)]
transition_kernel[1][1] = [(1,0,1)] 

for ss in sample_size:
    qGW = qSS(states=states,actions=actions,tKernel=transition_kernel,gamma=0.9,env="grid")
    qGW.step(initState=0,horizon=steps)
    qGW.solve(shots=ss,algorithm="Exponential",outpath="/home/andre/Desktop/quantum/QRL/empiric_data/",filename="mdp3_"+str(ss)+"pi_over_4"+".png")

qGW = qSS(states=states,actions=actions,tKernel=transition_kernel,gamma=0.9,env="grid")
qGW.step(initState=0,horizon=steps)
qGW.solve(shots=k,algorithm="Exponential",outpath="/home/andre/Desktop/quantum/QRL/empiric_data/",filename="mdp3_optimal_"+str(k)+"pi_over_4"+".png")

#########################################################################################
######################################### MDP 4 #########################################
#########################################################################################

states = 3
actions = 2
steps = 3

sigma = 2.58
epsilon = 0.01
k = round(((sigma**2)/(8*(epsilon**2)))*(np.log2(actions))*(np.sqrt(16*(epsilon**2) + 1) + 1))

transition_kernel= np.zeros((states,actions),dtype=object)

#probality, sprime, reward
transition_kernel[0][0] = [(1,0,1)]
transition_kernel[0][1] = [(0.5,0,2),(0.5,1,2)]
transition_kernel[1][0] = [(0.5,0,1),(0.5,1,1)]
transition_kernel[1][1] = [(1,2,-1)] 
#state overheated is an absorving state, so in the quantum setting we leave the quantum agent in the same state with reward 0
transition_kernel[2][0] = [(1,2,0)] 
transition_kernel[2][1] = [(1,2,0)]  

for ss in sample_size:
    qGW = qSS(states=states,actions=actions,tKernel=transition_kernel,gamma=0.9,env="grid")
    qGW.step(initState=0,horizon=steps)
    qGW.solve(shots=ss,algorithm="Exponential",outpath="/home/andre/Desktop/quantum/QRL/empiric_data/",filename="mdp4_"+str(ss)+"pi_over_4"+".png")

qGW = qSS(states=states,actions=actions,tKernel=transition_kernel,gamma=0.9,env="grid")
qGW.step(initState=0,horizon=steps)
qGW.solve(shots=k,algorithm="Exponential",outpath="/home/andre/Desktop/quantum/QRL/empiric_data/",filename="mdp4_optimal_"+str(k)+"pi_over_4"+".png")

#########################################################################################
######################################### MDP 5 #########################################
#########################################################################################

states = 4
actions = 4
steps = 2
sigma = 2.58
epsilon = 0.01
k = round(((sigma**2)/(8*(epsilon**2)))*(np.log2(actions))*(np.sqrt(16*(epsilon**2) + 1) + 1))

tk = np.zeros((states,actions),dtype=object)

tk[0][0] = [(0.8,0,0),(0.10,1,0),(0.10,0,0)]
tk[0][1] = [(0.8,2,-1),(0.10,1,0),(0.10,0,0)]
tk[0][2] = [(0.8,0,0),(0.10,0,0),(0.10,2,-1)]
tk[0][3] = [(0.8,1,0),(0.10,0,0),(0.10,2,-1)]
tk[1][0] = [(0.8,1,0),(0.10,0,0),(0.10,1,0)]
tk[1][1] = [(0.8,3,2),(0.10,0,0),(0.10,1,0)]
tk[1][2] = [(0.8,0,0),(0.10,1,0),(0.10,3,2)]
tk[1][3] = [(0.8,1,0),(0.10,1,0),(0.10,3,2)]
#state 2 is a hole , in the quantum setting we remain the agent in the same state with zero reward.
tk[2][0] = [(1,1,1)]
tk[2][1] = [(1,1,1)]
tk[2][2] = [(1,1,1)]
tk[2][3] = [(1,1,1)]
#state 3 is the goal state, so the agent remains in the goal state with 2 reward
tk[3][0] = [(1,3,2)]
tk[3][1] = [(1,3,2)]
tk[3][2] = [(1,3,2)]
tk[3][3] = [(1,3,2)]


for ss in sample_size:
    qGW = qSS(states=states,actions=actions,tKernel=tk,gamma=0.9,env="grid")
    qGW.step(initState=0,horizon=steps)
    qGW.solve(shots=ss,algorithm="Exponential",outpath="/home/andre/Desktop/quantum/QRL/empiric_data/",filename="mdp5_"+str(ss)+"pi_over_4"+".png")

qGW = qSS(states=states,actions=actions,tKernel=tk,gamma=0.9,env="grid")
qGW.step(initState=0,horizon=steps)
qGW.solve(shots=k,algorithm="Exponential",outpath="/home/andre/Desktop/quantum/QRL/empiric_data/",filename="mdp5_optimal_"+str(k)+"pi_over_4"+".png")