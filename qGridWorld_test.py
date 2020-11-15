from qiskit import QuantumCircuit,QuantumRegister,ClassicalRegister
from qiskit import Aer,IBMQ
from qiskit import execute
from qiskit.tools import visualization
from qiskit.tools.visualization import circuit_drawer, plot_histogram
import matplotlib.pyplot as plt
from executeCircuit import execute_locally,extractClassical
from qEnvironments import quantum_sparse_sampling

##########################################################################
###### Deterministic-MDP with 2 states with 2 possible actions each ######
############################## FUNCTIONAL ################################ 
##########################################################################

'''
The result should be for 2 steps:
	1) take action 1 , move to state 1 and collect reward 1
	2) take action 0 , stay in state 1 and collect reward 1

The maximum reward it's 2 and the best actions (best policy) for the mdp are 
action 1 for state 0 and action 0 for state 1


nstates = 2
nactions = 2
steps = 3

states=[i for i in range(nstates)]
actions=[i for i in range(nactions)]
#[0,1] -> [stay,move]

transition_kernel=[(0,1,1),(0,0,0),(1,1,0),(1,0,1)]
reward_model=[0,1]

qGW = quantumGridWorld(states=states,actions=actions,tKernel=transition_kernel,rewardModel=reward_model)
qGW.step(initState=0,horizon=steps)

maximum, top_measurement = qGW.solve()

print("TRANSITION_KERNEL\n",transition_kernel,"\n")
print("REWARD_MODEL -->> ",reward_model,"\n")
print("REWARD_COLLECTED -->> ",maximum,"\n")
print("---TOP_MEASUREMENT---\n\n",top_measurement)
'''

####################################################################
############# MDP with 2 equally optimal trjectories ###############
####################################################################

# States {0,1,3} will have reward 0 and state 3 will have reward 1
# making possible to have 2 optimal strategies to achieve maximum reward

# NOTE: With 2 steps the agent has allready seen the entire mdp in superposition
# so it suffices to have 2 steps to achieve the optimal strategy

nstates = 4

#########################
#			#			#
# state 3	# state 2	#	
#########################
#			#			#
# state 0	# state 1	#
#########################

nactions = 4
steps = 3

states=[i for i in range(nstates)]
actions=[i for i in range(nactions)]
#[0,1,2,3] -> [up,down,left,right]

transition_kernel=[(0,0,3),(0,1,0),(0,2,0),(0,3,1),(1,0,2),(1,1,1),(1,2,0),(1,3,1),(2,0,2),(2,1,1),(2,2,3),(2,3,2),(3,0,3),(3,1,0),(3,2,3),(3,3,2)]
reward_model=[0,0,1,0]

qGW = quantumGridWorld(states=states,actions=actions,tKernel=transition_kernel,rewardModel=reward_model)
qGW.step(initState=0,horizon=steps)

maximum, top_measurement = qGW.solve()

print("TRANSITION_KERNEL\n",transition_kernel,"\n")
print("REWARD_MODEL -->> ",reward_model,"\n")
print("REWARD_COLLECTED -->> ",maximum,"\n")
print("---TOP_MEASUREMENT---\n\n",top_measurement)

