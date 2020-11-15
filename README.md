# Quantum-Reinforcement-Learning

Work done in my master thesis "Quantum-enhanced Reinforcement Learning". Developed a quantum version of the classical RL algorithm Sparse Sampling by Kearns et.al
Simulations were made using the IBM Qiskit software version:
  - qiskit 0.6.0
  - qiskit-aqua 0.4.0

Implementation of the quantum maximum finding algorithm compatible with qiskit aqua.
Tests of the quantum sparse sampling algorithm for different stochastic MDP's are present in the stochastic_mdp_test.ipynb notebook

######################################################
################# qAlgorithms.py #####################
######################################################

->Qiskit Aqua's Grover implementation doesn't allow one to pass a predefined circuit , a complex circuit perhaps as an oracle instance with the purpose of measuring more than just the target register(the one that will be amplitude amplified). Given that in real world applications we want in many times see the result of other quantum register when collapsing the target register (obtain the full quantum state and not just the registr amplified), in aqua that's not possible because inside of Grover's routine, Aqua creates a bunch of new quantum circuits, and even if the oracle contains ALL the complex predefined circuit, we can't tell aqua to measure registers other than the amplified one. 
Said that, in this file is a new implementation which tackles this problems, rising a new level of abstraction for grovers algorithm within qiskit.

-> This implementation of Grover's Algorithm also gives the programmer the possibility of calling grover's algorithm for some quantum state without needing to implement the oracle for the search itself. For this, the programmer just needs to pass grover the search parameter, and behind the scenes, grover implements the oracle, that is implemented in the qOracles.py file.

->There's also an impplementation of Quantum Maximum finding compatible with Qiskit Aqua.

'''''''''''''''''''''''''''''''''''''''''''''''
initialization:

def __init__(self, oracle=None, init_state=None, init_state_circuit=None,
				 incremental=False, num_iterations=1, search_register=None, search_index=None, extra_registers=None,mct_mode='basic'):

In a python file import algorithms as :

	from qAlgorithms import grover
	from qAlgorithms import QMF
'''''''''''''''''''''''''''''''''''''''''''''''

######################################################
################# qEnvironments.py ###################
######################################################

-> In this file there's an implementation of the GridWorld environment in the quantum framework - quntumGridWorld
->For now there's only an implementation of the deterministic Markov Decision Process but in near the future, the stochastic MDP will be available.

'''''''''''''''''''''''''''''''''''
Methods within the class:

qgw = quantumGridWorld(states,actions,reward_model,transition_kernel) to create the quantum environment .

qgw.step(initState,horizon) to make the agent transitioning between states under the superposition policy, horizon being the time step.

reward, top_measurement = qgw.solve() solves the MDP, giving the reward collected and the series of actions that lead to the maximum reward collected.

In a python file import algorithms as :
	from qEnvironments import quantumGridWorld

''''''''''''''''''''''''''''''''''''

######################################################
################# qOracles.py ########################
######################################################

-> Implementation of general oracles as oracle instances for Qiskit Aqua for the the grover search and the quantum maximum finding routines.

'''''''''''''''''''''''''''''''''''''
In a python file import algorithms as :
	from qOracles import searchOracle, qmfOracle
'''''''''''''''''''''''''''''''''''''

######################################################
################# qArithmetic.py #####################
######################################################

->Implementation of quantum arithmetic routines including the adder and the increment operation

'''''''''''''''''''''''''''''''''''''
In a python file import algorithms as :

	from qArithmetic adder, increment
'''''''''''''''''''''''''''''''''''''

######################################################
################# mag_compare_gates.py ###############
######################################################
	Implemented by Luis Paulo Santos (DI,Uminho)
			
-> Magnitude comparator operations in a quantum computer.

->greater than n:
Compares the value of a quantum state with some integer n and stores the result in a quantum state, |1> if n is greater than the binary reperesentation in the quantum state, otherwise |0>.

'''''''''''''''''''''''''''''''''''''''
In a python file import algorithms as :

	from mag_compare_gates import mag_gt_k
'''''''''''''''''''''''''''''''''''''''

######################################################
################# qGridWorld_test.py #################
######################################################

->test the quantum gridworld environment. for running the file execute in the terminal:
python qGridWorld_test.py



