# Quantum-Tree-based-Planning

Work done in my master thesis "Quantum-enhanced Reinforcement Learning". Developed a quantum version of the classical RL algorithm Sparse Sampling by Kearns et.al. Library for Quantum Tree-based Reinforcement Learning for general stochastic environments, implemented for the IBM Qiskit platform version :

	qiskit 0.28.0

To use the quantum algorithm import: 
	
	from qEnvironments import quantum_sparse_sampling as qSS
	
creting an object:
	
	qAgent = qSS(states = N_states , n_actions = N_actions , tKernel = transition_kernel)

Let the qAgent interact with the environment for horizon h:
	
	qAgent.step(initState = initial_state , horizon = h)
	
Use the exponential search to reach distribution over the set of possible actions to take in initial state, A, and the correspondent approximately optimal action a*

	A, a*, grover_iterations = qAgent.solve(shots=shots)
	

#### File random_mdp_generator can be used to generate random mdp's
#### Stochastic_mdp_test.ipynb is a notebook with a couple of simple environments
#### quantum_vs_classical_analysis.ipynb is an attempt to empirically verify the difference between the number of operations of both quantum and classical algorithms.
