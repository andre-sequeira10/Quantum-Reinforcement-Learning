from qiskit import QuantumCircuit,QuantumRegister,ClassicalRegister
from qiskit import Aer,IBMQ
from qiskit import execute
from qiskit.tools import visualization
from qiskit.tools.visualization import circuit_drawer, plot_histogram
import matplotlib.pyplot as plt
from executeCircuit import execute_locally,extractClassical
import random as r
from qAlgorithms import QMF
import math
import random 
import numpy as np
import sys 
from qiskit.aqua.components.initial_states import Custom

#######################################################################
############ QMF TEST FOR A UNIFORM SUPERPOSITION STATE ###############
########################## FUNCTIONAL #################################
#######################################################################

n = 4 # quantum state will have 2**n superposition states 
superpositions = 2**n
maximum_expected = superpositions-1

#IN THIS CASE THE MAXIMUM WILL BE |1111> = 15 

qstate = QuantumRegister(n,name="qstate") #quantum register that contains the superposition
new_qr = QuantumRegister(n,name="new_qr") #qantum register for qmf measuring test 
qc = QuantumCircuit(qstate,new_qr)

qc.h(qstate)
qc.h(new_qr) #new_qr will be measured as |0001> = 1

'''
	INIT PARAMS FOR THE QUANTUM MAXIMUM FINDING ALG 

__init__(self, circuit=None,search_register=None,num_iterations=None, max_index=None, extra_registers=None,size=None):



#############################################################################################
	IF WITH A UNIFORM SUPERPOSITION STATE LIKE qstate WE CHOOSE INITIAL index = 0 or 1, 
	WHAT HAPPENS IS THAT THE ORACLE WILL MARK ABOUT N-1 OR N-2 TERMS IN THE SUPERPOSITION
	AND SO, THE AVERAGE WILL BECOME NEGATIVE, AND THE INVERSION ABOUT MEAN OPERATOR WILL 
	NOT WORK PROPERLY, MEANING THAT THE PROBABILITY OF COLLAPSING INTO 0 OR 1 INSTEAD OF 
	OF AN ACTUAL SOLUTION WILL BECOME LARGER AND LARGER MAKING REACHING THE MAXIMUM IMPOSSIBLE
	IN THESE CASES RANDOMLY CHOOSING AN INTIAL GUESS, IS THE BEST STRATEGY FOR ACHIEVING 
	THE MAXIMUM.
############################################################################################## 
'''

index = int(sys.argv[1])
alg = QMF(circuit=qc,search_register=qstate,size=superpositions,max_index=index,extra_registers=[new_qr])

Backend = Aer.get_backend('qasm_simulator')
maximum, top_measurement = alg.run(backend=Backend,shots=1024)

#measured_val = result['result']
#maximum = result['top_measurement']

print("MAXIMUM SHOULD BE |",np.binary_repr(maximum_expected,n),"> = %d" % maximum_expected)
print("MAXIMUM ACHIEVED - %d\n" % maximum)
print("TOP MEASUREMENT\n",top_measurement)


#######################################################################
############ QMF TEST FOR AN ARBITRAY CUSTOM INTIAL STATE #############
########################## FUNCTIONAL #################################
#######################################################################
'''
state vector array - > numpy array [[Real,imaginary], ... ] for all superposition state
For n=4 qubits the first term encodes the amplitude of the state |0000> the second term encodes |0001> and so one.

For n=4 i'll create the state = |s> = 1/sqrt(2) ( |0100> + |0010> ) and QMF should return
|0100> = 4 as the maximum value within the state.
'''

state=[]
for i in range(2**n):
	if i == 4 or i == 2:
		state.append(complex(1/math.sqrt(2),0.0))
	else:
		state.append(complex(0.0,0.0))

state = np.asarray(state)

quantum_state = Custom(n,state_vector=state)

index = int(sys.argv[1])
alg = QMF(initial_state=quantum_state,search_register=qstate,size=superpositions,max_index=index)

Backend = Aer.get_backend('qasm_simulator')
maximum, top_measurement = alg.run(backend=Backend,shots=1024)

#measured_val = result['result']
#maximum = result['top_measurement']

print("MAXIMUM SHOULD BE |0100> = 4\n")
print("MAXIMUM ACHIEVED - %d\n" % maximum)
print("TOP MEASUREMENT\n",top_measurement)


###########################################################################
############ QMF TEST FOR A STATE THATS NOT IN SUPERPOSITION ##############
########################## FUNCTIONAL #####################################
###########################################################################
'''
Test if the quantum maximum finding algorithm can find the maximum in a register 
tht's not in superposition, qstate, however it is conditioned by another register, h_state, that's in superposition.

qc has 2 CNOT gates , the result should be |psi> = |11>|11> , giving the maximum on the qstate register of |qstate> = |11> = 3
'''
h_state = QuantumRegister(2,name="hstate")
qstate = QuantumRegister(2,name="qstate")
qc= QuantumCircuit(h_state,qstate)

h_size = 4

qc.h(h_state)
qc.cx(h_state[0],qstate[0])
qc.cx(h_state[1],qstate[1])

alg = QMF(circuit=qc,search_register=qstate,size=h_size,extra_registers=[h_state],draw_circuit=True)

Backend = Aer.get_backend('qasm_simulator')
maximum, top_measurement = alg.run(backend=Backend,shots=1024)

#measured_val = result['result']
#maximum = result['top_measurement']

print("MAXIMUM SHOULD BE |1> = 1\n")
print("MAXIMUM ACHIEVED - %d\n" % maximum)
print("TOP MEASUREMENT\n",top_measurement)

