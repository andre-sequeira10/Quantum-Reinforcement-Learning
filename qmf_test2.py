from qiskit import QuantumCircuit,QuantumRegister,ClassicalRegister
from qiskit import Aer,IBMQ
from qiskit import execute
from qiskit.tools import visualization
from qiskit.tools.visualization import circuit_drawer, plot_histogram
import matplotlib.pyplot as plt
from executeCircuit import execute_locally,extractClassical
import random as r
from qAlgorithms import quantumMaximumFinding
import math as m 
import random 
from qArithmetic import increment

size = 4 # quantum state will have 2**size superposition states 

#Code for testing qmf on a register r, that will be incremented controlled by a 
#superpostition state s
#If |s> == |10> increment r 1x 
#If |s> == |11> increment r 2x 

#SO IN THIS CASE THE MAXIMUM SHOULD BE |r> = |10> = 2 

s = QuantumRegister(size) #quantum register that contains the superposition
r = QuantumRegister(size) #quantum register for the increment
o = QuantumRegister(1) #quantum register for mark elements 
aux = QuantumRegister(size) #register for mag compare circuit
qinc = QuantumRegister(1) #aux register for the increment routine
sc = ClassicalRegister(size) #measure 
rc = ClassicalRegister(size) #measure 
oc = ClassicalRegister(1)

def measure(qc,a,b,o,ac,bc,oc):
    for i in range(len(a)):
        qc.measure(a[i],ac[i])
    for i in range(len(b)):
        qc.measure(b[i],bc[i])
    for i in range(len(o)):
        qc.measure(o[i],oc[i])

# # of iterations will be sqrt(2**size)
#qmf gives the correct answer with prob 1 - (1/2)^k , with k being the number of times 
# the algorithm is repeated 
k=2
qmf_iterations = k*(m.ceil(m.sqrt(2**size)))

#count = 0
#ite = 100
#for it in range(ite):
for i in range(10):
    index = 0
    for i in range(qmf_iterations):
            #Need to create new quantum circuit each iteration because if
            #i just do qc.reset(), the same circuit grows exponentially in size 
            # and executes the same circuit i times  
            qc = QuantumCircuit(s,r,o,qinc,aux,sc,rc,oc)
            #while not solution:
            qc.h(s)
            '''
            qc.x(s[0])
            qc.x(s[1])
            qc.ccx(s[0],s[1],qinc[0])
            increment(qc,r,control=qinc)
            increment(qc,r,control=qinc)
            qc.x(s[0])
            qc.x(s[1])
            qc.x(s[0])
            qc.ccx(s[0],s[1],qinc[0])
            increment(qc,r,control=qinc)
            qc.x(s[0])
            '''
            new_index = quantumMaximumFinding(qc,s,index,o,aux)
            print(new_index)
            if new_index > index:
                #solution = True
                index = new_index
                #else:
                #   qc = QuantumCircuit(a,o,aux,ac,oc)
                #  measure(qc,a,o,ac,oc)
    measure(qc,s,r,o,sc,rc,oc)

    print("MAXIMUM SHOULD BE |r> = |10> = 2 : \n")
    print("MAXIMUM ACHIEVED - %d\n" % index)
print(execute_locally(qc,nshots=size,draw_circuit=True,show=True))


