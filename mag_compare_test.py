from qiskit import QuantumCircuit,QuantumRegister,ClassicalRegister
from qiskit import Aer,IBMQ
from qiskit import execute
from qiskit.tools import visualization
from qiskit.tools.visualization import circuit_drawer, plot_histogram
import matplotlib.pyplot as plt
from executeCircuit import execute_locally,extractClassical
from mag_compare_gates import mag_gt_k

#test |a> > b 
#mag_gt_k puts the result in |o> = |1> if a gt b 

#|a> = |100> = 4
a = QuantumRegister(3)
aux = QuantumRegister(3)
b = -1
o = QuantumRegister(1)
oc = ClassicalRegister(1)
ac = ClassicalRegister(3)
qc = QuantumCircuit(a,aux,o,ac,oc)

#qc.x(a[2])
qc.h(a)
mag_gt_k(qc,a,b,o,aux)

qc.measure(o[0],oc[0])
qc.barrier()
for i in range(3):
    qc.measure(a[i],ac[i])

execute_locally(qc,nshots=1024,show=True)
