# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
from qiskit import QuantumCircuit,QuantumRegister,ClassicalRegister
from qiskit import Aer,IBMQ
from qiskit import execute
from qiskit.tools import visualization
from qiskit.tools.visualization import circuit_drawer, plot_histogram
import matplotlib.pyplot as plt
from executeCircuit import execute_locally,extractClassical
#from qEnvironments import quantum_sparse_sampling
import numpy as np
import math
from controlled_init import ctrl_initialize
#from qiskit.extensions.standard.ry import RYGate,CRYGate


n=2

qrs = QuantumRegister(n)
qra = QuantumRegister(1)

qrsp = QuantumRegister(n)


qc = QuantumCircuit(qrs,qra,qrsp)

qc.h(qra)

state0 = [complex(0.0,0.0) , complex(math.sqrt(0.9),0.0), complex(math.sqrt(0.1),0.0), complex(0.0,0.0)]

qc.ctrl_initialize(statevector=state0,ctrl_state = '000' , ctrl_qubits=[i for i in qrs]+[i for i in qra],qubits=qrsp)

#qc.measure_all()
qc2 = qc.reverse_ops().copy()
qc2.draw(output="mpl")
qc2.measure_all()
r,rc = execute_locally(qc2,nshots=100,show=True)
plt.show()


