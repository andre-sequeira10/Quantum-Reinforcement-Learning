from qiskit import QuantumCircuit,QuantumRegister,ClassicalRegister
from qiskit import Aer,IBMQ
from qiskit import execute
from qiskit.tools import visualization
from qiskit.tools.visualization import circuit_drawer, plot_histogram
import matplotlib.pyplot as plt

# n is the number o qubits in the registers
#summing registers a to b with c being the ancilla register or carry register also with n qubits 

def carry(qc, a, b, c, n):
    for i in range(n-1):
        qc.ccx(a[i], b[i], c[i+1])
        qc.cx(a[i], b[i])
        qc.ccx(c[i], b[i], c[i+1])

    #last carry bit directly to b instead of c
    qc.ccx(a[n-1], b[n-1],  b[n])
    qc.cx(a[n-1], b[n-1])
    qc.ccx(c[n-1], b[n-1], b[n])

def adder(qc, a, b, c, n):
    carry(qc,a,b,c,n)
    for i in range(n-1):
        #Reverse the operations in carry for the correct input bit
        qc.ccx(c[(n-2)-i], b[(n-2)-i], c[(n-1)-i])
        qc.cx(a[(n-2)-i], b[(n-2)-i])
        qc.ccx(a[(n-2)-i], b[(n-2)-i], c[(n-1)-i])
        #These two operations act as a sum gate; if a control bit is at                
        #the |1> state then the target bit b[(n-2)-i] is flipped
        qc.cx(c[(n-2)-i], b[(n-2)-i])
        qc.cx(a[(n-2)-i], b[(n-2)-i])

def increment(qc,reg,control=None):
    n=reg.size
    if control==None:
        for i in reversed(range(1,n)):
            qubits = [reg[j] for j in range(i)]
            qc.mct(qubits,reg[i],None,mode='noancilla')
        qc.x(reg[0])
    else:
        for i in reversed(range(1,n)):
            qubits = [reg[j] for j in range(i)]
            qubits.insert(0,control[0])
            qc.mct(qubits,reg[i],None,mode='advanced')
        qc.cx(control[0],reg[0])
    