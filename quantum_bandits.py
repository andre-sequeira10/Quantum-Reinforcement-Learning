from qiskit import QuantumCircuit,QuantumRegister,ClassicalRegister
from qiskit import Aer,IBMQ
from qiskit import execute
from qiskit.tools import visualization
from qiskit.tools.visualization import circuit_drawer, plot_histogram
import matplotlib.pyplot as plt
from executeCircuit import execute_locally,extractClassical
from qAlgorithms import ctrl_initialize
import numpy as np
import operator

n=1

a = QuantumRegister(n,"arm")
r = QuantumRegister(n,"reward")
qc = QuantumCircuit(a,r)

new_circ=QuantumCircuit(name=r"$\psi$")
a1 = QuantumRegister(n)
r1 = QuantumRegister(n)
new_circ.add_register(a1,r1)
state0 = [complex(np.sqrt(0.3),0.0),complex(np.sqrt(0.7),0.0)]
state1 = [complex(np.sqrt(1),0.0),complex(0.0,0.0)]
new_circ.h(a1)
new_circ.ctrl_initialize(statevector=state0,ctrl_state = '0' , ctrl_qubits=a1,qubits=r1)
new_circ.ctrl_initialize(statevector=state1,ctrl_state = '1' , ctrl_qubits=a1,qubits=r1)

'''
#SAME AS ctrl_initialize
#Rotation for 0->30% / 1->70% controlled by action 1
new_circ.cry(2*np.pi/6.77,a1[0],r1[0])
#Rotation for 0->80% / 1->20% controlled by action 0
new_circ.barrier()
new_circ.x(a1[0])
new_circ.cry(2*np.pi/3.17,a1[0],r1[0])
new_circ.x(a1[0])
'''
qc.append(new_circ,[a,r])

def Grover_Iterate():
    ###### mark |r> == \1> 
    qc.z(r)
    qc.barrier()
    ######
    qc.append(new_circ.inverse(),[i for i in a]+[i for i in r])

    qc.x(a)
    qc.x(r)
    #qc.h(r)
    #qc.ccx(a[0],a[1],r[0])
    qc.cz(a[0],r[0])
    #qc.h(r)
    qc.x(a)
    qc.x(r)

    qc.append(new_circ,[i for i in a]+[i for i in r])
    
def QSearch(n,shots=1,backend="qasm_simulator"):
    it, lambd = 1, 6/5
    max_it = 2**n + 2**n
    
    qcGrover = {}
    c=0
    while it < max_it:
        qcGrover = QuantumCircuit()
        qcGrover+=qc
        g_it = np.random.randint(it) + 1
        for i in range(g_it):
            Grover_Iterate()
        
        qcGrover.measure_all()
        r,rc = execute_locally(qcGrover,nshots=shots,backend=backend)
        best_measure = max(rc.items(), key=operator.itemgetter(1))[0]
        print(best_measure)
        reward_measure = int(best_measure[0],2)
        if reward_measure:
            break
        else:
            it = min(lambd*it,max_it)
            c+=1
        
    if it >= max_it:
        raise ValueError("Search not Worked! Aborted!")
    
    return rc,best_measure

shots = 1000
'''
new_circ,qbandits = create_bandits(n)
qbandits.measure_all()
execute_locally(qbandits,nshots=shots,show=True,savefig=True,label=r"$\mathbf{|reward\rangle|action\rangle}$")
'''
'''
qc.measure_all()
#qc.decompose().draw(output="mpl",filename="bandits_decomposed.png")
qc.draw(output="mpl",filename="bandits.png")

execute_locally(qc,nshots=shots,show=True,savefig=True,filename="bandits_no_amplification.png",label=r"$\mathbf{|reward\rangle|arm\rangle}$")
qc.remove_final_measurements()

rc,best_measure = QSearch(n,shots=shots)
from executeCircuit import show_results
qc.draw(output="mpl")#,filename="bandits_qsearch_circuit.png")
show_results(rc,shots,plot_show=True,label=r"$\mathbf{|reward\rangle|arm\rangle}$")#,filename="bandits_qsearch.png")
'''
a_m = ClassicalRegister(1)
qc.add_register(a_m)
qc.measure(a,a_m)
execute_locally(qc,nshots=shots,show=True,savefig=True)#,filename="bandits_no_amplification.png",label=r"$\mathbf{|reward\rangle|arm\rangle}$")

qc.remove_final_measurements()
for i in range(1):
    Grover_Iterate()

qc.add_register(a_m)
qc.measure(a,a_m)

qc.draw(output="mpl")#,filename="bandits_qsearch_circuit.png")
execute_locally(qc,nshots=shots,show=True,savefig=True,filename="bandits_qsearch",label=r"$\mathbf{|arm\rangle}$")

#CASE OF k=4

'''
n=2

a = QuantumRegister(n,"arm")
r = QuantumRegister(1,"reward")
qc = QuantumCircuit(a,r)

new_circ=QuantumCircuit(name=r"$\psi$")
a1 = QuantumRegister(n)
r1 = QuantumRegister(1)
new_circ.add_register(a1,r1)
state0 = [complex(np.sqrt(0.5),0.0),complex(np.sqrt(0.5),0.0)]
state1 = [complex(np.sqrt(0.8),0.0),complex(np.sqrt(0.2),0.0)]
state2 = [complex(np.sqrt(0.9),0.0),complex(np.sqrt(0.1),0.0)]
state3 = [complex(np.sqrt(0.3),0.0),complex(np.sqrt(0.7),0.0)]
new_circ.h(a1)
new_circ.ctrl_initialize(statevector=state0,ctrl_state = '00' , ctrl_qubits=a1,qubits=r1)
new_circ.ctrl_initialize(statevector=state1,ctrl_state = '01' , ctrl_qubits=a1,qubits=r1)
new_circ.ctrl_initialize(statevector=state2,ctrl_state = '10' , ctrl_qubits=a1,qubits=r1)
new_circ.ctrl_initialize(statevector=state3,ctrl_state = '11' , ctrl_qubits=a1,qubits=r1)

qc.append(new_circ,[i for i in a]+[i for i in r])

qc.measure_all()
execute_locally(qc,nshots=shots,show=True,savefig=True,label=r"$\mathbf{|reward\rangle|arm\rangle}$")
qc.remove_final_measurements()

a_m = ClassicalRegister(n)
qc.add_register(a_m)

for i in range(1):
    Grover_Iterate()

qc.measure(a,a_m)
execute_locally(qc,nshots=shots,show=True,savefig=True,label=r"$\mathbf{|arm\rangle}$")
'''