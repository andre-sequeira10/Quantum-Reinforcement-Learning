
from qiskit import QuantumCircuit,QuantumRegister,ClassicalRegister
from qiskit import Aer,IBMQ
from qiskit import execute
from qiskit.tools import visualization
from qiskit.tools.visualization import circuit_drawer, plot_histogram
import matplotlib.pyplot as plt
import warnings

# Plot results
def show_results(D,nshots,outpath=None,filename=None,plot_show=False,label=None):
    # D is a dictionary with classical bits as keys and count as value
    # example: D = {'000': 497, '001': 527}
    '''
    plt.bar(range(len(D)), list(D.values()), align='center')
    plt.xticks(range(len(D)), list(D.keys()))
    plt.show()
    '''

    if len(D) > 15:
        fig, ax = plt.subplots(figsize=(len(D) * 0.45,12))
    else:
        fig, ax = plt.subplots()
        
    # Create bar plot
    bar1 = ax.bar(range(len(D)), list(D.values()),color="darkorange")
    ax.set_facecolor("lightgray")
    ax.set_xticks(range(len(D)))
    ax.set_xticklabels(D.keys())#,rotation=45)
    ax.set_title('# Samples = '+str(nshots),fontweight="bold",fontsize=14)
    ax.set_ylabel("Counts",fontweight="bold",fontsize=14)
    if label is not None:
        ax.set_xlabel(label,fontsize=14)
    else:
        ax.set_xlabel(r"$\mathbf{|\psi\rangle}$",fontsize=14)

    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
    
    autolabel(bar1)
    
    from os import path 

    if filename is not None:
        if outpath is not None:
            plt.savefig(path.join(outpath,filename))
        else:
            plt.savefig(filename)
    
    if plot_show:			
        plt.show()

    
def extractClassical(d):
    return [*d]
    #return [int(x,base=2) for x in l]

# Execute circuit, display a histogram of the results
def execute_locally(qc,nshots=1,draw_circuit=False,decompose=False,show=False,savefig=False,backend="qasm_simulator",outpath=None,filename=None,label=None):
    if backend == "qasm_simulator":
        # Compile and run the Quantum circuit on a simulator backend
        backend_sim = Aer.get_backend('qasm_simulator')
        job_sim = execute(qc, backend_sim, shots=nshots)
        result_sim = job_sim.result()
        result_counts = result_sim.get_counts(qc)

        if draw_circuit:
            if decompose:
                qc.decompose().decompose().decompose().decompose().draw(output="mpl")
            else:   
                qc.draw(output="mpl")
        if show:
            show_results(result_counts,nshots,outpath=outpath,filename=filename,plot_show=savefig,label=label)
        elif savefig:
            show_results(result_counts,nshots,outpath=outpath,filename=filename,plot_show=False,label=label)
        else:
            pass

    elif backend == "statevector_simulator":
        backend_sim = Aer.get_backend('statevector_simulator')
        job_sim = execute(qc, backend_sim,shots=nshots)
        result_sim = job_sim.result()
        result_counts = result_sim.get_statevector(qc)
    # Print the results
    #print("---counts---\n\n",result_counts)

    return [*result_counts] , result_counts
    
def execute_ibmq5(qc, draw_circuit=False, show=False):
    pass

