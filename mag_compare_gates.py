
# importing Qiskit
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister

### exports:
# mag_gt_k (circuit, a, k, g, aux)
# mag_ge_k (circuit, a, k, g, aux)
# mag_lt_k (circuit, a, k, g, aux)
# mag_le_k (circuit, a, k, g, aux)
# mag_eq_k (circuit, a, k, g, aux)
# mag_gt (circuit, a, b, g, pa, pb, aux)

                                #### SCRIPT WITH MCGATES IS MISSING ####
                                #### I' DONT USE THESE GATES IN MAG_GT_K ####
                                #### SO IMPORT WILL BE IGNORED ####
#from mCgates import mCX

# Magnitude comparator equal to k (eq_k)
# sets g to |1> if |a> == k

# requires nw-2 ancilliary qubits
# nw-2 ancilliary qubits are used by mCX
def mag_eq_k (circuit, a, k, g, aux):
    nw = len(a)

    if (len(aux) < (nw-2)):
        raise ValueError('gt_k gate: ', nw-2, ' ancilliary qubits are required (', len(aux),' available)')

    div = k  
    j=0 
    while div>0:
        if (div & 1)==0:
            circuit.x(a[j])
        j = j+1
        div = div >> 1 
    while j<nw:          # remaining bits
        circuit.x(a[j])
        j = j+1

    mCX (circuit, a, aux, g)

    div = k   
    j = 0
    while div>0:
        if (div & 1)==0:
            circuit.x(a[j])
        j = j+1
        div = div >> 1 
    while j<nw:          # remaining bits
        circuit.x(a[j])
        j = j+1

# -- end Magnitude eq comparator

# Magnitude comparator value in a greater than k (gt_k)
# sets g to |1> if |a> > k

def mag_gt_k (circuit, a, k, g, aux):
    nqb = len(a)

    if (len(aux) < nqb):
        raise ValueError('gt_k gate: ', nqb, ' ancilliary qubits are required!')

    ## LS bit
    kbit = k & 1     # is this k's bit 0 or 1

    if kbit==0:
        # we want to make aux[0]=a[0] 
        # we know aux[0]==0 therefore use aux[0] = a[0] XOR aux[0] (CNOT)
        circuit.cx (a[0], aux[0])
    else:
        pass     # do nothing

    ## bits 1 to nqb-2
    for qb in range(1,nqb-1):
        kbit = k & (2**qb)     # is this k's bit 0 or 1

        if kbit==0:
            circuit.x (a[qb])
            circuit.ccx (a[qb], aux[qb-1], aux[nqb-1])
            
            circuit.x (aux[nqb-1])
            circuit.ccx (a[qb], aux[nqb-1], aux[qb])
            circuit.x (aux[qb])

	    # UNDO EVERYTHING to recover aux[nqb-1]  -- BUT NOT aux[qb]
            circuit.x (aux[nqb-1]) 
            circuit.ccx (a[qb], aux[qb-1], aux[nqb-1])
            circuit.x (a[qb])
        else:
            # we want to make aux[qb]=aux[qb-1] 
            # we know aux[qb]==0 therefore use aux[qb] = aux[qb] XOR aux[qb-1] (CNOT)
            circuit.ccx (a[qb], aux[qb-1], aux[qb])
 
    ## MS bit (nqb-1)
    qb = nqb-1
    kbit = k & (2**qb)     # is this k's bit 0 or 1

    if kbit==0:
        circuit.x (a[qb])
        circuit.ccx (a[qb], aux[qb-1], aux[nqb-1])
            
        circuit.x (aux[nqb-1])
        circuit.ccx (a[qb], aux[nqb-1], g)
        circuit.x (g)

	# UNDO EVERYTHING to recover aux[nqb-1]  -- BUT NOT g
        circuit.x (aux[nqb-1]) 
        circuit.ccx (a[qb], aux[qb-1], aux[nqb-1])
        circuit.x (a[qb])
    else:
        # we want to make g=aux[qb-1] 
        # we know g==0 therefore use g = g XOR aux[qb-1] (CNOT)
        circuit.ccx (a[qb], aux[qb-1], g)

    ####  NOW UNDO EVERYTHING to recover qubits from aux[0] to aux[nqb-2]
    # bits nqb-2 to 1
    for qb in range(nqb-2, 0, -1):

        kbit = k & (2**qb)
        if kbit==0:
            circuit.x (a[qb])
            circuit.ccx (a[qb], aux[qb-1], aux[nqb-1])
            circuit.x (aux[nqb-1]) 
            circuit.x (aux[qb])
            circuit.ccx (a[qb], aux[nqb-1], aux[qb])
            circuit.x (aux[nqb-1])
            circuit.ccx (a[qb], aux[qb-1], aux[nqb-1])
            circuit.x (a[qb])
        else:
            circuit.ccx (a[qb], aux[qb-1], aux[qb])
    ## LS bit
    kbit = k & 1     # is this k's bit 0 or 1
    if kbit==0:
        circuit.cx (a[0], aux[0])
    else:
        pass     # do nothing
        
# -- end Magnitude gt_k comparator

# Magnitude comparator value in a greater or equal to k (ge_k)
# sets g to |1> if |a> >= k
# uses mag_gt_k (a, k-1) if k > 0

def mag_ge_k (circuit, a, k, g, aux):
    
    if (k == 0):  # always true
        circuit.x(g)
        return
    
    # k > 0
    
    nqb = len(a)

    if (len(aux) < nqb):
        raise ValueError('gt_k gate: ', nqb, ' ancilliary qubits are required!')
    
    mag_gt_k(circuit, a, k-1, g, aux)
# -- end Magnitude ge_k comparator


# mag_lt_k uses mag_gt_k
def mag_lt_k (circuit, a, k, g, aux):

    if (k>0):
        mag_gt_k (circuit, a, k-1, g, aux)
        circuit.x(g)
# -- end Magnitude lt_k comparator

# mag_le_k uses mag_lt_k
def mag_le_k (circuit, a, k, g, aux):

    mag_lt_k (circuit, a, k+1, g, aux)
# -- end Magnitude lt_k comparator

# Magnitude comparator greater than (gt)
# sets g to |1> if |a> > |b>

def mag_gt (circuit, a, b, g, aux):
    nqb = len(b)

    if (len(a) != nqb):
        raise ValueError('comparator gate: a and b must have the same number of qubits')

    naux = len(aux)
    if (naux < nqb+1):
        raise ValueError('comparator gate: requires ' + (nqb+1) + '  qubits for ' + nqb + ' qubits operands!')

    

    # ### LS bit
    circuit.x (b[0])
    circuit.ccx (a[0], b[0], aux[0])   # aux[0] is a[0] > b[0]
    circuit.x (b[0])

    # bits 1 to nqb-2
    for qb in range(1, nqb-1):
    
        circuit.x (b[qb])
        circuit.ccx (a[qb], b[qb], aux[nqb-1])  # aux[nqb-1] is a[qb] AND !b[qb]

        circuit.cx (a[qb], b[qb])     # b[qb] is a[qb] XOR !b[qb]

        circuit.ccx (b[qb], aux[qb-1], aux[nqb])   

        # ORing the 2 previous results by NANDing the negated values
        circuit.x (aux[nqb])
        circuit.x (aux[nqb-1])
        circuit.ccx (aux[nqb-1], aux[nqb], aux[qb])   
        circuit.x (aux[qb])                           # aux[qb] is whether the a bits from 0 to qb are larger than in b

        # UNDOING everything to restore aux[nqb], aux[nqb-1], b[qb] (BUT NOT aux[qb])
        circuit.x (aux[nqb-1])
        circuit.x (aux[nqb])
        circuit.ccx (b[qb], aux[qb-1], aux[nqb])   
        circuit.cx (a[qb], b[qb])     
        circuit.ccx (a[qb], b[qb], aux[nqb-1])  
        circuit.x (b[qb])

    # bit nqb-1 (last one: write onto g)
    qb = nqb-1
    circuit.x (b[qb])
    circuit.ccx (a[qb], b[qb], aux[nqb-1])  # aux[nqb-1] is a[qb] AND !b[qb]

    circuit.cx (a[qb], b[qb])     # b[qb] is a[qb] XOR !b[qb]

    circuit.ccx (b[qb], aux[qb-1], aux[nqb])   

    # ORing the 2 previous results by NANDing the negated values
    circuit.x (aux[nqb])
    circuit.x (aux[nqb-1])
    circuit.ccx (aux[nqb-1], aux[nqb], g)   
    circuit.x (g)                           # g is the final result

    # UNDOING everything to restore aux[nqb], aux[nqb-1], b[qb] (BUT NOT g)
    circuit.x (aux[nqb-1])
    circuit.x (aux[nqb])
    circuit.ccx (b[qb], aux[qb-1], aux[nqb])   
    circuit.cx (a[qb], b[qb])     
    circuit.ccx (a[qb], b[qb], aux[nqb-1])  
    circuit.x (b[qb])

    ####  NOW UNDO EVERYTHING to recover qubits from aux[0] to aux[nqb-2]
    # bits nqb-2 to 1
    for qb in range(nqb-2, 0, -1):
        circuit.x (b[qb])
        circuit.ccx (a[qb], b[qb], aux[nqb-1])  
        circuit.cx (a[qb], b[qb])     
        circuit.ccx (b[qb], aux[qb-1], aux[nqb])   
        circuit.x (aux[nqb])
        circuit.x (aux[nqb-1])
        circuit.x (aux[qb])                          
        circuit.ccx (aux[nqb-1], aux[nqb], aux[qb])   
        circuit.x (aux[nqb-1])
        circuit.x (aux[nqb])
        circuit.ccx (b[qb], aux[qb-1], aux[nqb])   
        circuit.cx (a[qb], b[qb])     
        circuit.ccx (a[qb], b[qb], aux[nqb-1])  
        circuit.x (b[qb])
    # bit 0
    circuit.x (b[0])
    circuit.ccx (a[0], b[0], aux[0]) 
    circuit.x (b[0])

# -- end Magnitude comparator


