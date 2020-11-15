import math
import numpy as np
import scipy

from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit import BasicAer, execute
from qiskit.exceptions import QiskitError
#from qiskit.circuit import CompositeGate
from qiskit.circuit import Gate
from qiskit.circuit import Instruction
from qiskit.circuit.reset import Reset
from qiskit.extensions.standard.u3 import U3Gate
from qiskit.extensions.standard.cx import CnotGate
from qiskit.extensions.standard.ry import RYGate,CryGate
from qiskit.extensions.standard.rz import RZGate,CrzGate

# ====================================================================
_EPS = 1e-10  # global variable used to chop very small numbers to zero

class ControlledInitialize(Instruction):
    """Complex amplitude initialization.

    Class that implements the (complex amplitude) initialization of some
    flexible collection of qubit registers (assuming the qubits are in the
    zero state).
    """

    def __init__(self, controlled_qubit, params):
        """Create new initialize composite.

        params (list): vector of complex amplitudes to initialize to
        """
        
        # WATCH ME: save controlled qubit register
        self.controlled_qubit = controlled_qubit
        
        num_qubits = math.log2(len(params))

        # Check if param is a power of 2
        if num_qubits == 0 or not num_qubits.is_integer():
            raise QiskitError("Desired statevector length not a positive power of 2.")

        # Check if probabilities (amplitudes squared) sum to 1
        if not math.isclose(sum(np.absolute(params) ** 2), 1.0,
                            abs_tol=_EPS):
            raise QiskitError("Sum of amplitudes-squared does not equal one.")

        num_qubits = int(num_qubits)

        super().__init__("controlledinitialize", num_qubits, 0, params) # +1 per il controllo

    def _define(self):
        """Calculate a subcircuit that implements this initialization

        Implements a recursive initialization algorithm, including optimizations,
        from "Synthesis of Quantum Logic Circuits" Shende, Bullock, Markov
        https://arxiv.org/abs/quant-ph/0406176v5

        Additionally implements some extra optimizations: remove zero rotations and
        double cnots.
        """
        # call to generate the circuit that takes the desired vector to zero
        disentangling_circuit = self.gates_to_uncompute()

        # invert the circuit to create the desired vector from zero (assuming
        # the qubits are in the zero state)
        initialize_instr = disentangling_circuit.to_instruction().inverse()
        
        q = QuantumRegister(self.num_qubits, 'q')
        initialize_circuit = QuantumCircuit(self.controlled_qubit, q, name='init_def')
        for qubit in q:
            initialize_circuit.append(Reset(), [qubit])
            
        # WATCH ME: cambiati registri
        temp_qubitsreg = [ self.controlled_qubit[0] ] + q[:]
        # initialize_circuit.append(initialize_instr, q[:])
        initialize_circuit.append(initialize_instr, temp_qubitsreg)

        self.definition = initialize_circuit.data

    def gates_to_uncompute(self):
        """
        Call to create a circuit with gates that take the
        desired vector to zero.

        Returns:
            QuantumCircuit: circuit to take self.params vector to |00..0>
        """
        q = QuantumRegister(self.num_qubits)
        # WATCH ME: aggiunto registro controlled_qubit
        circuit = QuantumCircuit(self.controlled_qubit, q, name='disentangler')

        # kick start the peeling loop, and disentangle one-by-one from LSB to MSB
        remaining_param = self.params

        for i in range(self.num_qubits):
            # work out which rotations must be done to disentangle the LSB
            # qubit (we peel away one qubit at a time)
            (remaining_param,
             thetas,
             phis) = ControlledInitialize._rotations_to_disentangle(remaining_param)
            # WATCH ME: Initialize._rotations_to_disentangle diventa ControlledInitialize._rotations_to_disentangle
            
            # perform the required rotations to decouple the LSB qubit (so that
            # it can be "factored" out, leaving a shorter amplitude vector to peel away)
            
            # WATCH ME: substitute RZ with CRZ
            # rz_mult = self._multiplex(RZGate, phis)
            rz_mult = self._multiplex(CrzGate, phis)
            
            # WATCH ME: substitute RY with CRY
            # ry_mult = self._multiplex(RYGate, thetas)
            ry_mult = self._multiplex(CryGate, thetas)
            
            # WATCH ME: cambiati registri
            temp_qubitsreg = [ self.controlled_qubit[0] ] + q[i:self.num_qubits]
            # circuit.append(rz_mult.to_instruction(), q[i:self.num_qubits])
            # circuit.append(ry_mult.to_instruction(), q[i:self.num_qubits])
            circuit.append(rz_mult.to_instruction(), temp_qubitsreg)
            circuit.append(ry_mult.to_instruction(), temp_qubitsreg)
            
            print("Z: ", phis, " | Y: ", thetas)
            
        return circuit

    @staticmethod
    def _rotations_to_disentangle(local_param):
        """
        Static internal method to work out Ry and Rz rotation angles used
        to disentangle the LSB qubit.
        These rotations make up the block diagonal matrix U (i.e. multiplexor)
        that disentangles the LSB.

        [[Ry(theta_1).Rz(phi_1)  0   .   .   0],
         [0         Ry(theta_2).Rz(phi_2) .  0],
                                    .
                                        .
          0         0           Ry(theta_2^n).Rz(phi_2^n)]]
        """
        remaining_vector = []
        thetas = []
        phis = []

        param_len = len(local_param)

        for i in range(param_len // 2):
            # Ry and Rz rotations to move bloch vector from 0 to "imaginary"
            # qubit
            # (imagine a qubit state signified by the amplitudes at index 2*i
            # and 2*(i+1), corresponding to the select qubits of the
            # multiplexor being in state |i>)
            (remains,
             add_theta,
             add_phi) = ControlledInitialize._bloch_angles(local_param[2 * i: 2 * (i + 1)])
            # WATCH ME: Initialize._bloch_angles diventa ControlledInitialize._bloch_angles
            
            remaining_vector.append(remains)

            # rotations for all imaginary qubits of the full vector
            # to move from where it is to zero, hence the negative sign
            thetas.append(-add_theta)
            phis.append(-add_phi)

        return remaining_vector, thetas, phis

    @staticmethod
    def _bloch_angles(pair_of_complex):
        """
        Static internal method to work out rotation to create the passed in
        qubit from the zero vector.
        """
        [a_complex, b_complex] = pair_of_complex
        # Force a and b to be complex, as otherwise numpy.angle might fail.
        a_complex = complex(a_complex)
        b_complex = complex(b_complex)
        mag_a = np.absolute(a_complex)
        final_r = float(np.sqrt(mag_a ** 2 + np.absolute(b_complex) ** 2))
        if final_r < _EPS:
            theta = 0
            phi = 0
            final_r = 0
            final_t = 0
        else:
            theta = float(2 * np.arccos(mag_a / final_r))
            a_arg = np.angle(a_complex)
            b_arg = np.angle(b_complex)
            final_t = a_arg + b_arg
            phi = b_arg - a_arg

        return final_r * np.exp(1.J * final_t / 2), theta, phi

    def _multiplex(self, target_gate, list_of_angles):
        """
        Return a recursive implementation of a multiplexor circuit,
        where each instruction itself has a decomposition based on
        smaller multiplexors.

        The LSB is the multiplexor "data" and the other bits are multiplexor "select".

        Args:
            target_gate (Gate): Ry or Rz gate to apply to target qubit, multiplexed
                over all other "select" qubits
            list_of_angles (list[float]): list of rotation angles to apply Ry and Rz

        Returns:
            DAGCircuit: the circuit implementing the multiplexor's action
        """
        list_len = len(list_of_angles)
        local_num_qubits = int(math.log2(list_len)) + 1

        q = QuantumRegister(local_num_qubits)
        # WATCH ME: aggiunto registro controlled_qubit
        circuit = QuantumCircuit(self.controlled_qubit, q, name="multiplex" + local_num_qubits.__str__())

        lsb = q[0]
        msb = q[local_num_qubits - 1]

        # case of no multiplexing: base case for recursion
        if local_num_qubits == 1:
            temp_qubitsreg = [ self.controlled_qubit[0], q[0] ]
            circuit.append(target_gate(list_of_angles[0]), temp_qubitsreg)
            return circuit
                
            
        # calc angle weights, assuming recursion (that is the lower-level
        # requested angles have been correctly implemented by recursion
        angle_weight = scipy.kron([[0.5, 0.5], [0.5, -0.5]],
                                  np.identity(2 ** (local_num_qubits - 2)))

        # calc the combo angles
        list_of_angles = angle_weight.dot(np.array(list_of_angles)).tolist()

        # recursive step on half the angles fulfilling the above assumption
        multiplex_1 = self._multiplex(target_gate, list_of_angles[0:(list_len // 2)])
        temp_qubitsreg =  [ self.controlled_qubit[0] ] + q[0:-1]
        circuit.append(multiplex_1.to_instruction(), temp_qubitsreg)

        # attach CNOT as follows, thereby flipping the LSB qubit
        circuit.append(CnotGate(), [msb, lsb])

        # implement extra efficiency from the paper of cancelling adjacent
        # CNOTs (by leaving out last CNOT and reversing (NOT inverting) the
        # second lower-level multiplex)
        multiplex_2 = self._multiplex(target_gate, list_of_angles[(list_len // 2):])
        temp_qubitsreg = [ self.controlled_qubit[0] ] + q[0:-1]
        if list_len > 1:
            circuit.append(multiplex_2.to_instruction().mirror(), temp_qubitsreg)
        else:
            circuit.append(multiplex_2.to_instruction(), temp_qubitsreg)

        # attach a final CNOT
        circuit.append(CnotGate(), [msb, lsb])

        return circuit


def controlled_initialize(self, controlled_qubit, params, qubits):
    if not isinstance(qubits, list):
        qubits = [qubits]
    return self.append(ControlledInitialize(controlled_qubit, params), qubits)


QuantumCircuit.controlled_initialize = controlled_initialize

# ====================================================================

desired_vector = [ 1 / math.sqrt(2), 0, 0, 1 / math.sqrt(2) ]
qctl = QuantumRegister(1, "qctl")
qreg = QuantumRegister(2, "qreg")
creg = ClassicalRegister(2, "creg")
circuit = QuantumCircuit(qctl, qreg, creg)

circuit.x(qctl)
circuit.controlled_initialize(qctl, desired_vector, qreg)
circuit.measure(qreg, creg)

job = execute(circuit, BasicAer.get_backend('qasm_simulator'), shots=10000)
print('Counts: ', job.result().get_counts(circuit))



