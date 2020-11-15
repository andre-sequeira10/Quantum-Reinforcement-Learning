from qiskit import QuantumCircuit,QuantumRegister,ClassicalRegister
from qiskit import Aer,IBMQ
from qiskit import execute
from qiskit.tools import visualization
from qiskit.tools.visualization import circuit_drawer, plot_histogram
import matplotlib.pyplot as plt
from executeCircuit import execute_locally
import random 
import math 
from qiskit.qasm import pi 
import numpy as np
from qiskit.aqua import AquaError
import matplotlib.pyplot as plt
from qiskit.aqua.components.oracles import Oracle
from qiskit.aqua.components.initial_states import Custom
from qiskit.aqua.algorithms import QuantumAlgorithm
from qOracles import searchOracle,qmfOracle
from qiskit.aqua.circuits.gates import mct  # pylint: disable=unused-import
from qiskit.aqua.utils import get_subsystem_density_matrix
from qiskit.aqua import AquaError, Pluggable, PluggableType, get_pluggable_class
import logging
import operator
from timeit import default_timer as timer


logger = logging.getLogger(__name__)

class Grover(QuantumAlgorithm):
	# -*- coding: utf-8 -*-

	# This code is part of Qiskit.
	#
	# (C) Copyright IBM 2018, 2019.
	#
	# This code is licensed under the Apache License, Version 2.0. You may
	# obtain a copy of this license in the LICENSE.txt file in the root directory
	# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
	#
	# Any modifications or derivative works of this code must retain this
	# copyright notice, and modified files need to carry a notice indicating
	# that they have been altered from the originals.

	'''
	Adapted by André Sequeira from https://qiskit.org/documentation/_modules/qiskit/aqua/algorithms/single_sample/grover/grover.html#Grover 


	The Grover's Search algorithm.

	If the `num_iterations` param is specified, the amplitude amplification
	iteration will be built as specified.

	If the `incremental` mode is specified, which indicates that the optimal
	`num_iterations` isn't known in advance,
	a multi-round schedule will be followed with incremental trial `num_iterations` values.
	The implementation follows Section 4 of Boyer et al. <https://arxiv.org/abs/quant-ph/9605034>

	Qiskit Aqua's Grover implementation doesn't allow one to pass a predefined circuit , a complex circuit perhaps as an oracle instance with the purpose of measuring more than just the target register(the one that will be amplitude amplified). Given that in real world applications we want in many times see the result of other quantum register when collapsing the target register (obtain the full quantum state and not just the registr amplified), in aqua that's not possible because inside of Grover's routine, Aqua creates a bunch of new quantum circuits, and even if the oracle contains ALL the complex predefined circuit, we can't tell aqua to measure registers other than the amplified one. Said that, we propose a new implementation which tackles this problems, rising a new level of abstraction for grovers algorithm within qiskit.
	'''

	PROP_INCREMENTAL = 'incremental'
	PROP_NUM_ITERATIONS = 'num_iterations'
	PROP_MCT_MODE = 'mct_mode'

	CONFIGURATION = {
		'name': 'Grover',
		'description': "Grover's Search Algorithm",
		'input_schema': {
			'$schema': 'http://json-schema.org/draft-07/schema#',
			'id': 'grover_schema',
			'type': 'object',
			'properties': {
				PROP_INCREMENTAL: {
					'type': 'boolean',
					'default': False
				},
				PROP_NUM_ITERATIONS: {
					'type': 'integer',
					'default': 1,
					'minimum': 1
				},
				PROP_MCT_MODE: {
					'type': 'string',
					'default': 'basic',
					'enum': [
						'basic',
						'basic-dirty-ancilla',
						'advanced',
						'noancilla',
					]
				},
			},
			'additionalProperties': False
		},
		'problems': ['search'],
		'depends': [
			{
				'pluggable_type': 'initial_state',
				'default': {
					'name': 'CUSTOM',
					'state': 'uniform'
				}
			},
			{
				'pluggable_type': 'oracle',
				'default': {
					'name': 'LogicalExpressionOracle',
				},
			},
		],
	}

	def __init__(self, oracle=None, init_state=None, init_state_circuit=None,
				 incremental=False, num_iterations=1, search_register=None, search_index=None, extra_registers=None,method="output",mct_mode='basic'):
		"""
		Constructor.

		Args:
			oracle (Oracle): the oracle pluggable component
			init_state (InitialState): the initial quantum state preparation
			incremental (bool): boolean flag for whether to use incremental search mode or not
			num_iterations (int): the number of iterations to use for amplitude amplification
			mct_mode (str): mct mode
		Raises:
			AquaError: evaluate_classically() missing from the input oracle
		"""
		self.validate(locals())
		super().__init__()

		self._extra_reg_len=0
		if oracle is None:
			if search_index is None:
				raise AquaError("Index For Grover search is missing")
			
			if search_register is None:
				raise AquaError("Variable register for the oracle is missing. Insert either the oracle or the variable register")
			if extra_registers is not None:
				for r in extra_registers:
					self._extra_reg_len+=len(r)
				self._oracle = searchOracle(variable_register=search_register,index=search_index,extra_regs=extra_registers)
			else:
				self._oracle = searchOracle(variable_register=search_register,index=search_index)
		else:
			
			if not callable(getattr(oracle, "evaluate_classically", None)):
				raise AquaError('Missing the evaluate_classically() method from the provided oracle instance.')

			self._oracle = oracle
			_extra_reg=self._oracle.extra_registers
			if _extra_reg is not None:
				for r in _extra_reg:
					self._extra_reg_len+=len(r)

		
		self._mct_mode = mct_mode
		self._method = method

		'''
		Qiskit Aqua _init_state does not allow one to pass a circuit for state preparation
		unless a user prepares the circuit a priori with Custom function from qiskit!
		IS it really necessary ? 
		This should be optimized in order for the programmer to have more freedom
		'''
		if init_state is not None:
			self._init_state = init_state

		else:
			if self._extra_reg_len is not None:
				self._init_state = Custom(len(self._oracle.variable_register)+self._extra_reg_len, circuit=init_state_circuit) if init_state_circuit else Custom(len(self._oracle.variable_register)+self._extra_reg_len, state='uniform')
			else:
				self._init_state = Custom(len(self._oracle.variable_register), circuit=init_state_circuit) if init_state_circuit else Custom(len(self._oracle.variable_register), state='uniform')

		self._init_state_circuit = \
			self._init_state.construct_circuit(mode='circuit', register=self._oracle.variable_register)
		self._init_state_circuit_inverse = self._init_state_circuit.inverse()

		self._diffusion_circuit = self._construct_diffusion_circuit()
		self._max_num_iterations = np.ceil(2 ** (len(self._oracle.variable_register) / 2))
		self._incremental = incremental
		self._num_iterations = num_iterations if not incremental else 1
		self.validate(locals())
		if incremental:
			logger.debug('Incremental mode specified, ignoring "num_iterations".')
		else:
			if num_iterations > self._max_num_iterations:
				logger.warning('The specified value %s for "num_iterations" '
							   'might be too high.', num_iterations)
		self._ret = {}
		self._qc_aa_iteration = None
		self._qc_amplitude_amplification = None
		self._qc_measurement = None

	def _construct_diffusion_circuit(self):
		qc = QuantumCircuit(self._oracle.variable_register)
		num_variable_qubits = len(self._oracle.variable_register)

		#If user wants to measure more than the target register, then the extra registers
		#need to be part of the diffusion circuit otherwise it will measure some superposition
		#sample instead.
		'''
		if self._extra_reg_len > 0:
			extra_registers = self._oracle.extra_registers
			extra_qubits=[]
			for r in extra_registers[:-1]:
				num_variable_qubits += len(r)
				for i in range(len(r)):
					extra_qubits.append(r[i])
		'''
		num_ancillae_needed = 0
		if self._mct_mode == 'basic' or self._mct_mode == 'basic-dirty-ancilla':
			num_ancillae_needed = max(0, num_variable_qubits - 2)
		elif self._mct_mode == 'advanced' and num_variable_qubits >= 5:
			num_ancillae_needed = 1

		# check self._oracle's existing ancilla and add more if necessary
		num_oracle_ancillae = \
			len(self._oracle.ancillary_register) if self._oracle.ancillary_register else 0
		num_additional_ancillae = num_ancillae_needed - num_oracle_ancillae
		if num_additional_ancillae > 0:
			extra_ancillae = QuantumRegister(num_additional_ancillae, name='a_e')
			qc.add_register(extra_ancillae)
			ancilla = list(extra_ancillae)
			if num_oracle_ancillae > 0:
				ancilla += list(self._oracle.ancillary_register)
		else:
			ancilla = self._oracle.ancillary_register

		if self._oracle.ancillary_register:
			qc.add_register(self._oracle.ancillary_register)
		qc.barrier(self._oracle.variable_register)
		qc += self._init_state_circuit_inverse
		'''
		if self._extra_reg_len > 0:
			qc.u3(pi, 0, pi, self._oracle.variable_register)
			for r in extra_registers:
				qc.u3(pi, 0, pi, r)

			reg_size = len(extra_registers[-1])
			qc.u2(0, pi, extra_registers[-1][reg_size-1])
			controls = self._oracle.variable_register[0:num_variable_qubits - 1]+extra_qubits
			qc.mct(controls,extra_registers[-1][reg_size-1],ancilla,mode=self._mct_mode)
			qc.u2(0, pi, extra_registers[-1][reg_size-1])
			for r in extra_registers:
				qc.u3(pi, 0, pi, r)
			qc.u3(pi, 0, pi, self._oracle.variable_register)'''
		
		qc.u3(pi, 0, pi, self._oracle.variable_register)
		qc.u2(0, pi, self._oracle.variable_register[num_variable_qubits - 1])
		qc.mct(self._oracle.variable_register[0:num_variable_qubits - 1],self._oracle.variable_register[num_variable_qubits - 1],ancilla,mode=self._mct_mode)
		qc.u2(0, pi, self._oracle.variable_register[num_variable_qubits - 1])
		qc.u3(pi, 0, pi, self._oracle.variable_register)
		
		qc += self._init_state_circuit
		qc.barrier(self._oracle.variable_register)
	
		return qc

	@classmethod
	def init_params(cls, params, algo_input):
		"""
		Initialize via parameters dictionary and algorithm input instance
		Args:
			params (dict): parameters dictionary
			algo_input (AlgorithmInput): input instance
		Returns:
			Grover: and instance of this class
		Raises:
			AquaError: invalid input
		"""
		if algo_input is not None:
			raise AquaError("Input instance not supported.")

		grover_params = params.get(Pluggable.SECTION_KEY_ALGORITHM)
		incremental = grover_params.get(Grover.PROP_INCREMENTAL)
		num_iterations = grover_params.get(Grover.PROP_NUM_ITERATIONS)
		mct_mode = grover_params.get(Grover.PROP_MCT_MODE)

		oracle_params = params.get(Pluggable.SECTION_KEY_ORACLE)
		self._oracle = get_pluggable_class(PluggableType.ORACLE,
									 oracle_params['name']).init_params(params)

		# Set up initial state, we need to add computed num qubits to params
		init_state_params = params.get(Pluggable.SECTION_KEY_INITIAL_STATE)
		init_state_params['num_qubits'] = len(self._oracle.variable_register)
		init_state = get_pluggable_class(PluggableType.INITIAL_STATE,
										 init_state_params['name']).init_params(params)

		return cls(self._oracle, init_state=init_state,
				   incremental=incremental, num_iterations=num_iterations, mct_mode=mct_mode)


	@property
	def qc_amplitude_amplification_iteration(self):
		""" qc amplitude amplification iteration """
		if self._qc_aa_iteration is None:
			self._qc_aa_iteration = QuantumCircuit()
			self._qc_aa_iteration += self._oracle.circuit
			self._qc_aa_iteration += self._diffusion_circuit
		return self._qc_aa_iteration

	def _run_with_existing_iterations(self):
		if self._quantum_instance.is_statevector:
			qc = self.construct_circuit(measurement=False)
			result = self._quantum_instance.execute(qc)
			complete_state_vec = result.get_statevector(qc)
			variable_register_density_matrix = get_subsystem_density_matrix(
				complete_state_vec,
				range(len(self._oracle.variable_register), qc.width())
			)
			variable_register_density_matrix_diag = np.diag(variable_register_density_matrix)
			max_amplitude = max(
				variable_register_density_matrix_diag.min(),
				variable_register_density_matrix_diag.max(),
				key=abs
			)
			max_amplitude_idx = \
				np.where(variable_register_density_matrix_diag == max_amplitude)[0][0]
			top_measurement = np.binary_repr(max_amplitude_idx, len(self._oracle.variable_register))
		else:
			qc = self.construct_circuit(measurement=True)
			measurement = self._quantum_instance.execute(qc).get_counts(qc)
			self._ret['measurement'] = measurement
			top_measurement = max(measurement.items(), key=operator.itemgetter(1))[0]
		
		#########################################################
		####### COMPLETE CIRCUIT APPEARS AT THIS STAGE ##########
		#########################################################
		'''
		qc.draw(output="mpl")
		plt.show()
		'''
		self._ret['complete_circuit'] = qc
		self._ret['top_measurement'] = top_measurement
		oracle_evaluation, assignment = self._oracle.evaluate_classically(top_measurement)
		return assignment, oracle_evaluation

	def construct_circuit(self, measurement=False):
		"""
		Construct the quantum circuit

		Args:
			measurement (bool): Boolean flag to indicate if
				measurement should be included in the circuit.

		Returns:
			QuantumCircuit: the QuantumCircuit object for the constructed circuit
		"""
		if self._qc_amplitude_amplification is None:
			self._qc_amplitude_amplification = \
				QuantumCircuit() + self.qc_amplitude_amplification_iteration
		
		qc = QuantumCircuit(self._oracle.variable_register, self._oracle.output_register)
		if self._method == "output":
			qc.u3(pi, 0, pi, self._oracle.output_register)  # x
			qc.u2(0, pi, self._oracle.output_register)  # h
		
		qc += self._init_state_circuit
		qc += self._qc_amplitude_amplification
		
		if measurement:
			qc_registers = self._oracle.get_registers
			for i in range(len(qc_registers)):
				measurement_cr = ClassicalRegister(len(qc_registers[i]), name="measurement_cr{0}".format(i))
				qc.add_register(measurement_cr)
				qc.measure(qc_registers[i], measurement_cr)


		self._ret['circuit'] = qc
		return qc


	def _run(self):
		if self._incremental:
			current_max_num_iterations, lam = 1, 6 / 5

			def _try_current_max_num_iterations():
				target_num_iterations = self.random.randint(current_max_num_iterations) + 1
				self._qc_amplitude_amplification = QuantumCircuit()
				for _ in range(target_num_iterations):
					self._qc_amplitude_amplification += self.qc_amplitude_amplification_iteration
				return self._run_with_existing_iterations()

			while current_max_num_iterations < self._max_num_iterations:
				assignment, oracle_evaluation = _try_current_max_num_iterations()
				if oracle_evaluation:
					break
				current_max_num_iterations = \
					min(lam * current_max_num_iterations, self._max_num_iterations)
		else:
			self._qc_amplitude_amplification = QuantumCircuit()
			for _ in range(self._num_iterations):
				self._qc_amplitude_amplification += self.qc_amplitude_amplification_iteration
			assignment, oracle_evaluation = self._run_with_existing_iterations()

		self._ret['result'] = assignment
		if self._method == "no_output":
			self._ret['top_measurement'] = assignment[self._oracle._output_register.name]
		else:
			self._ret['top_measurement'] = assignment[self._oracle.variable_register.name]
		self._ret['oracle_evaluation'] = oracle_evaluation
		return self._ret

class QMF:
	
	'''
	Quantum Maximum Finding algorithm developed by Hoyer, P. et.al https://arxiv.org/pdf/quant-ph/9607014.pdf implementation
	# of iterations will be sqrt(size) size being the # of superposition terms.
	The algorithm gives the correct answer with prob 1 - (1/2)^k , with k being the number of times the algorithm is repeated . 
	So, to have at least 75% probability of finding the maximum we need k=2 repetitions of the algorithm.
	If the user don't pass num_iterations k for QMF then it will be used k=2 for 75% certainty result in order to preserve the complexity growth of the algorithm. 
	'''

	def __init__(self, initial_state=None,circuit=None,search_register=None,num_iterations=None, max_index=None, extra_registers=None,size=None,draw_circuit=False,max_observation=None):

		if size is None:
			raise AquaError("Superposition state size is missing")		

		self._qcircuit = circuit  
		self._initial_state = initial_state
		self._search_reg = search_register
		self._size=size
		self._qmf_iterations = num_iterations*(math.ceil(math.sqrt(self._size))) if num_iterations is not None else 2*(math.ceil(math.sqrt(self._size)))
		self._max_index = max_index if max_index is not None else 0
		self._extra_regs = extra_registers
		self._draw_circuit=draw_circuit 
		self._max_observation = max_observation

	def run(self,backend=None,shots=1,debug=False):
		
		current_best_measures = None 
		for i in range(self._qmf_iterations):
			oraculo = qmfOracle(variable_register=self._search_reg,index=self._max_index,extra_regs=self._extra_regs)

			#initial_state = Custom(num_qubits,state_vector = stateV)
			if self._qcircuit is not None:
				algorithm = Grover(oracle=oraculo,init_state_circuit=self._qcircuit,search_register=self._search_reg,incremental=True,extra_registers=self._extra_regs)
			else:
				algorithm = Grover(oracle=oraculo,init_state=self._initial_state,search_register=self._search_reg,incremental=True,extra_registers=self._extra_regs)

			start = timer()
			if backend is None:
				backend=Aer.get_backend('qasm_simulator')
				result = algorithm.run(backend,shots=shots)
			else:
				result = algorithm.run(backend,shots=shots)
			end = timer()

			r=result['top_measurement']

			if debug:
				print("TIME ELAPSED -->> ",(end-start))
				print("TOP-- ",r)
				print("RESULT -- ", result['result'])

			if r > self._max_index:
				self._max_index = r
				current_best_measures = result['result']
			
			if r == self._max_observation:
				break

		if self._draw_circuit:
			result['complete_circuit'].draw()
			plt.show()

		return self._max_index , current_best_measures

##########################################################################################################################
######################## CONTROLLED VERSION OF THE QISKIT INITIALIZE FUNCTION ############################################
##########################################################################################################################

from qiskit.exceptions import QiskitError
from qiskit.circuit import Instruction
from qiskit.extensions.standard.x import XGate, CXGate
from qiskit.extensions.standard.ry import RYGate,CRYGate
from qiskit.extensions.standard.rz import RZGate,CRZGate
from qiskit.circuit.reset import Reset

_EPS = 1e-10  # global variable used to chop very small numbers to zero

class ctrl_Initialize(Instruction):
	# -*- coding: utf-8 -*-

	# This code is part of Qiskit.
	#
	# (C) Copyright IBM 2017.
	#
	# This code is licensed under the Apache License, Version 2.0. You may
	# obtain a copy of this license in the LICENSE.txt file in the root directory
	# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
	#
	# Any modifications or derivative works of this code must retain this
	# copyright notice, and modified files need to carry a notice indicating
	# that they have been altered from the originals.

	"""
	Adapted by André Sequeira from https://qiskit.org/documentation/_modules/qiskit/extensions/quantum_initializer/initializer.html#Initialize

	Complex amplitude initialization.

	Class that implements the (complex amplitude) initialization of some
	flexible collection of qubit registers (assuming the qubits are in the
	zero state) controlled by an arbitrary n-qubit state .
	Note that Initialize is an Instruction and not a Gate since it contains a reset instruction,
	which is not unitary.
	"""

	def __init__(self, params, ctrl_state):
		"""Create new initialize composite.

		params (list): vector of complex amplitudes to initialize to
		"""
		num_qubits = math.log2(len(params))
		self.num_ctrl_qubits = len(ctrl_state)
		self.ctrl_state = ctrl_state
		
		# Check if param is a power of 2
		if num_qubits == 0 or not num_qubits.is_integer():
			raise QiskitError("Desired statevector length not a positive power of 2.")

		# Check if probabilities (amplitudes squared) sum to 1
		if not math.isclose(sum(np.absolute(params) ** 2), 1.0,
							abs_tol=_EPS):
			raise QiskitError("Sum of amplitudes-squared does not equal one.")

		num_qubits = int(num_qubits)

		super().__init__("ctrl_initialize", num_qubits, 0, params)

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
		#disentangling_circuit.decompose().draw(output="text",filename="CIRCUIT1.txt")
		# invert the circuit to create the desired vector from zero (assuming
		# the qubits are in the zero state)
		
		#DONT NEED TO INVERT THE CIRCUIT BECAUSE THE CIRCUIT IS ALREADY THE INVERSE
		#INVERSE DONT RESPECT THE CONTROL STATES OF CONTROLLED GATES 

		initialize_instr = disentangling_circuit.to_instruction().inverse()
		#disentangling_circuit.inverse().decompose().draw(output="text",filename="CIRCUIT1inv.txt")

		#initialize_instr = disentangling_circuit.to_instruction()

		q = QuantumRegister(self.num_qubits, 'q')
		cq = QuantumRegister(self.num_ctrl_qubits, 'cq')
		initialize_circuit = QuantumCircuit(cq,q, name='init_def')
		#REMOVE RESET OPERTATION BECAUSE IF WE WANT TO USE THE CTRL_INITIALIZE IN SUPERPOSITION IT WILL COLLAPSE THE STATE INTO |0>
		'''
		for qubit in q:
			initialize_circuit.append(Reset(), [qubit])
		'''
		qregs = [i for i in cq]+[i for i in q]
		initialize_circuit.append(initialize_instr, qregs)
		
		self.definition = initialize_circuit.data

	def gates_to_uncompute(self):
		"""Call to create a circuit with gates that take the desired vector to zero.

		Returns:
			QuantumCircuit: circuit to take self.params vector to :math:`|{00\\ldots0}\\rangle`
		"""
		q = QuantumRegister(self.num_qubits)
		cq = QuantumRegister(self.num_ctrl_qubits)
		circuit = QuantumCircuit(cq,q, name='disentangler')

		# kick start the peeling loop, and disentangle one-by-one from LSB to MSB
		remaining_param = self.params

		for i in range(self.num_qubits):
			# work out which rotations must be done to disentangle the LSB
			# qubit (we peel away one qubit at a time)
			(remaining_param,
			 thetas,
			 phis) = ctrl_Initialize._rotations_to_disentangle(remaining_param)

			# perform the required rotations to decouple the LSB qubit (so that
			# it can be "factored" out, leaving a shorter amplitude vector to peel away)
			
			#rz_mult = self._multiplex(RZGate, phis)
			#ry_mult = self._multiplex(RYGate, thetas)

			qregs = [i for i in cq] + q[i:self.num_qubits]
			
			#circuit.append(rz_mult.to_instruction(), qregs)
			#circuit.append(ry_mult.to_instruction(), qregs)
			#TRY TO REVERSE THE OPERATION BECAUSE INVERSE DOESNT RESPECT THE CONTROL STATES
			circuit.append(self._multiplex(RZGate, phis).to_instruction(), qregs)
			circuit.append(self._multiplex(RYGate, thetas).to_instruction(),qregs)
			#thetas = [-theta for theta in thetas]
			#phis = [-phi for phi in phis]
			#circuit.append(self._multiplex(RYGate, thetas).to_instruction(), qregs)
			#circuit.append(self._multiplex(RZGate, phis).to_instruction(),qregs)

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
			 add_phi) = ctrl_Initialize._bloch_angles(local_param[2 * i: 2 * (i + 1)])

			remaining_vector.append(remains)

			# rotations for all imaginary qubits of the full vector
			# to move from where it is to zero, hence the negative sign
			thetas.append(-add_theta)
			phis.append(-add_phi)

		return remaining_vector, thetas, phis

	@staticmethod
	def _bloch_angles(pair_of_complex):
		"""
		Static internal method to work out rotation to create the passed-in
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

		###############################################################################
		######## we  need to convert the target gate into a controlled gate ###########
		###############################################################################

		list_len = len(list_of_angles)
		local_num_qubits = int(math.log2(list_len)) + 1

		cq = QuantumRegister(self.num_ctrl_qubits)
		q = QuantumRegister(local_num_qubits)
		circuit = QuantumCircuit(cq,q, name="multiplex" + local_num_qubits.__str__())

		lsb = q[0]
		msb = q[local_num_qubits - 1]

		# case of no multiplexing: base case for recursion
		if local_num_qubits == 1:
	
			qregs = [i for i in cq] + [q[0]]
			
			for i,j in zip(range(self.num_ctrl_qubits),reversed(range(self.num_ctrl_qubits))):
				if self.ctrl_state[i] == '0':
					circuit.x(cq[j])
				
			#circuit.append(target_gate(list_of_angles[0]), [q[0]])
			#ctrl_gate = target_gate(list_of_angles[0]).control(num_ctrl_qubits = self.num_ctrl_qubits,ctrl_state=self.ctrl_state)
			ctrl_gate = target_gate(list_of_angles[0]).control(num_ctrl_qubits = self.num_ctrl_qubits)
			circuit.append(ctrl_gate,qregs)

			for i,j in zip(range(self.num_ctrl_qubits),reversed(range(self.num_ctrl_qubits))):
				if self.ctrl_state[i] == '0':
					circuit.x(cq[j])

			return circuit

		# calc angle weights, assuming recursion (that is the lower-level
		# requested angles have been correctly implemented by recursion
		angle_weight = np.kron([[0.5, 0.5], [0.5, -0.5]],
							   np.identity(2 ** (local_num_qubits - 2)))

		# calc the combo angles
		list_of_angles = angle_weight.dot(np.array(list_of_angles)).tolist()

		# recursive step on half the angles fulfilling the above assumption
		multiplex_1 = self._multiplex(target_gate, list_of_angles[0:(list_len // 2)])
		qregs = [i for i in cq] + [q[0:-1]]
		circuit.append(multiplex_1.to_instruction(), qregs)

		# attach CNOT as follows, thereby flipping the LSB qubit
		circuit.append(CXGate(), [msb, lsb])

		# implement extra efficiency from the paper of cancelling adjacent
		# CNOTs (by leaving out last CNOT and reversing (NOT inverting) the
		# second lower-level multiplex)
		multiplex_2 = self._multiplex(target_gate, list_of_angles[(list_len // 2):])
		qregs = [i for i in cq] + [q[0:-1]]
		if list_len > 1:
			circuit.append(multiplex_2.to_instruction().mirror(), qregs)
		else:

			circuit.append(multiplex_2.to_instruction(), qregs)

		# attach a final CNOT
		circuit.append(CXGate(), [msb, lsb])

		return circuit

	def broadcast_arguments(self, qargs, cargs):
		flat_qargs = [qarg for sublist in qargs for qarg in sublist]
		if (self.num_qubits + self.num_ctrl_qubits) != len(flat_qargs):
			raise QiskitError("Initialize parameter vector has %d elements, therefore expects %s "
							  "qubits. However, %s were provided." %
							  (2**self.num_qubits, self.num_qubits, len(flat_qargs)))
		yield flat_qargs, []

def ctrl_initialize(self, statevector=None, ctrl_state=None, ctrl_qubits=None, qubits=None):
	"""Apply initialize to circuit."""
	if statevector is None:
		raise QiskitError("Null State vector")
	if ctrl_qubits is None:
		raise QiskitError("Control qubits not specified")
	if ctrl_state is None:
		ctrl_state = "1"*len(ctrl_qubits)
	if qubits is None:
		raise QiskitError("No target qubits specified")

	if not isinstance(qubits, list):
		qubits = [qubits]
	if not isinstance(ctrl_qubits,list):
		ctrl_qubits = [ctrl_qubits]
	
	#regs=[i for i in ctrl_qubits]+[i for i in qubits]
	
	return self.append(ctrl_Initialize(statevector,ctrl_state),ctrl_qubits+qubits)


QuantumCircuit.ctrl_initialize = ctrl_initialize


def QSearch():
	pass
