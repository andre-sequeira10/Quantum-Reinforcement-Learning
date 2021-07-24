# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
Adapted by André Sequeira from

The Custom Circuit-based Quantum Oracle.

to

qmfOracle which marks elements of a superposition state who are greater than
a predefined value k 

&

searchOracle which marks the elements of a superposition state that are equal to a certain index 
"""

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.aqua import AquaError
from qiskit.aqua.components.oracles import Oracle
from mag_compare_gates import mag_gt_k
import numpy as np 


class qmfOracle(Oracle):
	"""
	From the helper class for creating oracles from user-supplied quantum circuits
	"""

	def __init__(self,oracle_circuit=None,variable_register=None, output_register=None, index=None , ancillary_register=None,ancillary_mag=None,extra_regs=None):
		"""
		Constructor.

		Args:
			variable_register (QuantumRegister): The register holding variable qubit(s) for the oracle function -> Reward Register 
			output_register (QuantumRegister): The register holding output qubit(s) for the oracle function -> marked elements
			circuit (QuantumCircuit): The quantum circuit corresponding to the intended oracle function
			typically the circuit already been built
		"""

		super().__init__()
		if variable_register is None:
			raise AquaError('Missing QuantumRegister for variables.')

		if index is None:
			raise AquaError('Missing default maximum index.')
				
		self._variable_register = variable_register
		self._output_register = output_register if output_register is not None else QuantumRegister(1,name="out")
		self._ancillary_register = ancillary_register 
		self._index = index
		self._ancillary_mag=ancillary_mag if ancillary_mag is not None else QuantumRegister(len(self._variable_register),name="ancillary_mag")
		self._extra_regs=extra_regs

		if oracle_circuit is None:
			self._circuit = self.construct_circuit()
		else:
			self._circuit = circuit
			self._circuit.add_register(self._output_register,self._ancillary_mag)
			if ancillary_register is not None:
				self._circuit.add_register(self._ancillary_register)


	@property
	def variable_register(self):
		return self._variable_register

	@property
	def output_register(self):
		return self._output_register

	@property
	def ancillary_register(self):
		return self._ancillary_register

	@property
	def index(self):
		return self._index

	@property
	def circuit(self):
		return self._circuit

	@property
	def extra_registers(self):
		return self._extra_regs
		
	@property
	def get_registers(self):
		regs = [self._variable_register]
		if self._extra_regs is not None:
			for r in self._extra_regs:
				regs.append(r)

		return regs

	def construct_circuit(self):
		"""Construct the oracle circuit.

		Returns:
			A quantum circuit for the oracle.
		"""
		qcircuit = QuantumCircuit(self._variable_register,self._output_register,self._ancillary_mag)
		if self._ancillary_register is not None:
			qcircuit.add_register(self._ancillary_register)

		###circuit,k,output,ancilla###
		mag_gt_k(qcircuit,self._variable_register,self._index,self._output_register,self._ancillary_mag)

		return qcircuit

	def evaluate_classically(self, measurement):
		meas = measurement.split(" ")
		registers={}
		j=len(meas[:-1])-1
		if self._extra_regs is not None:
			for i in range(len(self._extra_regs)):
				registers[self._extra_regs[i].name] = meas[j]
				j-=1

		val = int(meas[-1],2)
		registers[self._variable_register.name] = val  
		return val>self._index , registers

class searchOracle(Oracle):
	def __init__(self,oracle_circuit=None,variable_register=None, index=None , ancillary_register=None,extra_regs=None,output_register=None,method="output"):
		
		"""
		Constructor.

		Args:
			variable_register (QuantumRegister): The register holding variable qubit(s) for the oracle function -> Reward Register 
			output_register (QuantumRegister): The register holding output qubit(s) for the oracle function -> marked elements
			circuit (QuantumCircuit): The quantum circuit corresponding to the intended oracle function
			typically the circuit already been built
		"""

		super().__init__()
		if variable_register is None:
			raise AquaError('Missing QuantumRegister for variables.')
		if index is None:
			raise AquaError('Missing default index.')
		
		self._variable_register = QuantumRegister(len(variable_register),"var_reg")
		self._ancillary_register = ancillary_register
		self._index = index
		self._method = method
		if self._method == "output":
			self._output_register = QuantumRegister(1,name="out")
			self._circuit = self.construct_circuit()
		
		elif output_register is None:
			raise AquaError(' Missing output ')
		else:
			self._output_register = output_register
			self._circuit = self.construct_circuit()

		self._extra_regs = extra_regs
	@property
	def variable_register(self):
		return self._variable_register

	@property
	def extra_registers(self):
		return self._extra_regs

	@property
	def state_prep_circuit(self):
		return self._state_prep_circuit

	@property
	def output_register(self):
		return self._output_register

	@property
	def ancillary_register(self):
		return self._ancillary_register

	@property
	def index(self):
		return self._index

	@property
	def circuit(self):
		return self._circuit

	@property
	def get_registers(self):
		if self._method == "no_output":
			regs = [self._output_register]+[self._variable_register]
		else:
			regs = [self._variable_register]
		
		if self._extra_regs is not None:
			for r in self._extra_regs:
				regs.append(r)

		return regs

	def new_index(self,index):
		self._index = index 

	def construct_circuit(self):
		"""Construct the oracle circuit.

		Returns:
			A quantum circuit for the oracle.
		"""
		if self._method == "output":
			n = len(self._variable_register)
			indexB = np.binary_repr(self._index,width=n)

			self._circuit = QuantumCircuit(self._variable_register,self._output_register)
			if self._ancillary_register is not None:
				self._circuit.add_register(self._ancillary_register)

			j=n-1
			for q in indexB:
				if not int(q):
					self._circuit.x(self._variable_register[j])
				j-=1

			self._circuit.mct (self._variable_register, self._output_register[0],None,mode='advanced')

			j=n-1
			for q in indexB:
				if not int(q):
					self._circuit.x(self._variable_register[j])
				j-=1
		else:
			self._circuit = QuantumCircuit(self._variable_register,self._output_register)
			self._circuit.z(self._output_register)

		return self._circuit

	def evaluate_classically(self, measurement):
		meas = measurement.split(" ")
		registers={}
		if self._extra_regs is not None:
			for i in range(len(self._extra_regs)):
				registers[self._extra_regs[i].name] = meas[i]

		val = int(meas[-1],2)
		if self._method == "no_output":
			registers[self._output_register.name] = val  
			registers[self._variable_register.name] = meas[-2]  
		else:
			registers[self._variable_register.name] = val  
		return val==self._index , registers
