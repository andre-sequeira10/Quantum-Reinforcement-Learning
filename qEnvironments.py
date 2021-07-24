from qiskit import QuantumCircuit,QuantumRegister,ClassicalRegister
import matplotlib.pyplot as plt
from executeCircuit import execute_locally
import math 
import numpy as np 
from qiskit.aqua import AquaError
from qArithmetic import increment
#from qiskit.exceptions import QiskitError
#from qiskit.circuit import Instruction
from qiskit.circuit.library import RYGate,CRYGate, XGate, CXGate,RZGate,CRZGate
from qiskit.circuit.library import MCMT
#from qiskit.circuit.reset import Reset
#from qAlgorithms import Grover , QMF
#from qOracles import searchOracle
from controlled_init import ctrl_initialize
import operator 

'''
CLASS for the Quantum GridWorld Environment 
elements are passed from the classical environment:
	-> # states |S|  and # actions |A| 
	-> reward_model is passed in the transition kernel
	-> transitionKernel given as tuples [s][a] = [(p,s',r),  ... ]
'''

class quantum_sparse_sampling:
	def __init__(self,states=None,actions=None,tKernel=None,rewardModel=None, gamma=1.0,is_stochastic=True,env=None):
		
		if states is None:
			raise AquaError("Missing State space")
		if actions is None:
			raise AquaError("Missing Action space")
		if tKernel is None:
			raise AquaError("Missing Transition Kernel")

		self.states=states
		self.actions=actions
		self.is_stochastic = is_stochastic
		self.gamma = gamma

		self.sqbits = math.ceil(math.log2(self.states))
		self.aqbits = math.ceil(math.log2(self.actions))

		self.qa={}
		self.st={}
		self.st["st{0}".format(0)]=QuantumRegister(self.sqbits,name="initialState")
		self.qc=QuantumCircuit(self.st["st{0}".format(0)])

		#Transition Kernel tKernel[state][action] -> Probability | sprime | reward
		#Turn states into into binary and probability into sqrt(probability)
		if self.is_stochastic:
			self.tk = np.zeros((self.states,self.actions),dtype=object)
		else:
			self.tk = []
			for (s,a,s_) in tKernel:
				self.tk.append((bin(s)[2:].zfill(self.sqbits), bin(a)[2:].zfill(self.aqbits), bin(s_)[2:].zfill(self.sqbits)))
			if rewardModel is None:
				raise AquaError("Missing reward model")

			self.reward_model = rewardModel

		self.reward_max = -1
		self.rewards = []
		if self.is_stochastic:
			self.is_openai = len(tKernel[0][0][0]) == 1100
		else:
			self.is_openai = False
		self.env = env

		if self.is_openai:
			for s in range(self.states):
				for a in range(self.actions):
					self.tk[s][a] = [(p,sprime,reward) for (p,sprime,reward,_) in tKernel[s][a]]
					for (p,sp,r) in self.tk[s][a]:
						self.rewards.append((bin(sp)[2:].zfill(self.sqbits),r))
						if r > self.reward_max: 
							self.reward_max = r

		if self.env == "grid":
			for s in range(self.states):
				for a in range(self.actions):
					self.tk[s][a] = [(p,sprime,reward) for (p,sprime,reward) in tKernel[s][a]]
					for (p,sp,r) in self.tk[s][a]:
						self.rewards.append((bin(sp)[2:].zfill(self.sqbits),r))
						if r > self.reward_max: 
							self.reward_max = r
			self.rewards = set(self.rewards)

		elif self.is_stochastic:
			for s in range(self.states):
				for a in range(self.actions):
					self.tk[s][a] = [(math.sqrt(p),sprime,reward) for (p,sprime,reward) in tKernel[s][a]]
					for (p,sp,r) in self.tk[s][a]:
						self.rewards.append((bin(sp)[2:].zfill(self.sqbits),r))
						if r > self.reward_max: 
							self.reward_max = r
		
		self.was_executed=False
		if self.is_openai:
			self.positions = np.arange(self.states).reshape(self.sqbits,self.sqbits)
		#self.rewards = set(self.rewards)

	def stochastic_transition_oracle(self,step,R,reward_oracle="state-action"):
		#create new circuit to make call to the oracle efficient. Theres no need to run this for loops if we 
		#can just past a circuit with these logic gates combinations
		st_circuit = QuantumCircuit(name="qEnv")
		st_s = QuantumRegister(self.sqbits)
		st_sprime = QuantumRegister(self.sqbits)
		st_a = QuantumRegister(self.aqbits)
		st_r = QuantumRegister(1)
		st_circuit.add_register(st_s,st_a,st_sprime,st_r)

		if self.env == "grid":
			
			new_neighbours=[]
			for s in self.neighbours:
				for a in range(self.actions):
					#prepare complex vector from the transition probabilities 
					#If (s,a) doesnt happn then we need to make sprime == s 
					state=[complex(0.0,0.0) for i in range(2**self.sqbits)]
					for (p,sp,r) in self.tk[s][a]:
						state[sp] += complex(p,0.0)
						if sp not in self.states_visited:
							new_neighbours.append(sp)
							self.states_visited.append(sp)

					state = [complex(math.sqrt(p.real),0.0) for p in state]
					sbin=bin(s)[2:].zfill(self.sqbits)
					abin=bin(a)[2:].zfill(self.aqbits)
					ctrls = [i for i in st_s]+[i for i in st_a]

					#ctr_initialization to make the transition
					st_circuit.ctrl_initialize(statevector=state,ctrl_state=abin+sbin,ctrl_qubits=ctrls,qubits=st_sprime)
					if reward_oracle == "state-action":
						regs = [i for i in st_s]+[i for i in st_a]+[i for i in st_sprime]+[i for i in st_r]
					
					else:
						regs = [i for i in st_sprime]+[i for i in st_r]

					for (p,sp,r) in self.tk[s][a]:
						
						if r>0:
							spbin = bin(sp)[2:].zfill(self.sqbits)
							st_circuit.append(self.stochastic_reward_oracle(sbin,abin,spbin,r,step-1,R,reward_oracle=reward_oracle),regs)
						'''
						spbin = bin(sp)[2:].zfill(self.sqbits)
						st_circuit.append(self.stochastic_reward_oracle(sbin,abin,spbin,r,step-1,R),regs)
						'''
			self.neighbours = new_neighbours
		else:	
			if self.is_openai:
				possible_transitions = list(np.fliplr(self.positions).diagonal(self.sqbits-step))

				for s in possible_transitions:
					for a in range(self.actions):	
						#prepare complex vector from the transition probabilities 
						#If (s,a) doesnt happn then we need to make sprime == s 
						state=[complex(0.0,0.0) for i in range(2**self.sqbits)]
						for (p,sp,r) in self.tk[s][a]:
							state[sp] += complex(p,0.0)

						state = [complex(math.sqrt(p.real),0.0) for p in state]
						#We need to turn s,a into binary repr to make the ctrl_state in the initialization
						sbin=bin(s)[2:].zfill(self.sqbits)
						abin=bin(a)[2:].zfill(self.aqbits)
						ctrls = [i for i in st_s]+[i for i in st_a]

						#ctr_initialization to make the transition
						st_circuit.ctrl_initialize(statevector=state,ctrl_state=abin+sbin,ctrl_qubits=ctrls,qubits=st_sprime)
						
						if reward_oracle == "state-action":
							regs = [i for i in st_s]+[i for i in st_a]+[i for i in st_sprime]+[i for i in st_r]
						
						else:
							regs = [i for i in st_sprime]+[i for i in st_r]

						for (p,sp,r) in self.tk[s][a]:
							spbin = bin(sp)[2:].zfill(self.sqbits)
							st_circuit.append(self.stochastic_reward_oracle(sbin,abin,spbin,r,step-1,R,reward_oracle=reward_oracle),regs)
			else:		
				for s in range(self.states):
					for a in range(self.actions):
						#prepare complex vector from the transition probabilities 
						#If (s,a) doesnt happn then we need to make sprime == s 
						state=[complex(0.0,0.0) for i in range(2**self.sqbits)]
						for (p,sp,r) in self.tk[s][a]:
							state[sp] += complex(p,0.0)

						sbin=bin(s)[2:].zfill(self.sqbits)
						abin=bin(a)[2:].zfill(self.aqbits)
						ctrls = [i for i in st_s]+[i for i in st_a]

						#ctr_initialization to make the transition
						st_circuit.ctrl_initialize(statevector=state,ctrl_state=abin+sbin,ctrl_qubits=ctrls,qubits=st_sprime)
						
						if reward_oracle == "state-action":
							regs = [i for i in st_s]+[i for i in st_a]+[i for i in st_sprime]+[i for i in st_r]
					
						else:
							regs = [i for i in st_sprime]+[i for i in st_r]

						for (p,sp,r) in self.tk[s][a]:
							spbin = bin(sp)[2:].zfill(self.sqbits)
							st_circuit.append(self.stochastic_reward_oracle(sbin,abin,spbin,r,step-1,R,reward_oracle=reward_oracle),regs)

		return st_circuit

	def stochastic_reward_oracle(self,s,a,sp,r,step,R,reward_oracle="state-action"):

		#APPEND REWARD AS A ROTATION IN THE Y AXIS TO TURN INTO PROBABILITY
		
		if reward_oracle == "state-action":
			qr = QuantumRegister(1)
			qrs = QuantumRegister(self.sqbits)
			qra = QuantumRegister(self.aqbits)
			qrsp = QuantumRegister(self.sqbits)

			qc = QuantumCircuit(qrs,qra,qrsp,qr,name="Reward Oracle t={}".format(step))
			
			#reward function: Ry gates rotates by theta/2
			gamma_norm = (self.gamma - 1)/((self.gamma**self.horizon) - 1)
			reward_step = np.pi * gamma_norm * (self.gamma**step) * (r/self.reward_max)
			#reward_step = (pi/10) * ((self.gamma**step * r)/self.reward_max)
			for i,j in zip(range(len(s)),reversed(range(len(s)))):
				if s[i] == '0':
					qc.x(qrs[j])
			for i,j in zip(range(len(a)),reversed(range(len(a)))):
				if a[i] == '0':
					qc.x(qra[j])
			for i,j in zip(range(len(sp)),reversed(range(len(sp)))):
				if sp[i] == '0':
					qc.x(qrsp[j])

			#cry_gate = RYGate(reward_step).control(num_ctrl_qubits=2*self.sqbits+self.aqbits,ctrl_state=sp+a+s) 
			cry_gate = RYGate(reward_step).control(num_ctrl_qubits=2*self.sqbits+self.aqbits) 
			qc.append(cry_gate,[i for i in qrs]+[i for i in qra]+[i for i in qrsp]+[i for i in qr])
			for i,j in zip(range(len(s)),reversed(range(len(s)))):
				if s[i] == '0':
					qc.x(qrs[j])
			for i,j in zip(range(len(a)),reversed(range(len(a)))):
				if a[i] == '0':
					qc.x(qra[j])
			for i,j in zip(range(len(sp)),reversed(range(len(sp)))):
				if sp[i] == '0':
					qc.x(qrsp[j])

		else:
			
			qr = QuantumRegister(1)
			qrsp = QuantumRegister(self.sqbits)

			qc = QuantumCircuit(qrsp,qr,name="Reward Oracle t={}".format(step))
			
			#reward function: Ry gates rotates by theta/2
			gamma_norm = (self.gamma - 1)/((self.gamma**self.horizon) - 1)
			reward_step = np.pi * gamma_norm * (self.gamma**step) * (r/self.reward_max)
			#reward_step = (pi/10) * ((self.gamma**step * r)/self.reward_max)

			for i,j in zip(range(len(sp)),reversed(range(len(sp)))):
				if sp[i] == '0':
					qc.x(qrsp[j])

			#cry_gate = RYGate(reward_step).control(num_ctrl_qubits=2*self.sqbits+self.aqbits,ctrl_state=sp+a+s) 
			cry_gate = RYGate(reward_step).control(num_ctrl_qubits=self.sqbits) 

			qc.append(cry_gate,[i for i in qrsp]+[i for i in qr])
			
			for i,j in zip(range(len(sp)),reversed(range(len(sp)))):
				if sp[i] == '0':
					qc.x(qrsp[j])
					
		return qc


	def stateActionTransitionOracle(self,currentState,newState,step):
		self.qc.barrier()
		for t in self.tk:
			i=self.sqbits-1
			for s in t[0]:
				if not int(s):
					self.qc.x(currentState[i])
				i-=1

			i=self.aqbits-1
			#state=int(t[0],base=2)
			for a in t[1]:
				if not int(a):
					self.qc.x(self.qa["action{0}".format(step)][i])
				i-=1

			i=self.sqbits-1
			for ss in t[2]:
				if int(ss):
					controls=[currentState[i] for i in range(len(currentState))]
					controls=controls+[self.qa["action{0}".format(step)][i] for i in range(self.aqbits)]
					self.qc.mct(controls,newState[i],None,mode="noancilla")
				i-=1

			i=self.aqbits-1
			#state=int(t[0],base=2)
			for a in t[1]:
				if not int(a):
					self.qc.x(self.qa["action{0}".format(step)][i])
				i-=1

			i=self.sqbits-1
			for s in t[0]:
				if not int(s):
					self.qc.x(currentState[i])
				i-=1

			
		self.qc.barrier()				
	
	def rewardOracle(self,state,rewardReg,qinc):
		self.qc.barrier()
		for i in range(len(self.reward_model)):
			if self.reward_model[i]:
				b = np.binary_repr(i,width=self.sqbits)
				j=self.sqbits-1
				for q in b:
					if not int(q):
						self.qc.x(state[j])
					j-=1
				###Decompose multi control increment with an ancilla first###
				self.qc.mct(state,qinc,None,mode='advanced')
				#self.qc.reset(qinc)
				j=self.sqbits-1
				for q in b:
					if not int(q):
						self.qc.x(state[j])
					j-=1
		increment(self.qc,rewardReg,control=qinc)
		self.qc.barrier()

		

	def step(self,initState=0, horizon="infinite", reward_oracle="state-action"):

		#if horizon is infinite then the system evolves until the end of the computer resources 
		#self.horizon = self.nstates/2 if horizon=="infinite" else horizon
		self.horizon=horizon
		if initState != 0:    
			s = bin(initState)[2:].zfill(self.sqbits)
			i=self.sqbits-1
			for q in s:
				if int(q):
					self.qc.x(self.st["st{0}".format(0)][i])
				i-=1
		
		#reward_model register size will depend on the horizon    
		#given that we're working with binary rewards then for a horizon t,
		#the maximum reward_model collected is t, so the register has log2(t) qubits 
		#create dictionary to save state transition registers for easy acess in the future
		
		for i in range(self.horizon):
			self.qa["action{0}".format(i)] = QuantumRegister(self.aqbits,"action{0}".format(i))
			self.qc.add_register(self.qa["action{0}".format(i)])
		
		#Apply the superposition policy - hadamard on action register 
		#Same as applying an hadamard at each state transition
		for i in range(self.horizon):
			self.qc.h(self.qa["action{0}".format(i)])

		if self.is_stochastic:
			
			#Just one qubit for reward because reward will be amplitude encoded 
			self.rqbits = 1
			self.qr = QuantumRegister(self.rqbits,"reward")
			self.qc.add_register(self.qr)
			
			R = 0
			for i in range(horizon):
				R+=(self.gamma)**i
		
			#Apply transition and reward_model oracles
			self.neighbours = [initState]
			self.states_visited = [initState]
			oracles={}
			for i in range(1,self.horizon+1):
				oracles[i] = self.stochastic_transition_oracle(i,R,reward_oracle=reward_oracle)
				self.st["st{0}".format(i)] = QuantumRegister(self.sqbits,"stateTransition{0}".format(i))
				self.qc.add_register(self.st["st{0}".format(i)])
				
				regs = [i for i in self.st["st{0}".format(i-1)]] + [i for i in self.qa["action{0}".format(i-1)]] + [i for i in self.st["st{0}".format(i)]]+[i for i in self.qr]
				
				if self.is_openai or self.env=="grid":
					for j in range(1,i+1):
						self.qc.append(oracles[j],regs)
				else:
					self.qc.append(oracles[i],regs)
				#regs_reward = [i for i in self.st["st{0}".format(i)]]+[i for i in self.qr]
				#self.qc.append(self.stochastic_reward_oracle(i-1,self.horizon), regs_reward)

				self.qc.barrier()
			
		else:
			self.rqbits = math.ceil(math.log2(self.horizon))
				
			self.qr = QuantumRegister(self.rqbits,"reward")
			
			self.qinc={}
			self.clqinc=ClassicalRegister(1,"controlInc")
			self.qc.add_register(self.qr)
			#create action register - Depend on the horizon:
			#Horizon T - we have Tlog(aqbits) qubits to represent the action register

			for i in range(1,self.horizon+1):
				self.st["st{0}".format(i)] = QuantumRegister(self.sqbits,"stateTransition{0}".format(i))
				self.qc.add_register(self.st["st{0}".format(i)])
				self.stateActionTransitionOracle(self.st["st{0}".format(i-1)],self.st["st{0}".format(i)],i-1)
				self.qinc["qinc{0}".format(i-1)] = QuantumRegister(1,"qinc{0}".format(i-1))
				self.qc.add_register(self.qinc["qinc{0}".format(i-1)])
				self.rewardOracle(self.st["st{0}".format(i)],self.qr,self.qinc["qinc{0}".format(i-1)])
				self.qc.barrier()

	def draw_circuit(self,decompose_circuit=False,mode="text",layers=1,inverse=False):
		if inverse:
			if decompose_circuit:
				if layers==1:
					return self.qc_inverse.decompose().draw(output=mode)
				elif layers==2:
					return self.qc_inverse.decompose().decompose().draw(output=mode)
				elif layers==3:
					return self.qc_inverse.decompose().decompose().decompose().draw(output=mode)
				else:
					return self.qc_inverse.decompose().decompose().decompose().decompose().draw(output=mode)
			else:
				return self.qc_inverse.draw(output=mode)
		else:
			if decompose_circuit:
				if layers==1:
					return self.qc.decompose().draw(output=mode)
				elif layers==2:
					return self.qc.decompose().decompose().draw(output=mode)
				elif layers==3:
					return self.qc.decompose().decompose().decompose().draw(output=mode)
				else:
					return self.qc.decompose().decompose().decompose().decompose().draw(output=mode)

			else:
				return self.qc.draw(output=mode)

	#AMPLITUDE AMPLIFCATION BY HOYER,P BRASSARD ET.AL https://arxiv.org/pdf/quant-ph/0005055.pdf
	def Grover_Iterate(self,circuit,ctrls,regs,ancilla_reg):
					
					#invert phase o \1> terms
					circuit.z(self.qr)
					circuit.barrier()
					
					circuit.append(self.qc_pre_amplification_inverse,regs)

					for j in range(self.horizon+1):
						circuit.x(self.st["st{0}".format(j)])
					circuit.x(self.qr)
					for j in range(self.horizon):
						circuit.x(self.qa["action{0}".format(j)])
					
					circuit.barrier()
					
					circuit.h(self.st["st{0}".format(self.horizon)][-1])
					
					circuit.barrier()
					
					#circuit.append(CXGate().control(num_ctrl_qubits=len(ctrls)),ctrls+[self.st["st{0}".format(self.horizon)][-1]])
					circuit.mcx([q for q in ctrls] , len(ctrls), ancilla_qubits=[q for q in ancilla_reg], mode="v-chain")
					#circuit.MCMT(XGate(),ctrls,self.st["st{0}".format(self.horizon)][-1])#,None,mode="noancilla")
					circuit.barrier()

					circuit.h(self.st["st{0}".format(self.horizon)][-1])
					'''
					for j in range(1,self.horizon+1):
						circuit.h(self.st["st{0}".format(j)])
					
					circuit.h(self.qr)

					for j in range(self.horizon):
						circuit.h(self.qa["action{0}".format(j)])
					'''

					circuit.barrier()

					for j in range(self.horizon):
						circuit.x(self.qa["action{0}".format(j)])
				
					circuit.x(self.qr)
				
					for j in range(self.horizon+1):
						circuit.x(self.st["st{0}".format(j)])
					circuit.barrier()

					circuit.append(self.qc_pre_amplification,regs)

	def QSearch(self,shots=1,backend="qasm_simulator"):
			it, lambd = 1, 6/5
			max_it = ((2**(self.aqbits)*self.horizon)+(2**(self.sqbits)*(self.horizon+1)))/2
			
			qcGrover = {}
			c=0
			measured=False
			best_measure_counts = 0
			while it < max_it:
				qcGrover = QuantumCircuit()
				qcGrover+=self.qc
				ancilla_qubits = len(self.ctrls) - 2
				ancilla_mct = QuantumRegister(ancilla_qubits)
				qcGrover.add_register(ancilla_mct)

				
				g_it = np.random.randint(it) + 1
				print("g_it - ",g_it)
				#print("INSIDE QSEARCH - {}".format(g_it))

				for i in range(g_it):
					self.Grover_Iterate(qcGrover,self.ctrls,self.regs, ancilla_mct)
				
				qcGrover.measure_all()
				r,rc = execute_locally(qcGrover,nshots=shots,backend=backend)
				measure = max(rc.items(), key=operator.itemgetter(1))[0]
				print(measure)
				measure_c = max(rc.items(), key=operator.itemgetter(1))[1]
				print(measure_c)
				measure_counts = rc[measure]
				print(measure_counts)
				#print(best_measure)
				reward_measure = int(measure[(self.sqbits*self.horizon) + ancilla_qubits],2)
				print("reward - {}".format(reward_measure))
				if reward_measure:
					c+=g_it
					if measure_counts > best_measure_counts:
						best_measure_counts = measure_counts

						best_measure = measure
						measured=True
					elif measured:
						break

					#break
				else:
					#it = min(lambd*it,max_it)
					#it += 1
					c+=g_it
					if measured:
						break
				it = min(lambd*it,max_it)
				
			if it >= max_it:
				raise ValueError("Search not Worked! Aborted!")
			
			return rc,best_measure, c

	def solve(self,backend="qasm_simulator",shots=1,iterations=None,draw_circuit=False,amplify=False,measure_all=True,outpath=None,filename=None,plot_show=False,algorithm=None,n_samples=None):
		
		#reward_model that will serve as index for the quantum maximum finding (qmf) routine
		#qmf - apply ST and reward_model Oracles again and circuit is the oracle for qmf 
		#we then mark all the elements in the reward_model register that are greater than the test_reward
		#measurement of qmf will result in a new_reward > test_reward -> test_reward = new_reward
		#we do this procedure for O(sqrt |A|.T)
		#At each measurement given by the qmf we need to construct the oracle responsible for marking 
		#the elements in the superposition greater than test_reward
		
		# round up the number of iterations
		# qmf gives the correct answer with prob 1 - (1/2)^k , with k being the number of times 
		# the algorithm is repeated 

		#self.qc.measure_all()
		

		if self.is_stochastic:
			#self.qc.measure_all()
			'''
			result,result_counts = execute_locally(self.qc,nshots=shots,show=True)
			return result, result_counts
			'''
			self.regs = [i for i in self.st["st{0}".format(0)]]
			self.ctrls = [i for i in self.st["st{0}".format(0)]]
			for j in range(self.horizon):
				self.regs += [i for i in self.qa["action{0}".format(j)]]
				self.ctrls += [i for i in self.qa["action{0}".format(j)]]

			self.regs+=[i for i in self.qr]
			self.ctrls+=[i for i in self.qr]

			for j in range(1,self.horizon+1):
				self.regs += [i for i in self.st["st{0}".format(j)]]

			for j in range(1,self.horizon):
				self.ctrls += [i for i in self.st["st{0}".format(j)]]
			
			self.ctrls += [i for i in self.st["st{0}".format(self.horizon)][:-1]]
			self.ancilla_qubits = len(self.ctrls) - 2
			if algorithm is None:
				if not self.was_executed:
					
									
					self.qc_pre_amplification = self.qc.copy()
					self.qc_pre_amplification_inverse=self.qc.reverse_ops().copy()
					if amplify:
						if not iterations:
							for i in range(int(round(np.sqrt(2*self.actions**self.horizon)))):
								self.Grover_Iterate(self.ctrls,self.regs)
						else:
							for i in range(iterations):
								self.Grover_Iterate(self.ctrls,self.regs)

					if measure_all:		
						self.qc.measure_all()
					else:
						self.aclassical = ClassicalRegister(self.aqbits)
						self.rclassical = ClassicalRegister(1)
						self.qc.add_register(self.aclassical,self.rclassical)
						self.qc.measure(self.qa["action{0}".format(0)],self.aclassical)
						self.qc.measure(self.qr,self.rclassical)
					self.was_executed = True

						
				else:	
					if amplify:
						self.qc.remove_final_measurements()
						if not iterations:
							for i in range((int(round(np.sqrt(2*self.actions**self.horizon))))):
								self.Grover_Iterate(self.ctrls,self.regs)
						else:
							for i in range(iterations):
								self.Grover_Iterate(self.ctrls,self.regs)
						
						if measure_all:
							self.qc.measure_all()
						else:
							self.aclassical = ClassicalRegister(self.aqbits)
							self.rclassical = ClassicalRegister(1)
							self.qc.add_register(self.aclassical,self.rclassical)
							self.qc.measure(self.qa["action{0}".format(0)],self.aclassical)
							self.qc.measure(self.qr,self.rclassical)

				
				import operator 
				r,rc = execute_locally(self.qc,nshots=shots,backend=backend)
				best_measure = max(rc.items(), key=operator.itemgetter(1))[0]
				
				if measure_all:
					best_action = best_measure[(self.sqbits*self.horizon)+1+(self.aqbits*(self.horizon-1)):-self.sqbits]
				else:
					best_action = best_measure[1:]
				'''
				if len(rc) > 15:
					plt.figure(figsize=(len(rc) * 0.45,12))
				
				plt.gca().set_facecolor("gray")
				plt.gcf().set_facecolor("gray")
				#plt.bar(range(len(rc)), list(rc.values()), color="darkorange" ,align='center')
				'''		
				action_reward = []
				new_dict={}
				if measure_all:
					for p in list(rc.keys()):
						reward = p[(self.sqbits*self.horizon):(self.sqbits*self.horizon)+1]
						action = p[(self.sqbits*self.horizon)+1+(self.aqbits*(self.horizon-1)):-self.sqbits]
						action_reward.append(str(int(action,2))+" "+str(int(reward)))
						### new code added from here ###
						new_key = str(int(action,2))+" "+str(int(reward))
						if new_key in new_dict:
							new_dict[new_key] += rc.get(p)
						else:
							new_dict[new_key] = rc.get(p)
						### until here ###
				else:
					for p in list(rc.keys()):
						reward = p[-1]
						action = p[1:]
						action_reward.append(str(int(action,2))+" "+str(int(reward)))

				'''
				plt.xticks(range(len(rc)), action_reward)
				plt.xlabel(r"$\mathbf{|action\rangle |reward\rangle}$",fontsize=14)
				plt.ylabel(r'$\mathbf{\mathbb{E}}$'+" cumulative reward",fontweight="bold",fontsize=14)

				plt.show()
				'''

				### replace rc by new_dict
				#if len(rc) > 15:
				if len(new_dict) > 15:
					#fig, ax = plt.subplots(figsize=(len(rc) * 0.45,12))
					fig, ax = plt.subplots(figsize=(len(new_dict) * 0.45,12))
				else:
					fig, ax = plt.subplots()

				# Create bar plot
				#bar1 = ax.bar(range(len(rc)), list(rc.values()),color="darkorange")
				bar1 = ax.bar(range(len(new_dict)), list(new_dict.values()),color="darkorange")
				ax.set_facecolor("lightgray")
				#ax.set_xticks(range(len(rc)))
				ax.set_xticks(range(len(new_dict)))
				#ax.set_xticklabels(action_reward)
				ax.set_xticklabels(list(new_dict.keys()))
				ax.set_title('# Samples = '+str(shots),fontweight="bold",fontsize=14)
				ax.set_xlabel(r"$\mathbf{|action\rangle |reward\rangle}$",fontsize=14)
				ax.set_ylabel("Counts",fontweight="bold",fontsize=14)
				#ax.axhline(np.mean(counts),linestyle="dashed",color="r",label="Average")
				#ax.legend()
				#plt.savefig("random_dist.png")
				def autolabel(rects):
					"""Attach a text label above each bar in *rects*, displaying its height."""
					for rect in rects:
						height = rect.get_height()
						ax.annotate("{}".format(height),
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

				
				return rc,int(best_action,2)
			
			else:
				if self.was_executed:
					self.qc.remove_final_measurements()
				else:
					self.qc_pre_amplification = self.qc.copy()
					self.qc_pre_amplification_inverse=self.qc.reverse_ops().copy()
				
				if shots == 1:
					samples_dict={}
					iterations_per_sample = []
					for s in range(n_samples):
						rc,best_measure, it_sample = self.QSearch(shots=shots)
						print(it_sample)
						iterations_per_sample.append(it_sample)
						if best_measure in samples_dict:
							samples_dict[best_measure] += 1
						else:
							samples_dict.update(rc)
				else:
					rc,best_measure,iterations_per_sample = self.QSearch(shots=shots)

				'''
				if measure_all:
					best_action = best_measure[(self.sqbits*self.horizon)+1+(self.aqbits*(self.horizon-1)):-self.sqbits]
				else:
					best_action = best_measure[1:]
				'''
				'''
				if len(rc) > 15:
					plt.figure(figsize=(len(rc) * 0.45,12))
				
				plt.gca().set_facecolor("gray")
				plt.gcf().set_facecolor("gray")
				#plt.bar(range(len(rc)), list(rc.values()), color="darkorange" ,align='center')
				'''		
				#action_reward = []
				action_s = []
				new_dict={}
				if measure_all:
					if shots == 1:
					#for p in list(rc.keys()):
						for p in list(samples_dict.keys()):
							#reward = p[(self.sqbits*self.horizon):(self.sqbits*self.horizon)+1]
							action = p[self.ancilla_qubits + (self.sqbits*self.horizon)+1+(self.aqbits*(self.horizon-1)):-self.sqbits]
							#action_reward.append(str(int(action,2))+" "+str(int(reward)))
							action_s.append(str(int(action,2)))
							### new code added from here ###
							new_key = str(int(action,2))
							if new_key in new_dict:
								new_dict[new_key] += samples_dict.get(p)
							else:
								new_dict[new_key] = samples_dict.get(p)
							### until here ###
					else:
						for p in list(rc.keys()):
							#reward = p[(self.sqbits*self.horizon):(self.sqbits*self.horizon)+1]
							action = p[self.ancilla_qubits + (self.sqbits*self.horizon)+1+(self.aqbits*(self.horizon-1)):-self.sqbits]
							#action_reward.append(str(int(action,2))+" "+str(int(reward)))
							action_s.append(str(int(action,2)))
							### new code added from here ###
							new_key = str(int(action,2))
							if new_key in new_dict:
								new_dict[new_key] += rc.get(p)
							else:
								new_dict[new_key] = rc.get(p)
							### until here ###
				else:
					for p in list(rc.keys()):
						reward = p[-1]
						action = p[1:]
						action_s.append(str(int(action,2))+" "+str(int(reward)))

				'''
				plt.xticks(range(len(rc)), action_reward)
				plt.xlabel(r"$\mathbf{|action\rangle |reward\rangle}$",fontsize=14)
				plt.ylabel(r'$\mathbf{\mathbb{E}}$'+" cumulative reward",fontweight="bold",fontsize=14)

				plt.show()
				'''
				### replace rc by new_dict
				#if len(rc) > 15:
				if len(new_dict) > 15:
					#fig, ax = plt.subplots(figsize=(len(rc) * 0.45,12))
					fig, ax = plt.subplots(figsize=(len(new_dict) * 0.45,12))
				else:
					fig, ax = plt.subplots()

				# Create bar plot
				#bar1 = ax.bar(range(len(rc)), list(rc.values()),color="darkorange")
				bar1 = ax.bar(range(len(new_dict)), list(new_dict.values()),color="darkorange")
				ax.set_facecolor("lightgray")
				#ax.set_xticks(range(len(rc)))
				ax.set_xticks(range(len(new_dict)))
				#ax.set_xticklabels(action_reward)
				ax.set_xticklabels(list(new_dict.keys()))
				ax.set_title('# Samples = '+str(shots),fontweight="bold",fontsize=14)
				#ax.set_xlabel(r"$\mathbf{|action\rangle |reward\rangle}$",fontsize=14)
				ax.set_xlabel(r"$\mathbf{|action\rangle}$",fontsize=14)
				ax.set_ylabel("Samples",fontweight="bold",fontsize=14)
				
				def autolabel(rects):
					"""Attach a text label above each bar in *rects*, displaying its height."""
					for rect in rects:
						height = rect.get_height()
						ax.annotate("{}".format(height),
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
				
				import operator
				best_action = max(new_dict.items(), key=operator.itemgetter(1))[0]

				if plot_show:			
					plt.show()

				return new_dict,int(best_action),iterations_per_sample

		else:	
			self.draw_circuit=draw_circuit
			qmf_size = self.actions*self.horizon
			# We need to pass QMF the extra registers for measuring them
			# We want to measure the action register
			actions = [self.qa["action{0}".format(i)] for i in range(self.horizon)]
			#states = [self.st["st{0}".format(i)] for i in range(self.horizon+1)]
			extra_measure = actions 
			qmf = QMF(circuit=self.qc,search_register=self.qr,size=qmf_size,extra_registers=extra_measure,draw_circuit=self.draw_circuit,max_observation=self.horizon)
			
			maximum, top_measurement = qmf.run(backend=backend,shots=shots)
			return maximum, top_measurement        
