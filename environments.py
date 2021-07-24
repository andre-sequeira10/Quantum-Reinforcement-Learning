import numpy as np 

class openAI():

    def __init__(self, initial_state=None, action_set=None, transition_kernel=None, terminal=None):
        
        if transition_kernel is None:
            raise ValueError("State Transition matrix is missing")        
        
        if initial_state is None:
            raise ValueError("Environment initial state is missing")        
        
        if action_set is None:
            raise ValueError("Action set is missing")        
    

        self.initial_state = initial_state
        self.action_set = action_set
        self.st = transition_kernel
        self.terminal = terminal

    def reset(self,init_state=None):
        if init_state is None:
            self.current_state = self.initial_state
            return self.current_state
        else:
            self.current_state = init_state
            return self.current_state

    def step(self,action=None):
        if action is None:
            raise ValueError("Action missing")

        state_transition = self.st[self.current_state][action]
        next_states=[]
        next_states_prob=[]

        for (p,sprime,reward) in state_transition:
            next_states.append(sprime)
            next_states_prob.append(p)

        next_state = np.random.choice(next_states , p=next_states_prob)

        for (p,sprime,r) in state_transition:
            if sprime == next_state:
                reward = r
        
        if self.terminal is None:
            done = False
        else: 
            done = next_state in self.terminal

        self.current_state = next_state
        
        return self.current_state, reward, done

    def sample(self,p=None):
        if p is None:
            return np.random.choice(self.action_set)
        else:
            return np.random.choice(self.action_set, p=p)
            

