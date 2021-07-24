import numpy as np

class random_mdp:
    def __init__(self, n_states, n_actions, is_deterministic=False) -> None:
        self.n_states = n_states
        self.n_actions = n_actions
        self.is_deterministic = is_deterministic
    

    def create(self, actions_per_state="all" , reachability="all" , sensibility=0.1, r_max = 1 , reward_per_step=None ):

        if actions_per_state == "all":
            self.mdp = np.zeros((self.n_states, self.n_actions), dtype=object)
        else:
            pass
        
        if reward_per_step is not None:
            self.reward_model = np.repeat(reward_per_step , self.n_states)
        else:
            self.reward_model = np.random.randint(0,r_max+1 , self.n_states)

        if reachability == "all":
            allowed_states = np.zeros((self.n_states, self.n_actions), dtype=object)
            for s in range(self.n_states):
                for a in range(self.n_actions):
                    allowed_states[s][a] = [i for i in range(self.n_states)]
        else:
            allowed_states = reachability

        for s in range(self.n_states):
            for a in range(self.n_actions):
                state_action_pairs = []
                allowed_states_sa = allowed_states[s][a]
                p=1
                while p > sensibility:
                    p_sa = np.round(np.random.uniform(0.1,p) , 2)
                    s_prime = np.random.choice(allowed_states_sa)
                    allowed_states_sa.remove(s_prime)

                    if allowed_states_sa == []:
                        state_action_pairs.append((p,s_prime))
                        break

                    
                    if p_sa > 0.9:
                        p_sa = 1
                        state_action_pairs.append((p_sa,s_prime))
                        break
                    
                    else:
                        state_action_pairs.append((p_sa,s_prime))
                        
                    p = np.round(p - p_sa,2)
                                    
                if (p<1) and (p!=0.0) and (allowed_states_sa != []):
                    s_prime = np.random.choice(allowed_states_sa)
                    state_action_pairs.append((p,s_prime))
            
                self.mdp[s][a] = np.array(state_action_pairs, dtype=object)
        self.mdp = np.array(self.mdp)
        return self.mdp , self.reward_model

    def get_mdpR(self):
        self.tkr = np.zeros((self.n_states,self.n_actions), dtype=object)
        for s in range(self.n_states):
            for a in range(self.n_actions):
                tuples = []
                for (p,s_) in self.mdp[s][a]:
                    tuples.append((p,s_,self.reward_model[s_]))

                self.tkr[s][a] = np.array(tuples,dtype=object)

        return np.array(self.tkr, dtype=object)           
    
    def sample_trajectory(self,horizon):
        pass

'''
a = random_mdp(4,2)
mdp, reward_model = a.create(r_max=5)

print(mdp)
print(reward_model)
'''