
import random
import numpy as np

class E_Greedy_Policy():
    
    def __init__(self, env, epsilon, decay):
        self.epsilon = epsilon
        self.decay = decay
        self.action_dict = {0:"up", 1: "down", 2:"left", 3:"right"}
        self.action_id_dict = {v: k for k, v in self.action_dict.items()}
        self.coord_to_state = {}

        
    def __call__(self, state, q_values):
        # Sample an action from the policy, given a state
        is_greedy = random.random() > self.epsilon
        if is_greedy:
            index_action = np.argmax(q_values[state])
        else:
            index_action = random.randint(0, 3)
        
        action = self.action_dict[index_action]
        return action
        
    def update_epsilon(self):
        # call for each episode
        self.epsilon *= self.decay
        return self.epsilon
        