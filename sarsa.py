

# # Exercise 2 - SARSA
# 
# Now that we have our e-greedy policy, we need to refine our q-values little by little.
# 
# Create a new class SARSA that inherits from the E-greedy policy, and implements a method to update Q-values based on TD-learning.
# 
# Additionally, create a function to display the values learned. As you have 4 q-values per state, you can decide to display the values of the state when taking the best action.
# 
# Once this is implemented, you can have your policy control the agent and observe how the q-values (when taking the greedy action) look like.
# 
# You can also print the mean reward every k episodes, to see how the agent performs.
# 
from policy import E_Greedy_Policy
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import math

class SARSA(E_Greedy_Policy):
    # alpha - step size, learning rate, how fast to modify the values
    def __init__(self, env, epsilon, decay, alpha, gamma, q_values):
        super().__init__(env, epsilon, decay)
        self.alpha = alpha
        self.gamma = gamma
        self.q_values = q_values
        
    def update_values(self, s_current, a_next, r_next, s_next, a_next_next):
        # update Q-values based on TD-learning
        a_next_id = self.action_id_dict.get(a_next)
        a_next_next_id  = self.action_id_dict.get(a_next_next)
        q_current = self.q_values[s_current, a_next_id]
        q_next = self.q_values[s_next, a_next_next_id]
        q_current += self.alpha * (r_next + (self.gamma * q_next) - q_current)
        self.q_values[s_current, a_next_id] = q_current
        
    def display_values(self, no_of_steps, epsilons):
        # display the values learned. As you have 4 q-values per state, 
        # you can decide to display the values of the state when taking the best action.
        plt.plot(no_of_steps,'r-')
        plt.show()

        #plt.imshow(self.q_values, cmap='hot', interpolation='nearest')
        ax = sns.heatmap(self.q_values, linewidth=0.1)
        ax.axis([0, len(self.q_values), 0, 3])
        plt.show()

        plt.plot(epsilons, 'b-')
        plt.show()