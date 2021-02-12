
# # Exercise 1 - Implementing epsilon-greedy policy
# 
# This lab re-uses concepts and components from Lab 02 and Lab 03.
# 
# In order to use model-free control techniques, we learn the Q-values either trough MC-learning or TD-learning.
# The environment is explored using an epsilon-greedy policy.
# 
# Implement a policy class that selects actions within the probabilities of actions given values, using the e-greedy approach:
# - epsilon is an attribute
# - add a method to decrease the value of epsilon (this method might depend on a hyperparameter)
# - add a method to sample the action based on the q-values
# - the instance of the class returns a sampled action when called.
# 
# We will use the Ice dungeon developped during the last lab.
# You can use your own or use the one provided in this lab.
# 

from dungeon.dungeon import Dungeon, Action
from sarsa import SARSA
from mc import MC_control
import numpy as np

def main():
    total_episodes = 500
    N = 20
    gamma = 0.9
    epsilon = 0.999
    decay = 0.99
    alpha = 0.5
    _dungeon = Dungeon(N)
    no_actions = 4
    no_states = N * N
    q_values = np.zeros((no_states, no_actions))
    state_position_dict = {i*N + j: (i, j) for i in range(N) for j in range(N)}
    position_state_dict = {v: k for k, v in state_position_dict.items()}
    no_of_steps = []
    epsilons = []
    for _ep in range(total_episodes):
        no_steps = 0
        _ = _dungeon.reset()
        sarsa = SARSA(_dungeon, epsilon, decay, alpha, gamma, Action, q_values)
        position_agent = _dungeon.position_agent
        s_current = position_state_dict[position_agent[0], position_agent[1]]
        position_exit = _dungeon.position_exit
        s_exit = position_state_dict[position_exit[0], position_exit[1]]
        if s_current == s_exit:
            continue
        a_next = sarsa(s_current, sarsa.q_values)     # gets the action from the policy
        while s_current != s_exit:
            no_steps += 1
            _, r_next, _ = _dungeon.step(a_next)
            position_agent = _dungeon.position_agent
            s_next = position_state_dict[position_agent[0], position_agent[1]]
            a_next_next = sarsa(s_next, sarsa.q_values)
            sarsa.update_values(s_current, a_next, r_next, s_next, a_next_next)
            s_current = s_next
            a_next = a_next_next
        q_values = sarsa.q_values
        print("For episode " + str(_ep + 1) + " the number of steps were " + str(no_steps) + " epsilon was " + str(sarsa.epsilon))
        no_of_steps.append(no_steps)
        epsilons.append(sarsa.epsilon)
        epsilon = sarsa.update_epsilon()
    print("Q values")
    print(q_values)
    sarsa.display_values(no_of_steps, epsilons)

if __name__ == "__main__":
    main()





