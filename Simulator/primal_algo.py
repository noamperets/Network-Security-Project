from network_simulator import Simulator, Link, Node

import math
import random
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pickle

ITER_COUNT = 50000
STEP_SIZE = 0.001

class alpha_fairness_user_state:
    def __init__(self, user: Node, utility: float = 1.0, rate: float = 1.0, derv_util:float = 1.0):
        self.user = user                # User
        self.utility = utility          # Ur
        self.rate = rate                # Xr
        self.derv_util = derv_util      # Ur'

def init_primal_algorthm(node_list: list[Node], alpha: int = 1) -> list[alpha_fairness_user_state]:
    user_rate = 1 # this is the init value for the user rate
    users_state = []
    for user_index in range(len(node_list)):
        user_utility_rate = (user_rate ** (1 - alpha)) / (1 - alpha)
        derv_user_util = 1 / (user_rate ** alpha)
        user_state = alpha_fairness_user_state(node_list[user_index], 
                                               utility=user_utility_rate,
                                               rate=user_rate,
                                               derv_util=derv_user_util)
        users_state.append(user_state)
    
    return users_state

#TODO: check cost funch
def primal_cost_func(x: float):
    return math.exp(0.2 * x)

#TODO: change name too confusing
def run_primal_iteration(user_state: alpha_fairness_user_state, user_states: list[alpha_fairness_user_state], alpha_case):
    dirv = 1 / (user_state.rate ** alpha_case)

    link_rate_sum = 0
    for neighber_node in user_state.user.linked_list:
        neighber_state = user_states[neighber_node.id]
        link_rate_sum += neighber_state.rate 
    
    cost = primal_cost_func(link_rate_sum) 

    user_state.rate += STEP_SIZE * (dirv - cost)

def run_primal_alghorithm(sim: Simulator):   # name in english comunication networks book, maybe chehck n book for another name
    alpha_case = 1.001 # TODO: change by fairness/minmax etc.

    user_rates_by_time = [[] for _ in range(sim.n)]
    user_states = init_primal_algorthm(sim.node_list, alpha=alpha_case)
    for iter_index in range(ITER_COUNT):
        user_index = random.randint(0, sim.n - 1)
        user = sim.node_list[user_index]
        user_state = user_states[user_index]
        
        run_primal_iteration(user_state, user_states, alpha_case)

        # Add the rate after calc of primal algo
        user_rates_by_time[user_index].append( user_state.rate)

    plt.title('Primal Algorithm - α =' + str(alpha_case) + ', L=' + str(len(sim.link_list) // 2))
    for index, user in enumerate(user_rates_by_time):
        plt.plot(user, label=f'User {index+1}')

    
    plt.legend()
    plt.show()

    print('Primal Algorithm - α =' + str(alpha_case) + ', L=' + str(len(sim.link_list) // 2))
    for index in range(sim.n):
        print(f'User {index+1} rate: {user_rates_by_time[index][-1]}')
    #print('User rates are')
    


def main():
    sim = Simulator()
    #sim.initialize_sim(n=5, m=20, r=6)
    sim = sim.load_sim_state(Path.cwd() / 'sim_state1.pkl')
    sim.show_network_grath()

    run_primal_alghorithm(sim)

 

if __name__ == '__main__':
    main()

