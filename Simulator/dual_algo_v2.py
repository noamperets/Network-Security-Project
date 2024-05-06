from network_simulator import Simulator, Link, Node

import math
import random
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pickle

ITER_COUNT = 100000
STEP_SIZE = 0.001

"""
Init matrix of Link
Steps:
1. Init User rates to 1's vector
2. for each link assigned to user: set the user rates

"""

def create_usage_matrix(links_count, user_count, indexes):
    matrix = np.zeros((user_count, links_count), dtype=int)
    #print(f'In usage matrix {indexes=} ')
    for i,j in indexes:
        matrix[i, j] = 1

    return matrix

def run_dual_alghorithm(sim):
    alpha = 1.001  # Alpha fairness
    N_dual = 50000  # Number of iterations for Dual Algorithm

    # Initialization
    #U = len(sim.node_list)
    #L = len(sim.link_list) // 2

    U = 6
    L = 5

    C_l = np.ones(L)  # Capacitance of links
    X_r = np.ones(U)  # Initial X_r rates per users
    Links = np.zeros((L, U))  # Link rates per user

    Users = np.zeros((U, L))  # Link rates per user

    X_r_dual = np.ones(U)  # Initial X_r rates per users for Dual Algorithm
    H_r_dual = np.full(U, 0.001)  # Step size
    U_r_dual = (X_r_dual ** (1 - alpha)) / (1 - alpha)  # alpha fairness
    derivative_U_r_dual = 1 / (X_r_dual ** alpha)
    Lagranz = np.zeros((L, U))  # Initial Lagrange multiplier

    index_list = [(link.linked_node[0].id, link.linked_node[1].id) for link in sim.link_list]
    user_count = len(sim.node_list)
    links_count = len(sim.link_list) // 2
    usage_matrix = create_usage_matrix(links_count, user_count, index_list)
    #Users = np.asmatrix(np.array(usage_matrix))

    # Init users for Lecture graph
    for user_index in range(U):
        for link_index in range(L):
            if user_index == 0:
                # first user use all the links
                Users[user_index][link_index] = 1
            elif user_index == link_index + 1:
                # other user use just one link
                Users[user_index][link_index] = 1

    # Init User and link matrix and lagarnz
    #print(f'{U=}, {L=}')
    for link in range(L):
        for user in range(U):
            if Users[user, link] == 1:  # if the user uses this link
                Links[link, user] = X_r_dual[user]
                Lagranz[link, user] = 0.2

    # Dual Algorithm

    # Get initial rates for Dual Algorithm
    X_r_dual_iter = np.zeros((U, N_dual + 1))
    X_r_dual_iter[:, 0] = X_r_dual

    for iteration_algo in range(N_dual):
        user_iter = random.randint(0, U - 1)  # Random user

        # Sum of Lagranz
        sum_lagranz = 0
        for link in range(L):
            if Links[link, user_iter] != 0:
                sum_lagranz += Lagranz[link, user_iter]

        # Inverse derivative utility function with q_r
        X_r_dual[user_iter] = (sum_lagranz ** ((-1) / alpha))

        # y_l - c_l
        for link in range(L):
            if Links[link, user_iter] != 0:
                Links[link, user_iter] = X_r_dual[user_iter]

                y_l = np.sum(Links[link])
                if Lagranz[link, user_iter] == 0:
                    y_l_c_l = max((y_l - C_l[link]), 0)
                elif Lagranz[link, user_iter] > 0:
                    y_l_c_l = y_l - C_l[link]
                else:
                    y_l_c_l = 0

                # Update Lagrange multiplier
                Lagranz[link, user_iter] = Lagranz[link, user_iter] + H_r_dual[user_iter] * y_l_c_l

        # Update rates per iteration
        X_r_dual_iter[:, iteration_algo + 1] = X_r_dual[:]

    # Plot Dual Algorithm
    x_index_dual = np.arange(N_dual + 1)
    plt.figure('Dual')
    for i in range(U):
        plt.plot(x_index_dual, X_r_dual_iter[i], label=f'User {i + 1}')
    plt.legend(loc='upper right')

    plt.xlabel('Iteration')
    plt.ylabel('X_r')
    plt.title('Dual Algorithm - α =' + str(alpha))
    plt.show()

    print('Final rates are ', X_r_dual_iter[:, -1])


def main():
    sim = Simulator()

    #sim.initialize_sim(n=6, m=20, r=6)

    #sim.save_sim_state(Path.cwd() / 'sim_state1.pkl')
    sim = sim.load_sim_state(Path.cwd() / 'sim_state3.pkl')

    #sim.show_network_grath()
    sim.save_sim_state(Path.cwd() / 'sim_state3.pkl')

    src, dst = sim.link_list[0].linked_node[0].id, sim.link_list[0].linked_node[1].id
    #print(sim.link_list[0], "type is ", src, dst)

    #index_list = [(link.linked_node[0].id, link.linked_node[1].id) for link in sim.link_list]
    #print(sim.link_list, "\nU = ", len(sim.node_list), "L=", len(sim.link_list) // 2)
    #user_count = len(sim.node_list)
    #links_count = len(sim.link_list) // 2
    #usage_matrix = create_usage_matrix(links_count, user_count, index_list)
    #print(np.asmatrix(np.array(usage_matrix)))

    run_dual_alghorithm(sim)


if __name__ == '__main__':
    main()

