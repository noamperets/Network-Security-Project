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


class alpha_fairness_user_state:
    def __init__(self, user: Node, utility: float = 1.0, rate: float = 1.0, derv_util: float = 1.0, lag: float = 0.2):
        self.user = user  # User
        self.utility = utility  # Ur
        self.rate = rate  # Xr
        self.derv_util = derv_util  # Ur'


class alpha_fairness_link_state:
    def __init__(self, link: Link, lag: float = 0.2):
        self.link = link  # User
        self.lag = lag  # lamda
        self.rate = 1  # Xr


def init_dual_algorthm(node_list: list[Node], link_list: list[Link], alpha: int = 1) -> tuple[
    list[alpha_fairness_user_state], list[alpha_fairness_link_state]]:
    user_rate = 1  # this is the init value for the user rate
    users_state = []
    links_state = []
    lag = 0.2

    for user_index in range(len(node_list)):
        user_utility_rate = (user_rate ** (1 - alpha)) / (1 - alpha)
        derv_user_util = 1 / (user_rate ** alpha)
        user_state = alpha_fairness_user_state(node_list[user_index],
                                               utility=user_utility_rate,
                                               rate=user_rate,
                                               derv_util=derv_user_util)
        users_state.append(user_state)

    for link in link_list:
        link_state = alpha_fairness_link_state(link, lag)
        links_state.append(link_state)

    return users_state, links_state


# def run_dual_iteration(user_state: alpha_fairness_user_state, user_states: list[alpha_fairness_user_state], link_states: list[alpha_fairness_link_state], alpha_case):
#     lag_sum = 0
#     link_rates_sum = 0


#     # sum all lagarnz multipler of connected links
#     for link in user_state.user.links_list:
#         for link_state_alpha in link_states:
#             if link_state_alpha.link == link:
#                 lag_sum += link_state_alpha.lag


#     # change the rate by the dual iteration algo rule
#     user_state.rate = (lag_sum ** (-1 / alpha_case))
#     temp = user_state.rate

#     print("Lag sum is ", lag_sum, " user rate ", user_state.rate)
#     """    # sum all lagarnz multipler of connected links
#         for neighber_node in user_state.user.linked_list:
#             neighber_state = user_states[neighber_node.id]
#             link_rates_sum += neighber_state.rate
#     """

#     for link in user_state.user.links_list:
#         for link_state_alpha in link_states:
#             if link_state_alpha.link == link:
#                 # sum rate of user that has been used all links
#                 link_state_alpha.rate = temp

#             link_rates_sum += link_state_alpha.rate


#     """
#     # update lagarnz multipler
#     for link in  link_states:
#         if link.link in user_state.user.links_list:
#             # set y_l - c_l
#             if link.lag == 0:
#                 # yl_cl_diff >= 0
#                 yl_cl_diff = max(link_rates_sum - link.link.capacity, 0)
#                 print("Link lag is zero")
#             elif link.lag > 0:
#                 yl_cl_diff = link_rates_sum - link.link.capacity
#                 print("Pos Link lag")

#             else:
#                 link.lag = 0
#                 yl_cl_diff = 0

#             ## update link lag
#             print("Yl_cl diff: ", yl_cl_diff, " Capcity ", link.link.capacity)
#             link.lag += (yl_cl_diff * STEP_SIZE)
#     """

def run_dual_iteration(user_state: alpha_fairness_user_state, user_states: list[alpha_fairness_user_state],
                       link_states: list[alpha_fairness_link_state], alpha_case):
    lag_sum = 0  # [q_r]

    # sum all Lagrange multipliers of connected links
    for link in user_state.user.links_list:
        for link_state_alpha in link_states:
            if link_state_alpha.link == link:
                lag_sum += link_state_alpha.lag

                # update rate using the dual iteration algorithm
    user_state.rate = (lag_sum ** (-1 / alpha_case))

    # update rates of links connected to the user
    for link in user_state.user.links_list:
        for link_state_alpha in link_states:
            if link_state_alpha.link == link:
                link_state_alpha.rate = user_state.rate

                # we are assuming only 2 nodes use specfic link
                nodes_id = link.linked_node[0].id, link.linked_node[1].id
                yl = user_states[nodes_id[0]].rate + user_states[nodes_id[1]].rate

                # link_state_alpha.lag = np.abs(link_state_alpha.lag)*(link_state_alpha.lag / link_state_alpha.lag)

                if link_state_alpha.lag > 0:
                    lag_diff = yl - link_state_alpha.link.capacity
                elif link_state_alpha.lag == 0:
                    lag_diff = max(0, yl - link_state_alpha.link.capacity)

                else:
                    lag_diff = 0

                link_state_alpha.lag += lag_diff * STEP_SIZE

    # update Lagrange multipliers
    # for this case yl is conastant each link communicate only with its neighbor
    """
    for link_state_alpha in link_states:
        lag_diff = 0
        for link in user_state.user.links_list:
            if  link_state_alpha.link == link:
                yl = link_state_alpha.rate
                if link_state_alpha.lag > 0:
                    lag_diff = yl - link_state_alpha.link.capacity
                elif link_state_alpha.lag == 0:
                    lag_diff = max(0, yl - link_state_alpha.link.capacity)

                link_state_alpha.lag += lag_diff * STEP_SIZE
            yl_cl_diff = max(np.sum([link_state_beta.rate for link_state_beta in link_states if link_state_beta.link in user_state.user.links_list]) - link_state_alpha.link.capacity, 0)
            link_state_alpha.lag += yl_cl_diff * STEP_SIZE
    """


def run_dual_alghorithm(
        sim: Simulator):  # name in english comunication networks book, maybe chehck n book for another name
    alpha_case = 1.0001  # TODO: change by fairness/minmax etc.

    user_rates_by_time = [[] for _ in range(sim.n)]
    user_states, link_states = init_dual_algorthm(sim.node_list, sim.link_list, alpha=alpha_case)
    for i in range(ITER_COUNT):
        user_index = random.randint(0, sim.n - 1)
        user_state = user_states[user_index]

        run_dual_iteration(user_state, user_states, link_states, alpha_case)

        # Add the rate after calc of primal algo
        user_rates_by_time[user_index].append(user_state.rate)

        print(f"i = {i}, user_state.rate :{user_state.rate}")

    print("--------------------------------------------------")
    test = 0
    for index, user in enumerate(user_rates_by_time):
        plt.plot(user, label=f'User {index}')
        test += len(user)
        print(f"user = {index}, user_state.rate :{user_states[index].rate}")
    print("--------------------------------------------------")

    # print('Users final state:')
    # for user in user_states:
    #         print(f"user = {user}, user_state.rate :{user_state.rate}")

    plt.title(f'Dual Network Grath alpha = {alpha_case:.2f}')
    plt.xlabel('Iterations')
    plt.ylabel('Rate')

    plt.legend()
    plt.show()


def create_usage_matrix(links_count, user_count, indexes):
    matrix = np.zeros((user_count, links_count), dtype=int)
    print(f'In usage matrix {indexes=} ')
    for i,j in indexes:
        matrix[i, j] = 1

    return matrix


def main():
    sim = Simulator()

    sim.initialize_sim(n=6, m=20, r=6)

    #sim.save_sim_state(Path.cwd() / 'sim_state1.pkl')
    #sim = sim.load_sim_state(Path.cwd() / 'sim_state1.pkl')

    sim.show_network_grath()
    sim.save_sim_state(Path.cwd() / 'sim_state1.pkl')
    src, dst = sim.link_list[0].linked_node[0].id, sim.link_list[0].linked_node[1].id
    print(sim.link_list[0], "type is ", src, dst)

    index_list = [(link.linked_node[0].id, link.linked_node[1].id) for link in sim.link_list]
    print(sim.link_list, "\nU = ", len(sim.node_list), "L=", len(sim.link_list) // 2)
    user_count = len(sim.node_list)
    links_count = len(sim.link_list) // 2
    usage_matrix = create_usage_matrix(links_count, user_count, index_list)
    print(np.asmatrix(np.array(usage_matrix)))
    run_dual_alghorithm(sim)


if __name__ == '__main__':
    main()

