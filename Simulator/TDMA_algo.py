# TDMA_algo.py
import copy

import numpy as np

from network_simulator import Simulator
import matplotlib.pyplot as plt
from dijkstra import dijkstra_small_number_of_links

def allocate_rates(simulator, user_states, flows):

    #Initialize flows_num and data_sum to 0
    for link in simulator.link_list:
        for k in range(simulator.k):
            link.flows_num = np.zeros(simulator.k, dtype=int)
            link.data_sum = np.zeros(simulator.k, dtype=int)

    for user_state in user_states:
        user = user_state  # Access the node directly from user_state
        # Use Dijkstra's algorithm to compute routes from the user node
        valid_routes = dijkstra_small_number_of_links(simulator, root_node=user)

        # Calculate minimum service rate based on the computed routes
        if valid_routes:
            for flow in flows:
                if flow['source'] == user.id:
                    for route in user.Routes:
                        if route.destination.id == flow['destination']:
                            #Add the flow to all the links in the way
                            for link in route.Links:
                                min_index = np.argmin(link.data_sum)
                                link.flows_num[min_index] += 1
                                link.data_sum[min_index] += flow['data_amount']
                                flow['link_index'].append(tuple([link, min_index]))
                            break
    # Calculate the capacity of the links
    for user_state in user_states:
        user = user_state  # Access the node directly from user_state
        for flow in flows:
            if flow['source'] == user.id:
                for route in user.Routes:
                    if route.destination.id == flow['destination']:
                        for link in route.Links:
                            data_sum = 0
                            for flow_link in flow['link_index']:
                                if link == flow_link[0]:
                                    data_sum = link.data_sum[flow_link[1]]
                                    break
                            flow['capacity'] = min((flow['data_amount'] / data_sum) * link.capacity, flow['capacity']) # find the minimum capacity of the links

                        print(f"Flow from {flow['source']} to {flow['destination']} has rate {flow['capacity'] * 100} and data amount {flow['data_amount']}")
                        break

def run_tdma_algorithm(simulator, flows):
    user_rates_by_time = [[] for _ in range(len(simulator.node_list))]
    user_states = simulator.node_list  # Use node list directly as user_states

    #Calculate capacity of links using TDMA flow
    allocate_rates(simulator, user_states, flows)

def Q6_1():
    sim = Simulator()
    sim.initialize_sim(n=5, m=15, r=6, cap_func=sim.capacity_func)  # Initialize simulator with appropriate parameters
    flows = []
    min_rates = []

    for i in range(5, 51, 5):
        flow = sim.generate_random_flows(sim, num_flows=i)  # Generate random flows
        run_tdma_algorithm(sim, flow)  # Run TDMA algorithm
        flows.append(flow)
        min_rate = min(f['capacity'] for f in flow) * 100
        min_rates.append(min_rate)

    sim.show_network_grath()  # Display the network graph

    # plot the number of plots as a function of the flow with the minimum rate
    plt.plot([len(f) for f in flows], min_rates, 'o-')
    plt.title('Minimum Rate as a function of the number of flows')
    plt.ylabel('Minimum Rate [Mbps]')
    plt.xlabel('Number of Flows')
    plt.show()

def Q6_2():
    sim = Simulator()
    sim.initialize_sim(n=5, m=15, r=6, cap_func=sim.capacity_func)  # Initialize simulator with appropriate parameters
    flows = []
    rates = []
    data_amount = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    original_flow = sim.generate_random_flows(sim, num_flows=10)  # Generate random flows

    for data in data_amount:
        flow = copy.deepcopy(original_flow)  # Create a copy of the original flow
        flow[0]['data_amount'] = data
        run_tdma_algorithm(sim, flow)  # Run TDMA algorithm
        rate = flow[0]['capacity'] * 100
        rates.append(rate)
        flows.append(flow)

    sim.show_network_grath()  # Display the network graph

    # plot the rate as a function of data amount
    plt.plot(data_amount, rates, 'o-')
    plt.title('Rate as a function of data amount')
    plt.ylabel('Minimum Rate [Mbps]')
    plt.xlabel('Data Amount [MB]')
    plt.show()


def Q7_1(k):
    for i in range (1, k+1):
        sim = Simulator()
        sim.initialize_sim(n=20, m=10, r=6, k=i, cap_func=sim.capacity_func)  # Initialize simulator with appropriate parameters
        flows = []
        min_rates = []
        for i in range(5, 51, 5):
            flow = sim.generate_random_flows(sim, num_flows=i)  # Generate random flows
            run_tdma_algorithm(sim, flow)  # Run TDMA algorithm
            flows.append(flow)
            min_rate = min(f['capacity'] for f in flow) * 100
            min_rates.append(min_rate)
        sim.show_network_grath()  # Display the network graph
        # plot the number of plots as a function of the flow with the minimum rate
        plt.plot([len(f) for f in flows], min_rates, 'o-')
        plt.title(f'Minimum Rate as a function of the number of flows with k={sim.k}')
        plt.ylabel('Minimum Rate [Mbps]')
        plt.xlabel('Number of Flows')
        plt.ylim(0, 500)
        plt.show()

def Q7_2(max_k):
    k_values = []
    avg_rates = []

    for k in range(1, max_k+1):
        sim = Simulator()
        sim.initialize_sim(n=20, m=10, r=6, k=k, cap_func=sim.capacity_func)
        flow = sim.generate_random_flows(sim, num_flows=50)
        run_tdma_algorithm(sim, flow)
        avg_rate = sum(f['capacity'] for f in flow) / len(flow) * 100
        avg_rates.append(avg_rate)
        k_values.append(k)

    plt.plot(k_values, avg_rates, 'o-')
    plt.title('Average Rate as a function of the number of channels (k)')
    plt.ylabel('Average Rate [Mbps]')
    plt.xlabel('Number of Channels (k)')
    plt.show()


def main():
    Q6_1() # Graph of minimum rate as a function of the number of flows

    Q6_2() # Graph of minimum rate as a function of data amount

    k = 5
    Q7_1(k) # Graph of minimum rate as a function of the number of flows with channels k=1...5

    Q7_2(5) # Graph of average rate as a function of the number of channels (k=1...5)

if __name__ == '__main__':
    main()
