from __future__ import annotations

import math
import random
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pickle

"""
Notes from Kobis last lecture:
    - The graph needs to be a "connected graph" -> need to check for this
    - Power of each node can be picked by:
        - Avg length of all connected nodes
        - Changes with each connected node, depends on length from connected node <- encourage to do so
    - For each connected nodes Node_i, Node_j and Node_i Tx. The power Node_j rec is = p_i*|h_i,j|^2
        - |h_i,j| the gain of the chanel, we will use "path loss"(r^2/ r^3: free space, r^4:congested urban area) model + "small-scale fading". 
        - |h_i,j|^2 the power of the  incoming signal of interest
        - p_i Tx power
        - p_i*|h_i,j|^2 Rx power
        - |h^I_i,j|^2 the interference power
    - Chanel capacity
        - c_i,j = B_i*log2(1 + SINR)
        - SINR(Signal-to-interference-plus-noise ratio): ( p_i*|h_i,j|^2 ) / ( sum_of_all_power_in_network_multiplied_by_interference_power + channel_power_noise )
        - sum_of_all_power_in_network_multiplied_by_interference_power: the interference for node j: sum for all i {p_i * |h^I_i,j|^2}
    - F
        - source node
        - dst node
        - amount of data to Tx

    - db = 10log2(x) - because are calculations are powered centered
    - randomly generate antenna azimuth and ... , can be used to calculate path loss better.  need to talk about it

    - what is the typical interference power ? we can use -174 dbm
    - for alpha fairness, alpha cant be inf but can be a number large enough to be close to it. (4,5 from what kobi hinted to)
"""

# default values to be used:
default_power_value = 1
default_bw_value = 1
default_capacity_value = 1



trys = 200
# lamda = 900 * (10 ** 6) # 900 MHz Cellular frequency
lamda = 2400 * (10 ** 6)  # 2.4GHz Wifi
inf = 10 ** 10


def calculate_rayleigh_fading() -> float:  # (float scale = 1, int size = 0)
    # we can think of the gain and phase as I.I.D gaussian process with zero mean
    # (r^2/ r^3: free space, r^4:congested urban area)
    # TODO: need to be in absolute
    h_fading = (1 / np.sqrt(2)) * np.absolute(np.random.randn() + 1j * np.random.randn())
    return h_fading


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

def calc_angle_between_points(p1: Point, p2: Point) -> float:
    x = np.array([p1.x, p1.y])
    y = np.array([p2.x, p2.y])
    unit_x = x / np.linalg.norm(x)
    unit_y = y / np.linalg.norm(y)
    angle_rad = np.arccos(np.dot(unit_x, unit_y))
    angle_deg = np.degrees(angle_rad)

    return angle_deg


class Packets:
    def __init__(self, source: Node, destination: Node, location: Node, amount: int):
        self.source = source
        self.destination = destination
        self.location = location
        self.amount = amount

    def __str__(self):
        return f'{self.source} - {self.destination} - {self.amount}'


    def move(self, a: Node, b: Node, rate: int):
        if a == self.location:
            self.amount = max(self.amount - rate, 0)
            if b == self.destination:
                return None
            return Packets(self.source, self.destination, b, rate)




# TODO: buffer for how much data each node needs to Tx - for all nodes 
class Node:
    def __init__(self, point: Point, id: int):
        self.point = point
        # link list
        self.power = None
        self.bw = None
        self.id = id
        self.linked_list: list[Node] = []
        self.Routes: list[Route] = []

        self.links_list: list[Link] = []

    def __str__(self) -> str:
        return str(self.id)

    def __repr__(self) -> str:
        return str(self.id)

class Link:
    def __init__(self, linked_node: tuple[Node]):
        self.linked_node = linked_node
        self.gain = None  # |h_i,j|
        self.interference = None  # |h^I_i,j|
        self.capacity = None  # c_i,j
        self.flows_num = []
        self.data_sum = []
        self.id = None
        self.distance = None


    def __str__(self) -> str:
        return '(' + str(self.linked_node[0]) + ', ' + str(self.linked_node[1]) + ')'

    def __repr__(self) -> str:
        return '(' + str(self.linked_node[0]) + ', ' + str(self.linked_node[1]) + ')'


class Route:
    def __init__(self, destination: Node, weight: int):
        self.destination = destination
        self.Weight = weight
        self.Way = []
        self.Links = []


    def __str__(self) -> str:
        text = '(source:' + str(self.Way[0].linked_node[0]) + '  ,destination:' + str(self.destination) + '  ,Weight:' + str(self.Weight) + ')\n'
        text += 'Way: ' + str(self.Way[0].linked_node[0])
        for w in self.Way:
            text += ' --> ' + str(w.linked_node[1])

        return text



class Simulator():

    def node_genarator(self) -> list[Node]:
        """generate n (define as N in assignment) nodes placement randomly, around a circle of radios m (define as M in the assignment)

        Args:
            n (int): _description_
            m (float): radius of the graph

        Returns:
            list[Node]: _description_
        """

        # check that no node is generuted twice in the same place
        while True:
            node_list = []
            check_list = []
            for i in range(self.n):
                x = random.uniform(-self.m, self.m)
                y_limit = math.sqrt(self.m ** 2 - x ** 2)
                y = random.uniform(-y_limit, y_limit)
                node_list.append(Node(Point(x, y), i))
                check_list.append((x, y))

            if len(set(check_list)) > 1:
                break

        return node_list

    def node_linker(self) -> list[Link]:
        """finds all edges that connect nodes that are less than r

        Returns:
            list[Link]: list of all edges

        """
        self.try_num = + 1
        if self.try_num > trys:
            print(f"Number of trys exceeded: {trys}")
            exit()

        link_list = []
        self.link_matrix = np.diag(np.ones(self.n))

        for center_node in self.node_list:
            for node in self.node_list:
                if node == center_node:
                    continue

                if ((node.point.x - center_node.point.x) ** 2 + (
                        node.point.y - center_node.point.y) ** 2) < self.r ** 2:
                    link = Link((center_node, node))
                    self.link_matrix[center_node.id][node.id] = 1
                    link.distance = np.sqrt(
                        (node.point.x - center_node.point.x) ** 2 + (node.point.y - center_node.point.y) ** 2)
                    link_list.append(link)

                    center_node.linked_list.append(node)

                    center_node.links_list.append(link)


        # check for fully connected graph
        """
        Check the eigenvalues of the Laplacian Matrix
        The Laplacian Matrix is defined as the difference between the Diagonal matrix and the Adjacency Matrix.

        Once we know the Laplacian Matrix, our work is actually almost done. We’ll need to find the Fiedler value,
        which is simply the second smallest eigenvalue of our Laplacian matrix. If the Fiedler value is higher than zero, 
        then this means the graph is fully connected. If it isn’t, then the graph isn’t fully connected and some nodes are isolated from the others, 
        or form a subgraph.
        """
        degree_matrix = np.diag(np.sum(self.link_matrix, axis=0)) - np.diag(np.ones(self.n))
        laplacian_matrix = degree_matrix - self.link_matrix
        eigenvalues, _ = np.linalg.eig(laplacian_matrix)
        eigenvalues = np.sort(eigenvalues)

        print("~")
        return link_list, eigenvalues[1] > 0

    def show_network_grath(self):
        """visualize network graph using matplotlib

        Args:
            link_list (list[Link]): link list
        """
        for link in self.link_list:
            plt.plot([link.linked_node[0].point.x, link.linked_node[1].point.x],
                     [link.linked_node[0].point.y, link.linked_node[1].point.y], 'r-')

        for node in self.node_list:
            plt.plot(node.point.x, node.point.y, 'ro')
            plt.text(node.point.x, node.point.y, node.id)

        plt.show()

    def save_sim_state(self, file: Path = Path.cwd() / 'sim_state.pkl') -> None:
        # TODO: add doc string
        if file.exists():
            print("Overriding 'sim_state.pkl' with new simulator state.")
        with file.open('wb') as f:
            pickle.dump(self, f)

    def load_sim_state(self, file: Path) -> Simulator:
        # TODO: add doc string
        if not file.exists():
            exit(1)
        with file.open('rb') as f:
            return pickle.load(f)

    def __init__(self):
        self.n = None
        self.m = None
        self.r = None
        self.k = None
        self.link_matrix = None
        self.try_num = None
        self.node_list = None
        self.link_list = None

    @staticmethod
    def default_power_func(node_list: list[Node]) -> None:
        """default power function, enters power parameter for each node

        Args:
            node_list (list[Node]): list of all nodes in network
        """
        for node in node_list:
            node.power = default_power_value

    @staticmethod
    def default_bw_func(node_list: list[Node]) -> None:
        """default band-width function, enters bw parameter for each node

        Args:
            node_list (list[Node]): list of all nodes in network
        """
        for node in node_list:
            node.bw = default_bw_value

    @staticmethod
    def default_capacity_func(link_list: list[Link]) -> None:
        """default capacity function, enters capacity parameter for each node

        Args:
            link_list (list[Node]): list of all links in network
        """
        for link in link_list:
            link.capacity = default_capacity_value



    @staticmethod
    def default_gain_func(link_list: list[Link]) -> None:
        """default gain function, enters gain parameter for each node

        Args:
            link_list (list[Link]): list of all links in network
        """
        # TODO: emplemint "path loss"(r^2/ r^3: free space, r^4:congested urban area) model + "small-scale fading"
        # we used https://en.wikipedia.org/wiki/Rayleigh_fading#The_model
        # https://en.wikipedia.org/wiki/Path_loss
        # https://www.geeksforgeeks.org/fading-in-wireless-communication/
        # https://www.ahsystems.com/articles/Understanding-antenna-gain-beamwidth-directivity.php
        # https://study-ccnp.com/directional-antenna-vs-omnidirectional-antenna/
        # because we use Rayleigh model which model the vary of the signal
        # we encounter the problem of NLOS  - No line of sight
        # assuming urban environment with scatters
        # which is make sense in cellular/WIFI
        # we also consider antenna gains for typical antenna.
        # for calculation of the path-loss we need to
        # we are assuming fixed wavelength for the "carrier"

        # 2dbi antenna for wifi bands use like 5 for more info
        transmiter_antena_gain = 2
        reciver_antena_gain = 2

        for link in link_list:
            transmiter, reciver = link.linked_node
            transmitter_power, reciver_power = transmiter.power, reciver.power

            path_loss = (4 * math.pi * (link.distance ** 2)) / (
                    transmiter_antena_gain * reciver_antena_gain * (lamda ** 2))   

            link.gain = np.log10(path_loss) + np.log10(calculate_rayleigh_fading())

    @staticmethod
    def initialize_interference_null(link_list: list[Link]) -> None:
        """initialize interference to 0 for all links"""
        for link in link_list:
            link.interference = 0

    @staticmethod
    def default_interference_func(link_list: list[Link]) -> None:
        """default interference function, enters interference parameter for each node

        Args:
            link_list (list[Link]): list of all links in network
        """
        for link in link_list:
            angle = calc_angle_between_points(link.linked_node[0].point, link.linked_node[1].point)

            if -30.0 <= angle <= 30.0:
                link.interference = 0
            else:
                # small interference for outside of -30 , 30 radiation graph
                link.interference = 1e-3 * np.random.normal()


    @staticmethod
    def capacity_func(link_list: list[Link]) -> None:
        """default capacity function, enters capacity parameter for each node

        Args:
            link_list (list[Link]): list of all links in network
        """
        # TODO: implement capacity
        for link in link_list:
            noise = abs(random.gauss(0, 0.1))
            # sinr = (link.gain + link.linked_node[0].power) / (link.interference + noise)
            sinr = link.linked_node[0].power / (link.interference + noise)
            link.capacity = link.linked_node[0].bw * np.log2(1 + sinr)
    
    @staticmethod
    def generate_random_flows(self, num_flows):
        flows = []
        count = 0
        while count < num_flows:
            source = random.randint(0, self.n-1)  # source
            destination = random.randint(0, self.n-1)  # destination
            if source == destination:
                continue
            data_amount = random.randint(100, 1000)  # data TODO: what is the range 100 mbit/sec - 500 mbit/sec -> for 1 giga it will take 10 bursts -> in giga bite (1:10) looks good
            flows.append({'source': source, 'destination': destination, 'data_amount': data_amount,
                          'rate': 0, 'capacity': float('inf'), 'link_index': []})
            count += 1
        return flows

    ###############################################################
    ############### ADD YOUR FUNCTION BELOW #######################
    ### Add Power functions and Band-Width functions under here.###
    ### Notice, the functions take node_list and return None. #####
    ###############################################################

    @staticmethod
    def boiler_plate_power_func(node_list: list[Node]) -> None:
        pass

    @staticmethod
    def boiler_plate_bw_func(node_list: list[Node]) -> None:
        pass

    ###############################################################
    ###############################################################

    def initialize_sim(self, n: int, m: float, r: float, k: int = 1, power_func=default_power_func,
                       bw_func=default_bw_func, gain_func=default_gain_func, cap_func=default_capacity_func,
                       interference_func=default_interference_func) -> tuple[list[Node], list[Point]]:
        """initialize function for simulator

            Args:
                n (int): Number of Node, define as N in assignment
                m (float): Max radios for node placement, define as M in assignment
                r (float): Max radios for link
                k (int, optional): Number of orthogonal channels. Defaults to 1.
                power_func (_type_, optional): Power function to be used. Defaults to default_power_func.
                bw_func (_type_, optional): Band-width function to be used. Defaults to default_bw_func.

            Returns:
                tuple[list[Node], list[Point]]: (node list, link list)
            """
        self.n = n
        self.m = m
        self.r = r
        self.k = k
        self.link_matrix = None
        connected_grath = False
        self.try_num = 0

        while not connected_grath:
            self.node_list = self.node_genarator()
            self.link_list, connected_grath = self.node_linker()

        power_func(self.node_list)
        bw_func(self.node_list)

        gain_func(self.link_list)
        if interference_func:
            interference_func(self.link_list)
        else:
            self.initialize_interference_null(self.link_list)

        # self.capacity_func(self.link_list) #TODO: why in self?
        cap_func(self.link_list)

    # def dijkstra(self, root_node: Node):
    # # TODO: add doc string
    # # TODO: move out of sim
    # # we can look at the capacity with out any interference, min price for link.
    # # use interference = 0.
    # # another option is use fixed cost per link/ number of hops/
    # # use external function for picking route
        
    #     for node in self.node_list:
    #         if root_node == node:
    #             continue
    #         root_node.Routes.append(Route(None, node, inf))
    #     for link in self.link_list:
    #         if link.linked_node[0] == root_node:
    #             for route in root_node.Routes:
    #                 if route.destination == link.linked_node[1]:
    #                     route.weight = link.capacity
    #                     route.next_hop = link.linked_node[1]
    #                     break
        
    #     print(root_node.Routes)

    #     for r in root_node.Routes:
    #         print(r.weight)

    #     print(max(root_node.Routes, key=lambda x: x.weight))

        
    #     while max(root_node.Routes, key=lambda x: x.weight).weight < inf:
    #         for route1 in root_node.Routes:
    #             for link in self.link_list:
    #                 if link.linked_node[0] == route1.destination:
    #                     for route2 in root_node.Routes:
    #                         if route2.destination == link.linked_node[1]:
    #                             route2.next_hop = route1.next_hop
    #                             route2.weight = min(link.capacity, route1.weight)
    #                             break


def plot_graph_data(x_values, y_values, title_str, plot_string_color):
    # TODO: add doc string
    plt.plot(x_values, y_values, plot_string_color)
    plt.title(title_str)


def main():
    sim = Simulator()
    sim.initialize_sim(n=5, m=20, r=6)
    # sim.save_sim_state()
    sim.show_network_grath()
    # loaded_sim = sim
    # loaded_sim = Simulator()
    # loaded_sim = loaded_sim.load_sim_state(Path.cwd() / 'sim_state.pkl')
    # loaded_sim.show_network_grath()

    #print(sim.node_list[0].Routes)
    #sim.dijkstra(sim.node_list[2])
    #print(sim.node_list[0].Routes)

if __name__ == '__main__':
    main()
