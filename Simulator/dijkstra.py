import matplotlib

from network_simulator import Simulator, Link, Node, Route

import math
import random
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import sys

inf = sys.maxsize

def dijkstra_small_number_of_links(sim: Simulator, root_node: Node = None):
    if root_node is None:
        root_node = sim.node_list[0]

    # Initialize routes from the root_node to all other nodes
    root_node.Routes = []
    for node in sim.node_list:
        if node != root_node:
            # Start with a high initial weight (infinity) for all nodes except root_node
            root_node.Routes.append(Route(node, float('inf')))

    # Update routes based on direct links from root_node
    for link in sim.link_list:
        if link.linked_node[0] == root_node:
            for route in root_node.Routes:
                if route.destination == link.linked_node[1]:
                    # Update the route's weight and next_hop based on the link
                    route.Weight = 1  # Or use link distance if applicable
                    route.Way.append(link)
                    route.Links.append(link)

                    break

    # Perform Dijkstra's algorithm to update routes
    updated = True
    iteration = 0
    while updated:
        updated = False
        for route1 in root_node.Routes:
            for link in sim.link_list:
                if link.linked_node[0] == route1.destination and link.linked_node[1] != root_node:
                    for route2 in root_node.Routes:
                        if route2.destination == link.linked_node[1]:
                            # Calculate potential new weight via this route
                            new_weight = route1.Weight + 1  # Or other metric
                            if new_weight < route2.Weight:
                                # Update the route if a shorter path is found
                                route2.Weight = new_weight
                                route2.Way = route1.Way + [link]
                                route2.Links = route1.Links + [link]
                                updated = True
                                break
        iteration += 1
        # print(f"Iteration {iteration}: Routes from node {root_node.id}:")
        # for route in root_node.Routes:
        #     print(f"  To {route.destination.id} via {route.next_hop.id} (Weight: {route.Weight}, Way: {route.Way})")

    # Filter out duplicate routes and extract valid routes
    seen_destinations = set()
    valid_routes = []
    for route in root_node.Routes:
        if route.destination not in seen_destinations:
            seen_destinations.add(route.destination)
            valid_routes.append((route.destination, route.Weight, route.Way))

    return valid_routes


# def dijkstra_small_number_of_links(sim: Simulator, root_node: Node = None):
#     if root_node == None:
#         root_node = sim.node_list[0]
#
#     for node in sim.node_list:
#         if root_node == node:
#             continue
#         root_node.Routes.append(Route(node, 1000000))
#     for link in sim.link_list:
#         if link.linked_node[0] == root_node:
#             for route in root_node.Routes:
#                 if route.destination == link.linked_node[1]:
#                     route.Weight = 1
#                     route.Way.append(link)
#                     break
#
#     updated = True
#     while updated:
#         updated = False
#         for route1 in root_node.Routes:
#             for link in sim.link_list:
#                 if link.linked_node[0] == route1.destination and link.linked_node[1] != root_node:
#                     for route2 in root_node.Routes:
#                         if route2.destination == link.linked_node[1] and route2.Weight > route1.Weight + 1:
#                             route2.Weight = route1.Weight + 1
#                             route2.Way = []
#                             for w in route1.Way:
#                                 route2.Way.append(w)
#                             route2.Way.append(link)
#                             updated = True
#                             break


def dijkstra_on_capacity(sim: Simulator, root_node: Node = None):
    if root_node == None:
        root_node = sim.node_list[0]

    for node in sim.node_list:
        if root_node == node:
            continue
        root_node.Routes.append(Route(node, 0))
    for link in sim.link_list:
        if link.linked_node[0] == root_node:
            for route in root_node.Routes:
                if route.destination == link.linked_node[1]:
                    route.Weight = link.capacity
                    route.Way.append(link)
                    break

    updated = True
    while updated:
        updated = False
        for route1 in root_node.Routes:
            for link in sim.link_list:
                if link.linked_node[0] == route1.destination and link.linked_node[1] != root_node:
                    new_cap = min(route1.Weight, link.capacity)
                    for route2 in root_node.Routes:
                        if route2.destination == link.linked_node[1] and route2.Weight < new_cap:
                            route2.Weight = new_cap
                            route2.Way = []
                            for w in route1.Way:
                                route2.Way.append(w)
                            route2.Way.append(link)
                            updated = True
                            break

def distance_between_nodes(a: Node, b: Node):
    return ((a.point.x - b.point.x) ** 2 + (a.point.y - b.point.y) ** 2) ** 0.5

def dijkstra_on_distance(sim: Simulator, root_node: Node = None):
    if root_node == None:
        root_node = sim.node_list[0]

    for node in sim.node_list:
        if root_node == node:
            continue
        root_node.Routes.append(Route(node, 1000000))
    for link in sim.link_list:
        if link.linked_node[0] == root_node:
            for route in root_node.Routes:
                if route.destination == link.linked_node[1]:
                    route.Weight = distance_between_nodes(root_node, route.destination)
                    route.Way.append(link)
                    break

    updated = True
    while updated:
        updated = False
        for route1 in root_node.Routes:
            for link in sim.link_list:
                dis = distance_between_nodes(link.linked_node[0], link.linked_node[1])
                if link.linked_node[0] == route1.destination and link.linked_node[1] != root_node:
                    for route2 in root_node.Routes:
                        if route2.destination == link.linked_node[1] and route2.Weight > route1.Weight + dis:
                            route2.Weight = route1.Weight + dis
                            route2.Way = []
                            for w in route1.Way:
                                route2.Way.append(w)
                            route2.Way.append(link)
                            updated = True
                            break


# TODO: Not complited, Needs Nimrods help
# def print_dijkstra(sim: Simulator, root_node: Node):
#     """visualize dijkstra network graph using matplotlib
#
#         Args:
#             link_list (Node): Root node for dijkstra
#         """
#     for link in sim.link_list:
#         plt.plot([link.linked_node[0].point.x, link.linked_node[1].point.x],
#                  [link.linked_node[0].point.y, link.linked_node[1].point.y], 'b-')
#
#     for node in sim.node_list:
#         plt.plot(node.point.x, node.point.y, 'ro')
#         plt.text(node.point.x, node.point.y, node.id)
#
#     src_node = root_node
#     for route in sim.node_list[0].Routes:
#
#         plt.plot([src_node.point.x, route.next_hop.point.x],
#                  [src_node.point.y, route.next_hop.point.y], 'r-')
#
#         if (route.next_hop == route.Way[-1]):
#             src_node = root_node
#         else:
#             src_node = route.next_hop
#
#     plt.show()





# # TODO: Not complited, Needs to add the amount of undelivered packets in node a (U1) and b (U2) to node c
# def weight_DBDR(a: Node, b: Node, c: Node, w1=0.6):
#     w2 = 1 - w1
#     U1 = 1000
#     U2 = 1000
#     return w1 * U1 - w2 * U2 * distance_between_nodes(b, c) / distance_between_nodes(a, c)
#
#
# # TODO: Not complited, The algorithm is not converging and I still don't understand why
# def dijkstra_distance_weighted_backlog_differential(sim: Simulator, root_node: Node = None):
#     if root_node == None:
#         root_node = sim.node_list[0]
#
#     for node in sim.node_list:
#         if root_node == node:
#             continue
#         root_node.Routes.append(Route(node, 1000000000))
#     for link in sim.link_list:
#         if link.linked_node[0] == root_node:
#             for route in root_node.Routes:
#                 if route.destination == link.linked_node[1]:
#                     route.Weight = weight_DBDR(root_node, root_node, link.linked_node[1])
#                     route.Way.append(link)
#                     break
#
#     updated = True
#     while updated:
#         updated = False
#         for route1 in root_node.Routes:
#             for link in sim.link_list:
#                 if link.linked_node[0] == route1.destination and link.linked_node[1] != root_node:
#                     added_weight = weight_DBDR(root_node, link.linked_node[0], link.linked_node[1])
#                     print(root_node, link.linked_node[0], link.linked_node[1], added_weight)
#                     for route2 in root_node.Routes:
#                         if route2.destination == link.linked_node[1] and route2.Weight < route1.Weight + added_weight:
#                             route2.Weight = route1.Weight + added_weight
#                             route2.Way = []
#                             for w in route1.Way:
#                                 route2.Way.append(w)
#                             route2.Way.append(link)
#                             updated = True
#                             break

def print_dijkstra(sim: Simulator, root_node: Node):
    """visualize dijkstra network graph using matplotlib

        Args:
            link_list (Node): Root node for dijkstra
        """
    colors = ['b', 'g', 'c', 'm', 'y', 'k']
    shape = ['-', '--', ':']

    if root_node == None:
        root_node = sim.node_list[0]

    for link in sim.link_list:
        plt.plot([link.linked_node[0].point.x, link.linked_node[1].point.x],
                 [link.linked_node[0].point.y, link.linked_node[1].point.y], 'r-', linewidth=1,)

    for node in sim.node_list:
        plt.plot(node.point.x, node.point.y, 'ro')
        plt.text(node.point.x, node.point.y, node.id)

    for i in range(len(root_node.Routes)):
        for link in root_node.Routes[i].Way:
            plt.plot([link.linked_node[0].point.x, link.linked_node[1].point.x],
                     [link.linked_node[0].point.y, link.linked_node[1].point.y], colors[i % len(colors)]+shape[i*3//len(root_node.Routes)], linewidth=3)

    plt.show()


def main():
    sim = Simulator()
    sim.initialize_sim(n=20, m=20, r=12)
    # sim.show_network_grath()

    dijkstra_small_number_of_links(sim)

    # for l in sim.link_list:
    #     l.capacity = random.randint(1, 10)
    #     print(str(l), str(l.capacity))
    # dijkstra_on_capacity(sim)

    # dijkstra_distance_weighted_backlog_differential(sim)

    # dijkstra_on_distance(sim)
    print('dijkstra from node 0')
    for route in sim.node_list[0].Routes:
        print(route)

    # sim.show_network_grath()

    print_dijkstra(sim, sim.node_list[0])


        # In the first step calculate route from A to B using dijkstra


if __name__ == '__main__':
    main()
