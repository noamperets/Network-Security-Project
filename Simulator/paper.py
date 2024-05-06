import matplotlib

from network_simulator import Simulator, Link, Node, Route, Packets

import math
import random
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import sys

inf = sys.maxsize


class WQ:
    def __init__(self, a: Node, b: Node, destination: Node, source: Node, w1: float, packets_list):
        self.a = a
        self.b = b
        self.destination = destination
        self.source = source
        self.w1 = w1
        self.w2 = 1 - w1
        self.Ua = 0
        self.Ub = 0
        for packets in packets_list:
            if packets.destination == destination and packets.location == a:
                self.Ua = packets.amount
            if packets.destination == destination and packets.location == b:
                self.Ub = packets.amount
        self.Da = distance_between_nodes(a, destination)
        self.Db = max(distance_between_nodes(b, destination), 0.000001)
        self.value = max(self.w1 * self.Ua * (self.Da / self.Db) - self.w2 * self.Ub, 0)

    def __str__(self):
        return str(self.a) + '-->' + str(self.b) + '  for user: ' + str(self.destination) + '   with value: ' +str(self.value)


def init_packets_list(sim, to_do):
    packets_list = []
    if to_do == 'to':
        for i in range(1, sim.n):
            packets_list.append(Packets(sim.node_list[i], sim.node_list[0], sim.node_list[i], 100))
    elif to_do == 'from':
        for i in range(1, sim.n):
            packets_list.append(Packets(sim.node_list[0], sim.node_list[i], sim.node_list[0], 100))
    return packets_list


def clean_packets_list(packets_list):
    new_packets_list = []
    for i in range(len(packets_list)):
        if packets_list[i]:
            if packets_list[i].amount != 0:
                new_packets_list.append(packets_list[i])
    new_new_packets_list = []
    while  new_packets_list:
        for i in range(len(new_packets_list)):
            flag = False
            for j in range(i+1, len(new_packets_list)):
                if (new_packets_list[i].source == new_packets_list[j].source and
                        new_packets_list[i].destination == new_packets_list[j].destination and
                        new_packets_list[i].location == new_packets_list[j].location):
                    new_packets_list[i].amount += new_packets_list[j].amount
                    del new_packets_list[j]
                    flag = True
                    break
            if flag:
                break
        if i == len(new_packets_list) - 1:
            break

    return new_packets_list


def distance_between_nodes(a: Node, b: Node):
    return ((a.point.x - b.point.x) ** 2 + (a.point.y - b.point.y) ** 2) ** 0.5


def DBDR(sim, packets_list):
    w1 = 0.6
    packets_list = clean_packets_list(packets_list)
    DBDR = {}
    for link in sim.link_list:
        for packet in packets_list:
            if link.linked_node[0] == packet.location:
                DBDR[(link.linked_node[0], link.linked_node[1], packet.destination, packet.source)] = (
                    WQ(link.linked_node[0], link.linked_node[1], packet.destination, packet.source, w1, packets_list))

    next_move = []
    for destination in sim.node_list:
        for source in sim.node_list:
            DBDR_for_link = []
            for link in sim.link_list:
                if (link.linked_node[0], link.linked_node[1], destination, source) in DBDR.keys():
                    DBDR_for_link.append(DBDR[(link.linked_node[0], link.linked_node[1], destination, source)])
            if DBDR_for_link:
                next_move.append(max(DBDR_for_link, key=lambda x: x.value))

    while True:
        for i in range(len(next_move)):
            flag = False
            for j in range(i + 1, len(next_move)):
                if next_move[i].a == next_move[j].a and next_move[i].b == next_move[j].b:
                    if next_move[i].value >= next_move[j].value:
                        del next_move[j]
                    else:
                        del next_move[i]
                    flag = True
                    break
            if flag:
                break
        if i == len(next_move) - 1:
            break

    return next_move


def print_state(sim: Simulator, next_move):
    """visualize dijkstra network graph using matplotlib

        Args:
            link_list (Node): Root node for dijkstra
        """

    for link in sim.link_list:
        plt.plot([link.linked_node[0].point.x, link.linked_node[1].point.x],
                 [link.linked_node[0].point.y, link.linked_node[1].point.y], 'r-', linewidth=2)

    for node in sim.node_list:
        plt.plot(node.point.x, node.point.y, 'ro')
        plt.text(node.point.x, node.point.y, node.id)

    for move in next_move:
        plt.plot([move.a.point.x, move.b.point.x],
                        [move.a.point.y, move.b.point.y], 'b-', linewidth=2)
        for node in sim.node_list:
            plt.text((move.a.point.x+move.b.point.x)/2, (move.a.point.y+move.b.point.y)/2, str(move.source) + '-->' + str(move.destination), size=8)

    plt.show()


def DBDR_real_time(sim: Simulator, packets_list):
    while packets_list:
        next_move = DBDR(sim, packets_list)
        print('*'*30)
        for i in next_move:
            print(i)
        print_state(sim, next_move)
        for move in next_move:
            for packets in packets_list:
                if move.source == packets.source and move.destination == packets.destination and move.a == packets.location:
                    packets_list.append(packets.move(move.a, move.b, 100))
                    break
        packets_list=clean_packets_list(packets_list)
        input('Press Enter to continue')


def main():
    sim = Simulator()
    sim.initialize_sim(n=15, m=20, r=12)

    packets_list = init_packets_list(sim, 'from')
    # packets_list = init_packets_list(sim, 'to')
    DBDR_real_time(sim, packets_list)


if __name__ == '__main__':
    main()
