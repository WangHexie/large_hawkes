import json
from random import random

import numpy as np


def all_nodes(edge_list):
    return list(set([node for edge in edge_list for node in edge if type(node) == type("str")]))


def node_connected(node, edge_list):
    node_connected_list = []
    for node_i, node_j, _ in edge_list:
        if node_i == node:
            node_connected_list.append(node_j)
        elif node_j == node:
            node_connected_list.append(node_i)
    return node_connected_list


def node_not_connected(node, edge_list):
    nodes = all_nodes(edge_list)
    connected = node_connected(node, edge_list)
    not_connected = []
    for i in nodes:
        if i != node and i not in connected:
            not_connected.append(i)

    return not_connected


def list_similarity(list_primary, list2):
    num = len(set(list_primary))

    inter = set(list_primary).intersection(set(list2))
    return len(inter) / num if num != 0 else 0


def node_similarity(node, edge_list_1, edge_list_2):
    nodes = all_nodes(edge_list_1)

    node_connected_list_1 = node_connected(node, edge_list_1)
    node_connected_list_2 = node_connected(node, edge_list_2)

    node_not_connected_list_1 = node_not_connected(node, edge_list_1)
    node_not_connected_list_2 = node_not_connected(node, edge_list_2)

    connected_similarity = list_similarity(node_connected_list_1, node_connected_list_2)
    not_connected_similarity = list_similarity(node_not_connected_list_1, node_not_connected_list_2)

    my_f1 =  2 * connected_similarity * not_connected_similarity/ (connected_similarity+ not_connected_similarity) if connected_similarity+ not_connected_similarity != 0 else 0
    return connected_similarity, not_connected_similarity, my_f1

def graph_similarity(edge_list_1, edge_list_2):
    nodes = all_nodes(edge_list_1)

    all_result = []
    for i in nodes:
        all_result.append(node_similarity(i, edge_list_1, edge_list_2))

    return np.array(all_result).T.mean(-1)


def graph_sim_by_path(name1, name2):
    with open("../../cache/" + name1, "r") as f:
        base_graph = json.load(f)

    with open("../../cache/" + name2, "r") as f:
        compare_graph = json.load(f)

    print("acc", "TN/N", graph_similarity(base_graph, compare_graph))


def generate_random_graph(edge_list):
    nodes = all_nodes(edge_list)
    edges = []
    threshold = 0.7
    for node_i in nodes:
        for node_j in nodes:
            if random() > threshold:
                edges.append([node_i, node_j, 1])
    return edges


if __name__ == '__main__':
    # with open("../../cache/base_graph.json", "r") as f:
    #     base_graph = json.load(f)
    #
    # with open("../../cache/graph.json", "r") as f:
    #     compare_graph = json.load(f)
    #
    # print("acc", "TN/N", graph_similarity(base_graph, compare_graph))

    # graph_sim_by_path("graph_2_formal_run.json", "graph_3_formal_run.json")
    # graph_sim_by_path("base_graph.json", "graph_3_formal_run.json")
    # graph_sim_by_path("graph_3_formal_run.json", "base_graph.json")
    #
    #
    with open("../../cache/base_graph.json", "r") as f:
        base_graph = json.load(f)

    random_graph = generate_random_graph(base_graph)
    result = graph_similarity(base_graph, random_graph)
    print(result)

    graph_sim_by_path("base_graph.json", "graph_train_on_data.json")
    graph_sim_by_path("graph_train_on_data_2.json", "graph_train_on_data.json")

