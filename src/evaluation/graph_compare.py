import numpy as np


def all_nodes(edge_list):
    return list(set([node for edge in edge_list for node in edge if type(node) == type("str")]))


def node_connected(node, edge_list):
    node_connected_list = []
    for node_i, node_j, _ in edge_list:
        if node_i == node:
            node_connected_list.append(node_j)
        elif node_j == node:
            node_connected_list.append(node_connected_list)
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
    return len(inter) / num


def node_similarity(node, edge_list_1, edge_list_2):
    nodes = all_nodes(edge_list_1)

    node_connected_list_1 = node_connected(node, edge_list_1)
    node_connected_list_2 = node_connected(node, edge_list_2)

    node_not_connected_list_1 = node_not_connected(node, edge_list_1)
    node_not_connected_list_2 = node_not_connected(node, edge_list_2)

    connected_similarity = list_similarity(node_connected_list_1, node_connected_list_2)
    not_connected_similarity = list_similarity(node_not_connected_list_1, node_not_connected_list_2)

    return connected_similarity, not_connected_similarity


def graph_similarity(edge_list_1, edge_list_2):
    nodes = all_nodes(edge_list_1)

    all_result = []
    for i in nodes:
        all_result.append(node_similarity(i, edge_list_1, edge_list_2=))

    return np.array(all_result).T
