import json
from typing import List

import torch
import networkx as nx

from src.data.dataset import NewsDataset
from src.model.hawkes import MFHawkes
import pandas as pd
import matplotlib as mpl
import matplotlib
import numpy as np
import matplotlib.pyplot as plt


def save_graph(edges):
    with open("../../cache/graph.json", "w") as f:
        json.dump(edges, f)


def create_graph(node_name, adj):
    adj = adj.detach().numpy()
    np.fill_diagonal(adj, 0)
    threshold = pd.Series(adj.flatten()).quantile(0.5)
    mask = (adj - threshold) < 0
    adj[mask] = 0
    G = nx.DiGraph()

    edges = [[node_name[i], node_name[j], float(adj[i][j])] for i in range(len(adj)) for j in
             range(len(adj)) if adj[i][j] > 0]

    save_graph(edges)
    G.add_weighted_edges_from(edges)
    weight = [adj[i][j] for i in range(len(adj)) for j in range(len(adj)) if adj[i][j] > 0]
    return G, weight


def draw_graph_and_save(G, weight, path):
    mpl.rc("savefig", dpi=1000)
    mpl.rcParams['figure.dpi'] = 400
    mpl.rcParams['savefig.facecolor'] = "white"

    edge_width = [i * 0.1 for i in weight]
    plt.figure(figsize=(25, 25))

    nx.draw_networkx(G,
                     #                  with_labels = True,
                     width=edge_width
                     )

    plt.axis('off')
    plt.savefig(path, facecolor="white", transparent=True)

    # plt.tight_layout()


@torch.no_grad()
def visualize_graph_using_MFhawkes(model: MFHawkes, author: List[str], path):
    id_map = NewsDataset.load_id_map()
    ids = [id_map[i] for i in author]
    adj = model.connection_function(torch.LongTensor(ids))
    G, weight = create_graph(author, adj)
    draw_graph_and_save(G, weight, path)
