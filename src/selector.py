import os
import networkx as nx
import numpy as np
import joblib
import community as community_louvain

root = os.path.dirname(os.path.abspath(__file__))


def load_protein_similarity_matrix():
    uniprot_acs, M = joblib.load(
        os.path.dirname(
            root, "..", "data", "protein_protein_spearman_correlation.joblib"
        )
    )
    return uniprot_acs, M


global_uniprot_acs, M = load_protein_similarity_matrix()


def get_graph(uniprot_acs, cutoff):
    pid2idx = dict((k, i) for i, k in global_uniprot_acs.items())
    G = nx.Graph()
    G.add_nodes_from(uniprot_acs)
    for i, pid_0 in enumerate(uniprot_acs):
        for j, pid_1 in enumerate(uniprot_acs):
            if i >= j:
                continue
            v = M[pid2idx[pid_0], pid2idx[pid_1]]
            if v >= cutoff:
                G.add_edge((pid_0, pid_1))
    return G


def get_graph_partition(G):
    partition = community_louvain.best_partition(G)
    return partition
