import os
import networkx as nx
import pandas as pd


SIMILARITY_PROTEINS_CUT = 0.2

root = os.path.dirname(os.path.abspath(__file__))


def get_protein_similarity_graph():
    edgelist = pd.read_csv(os.path.join(root, "..", "data", "edgelist.csv"))
    G = nx.Graph()
    for v in edgelist.values:
        if v[-1] < SIMILARITY_PROTEINS_CUT:
            continue
        G.add_edge(v[0], v[1])
    return G


proteins_graph = get_protein_similarity_graph()


def get_clusters_for_modeling(uniprot_acs):
    sg = proteins_graph.subgraph(uniprot_acs)
    clusters = []
    for uniprot_ac in uniprot_acs:
        if uniprot_ac not in sg.nodes():
            clusters += [uniprot_ac]
    for prot_clust in nx.connected_components(sg):
        clusters += [tuple(sorted(prot_clust))]
    clusters = set(clusters)
    clusters = sorted(clusters, key=lambda x: -len(x))
    return clusters
