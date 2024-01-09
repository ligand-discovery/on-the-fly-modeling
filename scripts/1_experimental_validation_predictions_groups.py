import os
import joblib
import sys
import numpy as np
import networkx as nx
import pandas as pd
from tqdm import tqdm

root = os.path.dirname(os.path.abspath(__file__))

sys.path.append(os.path.join(root, "..", "src"))
from model import OnTheFlyModel, HitSelectorByOverlap, CommunityDetector


print("Loading molecules for prediction")
df = pd.read_csv(
    os.path.join(root, "..", "data", "experimental", "validation_hits_with_slc.tsv"),
    sep="\t",
)

print("Getting list of SLCs")

def load_hits():
    hits, fid_prom, pid_prom = joblib.load(
        os.path.join(root, "..", "data", "hits.joblib")
    )
    return hits, fid_prom, pid_prom

def pid2name_mapper():
    df = pd.read_csv(
        os.path.join(root, "..", "data/pid2name_primary.tsv"), sep="\t", header=None
    )
    df.columns = ["uniprot_ac", "gene_name"]
    name2pid = {}
    pid2name = {}
    any2pid = {}
    for r in df.values:
        name2pid[r[1]] = r[0]
        pid2name[r[0]] = r[1]
        any2pid[r[0]] = r[0]
        any2pid[r[1]] = r[0]
    return pid2name, name2pid, any2pid

def pids_to_dataframe(pids):
    R = []
    for pid in pids:
        r = [pid, pid2name[pid], pid_prom[pid]]
        R += [r]
    df = (
        pd.DataFrame(R, columns=["UniprotAC", "Gene Name", "Fragment Hits"])
        .drop_duplicates()
        .reset_index(drop=True)
    )
    return df

pid2name, name2pid, any2pid = pid2name_mapper()
hits, fid_prom, pid_prom = load_hits()

slcs_all = []
with open(
    os.path.join(root, "..", "data", "examples", "slc_cemm_interest.txt"), "r"
) as f:
    for l in f:
        slcs_all += [l.rstrip()]

input_tokens = slcs_all
input_pids = []
for it in input_tokens:
    if it in any2pid:
        pid = any2pid[it]
        if pid in pid_prom:
            input_pids += [any2pid[it]]

input_data = pids_to_dataframe(input_pids)


SIMILARITY_PERCENTILES = [95, 90]


def load_protein_spearman_similarity_matrix():
    uniprot_acs, M = joblib.load(
        os.path.join(root, "..", "data", "protein_protein_spearman_correlations.joblib")
    )
    values = np.triu(M, k=1).ravel()
    cutoffs = [np.percentile(values, p) for p in SIMILARITY_PERCENTILES]
    return uniprot_acs, M, cutoffs


def load_protein_hit_similarity_matrix():
    uniprot_acs, M = joblib.load(
        os.path.join(root, "..", "data", "protein_protein_hit_cosines.joblib")
    )
    values = np.triu(M, k=1).ravel()
    cutoffs = [np.percentile(values, p) for p in SIMILARITY_PERCENTILES]
    return uniprot_acs, M, cutoffs


global_uniprot_acs_0, M0, cutoffs_0 = load_protein_spearman_similarity_matrix()
global_uniprot_acs_1, M1, cutoffs_1 = load_protein_hit_similarity_matrix()


def get_protein_graph(uniprot_acs):
    G = nx.Graph()
    G.add_nodes_from(uniprot_acs)
    pid2idx_0 = dict((k, i) for i, k in enumerate(global_uniprot_acs_0))
    pid2idx_1 = dict((k, i) for i, k in enumerate(global_uniprot_acs_1))
    for i, pid_0 in enumerate(uniprot_acs):
        for j, pid_1 in enumerate(uniprot_acs):
            if i >= j:
                continue
            v = M0[pid2idx_0[pid_0], pid2idx_0[pid_1]]
            for cutoff in cutoffs_0:
                if v >= cutoff:
                    if not G.has_edge(pid_0, pid_1):
                        G.add_edge(pid_0, pid_1, weight=1)
                    else:
                        current_weight = G[pid_0][pid_1].get("weight")
                        G[pid_0][pid_1]["weight"] = current_weight + 1
            v = M1[pid2idx_1[pid_0], pid2idx_1[pid_1]]
            for cutoff in cutoffs_1:
                if v >= cutoff:
                    if not G.has_edge(pid_0, pid_1):
                        G.add_edge(pid_0, pid_1, weight=1)
                    else:
                        current_weight = G[pid_0][pid_1].get("weight")
                        G[pid_0][pid_1]["weight"] = current_weight + 1
    return G

uniprot_inputs = list(input_data["UniprotAC"])
graph = get_protein_graph(uniprot_inputs)
print(len(graph.nodes()))
auroc_cut = 0.7
tfidf = True

model = OnTheFlyModel()

community_detector = CommunityDetector(tfidf=tfidf, auroc_cut=auroc_cut)
clusters_of_proteins = community_detector.cluster(model, graph)
clusters_of_proteins = clusters_of_proteins["ok"]
clusters_of_proteins += [uniprot_inputs]

model_tasks = []
columns = []
R = []
for i, uniprot_acs in enumerate(clusters_of_proteins):
    for max_hit_fragments in tqdm([50, 100, 200]):
        for max_fragment_promiscuity in [100, 250, 500]:
            columns += ["clu_{0}_{1}_{2}".format(i, max_hit_fragments, max_fragment_promiscuity)]
            data = HitSelectorByOverlap(uniprot_acs, tfidf=tfidf).select(max_hit_fragments, max_fragment_promiscuity)
            model.fit(data["y"])
            y_hat = model.predict_proba(list(df["smiles"]))[:, 1]
            R += [list(y_hat)]

R = np.array(R).T

dr = pd.DataFrame(R, columns=columns)
df = pd.concat([df, dr], axis=1)

df.to_csv(
    os.path.join(root, "..", "results", "1_experimental_validation_predictions_groups.tsv"),
    sep="\t",
)
