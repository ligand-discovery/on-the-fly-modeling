import os
import joblib
import sys
import numpy as np
import networkx as nx
import pandas as pd
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import Descriptors

root = os.path.dirname(os.path.abspath(__file__))

sys.path.append(os.path.join(root, "..", "src"))
from model import OnTheFlyModel, HitSelectorByOverlap, CommunityDetector, task_evaluator, fragment_embedder

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

print("Loading molecules for prediction")
df = pd.read_csv(
    os.path.join(root, "..", "data", "slc_inhibitor_collection_gsf_with_auto_crf.tsv"),
    sep=",",
)

R = []
for r in df[["smiles", "smiles_with_crf"]].values:
    ik = Chem.MolToInchiKey(Chem.MolFromSmiles(r[0]))
    smiles = r[1].split("; ")
    for smi in smiles[:1]:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        R += [(ik, smi)]

print("Embeddings")
df = pd.DataFrame(R, columns=["inchikey", "smiles"])

ik2emb_file = os.path.join(root, "..", "data", "tmp_ik2emb.joblib")
if os.path.exists(ik2emb_file):
    ik2emb = joblib.load(ik2emb_file)
else:
    ik2emb = {}
    for i, r in enumerate(df.values):
        print(i)
        try:
            x = fragment_embedder.transform([r[1]])[0]
        except:
            continue
        ik2emb[r[0]] = x
        n = len(x)
    joblib.dump(ik2emb, ik2emb_file)

df = df[df["inchikey"].isin(ik2emb.keys())].reset_index(drop=True)
mw = [Descriptors.ExactMolWt(Chem.MolFromSmiles(smi)) for smi in list(df["smiles"])]
df["mw"] = mw

X = []
for ik in list(df["inchikey"]):
    X += [ik2emb[ik]]

X = np.array(X)
print(X)
print(X.shape)
MAX_HIT_FRAGMENTS_LIST = [50, 100, 200]
MAX_FRAGMENT_PROMISCUITY = [100, 250, 500]
model_tasks = []
columns = []
columns_metadata = []
R = []
iter_counts = 0

for i, uniprot_acs in enumerate(clusters_of_proteins):
    for max_hit_fragments in tqdm(MAX_HIT_FRAGMENTS_LIST):
        for max_fragment_promiscuity in MAX_FRAGMENT_PROMISCUITY:
            print(iter_counts)
            iter_counts += 1
            column = "clu{0}_{1}_{2}".format(i, max_hit_fragments, max_fragment_promiscuity)
            data = HitSelectorByOverlap(uniprot_acs, tfidf=tfidf).select(max_hit_fragments, max_fragment_promiscuity)
            task_evaluation = task_evaluator(model, data)
            if task_evaluation is None:
                continue
            columns += [column]
            columns_metadata += [task_evaluation]
            model.fit(data["y"])
            y_hat = model.classifier.predict_proba(X)[:, 1]
            R += [list(y_hat)]

R = np.array(R).T

dr = pd.DataFrame(R, columns=columns)
df = pd.concat([df, dr], axis=1)

df.to_csv(
    os.path.join(root, "..", "results", "2_gsf_predictions_groups.tsv"), sep="\t", index=False
)

joblib.dump(
    columns_metadata,
    os.path.join(root, "..", "results", "2_gsf_predictions_groups.metadata")
)
