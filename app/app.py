import os
import sys
import streamlit as st
import pandas as pd
import joblib
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import AllChem
from rdkit import RDLogger
import uuid

def get_session_id():
    if "session_id" not in st.session_state:
        st.session_state["session_id"] = str(uuid.uuid4())
    return st.session_state["session_id"]


session_id = get_session_id()

RDLogger.DisableLog("rdApp.*")

import networkx as nx

st.set_page_config(layout="wide")

root = os.path.dirname(os.path.abspath(__file__))

sys.path.append(os.path.join(root, "..", "src"))
from model import OnTheFlyModel, HitSelectorByOverlap, CommunityDetector, task_evaluator

cache_folder = os.path.join(root, "..", "cache")
if not os.path.exists(cache_folder):
    os.mkdir(cache_folder)

df = None

# Functions and variables

CRF_PATTERN = "CC1(CCC#C)N=N1"
CRF_PATTERN_0 = "C#CC"
CRF_PATTERN_1 = "N=N"


SIMILARITY_PERCENTILES = [95, 90]


@st.cache_data()
def load_protein_spearman_similarity_matrix():
    uniprot_acs, M = joblib.load(
        os.path.join(root, "..", "data", "protein_protein_spearman_correlations.joblib")
    )
    values = np.triu(M, k=1).ravel()
    cutoffs = [np.percentile(values, p) for p in SIMILARITY_PERCENTILES]
    return uniprot_acs, M, cutoffs


@st.cache_data()
def load_protein_hit_similarity_matrix():
    uniprot_acs, M = joblib.load(
        os.path.join(root, "..", "data", "protein_protein_hit_cosines.joblib")
    )
    values = np.triu(M, k=1).ravel()
    cutoffs = [np.percentile(values, p) for p in SIMILARITY_PERCENTILES]
    return uniprot_acs, M, cutoffs


global_uniprot_acs_0, M0, cutoffs_0 = load_protein_spearman_similarity_matrix()
global_uniprot_acs_1, M1, cutoffs_1 = load_protein_hit_similarity_matrix()


@st.cache_data()
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


@st.cache_data()
def load_hits():
    hits, fid_prom, pid_prom = joblib.load(
        os.path.join(root, "..", "data", "hits.joblib")
    )
    return hits, fid_prom, pid_prom


@st.cache_data()
def load_fid2smi():
    d = pd.read_csv(os.path.join(root, "..", "data", "cemm_smiles.csv"))
    fid2smi = {}
    for v in d.values:
        fid2smi[v[0]] = v[1]
    return fid2smi


@st.cache_data()
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


def is_valid_smiles(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
    except:
        mol = None
    if mol is None:
        return False
    else:
        return True


def has_crf(mol):
    pattern = CRF_PATTERN
    has_pattern = mol.HasSubstructMatch(Chem.MolFromSmarts(pattern))
    if not has_pattern:
        if mol.HasSubstructMatch(
            Chem.MolFromSmarts(CRF_PATTERN_0)
        ) and mol.HasSubstructMatch(Chem.MolFromSmarts(CRF_PATTERN_1)):
            return True
        else:
            return False
    return True


def get_fragment_image(smiles):
    m = Chem.MolFromSmiles(smiles)
    AllChem.Compute2DCoords(m)
    opts = Draw.DrawingOptions()
    opts.bgColor = None
    im = Draw.MolToImage(m, size=(200, 200), options=opts)
    return im


# App code

hits, fid_prom, pid_prom = load_hits()
pid2name, name2pid, any2pid = pid2name_mapper()
fid2smi = load_fid2smi()


st.title("On-the-Fly Modeling from Ligand Discovery")
st.write(
    "Welcome to the On-the-Fly Modeling tool! Select your proteins of interest and we'll build a quick ML model."
)

cols = st.columns([1, 1, 2.4])

col = cols[0]

col.subheader(":mag: Input your proteins")

text = col.text_area("Paste proteins separated by space or new line")
input_tokens = text.split()
input_pids = []
for it in input_tokens:
    if it in any2pid:
        pid = any2pid[it]
        if pid in pid_prom:
            input_pids += [any2pid[it]]

input_data = pids_to_dataframe(input_pids)

tfidf = col.checkbox(label="TFIDF", value=True)

if input_data.shape[0] == 0:
    has_input = False
    if len(input_tokens) > 0:
        col.warning(
            "None of your input proteins was found in the Ligand Discovery interactome.".format(
                len(input_pids), len(input_tokens)
            )
        )
else:
    has_input = True

if has_input:

    model = OnTheFlyModel()
    is_fitted = False

    col.info(
        "{0} out of {1} input proteins were found in the Ligand Discovery interactome, corresponding to all statistically significant fragment-protein pairs.".format(
            len(input_pids), len(input_tokens)
        )
    )

    col.dataframe(input_data, hide_index=True)

    uniprot_inputs = list(input_data["UniprotAC"])
    if len(uniprot_inputs) == 1:
        clusters_of_proteins = [uniprot_inputs]
    else:
        graph = get_protein_graph(uniprot_inputs)
        auroc_cut=0.7
        graph_key = "-".join(sorted(graph.nodes()))
        clusters_cache_file = os.path.join(
            root, "..", "cache", session_id + "_clusters.joblib"
        )
        clusters_of_proteins = None
        if os.path.exists(clusters_cache_file):
            gk, clu = joblib.load(clusters_cache_file)
            if gk == graph_key:
                clusters_of_proteins = clu
        if clusters_of_proteins is None:
            community_detector = CommunityDetector(tfidf=tfidf, auroc_cut=auroc_cut)
            clusters_of_proteins = community_detector.cluster(model, graph)
            clusters_of_proteins = clusters_of_proteins["ok"]
            joblib.dump((graph_key, clusters_of_proteins), clusters_cache_file)
        
    col = cols[1]

    col.subheader(":robot_face: Quick modeling")

    only_one_option = False
    if len(clusters_of_proteins) == 1:
        if sorted(clusters_of_proteins[0]) == sorted(list(input_data["UniprotAC"])):
            only_one_option = True

    options = []
    if not only_one_option:
        for prot_clust in clusters_of_proteins:
            options += [", ".join(sorted([pid2name[pid] for pid in prot_clust]))]
        options += ["Full set of proteins"]
    else:
        if len(clusters_of_proteins[0]) == 1:
            options = [pid2name[clusters_of_proteins[0][0]]]
        else:
            options = ["Full set of proteins"]
    
    selected_cluster = col.radio(
        "These are some suggested groups of proteins for modeling",
        options=options,
    )

    if selected_cluster == "Full set of proteins":
        selected_cluster = uniprot_inputs
    else:
        selected_cluster = [name2pid[n] for n in selected_cluster.split(", ")]

    # default_max_hit_fragments = get_default_max_hit_fragments(selected_cluster)
    # default_max_prom_fragments = get_default_max_prom_fragments(selected_cluster)
    default_max_hit_fragments = 10
    default_max_fragment_prom = 500

    max_hit_fragments = col.slider(
        "Maximum number of positives",
        min_value=10,
        max_value=200,
        step=10,
        value=default_max_hit_fragments,
        help="Fragments will be ranked by specificity, i.e. by ascending value of promiscuity.",
    )

    max_fragment_prom = col.slider(
        "Maximum promiscuity of included fragments",
        min_value=50,
        max_value=500,
        step=10,
        value=default_max_fragment_prom,
        help="Maximum number of proteins for included fragments.",
    )

    uniprot_acs = list(selected_cluster)

    data = HitSelectorByOverlap(uniprot_acs=uniprot_acs, tfidf=tfidf).select(max_hit_fragments=max_hit_fragments, max_fragment_promiscuity=max_fragment_prom)

    num_positives = len(data[data["y"] == 1])
    num_total = len(data[data["y"] != -1])

    subcols = col.columns(3)
    subcols[0].metric(
        "Positives", value=num_positives
    )

    subcols[1].metric("Total", value=num_total)

    subcols[2].metric("Rate", value="{0:.1f}%".format(num_positives/num_total*100))

    if num_positives == 0:
        col.error(
            "No positives available. We cannot build a model with no positive data."
        )
        is_fitted = False

    else:
        task_evaluation = task_evaluator(model, data)
        subcols[0].metric(label="Corr. other", value="{0:.3f}".format(task_evaluation["ref_rho"]))
        subcols[1].metric(label="Frag. promiscuity", value="{0:.1f}".format(task_evaluation["prom"]))
        subcols[2].metric(label="Interactors ({0})".format(len(uniprot_acs)), value="{0:.1f}".format(task_evaluation["hits"]))

        expander = col.expander("View positives")
        positives_data = data[data["y"] == 1]
        pos_fids = sorted(positives_data["fid"])
        pos_smis = [fid2smi[fid] for fid in pos_fids]
        expander.dataframe(
            pd.DataFrame({"FragmentID": pos_fids, "SMILES": pos_smis}), hide_index=True
        )
        expander = col.expander("View negatives")
        negatives_data = data[data["y"] == 0]
        neg_fids = sorted(negatives_data["fid"])
        neg_smis = [fid2smi[fid] for fid in neg_fids]
        expander.dataframe(
            pd.DataFrame({"FragmentID": neg_fids, "SMILES": neg_smis}), hide_index=True
        )

        if num_positives < 5:
            col.warning("Not enough data to estimate AUROC.")

        else:
            auroc = task_evaluation["auroc"]
            col.metric(
                "AUROC estimation", value="{0:.3f} ± {1:.3f}".format(auroc[0], auroc[1])
            )
            subcols = col.columns(3)

        model.fit(data["y"])
        is_fitted = True

    if is_fitted:
        col = cols[2]

        col.subheader(":crystal_ball: Make predictions")

        input_prediction_tokens = col.text_area(
            label="Input your SMILES of interest. They should have the diazirine fragment"
        )

        pred_tokens = [t for t in input_prediction_tokens.split("\n") if t != ""]

        smiles_list = []
        for token in pred_tokens:
            if not is_valid_smiles(token):
                continue
            smi = token
            if not has_crf(Chem.MolFromSmiles(smi)):
                continue
            smiles_list += [smi]

        if len(smiles_list) == 0:
            has_prediction_input = False
            if len(pred_tokens) > 0:
                col.warning(
                    "No valid inputs were found. Please make sure the CRF pattern is present: {0}".format(
                        CRF_PATTERN
                    )
                )
        else:
            has_prediction_input = True

        if has_prediction_input:
            col.info(
                "{0} out of {1} input molecules are valid".format(
                    len(smiles_list), len(pred_tokens)
                )
            )
            do_tau = False
            if do_tau:
                y_hat, tau_ref, tau_train = model.predict_proba_and_tau(smiles_list)
                dr = pd.DataFrame(
                    {
                        "SMILES": smiles_list,
                        "Score": y_hat,
                        "Tau": tau_ref,
                        "TauTrain": tau_train,
                    }
                )
                for v in dr.values:
                    expander = col.expander(
                        "Score: `{0:.3f}` | Tau: `{1:.2f}` | Tau Train: `{2:.2f}`| SMILES: `{3}`".format(
                            v[1], v[2], v[3], v[0]
                        )
                    )
                    expander.image(get_fragment_image(v[0]))
            else:
                y_hat = model.predict_proba(smiles_list)[:, 1]
                dr = pd.DataFrame({"SMILES": smiles_list, "Score": y_hat})
                for v in dr.values:
                    expander = col.expander(
                        "Score: `{0:.3f}` | SMILES: `{1}`".format(v[1], v[0])
                    )
                    expander.image(get_fragment_image(v[0]))
