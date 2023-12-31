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

RDLogger.DisableLog("rdApp.*")

import networkx as nx

st.set_page_config(layout="wide")

root = os.path.dirname(os.path.abspath(__file__))

sys.path.append(os.path.join(root, "..", "src"))
from model import OnTheFlyModel, HitSelector

df = None

# Functions and variables

CRF_PATTERN = "CC1(CCC#C)N=N1"
CRF_PATTERN_0 = "C#CC"
CRF_PATTERN_1 = "N=N"

SIMILARITY_PROTEINS_CUT = 0.2


@st.cache_data()
def get_protein_similarity_graph():
    edgelist = pd.read_csv(os.path.join(root, "..", "data", "edgelist.csv"))
    G = nx.Graph()
    for v in edgelist.values:
        if v[-1] < SIMILARITY_PROTEINS_CUT:
            continue
        G.add_edge(v[0], v[1])
    return G


@st.cache_data()
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
proteins_graph = get_protein_similarity_graph()


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
    col.info(
        "{0} out of {1} input proteins were found in the Ligand Discovery interactome.".format(
            len(input_pids), len(input_tokens)
        )
    )

    col.dataframe(input_data, hide_index=True)

    col = cols[1]

    col.subheader(":robot_face: Quick modeling")

    clusters_of_proteins = get_clusters_for_modeling(list(input_data["UniprotAC"]))

    options = []
    for prot_clust in clusters_of_proteins:
        options += [", ".join(sorted([pid2name[pid] for pid in prot_clust]))]

    type_options = ["At least one", "At least half", "All"]
    type_of_prediction = col.radio(
        "Within a group, predict...", options=type_options, index=0, horizontal=True
    )
    type_proportions = [0, 0.5, 1]
    type_selected_proportion = type_proportions[type_options.index(type_of_prediction)]

    max_hit_fragments = col.slider(
        "Maximum number of hits per group",
        min_value=10,
        max_value=200,
        step=10,
        value=100,
        help="Fragments will be ranked by specificity, i.e. by ascending value of promiscuity.",
    )

    selected_cluster = col.radio(
        "These are some suggested groups of proteins for modeling",
        options=options,
    )

    selected_cluster = [name2pid[n] for n in selected_cluster.split(", ")]

    uniprot_acs = list(selected_cluster)
    model = OnTheFlyModel()
    is_fitted = False

    hit_selector = HitSelector(uniprot_acs=uniprot_acs)
    data = hit_selector.select(
        min_prop_hit_proteins=type_selected_proportion,
        max_hit_fragments=max_hit_fragments,
    )

    num_positives = np.sum(data["y"])

    col.metric(
        "Number of positives in the Ligand Discovery dataset", value=num_positives
    )

    if num_positives == 0:
        col.error(
            "No positives available. We cannot build a model with no positive data."
        )
        is_fitted = False

    else:
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
            baseline = col.checkbox(label="Fast baseline AUROC estimation", value=True)
            auroc = model.estimate_performance(data["y"], baseline)

            col.metric(
                "AUROC estimation", value="{0:.3f} ± {1:.3f}".format(auroc[0], auroc[1])
            )

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
            do_tau = col.checkbox("Calculate Tau (slower)", value=False)
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
