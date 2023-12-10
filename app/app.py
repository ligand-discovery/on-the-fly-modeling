import os
import sys
import streamlit as st
import pandas as pd
import joblib
import numpy as np
from rdkit import Chem

st.set_page_config(layout="wide")

root = os.path.dirname(os.path.abspath(__file__))

sys.path.append(os.path.join(root, "..", "src"))
from model import OnTheFlyModel

df = None

# Functions and variables

CRF_PATTERN = "CC1(CCC#C)N=N1"
CRF_PATTERN_0 = "C#CC"
CRF_PATTERN_1 = "N=N"


@st.cache_data()
def load_hits():
    hits, fid_prom, pid_prom = joblib.load(
        os.path.join(root, "..", "data", "hits.joblib")
    )
    return hits, fid_prom, pid_prom


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


# App code

hits, fid_prom, pid_prom = load_hits()
pid2name, name2pid, any2pid = pid2name_mapper()

st.title("On-the-Fly Modeling from Ligand Discovery")
st.write(
    "Welcome to the On-the-Fly Modeling tool! Select your proteins of interest and we'll build a quick ML model."
)

cols = st.columns([1, 1, 1])

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

    col.dataframe(input_data)

    col = cols[1]

    col.subheader(":robot_face: Quick modeling")

    type_of_modeling = col.radio(
        "What king of modeling do yo want?", options=["Aggregated", "Any"]
    )

    if type_of_modeling == "Aggregated":
        aggregate = True
    else:
        aggregate = False

    model = OnTheFlyModel()
    uniprot_acs = list(input_data["UniprotAC"])
    model = OnTheFlyModel()
    is_fitted = False
    data = model.prepare_classification(uniprot_acs)
    auroc = model.estimate_performance(data["y"])

    col.metric("Number of positives", value=np.sum(data["y"]))
    col.metric(
        "Quick AUROC estimation", value="{0:.3f} ± {1:.3f}".format(auroc[0], auroc[1])
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

        valid_smiles = []
        for token in pred_tokens:
            if not is_valid_smiles(token):
                continue
            smi = token
            if not has_crf(Chem.MolFromSmiles(smi)):
                continue
            valid_smiles = smi

        if len(valid_smiles) == 0:
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
            model.predict(valid_smiles)
