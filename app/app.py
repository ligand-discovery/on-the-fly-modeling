import os
import streamlit as st
import pandas as pd
import joblib
st.set_page_config(layout="wide")

root = os.path.dirname(os.path.abspath(__file__))

df = None

@st.cache_data()
def load_hits():
    hits, fid_prom, pid_prom = joblib.load(os.path.join(root, "..", "data", "hits.joblib"))
    return hits, fid_prom, pid_prom

@st.cache_data()
def pid2name_mapper():
    df = pd.read_csv(os.path.join(root, "..", "data/pid2name_primary.tsv"), sep="\t", header=None)
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
    df = pd.DataFrame(R, columns=["UniprotAC", "Gene Name", "Fragment Hits"]).drop_duplicates().reset_index(drop=True)
    return df

hits, fid_prom, pid_prom = load_hits()
pid2name, name2pid, any2pid = pid2name_mapper()

st.title("Ligand Discovery On-the-Fly Modeling")
st.write("Welcome to the Ligand Discovery On-the-Fly Modeling tool! Select your proteins of interest and we'll build a quick ML model.")

cols = st.columns([1,1,1])

col = cols[0]

col.multiselect(label="Select proteins from the Ligand Discovery primary screening dataset", options=[])

col = cols[1]

text = col.text_area("Paste proteins separated by space or new line")
input_tokens = text.split()
input_pids = []
for it in input_tokens:
    if it in any2pid:
        pid = any2pid[it]
        if pid in pid_prom:
            input_pids += [any2pid[it]]

input_data = pids_to_dataframe(input_pids)
        
st.write(input_data)

col = cols[2]

uploaded_file = col.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    

if df is not None:
    st.write(df)


col = cols[1]



col = cols[2]


