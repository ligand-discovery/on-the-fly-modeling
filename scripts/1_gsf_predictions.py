import joblib
import pandas as pd
import os
import sys
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from tqdm import tqdm

root = os.path.dirname(os.path.abspath(__file__))

sys.path.append(os.path.join(root, "..", "src"))
from model import OnTheFlyModel, HitSelector, fragment_embedder

print("Loading protein precalcs")
protein_precalcs = joblib.load(
    os.path.join(root, "..", "data", "protein_precalcs.joblib")
)

print("Getting list of SLCs")
slcs_all = []
with open(
    os.path.join(root, "..", "data", "examples", "slc_cemm_interest.txt"), "r"
) as f:
    for l in f:
        slcs_all += [l.rstrip()]

print("Getting SLCs for modeling")
slcs = []
aurocs = []
for d in protein_precalcs:
    if d["uniprot_ac"] not in slcs_all:
        continue
    slcs += [d["uniprot_ac"]]
    aurocs += [d["auroc"]]

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

model = OnTheFlyModel()
R = []
slcs = slcs[68:]
for slc in tqdm(slcs):
    hit_selector = HitSelector([slc])
    data = hit_selector.select(max_hit_fragments=200)
    model.fit(data["y"])
    y_hat = model.classifier.predict_proba(X)[:, 1]
    R += [list(y_hat)]

R = np.array(R).T

dr = pd.DataFrame(R, columns=slcs)
df = pd.concat([df, dr], axis=1)

df.to_csv(
    os.path.join(root, "..", "results", "1_gsf_predictions.tsv"), sep="\t", index=False
)
