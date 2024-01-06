import joblib
import pandas as pd
import os
import sys
import numpy as np

root = os.path.dirname(os.path.abspath(__file__))

sys.path.append(os.path.join(root, "..", "src"))
from model import OnTheFlyModel, HitSelector

print("Loading protein precalcs")
protein_precalcs = joblib.load(os.path.join(root, "..", "data", "protein_precalcs.joblib"))

print("Getting list of SLCs")
slcs_all = []
with open(os.path.join(root, "..", "data", "examples", "slc_cemm_interest.txt"), "r") as f:
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
df = pd.read_csv(os.path.join(root, "..", "data", "experimental", "validation_hits_with_slc.tsv"), sep="\t")

model = OnTheFlyModel()
R = []
for slc in slcs:
    hit_selector = HitSelector([slc])
    data = hit_selector.select(max_hit_fragments=200)
    model.fit(data["y"])
    y_hat = model.predict_proba(list(df["smiles"]))[:,1]
    R += [list(y_hat)]

R = np.array(R).T

dr = pd.DataFrame(R, columns=slcs)
df = pd.concat([df, dr], axis=1)

df.to_csv(os.path.join(root, "..", "results", "1_experimental_validation_predictions.tsv"), sep="\t")