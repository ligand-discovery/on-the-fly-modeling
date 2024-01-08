import sys
import os
import joblib
import numpy as np
from tqdm import tqdm

root = os.path.dirname(os.path.abspath(__file__))

sys.path.append(os.path.join(root, "..", "src"))

from model import precalc_embeddings, precalc_embeddings_reference, pid_prom
from model import OnTheFlyModel, HitSelector

slcs = []
with open(os.path.join(root, "../data/examples/slc_cemm_interest.txt"), "r") as f:
    for l in f:
        slcs += [l.rstrip()]

mdl = OnTheFlyModel()

results = []
full_auc_count = 0
for uniprot_ac in tqdm(list(pid_prom.keys())):
    hit_selector = HitSelector(uniprot_acs=[uniprot_ac])
    data = hit_selector.select(max_hit_fragments=200)
    auroc_baseline_10 = mdl.estimate_performance(data["y"], baseline=True, n_splits=10)
    auroc_baseline = mdl.estimate_performance(data["y"], baseline=True, n_splits=3)
    if uniprot_ac in slcs:
        auroc = mdl.estimate_performance(data["y"], baseline=False, n_splits=3)
    else:
        auroc = (None, None)
    result = {
        "uniprot_ac": uniprot_ac,
        "auroc": auroc,
        "auroc_baseline": auroc_baseline,
        "auroc_baseline_10": auroc_baseline_10,
        "n_pos": int(np.sum(data["y"])),
    }
    results += [result]

file_name = os.path.join(root, "..", "data", "protein_precalcs_only_aucs.joblib")
joblib.dump(results, file_name)
