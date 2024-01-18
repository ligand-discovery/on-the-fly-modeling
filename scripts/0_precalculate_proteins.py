import sys
import os
import joblib
import numpy as np
from tqdm import tqdm

root = os.path.dirname(os.path.abspath(__file__))

sys.path.append(os.path.join(root, "..", "src"))

from model import precalc_embeddings, pid_prom
from model import OnTheFlyModel, HitSelector

mdl = OnTheFlyModel()

results = []
full_auc_count = 0
for uniprot_ac in tqdm(list(pid_prom.keys())):
    hit_selector = HitSelector(uniprot_acs=[uniprot_ac])
    data = hit_selector.select(max_hit_fragments=200)
    auroc_baseline_10 = mdl.estimate_performance(data["y"], baseline=True, n_splits=10)
    auroc_baseline = (None, None)
    auroc = (None, None)
    mdl.fit(data["y"])
    y_hat_train = [
        float(x) for x in mdl.classifier.predict_proba(precalc_embeddings)[:, 1]
    ]
    result = {
        "uniprot_ac": uniprot_ac,
        "auroc": auroc,
        "auroc_baseline": auroc_baseline,
        "auroc_baseline_10": auroc_baseline_10,
        "n_pos": int(np.sum(data["y"])),
        "y_hat_train": y_hat_train,
        "y_hat_ref": None,
    }
    results += [result]

file_name = os.path.join(root, "..", "data", "protein_precalcs.joblib")
joblib.dump(results, file_name)
