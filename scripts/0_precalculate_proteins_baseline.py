import sys
import os
import joblib
import numpy as np
from tqdm import tqdm
import copy

root = os.path.dirname(os.path.abspath(__file__))

sys.path.append(os.path.join(root, "..", "src"))

from model import precalc_embeddings, precalc_embeddings_reference, pid_prom
from model import OnTheFlyModel, HitSelector

mdl = OnTheFlyModel()

results = []
full_auc_count = 0
for uniprot_ac in tqdm(list(pid_prom.keys())):
    hit_selector = HitSelector(uniprot_acs=[uniprot_ac])
    data = hit_selector.select(max_hit_fragments=200)
    auroc_baseline = (None, None)
    auroc = mdl.estimate_performance(data["y"], baseline=True, n_splits=10)
    mdl.baseline_classifier.fit(precalc_embeddings, data["y"])
    y_hat_train = [
        float(x)
        for x in mdl.baseline_classifier.predict_proba(precalc_embeddings)[:, 1]
    ]
    y_hat_ref = [
        float(x)
        for x in mdl.baseline_classifier.predict_proba(precalc_embeddings_reference)[
            :, 1
        ]
    ]
    result = {
        "uniprot_ac": uniprot_ac,
        "n_pos": int(np.sum(data["y"])),
        "auroc": auroc,
        "y_hat_train": y_hat_train,
        "y_hat_ref": y_hat_ref,
    }
    results += [result]

file_name = os.path.join(root, "..", "data", "protein_precalcs_baseline.joblib")
joblib.dump(results, file_name)

protein_precalcs_baseline = joblib.load(
    os.path.join(root, "..", "data", "protein_precalcs_baseline.joblib")
)

proteome_reference_predictions_file = os.path.join(
    root, "..", "data", "proteome_reference_predictions_07.joblib"
)

proteome_reference_predictions = []
for r in protein_precalcs_baseline:
    auroc = r["auroc"][0]
    if auroc is None:
        continue
    if auroc < 0.7:
        continue
    proteome_reference_predictions += [r["y_hat_ref"]]
proteome_reference_predictions = np.array(proteome_reference_predictions).T

joblib.dump(proteome_reference_predictions, proteome_reference_predictions_file)
