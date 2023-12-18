import sys
import os
import joblib
import numpy as np
from tqdm import tqdm

root = os.path.dirname(os.path.abspath(__file__))

sys.path.append(os.path.join(root, "..", "src"))

from model import precalc_embeddings, precalc_embeddings_reference, pid_prom
from model import OnTheFlyModel

mdl = OnTheFlyModel()

results = []
for uniprot_ac in tqdm(list(pid_prom.keys())):
    data = mdl.prepare_classification([uniprot_ac])
    auroc = mdl.estimate_performance(data["y"], baseline=True)
    mdl.fit(data["y"])
    y_hat_train = [
        float(x) for x in mdl.classifier.predict_proba(precalc_embeddings)[:, 1]
    ]
    y_hat_ref = None
    # y_hat_ref = list(mdl.classifier.predict_proba(precalc_embeddings_reference)[:,1])
    result = {
        "uniprot_ac": uniprot_ac,
        "auroc": auroc,
        "n_pos": int(np.sum(data["y"])),
        "y_hat_train": y_hat_train,
        "y_hat_ref": y_hat_ref,
    }
    results += [result]

file_name = os.path.join(root, "..", "data", "protein_precalcs.joblib")
joblib.dump(results, file_name)
