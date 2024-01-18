import sys
import os
import joblib
import numpy as np
from tqdm import tqdm
import pandas as pd
import collections

root = os.path.dirname(os.path.abspath(__file__))

sys.path.append(os.path.join(root, "..", "src"))

from model import precalc_embeddings, precalc_embeddings_reference, fids
from model import OnTheFlyModel

mdl = OnTheFlyModel()

df0 = pd.read_csv(os.path.join(root, "..", "data", "promiscuity_pxf_fxp.csv"))
df1 = pd.read_csv(os.path.join(root, "..", "data", "promiscuity_pxf.csv"))

tasks = {}

fids_ = list(df0["fid"])
cols_ = list(df0.columns)[2:]
for col in cols_:
    vals = list(df0[col])
    d = {}
    for i, v in enumerate(vals):
        d[fids_[i]] = v
    tasks[col] = d

fids_ = list(df1["fid"])
cols_ = list(df1.columns)[2:]
for col in cols_:
    vals = list(df1[col])
    d = {}
    for i, v in enumerate(vals):
        d[fids_[i]] = v
    tasks[col] = d

results = []
full_auc_count = 0
for task, d in tqdm(tasks.items()):
    y = [d[fid] for fid in fids]
    auroc_baseline = (None, None)
    auroc = mdl.estimate_performance(y, baseline=True, n_splits=10)
    mdl.baseline_classifier.fit(precalc_embeddings, y)
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
        "task": task,
        "auroc": auroc,
        "y_hat_train": y_hat_train,
        "y_hat_ref": y_hat_ref,
    }
    results += [result]

file_name = os.path.join(root, "..", "data", "promiscuity_precalcs_baseline.joblib")
joblib.dump(results, file_name)

promiscuity_precalcs_baseline = joblib.load(
    os.path.join(root, "..", "data", "promiscuity_precalcs_baseline.joblib")
)

promiscuity_reference_predictions_file = os.path.join(
    root, "..", "data", "promiscuity_reference_predictions_07.joblib"
)

promiscuity_reference_predictions = []
for r in promiscuity_precalcs_baseline:
    auroc = r["auroc"][0]
    if auroc is None:
        continue
    if auroc < 0.7:
        continue
    promiscuity_reference_predictions += [r["y_hat_ref"]]
promiscuity_reference_predictions = np.array(promiscuity_reference_predictions).T

joblib.dump(promiscuity_reference_predictions, promiscuity_reference_predictions_file)
