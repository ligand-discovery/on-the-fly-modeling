import os
import pandas as pd
import joblib
import numpy as np
import collections
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from tabpfn import TabPFNClassifier
from lol import LOL

from fragmentembedding import FragmentEmbedder

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")

fragment_embedder = FragmentEmbedder()
fids, _, precalc_embeddings = joblib.load(os.path.join(DATA_PATH, "cemm_emb.joblib"))
hits, fid_prom, pid_prom = joblib.load(os.path.join(DATA_PATH, "hits.joblib"))


tabpfn_model = TabPFNClassifier(N_ensemble_configurations=32)


class LigandDiscoveryClassifier(object):
    def __init__(self):
        tabpfn_model.remove_models_from_memory()

    def fit(self, X, y):
        self.reducer = LOL(n_components=100)
        X = self.reducer.fit_transform(X, y)
        tabpfn_model.fit(X, y)

    def predict_proba(self, X):
        X = self.reducer.transform(X)
        y_hat = tabpfn_model.predict_proba(X)
        return y_hat

    def predict(self, X):
        X = self.reducer.transform(X)
        y_hat = tabpfn_model.predict(X)
        return y_hat


class OnTheFlyModel(object):
    def __init__(self):
        self.fragment_embedder = fragment_embedder
        self.precalc_embeddings = precalc_embeddings
        self.baseline_classifier = GaussianNB()
        self.classifier = LigandDiscoveryClassifier()
        self.fids = fids
        self._valid_prots = set(pid_prom.keys())
        self._fid_prom = []
        for fid in self.fids:
            if fid in fid_prom:
                self._fid_prom += [fid_prom[fid]]
            else:
                self._fid_prom += [0]

    def _check_prots(self, uniprot_acs):
        for pid in uniprot_acs:
            if pid not in self._valid_prots:
                raise Exception(
                    "{0} protein is not amongst our screening hits".format(pid)
                )

    def prepare_classification(self, uniprot_acs):
        self._check_prots(uniprot_acs)
        my_hits_dict = collections.defaultdict(int)
        pids_set = set(uniprot_acs)
        for k, _ in hits.items():
            if k[0] in pids_set:
                my_hits_dict[k[1]] += 1
        my_hits = []
        for fid in fids:
            if fid in my_hits_dict:
                my_hits += [my_hits_dict[fid]]
            else:
                my_hits += [0]
        y = []
        for h in my_hits:
            if h > 0:
                y += [1]
            else:
                y += [0]
        data = {"fid": self.fids, "prom": self._fid_prom, "hits": my_hits, "y": y}
        data = pd.DataFrame(data)
        return data

    def estimate_performance(self, y):
        y = np.array(y)
        skf = StratifiedKFold(n_splits=5, shuffle=True)
        aurocs = []
        for train_idx, test_idx in skf.split(self.precalc_embeddings, y):
            X_train = self.precalc_embeddings[train_idx]
            X_test = self.precalc_embeddings[test_idx]
            y_train = y[train_idx]
            y_test = y[test_idx]
            self.baseline_classifier.fit(X_train, y_train)
            y_hat = self.baseline_classifier.predict_proba(X_test)[:, 1]
            auroc = roc_auc_score(y_test, y_hat)
            print(auroc)
            aurocs += [auroc]
        return np.mean(aurocs), np.std(aurocs)

    def fit(self, y):
        y = np.array(y)
        self.classifier.fit(self.precalc_embeddings, y)

    def predict_proba(self, smiles_list):
        X = fragment_embedder.transform(smiles_list)
        y_hat = self.classifier.predict_proba(X)
        return y_hat

    def predict(self, smiles_list):
        X = fragment_embedder.transform(smiles_list)
        y_hat = self.classifier.predict(X)
        return y_hat
