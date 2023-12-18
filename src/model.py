import os
import pandas as pd
import joblib
import numpy as np
import collections
from scipy.stats import median_abs_deviation
from scipy.stats import rankdata
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score
from sklearn.naive_bayes import GaussianNB as BaselineClassifier
from tabpfn import TabPFNClassifier
from lol import LOL

from fragmentembedding import FragmentEmbedder

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")

fragment_embedder = FragmentEmbedder()
fids, _, precalc_embeddings = joblib.load(os.path.join(DATA_PATH, "cemm_emb.joblib"))

catalog_ids, _, precalc_embeddings_reference = joblib.load(
    os.path.join(DATA_PATH, "enamine_stock_emb.joblib")
)
hits, fid_prom, pid_prom = joblib.load(os.path.join(DATA_PATH, "hits.joblib"))


tabpfn_model = TabPFNClassifier(device="cpu", N_ensemble_configurations=32)


class LigandDiscoveryBaselineClassifier(object):
    def __init__(self):
        self.model = BaselineClassifier()

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict_proba(self, X):
        y_hat = self.model.predict_proba(X)
        return y_hat

    def predict(self, X):
        y_hat = self.model.predict(X)
        return y_hat


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


class HitSelector(object):
    def __init__(self, uniprot_acs):
        self._valid_prots = set(pid_prom.keys())
        self.uniprot_acs = [pid for pid in uniprot_acs if pid in self._valid_prots]
        self.fids = fids
        self.pids = self.uniprot_acs
        self._fid_prom = []
        for fid in self.fids:
            if fid in fid_prom:
                self._fid_prom += [fid_prom[fid]]
            else:
                self._fid_prom += [0]

    def select(self, min_prop_hit_proteins=0, max_hit_fragments=100):
        min_hit_proteins = int(min_prop_hit_proteins * len(self.uniprot_acs))
        print(min_hit_proteins)
        my_hits_dict = collections.defaultdict(int)
        pids_set = set(self.uniprot_acs)
        for k, _ in hits.items():
            if k[0] in pids_set:
                my_hits_dict[k[1]] += 1
        my_hits = []
        for fid in self.fids:
            if fid in my_hits_dict:
                my_hits += [my_hits_dict[fid]]
            else:
                my_hits += [0]
        y = []
        for h in my_hits:
            if h == 0:
                y += [0]
            else:
                if h >= min_hit_proteins:
                    y += [1]
                else:
                    y += [0]
        data = {"fid": self.fids, "prom": self._fid_prom, "hits": my_hits, "y": y}
        if np.sum(data["y"]) > max_hit_fragments:
            promiscuity_ranks = rankdata(data["prom"], method="ordinal")
            promiscuity_ranks = promiscuity_ranks / np.max(promiscuity_ranks)
            data_ = pd.DataFrame(data)
            data_["ranks"] = promiscuity_ranks
            data_ = data_.sort_values(by="ranks", ascending=True)
            data_1 = data_[data_["y"] == 1]
            data_1 = data_1.head(max_hit_fragments)
            fids_1 = set(data_1["fid"])
            y = []
            for fid in data["fid"]:
                if fid in fids_1:
                    y += [1]
                else:
                    y += [0]
            data["y"] = y
        data = pd.DataFrame(data)
        return data


class OnTheFlyModel(object):
    def __init__(self):
        self.fragment_embedder = fragment_embedder
        self.precalc_embeddings = precalc_embeddings
        self.precalc_embeddings_reference = precalc_embeddings_reference
        self.baseline_classifier = LigandDiscoveryBaselineClassifier()
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

    def _calculate_percentiles(self, y, yref):
        sorted_yref = np.sort(yref)
        percentiles = [
            np.searchsorted(sorted_yref, yi, side="right") / len(yref) for yi in y
        ]
        return percentiles

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

    def estimate_performance(self, y, baseline=True):
        try:
            y = np.array(y)
            if np.sum(y) < 2:
                return None, None
            skf = StratifiedShuffleSplit(n_splits=10, test_size=0.2)
            aurocs = []
            for train_idx, test_idx in skf.split(self.precalc_embeddings, y):
                X_train = self.precalc_embeddings[train_idx]
                X_test = self.precalc_embeddings[test_idx]
                y_train = y[train_idx]
                y_test = y[test_idx]
                if baseline:
                    self.baseline_classifier.fit(X_train, y_train)
                    y_hat = self.baseline_classifier.predict_proba(X_test)[:, 1]
                else:
                    self.classifier.fit(X_train, y_train)
                    y_hat = self.classifier.predict_proba(X_test)[:, 1]
                auroc = roc_auc_score(y_test, y_hat)
                print(auroc)
                aurocs += [auroc]
            return np.median(aurocs), median_abs_deviation(aurocs)
        except:
            return None, None

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

    def predict_proba_and_tau(self, smiles_list):
        X = fragment_embedder.transform(smiles_list)
        y_hat = self.classifier.predict_proba(X)[:, 1]
        sample_indices = np.random.choice(
            self.precalc_embeddings_reference.shape[0], size=1000, replace=False
        )
        reference_y_hat = self.classifier.predict_proba(
            self.precalc_embeddings_reference
        )[sample_indices, 1]
        train_y_hat = self.classifier.predict_proba(self.precalc_embeddings)[:, 1]
        tau_ref = self._calculate_percentiles(y_hat, reference_y_hat)
        tau_train = self._calculate_percentiles(y_hat, train_y_hat)
        return y_hat, tau_ref, tau_train
