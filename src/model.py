import os
import pandas as pd
import joblib
import numpy as np
import random
import collections
from scipy.stats import median_abs_deviation
from scipy.stats import rankdata
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score
from sklearn.naive_bayes import GaussianNB as BaselineClassifier
from sklearn.neighbors import NearestNeighbors
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


class BinaryBalancer(object):
    def __init__(self, proportion=0.3, n_samples=1000, smote=True):
        self.proportion = proportion
        self.n_samples = n_samples
        self.smote = smote

    def _resample(self, X, size, weights):
        idxs = [i for i in range(X.shape[0])]
        sampled_idxs = np.random.choice(idxs, size=(size - X.shape[0]), replace=True, p=weights)
        X_s = X[sampled_idxs]
        if self.smote:
            nn = NearestNeighbors(n_neighbors=4)
            nn.fit(X)
            neighs = nn.kneighbors(X_s, return_distance=False)[:, 1:]
            R = []
            w = np.array([0.75, 0.5, 0.25])
            w /= w.sum()
            for i in range(X_s.shape[0]):
                gap = random.random()
                j = int(np.random.choice([0, 1, 2], p=w))
                neigh_idx = neighs[i,j]
                d = X[neigh_idx] - X_s[i]
                R += [X_s[i] + gap*d]
            X_s = np.array(R)
        X = np.vstack([X, X_s])
        return X

    def transform(self, X, y, sample_weights=None):
        X = np.array(X)
        y = np.array(y)
        X_0 = X[y == 0]
        X_1 = X[y == 1]
        num_0_samples = int(self.n_samples*(1 - self.proportion))
        num_1_samples = int(self.n_samples*self.proportion)
        if sample_weights is None:
            sample_weights = np.array([1.]*X.shape[0])
        else:
            sample_weights = np.array(sample_weights)
        weights_0 = sample_weights[y == 0]
        weights_1 = sample_weights[y == 1]
        weights_0 = weights_0 / weights_0.sum()
        weights_1 = weights_1 / weights_1.sum()
        X_0 = self._resample(X_0, num_0_samples, weights_0)
        X_1 = self._resample(X_1, num_1_samples, weights_1)
        X = np.vstack([X_0, X_1])
        y = np.array([0]*X_0.shape[0] + [1]*X_1.shape[0])
        idxs = [i for i in range(len(y))]
        random.shuffle(idxs)
        X = X[idxs]
        y = y[idxs]
        return X, y


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


class BaselineClassifierReducer(object):
    def __init__(self):
        self.top_cuts = [50, 100, 250, 500]

    def fit(self, X, y, promiscuity_counts):
        self.baseline_classifiers = [LigandDiscoveryBaselineClassifier() for _ in range(len(self.top_cuts))]
        for i, top_cut in enumerate(self.top_cuts):
            idxs = []
            for j, pc in enumerate(promiscuity_counts):
                if pc > top_cut:
                    continue
                idxs += [j]
            X_ = X[idxs]
            y_ = y[idxs]
            print(np.sum(y_), len(idxs))
            if np.sum(y_) > 0:
                self.baseline_classifiers[i].fit(X_, y_)
            else:
                self.baseline_classifiers[i] = None

    def transform(self, X):
        R = []
        for model in self.baseline_classifiers:
            if model is None:
                y_hat = [0]*X.shape[0]
            else:
                y_hat = list(model.predict_proba(X)[:,1])
            R += [y_hat]
        X = np.array(R).T
        print(X.shape)
        return X
    
    def fit_transform(self, X, y, promiscuity_counts):
        self.fit(X, y, promiscuity_counts)
        return self.transform(X)


class LigandDiscoveryClassifier(object):
    def __init__(self):
        tabpfn_model.remove_models_from_memory()
        self.reducer_0 = LOL(n_components=5)
        self.reducer_1 = BaselineClassifierReducer()
        self.balancer = BinaryBalancer(0.5)

    def fit(self, X, y, promiscuity_counts):
        X_0 = self.reducer_0.fit_transform(X, y)
        X_1 = self.reducer_1.fit_transform(X, y, promiscuity_counts)
        X = np.hstack([X_0, X_1])
        X, y = self.balancer.transform(X, y)
        tabpfn_model.fit(X, y)

    def predict_proba(self, X):
        X_0 = self.reducer_0.transform(X)
        X_1 = self.reducer_1.transform(X)
        X = np.hstack([X_0, X_1])
        y_hat = tabpfn_model.predict_proba(X)
        return y_hat

    def predict(self, X):
        X_0 = self.reducer_0.transform(X)
        X_1 = self.reducer_1.transform(X)
        X = np.hstack([X_0, X_1])
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

    def estimate_performance(self, y, baseline=True):
        try:
            y = np.array(y)
            promiscuity_counts = np.array(self._fid_prom)
            if np.sum(y) < 2:
                print("Not enough positives data")
                return None, None
            skf = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
            aurocs = []
            print(aurocs)
            for train_idx, test_idx in skf.split(self.precalc_embeddings, y):
                X_train = self.precalc_embeddings[train_idx]
                X_test = self.precalc_embeddings[test_idx]
                prom_counts_train = promiscuity_counts[train_idx]
                y_train = y[train_idx]
                y_test = y[test_idx]
                if baseline:
                    self.baseline_classifier.fit(X_train, y_train)
                    y_hat = self.baseline_classifier.predict_proba(X_test)[:, 1]
                else:
                    self.classifier.fit(X_train, y_train, prom_counts_train)
                    y_hat = self.classifier.predict_proba(X_test)[:, 1]
                auroc = roc_auc_score(y_test, y_hat)
                print(auroc)
                aurocs += [auroc]
            return np.median(aurocs), median_abs_deviation(aurocs)
        except Exception as e:
            print("AUROC estimation went wrong", e)
            return None, None

    def estimate_performance_on_train(self, y, baseline=True):
        y = np.array(y)
        X = self.precalc_embeddings
        if baseline:
            self.baseline_classifier.fit(X, y)
            y_hat = self.baseline_classifier.predict_proba(X)[:, 1]
        else:
            y_hat = self.classifier.fit(X, y, self._fid_prom)
            y_hat = self.classifier.predict_proba(X)[:, 1]
        auroc = roc_auc_score(y, y_hat)
        return auroc

    def fit(self, y):
        y = np.array(y)
        self.classifier.fit(self.precalc_embeddings, y, self._fid_prom)

    def predict_proba(self, smiles_list):
        X = fragment_embedder.transform(smiles_list)
        y_hat = self.classifier.predict_proba(X)
        return y_hat

    def predict(self, smiles_list):
        X = fragment_embedder.transform(smiles_list)
        y_hat = self.classifier.predict(X)
        return y_hat
    
    def predict_proba_on_train(self):
        y_hat = self.classifier.predict_proba(self.precalc_embeddings)
        return y_hat
    
    def predict_on_train(self):
        y_hat = self.classifier.predict(self.precalc_embeddings)
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
