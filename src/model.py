import os
import pandas as pd
import joblib
import numpy as np
import random
import collections
from scipy.stats import median_abs_deviation
from scipy.stats import rankdata, pearsonr
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score
from sklearn.naive_bayes import GaussianNB as BaselineClassifier
from sklearn.neighbors import NearestNeighbors, KNeighborsRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from tabpfn import TabPFNClassifier
from lol import LOL
from community import community_louvain
from fragmentembedding import FragmentEmbedder

random.seed(42)
np.random.seed(42)

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")

promiscuity_reference_predictions_file = os.path.join(
    DATA_PATH, "promiscuity_reference_predictions_07.joblib"
)
promiscuity_reference_predictions = joblib.load(promiscuity_reference_predictions_file)

fragment_embedder = FragmentEmbedder()
fids, _, precalc_embeddings = joblib.load(os.path.join(DATA_PATH, "cemm_emb.joblib"))

catalog_ids, _, precalc_embeddings_reference = joblib.load(
    os.path.join(DATA_PATH, "enamine_stock_emb.joblib")
)
hits, fid_prom, pid_prom = joblib.load(os.path.join(DATA_PATH, "hits.joblib"))


tabpfn_model = TabPFNClassifier(device="cpu", N_ensemble_configurations=32)


class BinaryBalancer(object):
    def __init__(self, proportion=0.5, n_samples=1000, smote=False):
        self.proportion = proportion
        self.n_samples = n_samples
        self.smote = smote

    def _resample(self, X, size, weights):
        idxs = [i for i in range(X.shape[0])]
        sampled_idxs = np.random.choice(
            idxs, size=(size - X.shape[0]), replace=True, p=weights
        )
        X_s = X[sampled_idxs]
        if self.smote:
            n_neighbors = min(X.shape[0], 4)
            nn = NearestNeighbors(n_neighbors=n_neighbors)
            nn.fit(X)
            neighs = nn.kneighbors(X_s, return_distance=False)[:, 1:]
            R = []
            w = np.array([0.75, 0.5, 0.25])
            w = w[: neighs.shape[1]]
            idxs_to_sample = [i for i in range(neighs.shape[1])]
            w /= w.sum()
            for i in range(X_s.shape[0]):
                if len(idxs_to_sample) == 0:
                    R += [X_s[i]]
                else:
                    gap = random.random()
                    j = int(np.random.choice(idxs_to_sample, p=w))
                    neigh_idx = neighs[i, j]
                    d = X[neigh_idx] - X_s[i]
                    R += [X_s[i] + gap * d]
            X_s = np.array(R)
        X = np.vstack([X, X_s])
        return X

    def transform(self, X, y, promiscuity_counts):
        if promiscuity_counts is None:
            sample_weights = None
        else:
            promiscuity_counts = np.clip(promiscuity_counts, 10, 500)
            sample_weights = [1 / p for p in promiscuity_counts]
        X = np.array(X)
        y = np.array(y)
        X_0 = X[y == 0]
        X_1 = X[y == 1]
        num_0_samples = int(self.n_samples * (1 - self.proportion))
        num_1_samples = int(self.n_samples * self.proportion)
        if sample_weights is None:
            sample_weights = np.array([1.0] * X.shape[0])
        else:
            sample_weights = np.array(sample_weights)
        weights_0 = sample_weights[y == 0]
        weights_1 = sample_weights[y == 1]
        weights_0 = weights_0 / weights_0.sum()
        weights_1 = weights_1 / weights_1.sum()
        X_0 = self._resample(X_0, num_0_samples, weights_0)
        X_1 = self._resample(X_1, num_1_samples, weights_1)
        X = np.vstack([X_0, X_1])
        y = np.array([0] * X_0.shape[0] + [1] * X_1.shape[0])
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
        self.baseline_classifiers = [
            LigandDiscoveryBaselineClassifier() for _ in range(len(self.top_cuts))
        ]
        for i, top_cut in enumerate(self.top_cuts):
            idxs = []
            for j, pc in enumerate(promiscuity_counts):
                if pc > top_cut:
                    continue
                idxs += [j]
            X_ = X[idxs]
            y_ = y[idxs]
            if np.sum(y_) > 0:
                self.baseline_classifiers[i].fit(X_, y_)
            else:
                self.baseline_classifiers[i] = None
        R = []
        for model in self.baseline_classifiers:
            if model is None:
                y_hat = [0] * precalc_embeddings_reference.shape[0]
            else:
                y_hat = list(model.predict_proba(precalc_embeddings_reference)[:, 1])
            R += [y_hat]
        _X_transformed_reference = np.array(R).T
        self._kneigh_regressor = KNeighborsRegressor(n_neighbors=1)
        self._kneigh_regressor.fit(
            precalc_embeddings_reference, _X_transformed_reference
        )

    def transform(self, X):
        X = self._kneigh_regressor.predict(X)
        return X

    def fit_transform(self, X, y, promiscuity_counts):
        self.fit(X, y, promiscuity_counts)
        return self.transform(X)


class LigandDiscoveryClassifier(object):
    def __init__(self):
        tabpfn_model.remove_models_from_memory()

    def fit(self, X, y, promiscuity_counts):
        n_components = int(min(np.sum(y), 100))
        self.reducer = LOL(n_components=n_components)
        self.balancer = BinaryBalancer(0.5)
        X = self.reducer.fit_transform(X, y)
        X, y = self.balancer.transform(X, y, promiscuity_counts=promiscuity_counts)
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


class HitSelectorByOverlap(object):
    def __init__(self, uniprot_acs, tfidf):
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
        self.tfidf = tfidf
        self.fid2pid = collections.defaultdict(list)
        for k, v in hits.items():
            self.fid2pid[k[1]] += [(k[0], v)]

    def select_without_tfidf(self, max_hit_fragments, max_fragment_promiscuity):
        protein_overlaps = []
        my_hits = []
        for i, fid in enumerate(self.fids):
            prom = self._fid_prom[i]
            if prom > max_fragment_promiscuity:
                my_hits += [-1]
                protein_overlaps += [-1]
            else:
                if fid in self.fid2pid:
                    all_prots = [x[0] for x in self.fid2pid[fid]]
                    sel_prots = list(set(self.uniprot_acs).intersection(all_prots))
                    my_hits += [len(sel_prots)]
                    protein_overlaps += [len(sel_prots) / len(all_prots)]
                else:
                    my_hits += [0]
                    protein_overlaps += [0]
        y = [0] * len(fids)
        idxs = np.argsort(protein_overlaps)[::-1]
        idxs = idxs[:max_hit_fragments]
        for idx in idxs:
            if protein_overlaps[idx] == 0:
                continue
            y[idx] = 1
        y = np.array(y)
        protein_overlaps = np.array(protein_overlaps)
        y[protein_overlaps == -1] = -1
        data = {"fid": self.fids, "prom": self._fid_prom, "hits": my_hits, "y": list(y)}
        return pd.DataFrame(data)

    def select_with_tfidf(self, max_hit_fragments, max_fragment_promiscuity):
        corpus = []
        my_hits = []
        for i, fid in enumerate(self.fids):
            prom = self._fid_prom[i]
            if prom > max_fragment_promiscuity:
                my_hits += [-1]
                corpus += [""]
            else:
                if fid in self.fid2pid:
                    all_prots = [x[0] for x in self.fid2pid[fid]]
                    sel_prots = list(set(self.uniprot_acs).intersection(all_prots))
                    my_hits += [len(sel_prots)]
                    corpus += [
                        " ".join(
                            [x[0] for x in self.fid2pid[fid] for _ in range(int(x[1]))]
                        )
                    ]
                else:
                    my_hits += [0]
                    corpus += [""]
        vectorizer = TfidfVectorizer(min_df=1, lowercase=False)
        tfidf_matrix = vectorizer.fit_transform(corpus).toarray()
        feature_names = vectorizer.get_feature_names_out()
        idxs = []
        for i, n in enumerate(feature_names):
            if n in self.uniprot_acs:
                idxs += [i]
        all_vals = np.sum(tfidf_matrix, axis=1)
        sel_vals = np.sum(tfidf_matrix[:, idxs], axis=1)
        prop_vals = []
        for s, a in zip(sel_vals, all_vals):
            if a == 0:
                prop_vals += [0]
            else:
                prop_vals += [s / a]
        y = [0] * len(fids)
        idxs = np.argsort(prop_vals)[::-1]
        idxs = idxs[:max_hit_fragments]
        for idx in idxs:
            if prop_vals[idx] == 0:
                continue
            y[idx] = 1
        y = np.array(y)
        my_hits = np.array(my_hits)
        y[my_hits == -1] = -1
        data = {
            "fid": self.fids,
            "prom": self._fid_prom,
            "hits": list(my_hits),
            "y": list(y),
        }
        return pd.DataFrame(data)

    def select(self, max_hit_fragments, max_fragment_promiscuity):
        if self.tfidf:
            return self.select_with_tfidf(
                max_hit_fragments=max_hit_fragments,
                max_fragment_promiscuity=max_fragment_promiscuity,
            )
        else:
            return self.select_without_tfidf(
                max_hit_fragments=max_hit_fragments,
                max_fragment_promiscuity=max_fragment_promiscuity,
            )


class OnTheFlyModel(object):
    def __init__(self, verbose=False):
        self.verbose = verbose
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

    def estimate_performance(self, y, baseline=True, n_splits=10):
        try:
            promiscuity_counts = np.array(self._fid_prom)
            y = np.array(y)
            mask = y != -1
            precalc_embeddings = self.precalc_embeddings[mask]
            y = y[mask]
            if np.sum(y) < 2:
                if self.verbose:
                    print("Not enough positives data")
                return None, None
            skf = StratifiedShuffleSplit(
                n_splits=n_splits, test_size=0.2, random_state=42
            )
            aurocs = []
            for train_idx, test_idx in skf.split(precalc_embeddings, y):
                X_train = precalc_embeddings[train_idx]
                X_test = precalc_embeddings[test_idx]
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
                if self.verbose:
                    print(auroc)
                aurocs += [auroc]
            return np.median(aurocs), median_abs_deviation(aurocs)
        except Exception as e:
            if self.verbose:
                print("AUROC estimation went wrong", e)
            return None, None

    def estimate_performance_on_train(self, y, baseline=True):
        y = np.array(y)
        mask = y != -1
        y = y[mask]
        X = self.precalc_embeddings[mask]
        promiscuity_counts = np.array(self._fid_prom)[mask]
        if baseline:
            self.baseline_classifier.fit(X, y)
            y_hat = self.baseline_classifier.predict_proba(X)[:, 1]
        else:
            y_hat = self.classifier.fit(X, y, promiscuity_counts)
            y_hat = self.classifier.predict_proba(X)[:, 1]
        auroc = roc_auc_score(y, y_hat)
        return auroc

    def fit(self, y, baseline=False):
        y = np.array(y)
        mask = y != -1
        y = y[mask]
        X = self.precalc_embeddings[mask]
        promiscuity_counts = np.array(self._fid_prom)[mask]
        if not baseline:
            self.classifier.fit(X, y, promiscuity_counts)
        else:
            self.baseline_classifier.fit(X, y)
        self._is_fitted_baseline = baseline

    def predict_proba(self, smiles_list):
        X = fragment_embedder.transform(smiles_list)
        if not self._is_fitted_baseline:
            y_hat = self.classifier.predict_proba(X)
        else:
            y_hat = self.baseline_classifier.predict_proba(X)
        return y_hat

    def predict(self, smiles_list):
        X = fragment_embedder.transform(smiles_list)
        if not self._is_fitted_baseline:
            y_hat = self.classifier.predict(X)
        else:
            y_hat = self.baseline_classifier.predict(X)
        return y_hat

    def predict_proba_on_train(self):
        X = self.precalc_embeddings
        if not self._is_fitted_baseline:
            y_hat = self.classifier.predict_proba(X)
        else:
            y_hat = self.baseline_classifier.predict_proba(X)
        return y_hat

    def predict_on_train(self):
        X = self.precalc_embeddings
        if not self._is_fitted_baseline:
            y_hat = self.classifier.predict(X)
        else:
            y_hat = self.baseline_classifier.predict(X)
        return y_hat

    def predict_proba_and_tau(self, smiles_list):
        sample_indices = np.random.choice(
            self.precalc_embeddings_reference.shape[0], size=1000, replace=False
        )
        X = fragment_embedder.transform(smiles_list)
        if not self._is_fitted_baseline:
            y_hat = self.classifier.predict_proba(X)[:, 1]
            reference_y_hat = self.classifier.predict_proba(
                self.precalc_embeddings_reference
            )[sample_indices, 1]
            train_y_hat = self.classifier.predict_proba(self.precalc_embeddings)[:, 1]
        else:
            y_hat = self.baseline_classifier.predict_proba(X)[:, 1]
            reference_y_hat = self.baseline_classifier.predict_proba(
                self.precalc_embeddings_reference
            )[sample_indices, 1]
            train_y_hat = self.baseline_classifier.predict_proba(
                self.precalc_embeddings
            )[:, 1]
        tau_ref = self._calculate_percentiles(y_hat, reference_y_hat)
        tau_train = self._calculate_percentiles(y_hat, train_y_hat)
        return y_hat, tau_ref, tau_train


def evaluate_predictive_capacity(model, uniprot_acs, tfidf):
    prom_cuts = []
    hit_cuts = []
    aurocs = []
    n_pos = []
    n_tot = []
    for prom_cut in [50, 100, 250, 500]:
        for hit_cut in [10, 50, 100, 200]:
            prom_cuts += [prom_cut]
            hit_cuts += [hit_cut]
            data = HitSelectorByOverlap(uniprot_acs=uniprot_acs, tfidf=tfidf).select(
                hit_cut, prom_cut
            )
            n_pos += [len(data[data["y"] == 1])]
            n_tot += [len(data[data["y"] != -1])]
            auroc = model.estimate_performance(data["y"], baseline=True, n_splits=10)
            aurocs += [auroc[0]]
    data = {
        "hit_cut": hit_cuts,
        "prom_cut": prom_cuts,
        "n_pos": n_pos,
        "n_tot": n_tot,
        "auroc": aurocs,
    }
    return pd.DataFrame(data)


def evaluate_predictive_capacity_aggregate(
    model, uniprot_acs, tfidf, auroc_percentile=75
):
    res = evaluate_predictive_capacity(
        model=model, uniprot_acs=uniprot_acs, tfidf=tfidf
    )
    aurocs = []
    for auroc in list(res["auroc"]):
        if str(auroc) == "nan":
            aurocs += [0.5]
        else:
            aurocs += [float(auroc)]
    return np.percentile(aurocs, auroc_percentile)


class CommunityDetector(object):
    def __init__(self, auroc_cut=0.7, tfidf=True):
        self.auroc_cut = auroc_cut
        self.tfidf = tfidf

    def community_subgraphs(self, graph):
        partition = community_louvain.best_partition(
            graph, randomize=False, random_state=42
        )
        clusters = collections.defaultdict(list)
        for k, v in partition.items():
            clusters[v] += [k]
        clusters = [tuple(sorted(v)) for k, v in clusters.items()]
        clusters = set(clusters)
        clusters = sorted(clusters, key=lambda x: -len(x))
        subgraphs = []
        for nodes in clusters:
            subgraphs += [graph.subgraph(list(nodes)).copy()]
        return subgraphs

    def accept_graph(self, model, graph):
        uniprot_acs = graph.nodes()
        if len(uniprot_acs) == 1:
            return True
        auroc = evaluate_predictive_capacity_aggregate(
            model=model, uniprot_acs=uniprot_acs, tfidf=self.tfidf
        )
        if auroc > self.auroc_cut:
            return True
        return False

    def select_subgraphs(self, model, graph):
        if self.accept_graph(model, graph):
            result = {"ok": [graph], "ko": []}
            return result
        acc_subgraphs = []
        rej_subgraphs_0 = []
        for subgraph in self.community_subgraphs(graph):
            if self.accept_graph(model, subgraph):
                acc_subgraphs += [subgraph]
            else:
                rej_subgraphs_0 += [subgraph]
        rej_subgraphs_1 = []
        for rej_subgraph in rej_subgraphs_0:
            for subgraph in self.community_subgraphs(rej_subgraph):
                if self.accept_graph(model, subgraph):
                    acc_subgraphs += [subgraph]
                else:
                    rej_subgraphs_1 += [subgraph]
        rej_subgraphs_2 = []
        for rej_subgraph in rej_subgraphs_1:
            for subgraph in self.community_subgraphs(rej_subgraph):
                if self.accept_graph(model, subgraph):
                    acc_subgraphs += [subgraph]
                else:
                    rej_subgraphs_2 += [subgraph]
        rej_subgraphs = rej_subgraphs_2
        result = {"ok": acc_subgraphs, "ko": rej_subgraphs}
        return result

    def cluster(self, model, graph):
        result_graph = self.select_subgraphs(model, graph)
        ok_list = [[n for n in g.nodes()] for g in result_graph["ok"]]
        ko_list = [[n for n in g.nodes()] for g in result_graph["ko"]]
        result = {
            "ok": sorted(ok_list, key=lambda x: -len(x)),
            "ko": sorted(ko_list, key=lambda x: -len(x)),
        }
        return result


def task_evaluator(model, data, do_auroc=True):
    try:
        y = np.array(data["y"])
        mask = y != -1
        model.baseline_classifier.fit(precalc_embeddings[mask], y[mask])
        y_hat_ref = np.array(
            model.baseline_classifier.predict_proba(precalc_embeddings_reference)[:, 1]
        )
        rho = np.nanmean(
            [
                pearsonr(y_hat_ref, promiscuity_reference_predictions[:, j])[0]
                for j in range(promiscuity_reference_predictions.shape[1])
            ]
        )
        if do_auroc:
            auroc = model.estimate_performance(data["y"], baseline=True, n_splits=10)
        else:
            auroc = (None, None)
        prom = np.mean(data[data["y"] == 1]["prom"])
        hits = np.mean(data[data["y"] == 1]["hits"])
        result = {"auroc": auroc, "prom": prom, "hits": hits, "ref_rho": rho}
        return result
    except:
        return None
