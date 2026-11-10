from __future__ import annotations

from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import roc_auc_score

from typing import Optional
from tqdm.auto import tqdm

from sklearn.base import ClassifierMixin


class Boosting(ClassifierMixin):

    def __init__(
        self,
        base_model_class=DecisionTreeRegressor,
        base_model_params: Optional[dict] = None,
        n_estimators: int = 20,
        learning_rate: float = 0.05,
        random_state: int | None = None,
        verbose: bool = False,

        early_stopping_rounds: int | None = None,
        eval_metric: str | None = None,

        subsample: float = 1.0,
        bagging_temperature: float = 1.0,
        bootstrap_type: str | None = 'Bernoulli',
        rsm: float = 1.0
    ):
        super().__init__()

        self.base_model_class = base_model_class
        self.base_model_params = {} if base_model_params is None else base_model_params

        self.n_estimators = n_estimators
        self.learning_rate = learning_rate

        self.models = []
        self.gammas = []

        # не забудьте вставить его везде, где у вас возникает рандом
        self.random_state = random_state
        self.verbose = verbose
        self.classes_ = np.array([-1, 1])  # в нашей задаче классы захардкожены

        # {"train_roc_auc": [], "train_loss": [], ...}
        self.history = defaultdict(list)

        self.sigmoid = lambda x: 1 / (1 + np.exp(-x))
        self.loss_fn = lambda y, z: -np.log(self.sigmoid(y * z)).mean()
        self.loss_derivative = lambda y, z: - \
            y / (1 + np.exp(y * z))  # исправила

        self.train_predictions = None

        self.early_stopping_rounds = early_stopping_rounds
        self.eval_metric = eval_metric

        self.subsample = subsample
        self.bagging_temperature = bagging_temperature
        self.bootstrap_type = bootstrap_type
        self.rsm = rsm
        self.feature_masks = []

        self.feature_importances_ = None

    def partial_fit(self, X: np.ndarray, y: np.ndarray):

        if self.train_predictions is None:
            self.train_predictions = np.zeros(X.shape[0])

        residuals = -self.loss_derivative(y, self.train_predictions)

        # бутстрап
        rng = np.random.default_rng(self.random_state)

        if self.bootstrap_type == 'Bernoulli':
            mask = rng.random(X.shape[0]) < self.subsample
            sample_idx = np.where(mask)[0]

        elif self.bootstrap_type == 'Bayesian':
            U = rng.random(X.shape[0])
            weights = (-np.log(U)) ** self.bagging_temperature
            sample_idx = np.repeat(np.arange(X.shape[0]), weights.astype(int))
            if len(sample_idx) == 0:
                sample_idx = np.arange(X.shape[0])
            sample_idx = rng.choice(sample_idx, size=int(
                self.subsample * X.shape[0]), replace=True)

        else:
            sample_idx = np.arange(X.shape[0])

        n_features = X.shape[1]
        feature_mask = np.ones(n_features, dtype=bool)

        if self.rsm < 1.0:
            n_selected = max(1, int(self.rsm * n_features))
            feature_mask[:] = False
            selected_idx = rng.choice(
                n_features, size=n_selected, replace=False)
            feature_mask[selected_idx] = True

        X_sub, y_sub, residuals_sub = X[sample_idx][:,
                                                    feature_mask], y[sample_idx], residuals[sample_idx]

        base_model = self.base_model_class(
            random_state=self.random_state, **self.base_model_params)
        base_model.fit(X_sub, residuals_sub)
        new_predictions = base_model.predict(X[:, feature_mask])
        gamma = self.find_optimal_gamma(
            y, self.train_predictions, new_predictions)
        self.train_predictions += self.learning_rate * gamma * new_predictions

        self.models.append(base_model)
        self.gammas.append(gamma)
        self.feature_masks.append(feature_mask)

        loss = self.loss_fn(y, self.train_predictions)
        roc_auc = roc_auc_score(y == 1, self.sigmoid(self.train_predictions))

        self.history["train_loss"].append(loss)
        self.history["train_roc_auc"].append(roc_auc)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            eval_set: tuple[np.ndarray] | None = None,
            use_best_model: bool = False):
        """
        :param X_train: features array (train set)
        :param y_train: targets array (train set)
        """
        self.train_predictions = np.zeros(X_train.shape[0])

        if eval_set is not None:
            X_valid, y_valid = eval_set
            valid_predictions = np.zeros(X_valid.shape[0])

        best_score = -np.inf  # лучшее качество на val
        best_iter = 0
        bad_rounds = 0

        estimator_range = range(self.n_estimators)
        if self.verbose:
            estimator_range = tqdm(estimator_range)

        for i in estimator_range:
            self.partial_fit(X_train, y_train)

            if eval_set is not None:
                valid_predictions += (self.learning_rate *
                                      self.gammas[-1] * self.models[-1].predict(X_valid[:, self.feature_masks[-1]]))
                valid_loss = self.loss_fn(y_valid, valid_predictions)
                valid_roc_auc = roc_auc_score(
                    y_valid == 1, self.sigmoid(valid_predictions))
                self.history["valid_loss"].append(valid_loss)
                self.history["valid_roc_auc"].append(valid_roc_auc)

                if self.eval_metric is not None:
                    current_score = self.history[self.eval_metric][-1]

                    if current_score > best_score:
                        best_score = current_score
                        best_iter = i
                        bad_rounds = 0
                    else:
                        bad_rounds += 1

                    if (self.early_stopping_rounds is not None and bad_rounds >= self.early_stopping_rounds):
                        if self.verbose:
                            print(f"Early stopping at iteration {i}")
                        break

        if use_best_model and eval_set is not None:
            self.models = self.models[:best_iter + 1]
            self.gammas = self.gammas[:best_iter + 1]

        # важность признаков
        n_features = X_train.shape[1]
        importances = np.zeros(n_features)

        for model, gamma, mask in zip(self.models, self.gammas, self.feature_masks):
            tree_importances = model.feature_importances_

            used_features = np.where(mask)[0]

            importances[used_features] += np.abs(gamma) * tree_importances

        if importances.sum() > 0:
            importances /= importances.sum()

        self.feature_importances_ = importances

        # чтобы было удобнее смотреть
        for key in self.history:
            self.history[key] = np.array(self.history[key])

        return self

    def plot_history(self, keys):

        if isinstance(keys, str):  # если один аргумент
            keys = [keys]

        for key in keys:
            plt.plot(self.history[key], label=key)

        plt.xlabel("Iteration")
        plt.legend()
        plt.grid(True)
        plt.show()

    def predict_proba(self, X: np.ndarray):

        predictions = np.zeros(X.shape[0])

        for model, gamma, feature_mask in zip(self.models, self.gammas, self.feature_masks):
            predictions += self.learning_rate * \
                gamma * model.predict(X[:, feature_mask])

        probs = self.sigmoid(predictions)

        return np.vstack([1 - probs, probs]).T  # формат sklearn

    def find_optimal_gamma(self, y: np.ndarray, old_predictions: np.ndarray, new_predictions: np.ndarray) -> float:
        gammas = np.linspace(start=0, stop=1, num=100)
        losses = [
            self.loss_fn(y, old_predictions + gamma * new_predictions)
            for gamma in gammas
        ]
        return gammas[np.argmin(losses)]

    def score(self, X: np.ndarray, y: np.ndarray):
        return roc_auc_score(y == 1, self.predict_proba(X)[:, 1])


class CatEncoder:
    def __init__(
            self,
            cat_features
    ):
        super().__init__()

        self.cat_features = cat_features
        self.stats = {}

        # добавила тк столкнулась с ошибкой, что в val есть не все признаки из cat_features
        self.global_mean = None

    def fit(self, X: np.ndarray, y: np.ndarray, all_features):

        self.global_mean = (y == 1).mean()

        for feature in self.cat_features:
            idx = all_features.index(feature)
            feature_values = X[:, idx]

            unique_feature_values, counts = np.unique(
                feature_values, return_counts=True)
            sums_positive = np.zeros_like(counts, dtype=float)

            for i, v in enumerate(unique_feature_values):
                sums_positive[i] = (y[feature_values == v] == 1).sum()

            self.stats[feature] = {
                "values": unique_feature_values,
                "codes": sums_positive / counts
            }

        return self

    def transform(self, X: np.ndarray, all_features):
        X = X.copy()

        for feature in self.cat_features:
            idx = all_features.index(feature)
            feature_values = X[:, idx]

            values = self.stats[feature]["values"]
            codes = self.stats[feature]["codes"]

            encoded = np.full(feature_values.shape[0], self.global_mean)

            for v, c in zip(values, codes):
                encoded[feature_values == v] = c

            X[:, idx] = encoded

        return X


class Quantize:
    def __init__(
            self,
            quantization_type: str | None = None,
            nbins: int = 255
    ):
        super().__init__()

        self.quantization_type = quantization_type
        self.nbins = nbins

        self.bins_borders = None
        self.numeretic_features = None

    def fit(self, X: np.ndarray, numeretic_features):
        self.numeretic_features = numeretic_features
        self.bins_borders = {}

        for feature in numeretic_features:
            feature_values = X[:, feature]

            if self.quantization_type == "uniform":
                fmin = np.nanmin(feature_values)
                fmax = np.nanmax(feature_values)
                borders = np.linspace(fmin, fmax, self.nbins + 1)

            elif self.quantization_type == "quantile":
                quantiles = np.linspace(0, 1, self.nbins + 1)
                borders = np.quantile(
                    feature_values, quantiles, method="linear")

            self.bins_borders[feature] = borders

        return self

    def transform(self, X: np.ndarray):
        X = X.copy()

        for feature in self.numeretic_features:
            borders = self.bins_borders[feature]
            # индекс бина с 0 для каждого объекта X для конкретной фичи
            X[:, feature] = np.digitize(X[:, feature], borders) - 1

        return X
