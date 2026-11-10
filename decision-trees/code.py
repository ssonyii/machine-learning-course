import numpy as np
from collections import Counter


def find_best_split(feature_vector, target_vector, min_samples_leaf=None):
    """
    Указания:
    * Пороги, приводящие к попаданию в одно из поддеревьев пустого множества объектов, не рассматриваются
    * В качестве порогов нужно брать среднее двух соседних при сортировке значений признака
    * Поведение функции в случае константного признака может быть любым
    * При одинаковых приростах критерия Джини нужно выбирать минимальный сплит
    * Критерий Джини должен поддерживать многоклассовый случай:
    * H(R) = 1 - sum_k p_k^2, где p_k - доля класса k в узле. Для K=2 формула сводится к 1 - p0^2 - p1^2.
    * За наличие в функции циклов балл будет снижен. Векторизуйте! :)

    :param feature_vector: вещественнозначный вектор значений признака
    :param target_vector: вектор классов объектов, len(feature_vector) == len(target_vector)

    :return thresholds: отсортированный по возрастанию вектор со всеми возможными порогами, по которым объекты можно разделить на две различные подвыборки или поддерева
    :return ginis: вектор со значениями критерия Джини для каждого из порогов в thresholds, len(ginis) == len(thresholds)
    :return threshold_best: оптимальный порог (число)
    :return gini_best: оптимальное значение критерия Джини (число)
    """
    order = np.argsort(feature_vector)
    X = feature_vector[order]
    y = target_vector[order]

    classes = np.unique(y)

    is_class_k = (y[:, None] == classes[None, :]).astype(int)

    num_R_left = np.cumsum(is_class_k, axis=0) # [i, k] сколько объектов класса k будет слева, если мы сплитим после позиции i
    num_R_total = num_R_left[-1]
    num_R_right = num_R_total - num_R_left

    left_sizes = np.arange(1, len(X) + 1) # если мы сплитим после позиции i, то слева будет i+1 объектов
    right_sizes = len(X) - left_sizes

    mask = X[:-1] != X[1:]

    if min_samples_leaf is not None:
        valid_sizes = (
            left_sizes[:-1] >= min_samples_leaf) & (right_sizes[:-1] >= min_samples_leaf)
        split_indices = np.where(mask & valid_sizes)[0]
    else:
        split_indices = np.where(mask)[0]

    if len(split_indices) == 0:
        return np.array([]), np.array([]), None, None

    R_left = num_R_left[split_indices] # [i, k] кол-во объектов класса k слева для i-го допустимого порога
    R_right = num_R_right[split_indices]

    size_R_left = left_sizes[split_indices].reshape(-1, 1) # |R_left|
    size_R_right = right_sizes[split_indices].reshape(-1, 1)

    # gini
    H_left = 1 - np.sum((R_left / size_R_left) ** 2, axis=1)
    H_right = 1 - np.sum((R_right / size_R_right) ** 2, axis=1)

    R_m = len(X)
    ginis = -(size_R_left.flatten() / R_m) * H_left - \
        (size_R_right.flatten() / R_m) * H_right

    best_split_idx = np.argmax(ginis)
    gini_best = ginis[best_split_idx]
    idx_in_X = split_indices[best_split_idx]
    threshold_best = (X[idx_in_X] + X[idx_in_X + 1]) / 2

    thresholds = (X[split_indices] + X[split_indices + 1]) / 2

    return thresholds, ginis, threshold_best, gini_best


class DecisionTree:
    """
    Простое классификационное дерево, поддерживающее:
    * real / categorical признаки
    * binary и multiclass цели (метки могут быть числами или строками)
    * ограничения max_depth, min_samples_split, min_samples_leaf (как в sklearn по смыслу)

    ВНИМАНИЕ: в методе _fit_node ниже намеренно оставлены ошибки и бинарные допущения.
    Их нужно исправить в рамках задания.
    """

    def __init__(self, feature_types, max_depth=None, min_samples_split=None, min_samples_leaf=None):
        if np.any(list(map(lambda x: x != "real" and x != "categorical", feature_types))):
            raise ValueError("There is unknown feature type")

        self._tree = {}
        self._feature_types = feature_types
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf

    def _fit_node(self, sub_X, sub_y, node, depth=0):

        # добавила критерии останова еще
        if self._min_samples_split is not None and len(sub_X) < self._min_samples_split:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        if self._max_depth is not None and depth >= self._max_depth:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        if np.all(sub_y == sub_y[0]):  # исправила
            node["type"] = "terminal"
            node["class"] = sub_y[0]
            return

        feature_best, threshold_best, split = None, None, None
        gini_best = -np.inf

        for feature in range(sub_X.shape[1]):
            feature_type = self._feature_types[feature]
            categories_map = {}

            if feature_type == "real":
                feature_vector = sub_X[:, feature]
            elif feature_type == "categorical":
                # ВАЖНО: преобразование категориальных признаков должно работать и для multiclass.
                # counts = Counter(sub_X[:, feature])
                # clicks = Counter(sub_X[sub_y == 1, feature])
                # ratio = {}
                # for key, current_count in counts.items():
                #     if key in clicks:
                #         current_click = clicks[key]
                #     else:
                #         current_click = 0
                #     ratio[key] = current_count / current_click
                # sorted_categories = list(
                #     map(lambda x: x[1], sorted(ratio.items(), key=lambda x: x[1])))
                # categories_map = dict(
                #     zip(sorted_categories, list(range(len(sorted_categories)))))

                # feature_vector = np.array(
                #     [categories_map[x] for x in sub_X[:, feature]])

                # как в лекции
                categories = np.unique(sub_X[:, feature])
                class_ratio = {}
                for c in categories:
                    mask = sub_X[:, feature] == c
                    if np.sum(mask) > 0:
                        class_ratio[c] = np.mean(sub_y[mask])
                    else:
                        class_ratio[c] = 0

                sorted_categories = sorted(
                    categories, key=lambda x: class_ratio[x])

                categories_map = {c: i for i,
                                  c in enumerate(sorted_categories)}
                feature_vector = np.array(
                    [categories_map[x] for x in sub_X[:, feature]])
            else:
                raise ValueError

            thresholds, ginis, threshold, gini = find_best_split(
                feature_vector, sub_y)

            if threshold is None:
                continue

            if gini > gini_best:
                feature_best = feature
                gini_best = gini

                if feature_type == "real":
                    split = feature_vector < threshold
                    threshold_best = threshold

                elif feature_type == "categorical":  # исправила
                    threshold_best = [
                        c for c, rank in categories_map.items() if rank < threshold]
                    split = np.isin(sub_X[:, feature], threshold_best)
                else:
                    raise ValueError

        if feature_best is None:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        node["type"] = "nonterminal"
        node["feature_split"] = feature_best

        if self._feature_types[feature_best] == "real":
            node["threshold"] = threshold_best
        elif self._feature_types[feature_best] == "categorical":
            node["categories_split"] = threshold_best
        else:
            raise ValueError

        node["left_child"], node["right_child"] = {}, {}
        self._fit_node(sub_X[split], sub_y[split],
                       node["left_child"], depth + 1)
        self._fit_node(sub_X[~split], sub_y[~split],
                       node["right_child"], depth + 1)  # исправила

    def _predict_node(self, x, node):
        if node["type"] == "terminal":
            return node["class"]

        feature = node["feature_split"]

        if self._feature_types[feature] == "real":
            if x[feature] < node["threshold"]:
                return self._predict_node(x, node["left_child"])
            else:
                return self._predict_node(x, node["right_child"])

        else:
            if x[feature] in node["categories_split"]:
                return self._predict_node(x, node["left_child"])
            else:
                return self._predict_node(x, node["right_child"])

    def fit(self, X, y):
        self._fit_node(X, y, self._tree)

    def predict(self, X):
        predicted = []
        for x in X:
            predicted.append(self._predict_node(x, self._tree))
        return np.array(predicted)
