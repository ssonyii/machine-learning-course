import numpy as np
from scipy.sparse.linalg import svds
from descents import BaseDescent
from dataclasses import dataclass
from enum import auto, Enum
from typing import Dict, Type, Optional


class LossFunction(Enum):
    MSE = auto()
    MAE = auto()
    LogCosh = auto()
    Huber = auto()


class LinearRegression:
    def __init__(
        self,
        optimizer: Optional[BaseDescent | str] = None,
        l2_coef: float = 0.0,
        tolerance: float = 1e-6,
        huber_delta: float = 1.0,
        max_iter: int = 1000,
        loss_function: LossFunction = LossFunction.MSE
    ):
        self.optimizer = optimizer
        if isinstance(optimizer, BaseDescent):
            self.optimizer.set_model(self)
        self.l2_coef = l2_coef
        self.tolerance = tolerance
        self.huber_delta = huber_delta
        self.max_iter = max_iter
        self.loss_function = loss_function
        self.w = None
        self.X_train = None
        self.y_train = None
        self.loss_history = []

    def predict(self, X: np.ndarray) -> np.ndarray:
        return X @ self.w

    def compute_gradients(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        l = X.shape[0]
        errors = self.predict(X) - y.reshape(-1, 1)

        if self.loss_function is LossFunction.MSE:
            if self.l2_coef > 0:
                gradients = (2 / l) * X.T @ errors + self.l2_coef * self.w
            else:
                gradients = (2 / l) * X.T @ errors

        elif self.loss_function is LossFunction.MAE:
            gradients = X.T @ np.sign(errors)

        elif self.loss_function is LossFunction.LogCosh:
            gradients = (1.0 / l) * X.T @ np.tanh(errors)

        elif self.loss_function is LossFunction.Huber:
            huber_delta = self.huber_delta
            g = np.where(np.abs(errors) < huber_delta, errors,
                         huber_delta * np.sign(errors))
            gradients = (1.0 / l) * (X.T @ g)

        return gradients

    def compute_loss(self, X: np.ndarray, y: np.ndarray) -> float:
        l = X.shape[0]
        errors = self.predict(X) - y.reshape(-1, 1)

        if self.loss_function is LossFunction.MSE:
            mse_loss = float((1/l) * (errors.T @ errors))
            if self.l2_coef > 0:
                l2_penalty = self.l2_coef * np.sum(self.w ** 2)
                loss = float(mse_loss + l2_penalty)
            else:
                loss = mse_loss

        elif self.loss_function is LossFunction.MAE:
            loss = np.mean(np.abs(errors))

        elif self.loss_function is LossFunction.LogCosh:
            loss = np.mean(np.log(np.cosh(errors)))

        elif self.loss_function is LossFunction.Huber:
            huber_delta = self.huber_delta
            mask = np.abs(errors) < huber_delta
            loss = np.mean(0.5 * (errors[mask]**2) + np.sum(
                huber_delta * np.abs(errors[~mask]) - 0.5 * huber_delta**2) / l)

        return loss

    def fit(self, X: np.ndarray, y: np.ndarray):

        self.X_train, self.y_train = X, y.reshape(-1, 1)
        num_objects, num_features = X.shape

        if isinstance(self.optimizer, BaseDescent):
            self.optimizer.set_model(self)
            self.w = np.zeros((num_features, 1))
            self.loss_history = [self.compute_loss(X, y)]

            for i in range(self.max_iter):
                old_w = self.w.copy()
                self.optimizer.step()
                loss = self.compute_loss(X, y)
                self.loss_history.append(loss)

                if np.any(np.isnan(self.w)):
                    # print(f'NaN в весах на итерации {i+1}')
                    break
                if np.sum((self.w - old_w) ** 2) < self.tolerance:
                    # print(f'сходимость достигнуа на итерации {i+1}')
                    break
            return self

        elif self.optimizer is None:
            XTX = X.T @ X + num_objects * self.l2_coef * np.eye(X.shape[1])
            XTy = X.T @ y
            XTX_inv = np.linalg.inv(XTX)
            self.w = XTX_inv @ XTy
            return self

        elif self.optimizer == "SVD":

            k = 4
            u, s, vh = svds(X, k=k)
            s_inv = np.diag(1.0 / s + self.l2_coef)
            self.w = vh.T @ s_inv @ u.T @ y
            return self
