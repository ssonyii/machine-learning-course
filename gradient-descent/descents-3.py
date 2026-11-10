import numpy as np
from abc import ABC, abstractmethod

# ===== Learning Rate Schedules =====


class LearningRateSchedule(ABC):
    @abstractmethod
    def get_lr(self, iteration: int) -> float:
        pass


class ConstantLR(LearningRateSchedule):
    def __init__(self, lr: float):
        self.lr = lr

    def get_lr(self, iteration: int) -> float:
        return self.lr


class TimeDecayLR(LearningRateSchedule):
    def __init__(self, lambda_: float = 1.0):
        self.s0 = 1
        self.p = 0.5
        self.lambda_ = lambda_

    def get_lr(self, iteration: int) -> float:
        k = iteration
        step_lr = self.lambda_ * np.power(self.s0 / (self.s0 + k), self.p)
        return step_lr


# ===== Base Optimizer =====
class BaseDescent(ABC):
    def __init__(self, lr_schedule: LearningRateSchedule = TimeDecayLR):
        self.lr_schedule = lr_schedule()
        self.iteration = 0
        self.model = None

    def set_model(self, model):
        self.model = model

    @abstractmethod
    def update_weights(self):
        pass

    def step(self):
        delta_w = self.update_weights()
        self.iteration += 1
        return delta_w
        


# ===== Specific Optimizers =====
class VanillaGradientDescent(BaseDescent):
    def update_weights(self):
        X_train = self.model.X_train
        y_train = self.model.y_train

        lr = self.lr_schedule.get_lr(self.iteration)
        gradient = self.model.compute_gradients(X_train, y_train)
        weight_delta = -lr * gradient
        self.model.w = self.model.w + weight_delta
        return weight_delta


class StochasticGradientDescent(BaseDescent):
    def __init__(self, lr_schedule: LearningRateSchedule = TimeDecayLR, batch_size=1):
        super().__init__(lr_schedule)
        self.batch_size = batch_size

    def update_weights(self):
        X_train = self.model.X_train
        y_train = self.model.y_train

        # 1) выбрать случайный батч
        l = X_train.shape[0]
        batch_indices = np.random.randint(0, l, self.batch_size)
        X_batch = X_train[batch_indices]
        y_batch = y_train[batch_indices]
        # 2) вычислить градиенты на батче
        stochastic_gradient = self.model.compute_gradients(X_batch, y_batch)
        # 3) обновить веса модели
        lr = self.lr_schedule.get_lr(self.iteration)
        weight_delta = -lr * stochastic_gradient
        self.model.w = self.model.w + weight_delta
        return weight_delta


class SAGDescent(BaseDescent):
    def __init__(self, lr_schedule: LearningRateSchedule = TimeDecayLR):
        super().__init__(lr_schedule)
        self.grad_memory = None
        self.grad_sum = None

    def update_weights(self):
        X_train = self.model.X_train
        y_train = self.model.y_train
        num_objects, num_features = X_train.shape

        if self.grad_memory is None:
            self.grad_memory = np.zeros((num_objects, num_features))
            self.grad_sum = np.zeros((num_features))

            # добавила изначальный градиент, чтобы сошелся
            j0 = np.random.randint(0, num_objects)
            X0, y0 = X_train[j0:j0+1], y_train[j0:j0+1]
            g0 = self.model.compute_gradients(X0, y0).reshape(num_features)
            self.grad_memory[j0] = g0
            self.grad_sum = g0.copy()

        j = np.random.randint(0, num_objects)
        X_j = X_train[j:j+1]
        y_j = y_train[j:j+1]

        g_j_new = self.model.compute_gradients(X_j, y_j).reshape(num_features)
        g_j_old = self.grad_memory[j].copy()
        self.grad_sum = self.grad_sum + (g_j_new - g_j_old)
        self.grad_memory[j] = g_j_new
        avg_grad = self.grad_sum / num_objects
        lr = self.lr_schedule.get_lr(self.iteration)
        weight_delta = -lr * avg_grad.reshape(-1, 1)
        self.model.w = self.model.w + weight_delta
        return weight_delta


class MomentumDescent(BaseDescent):
    def __init__(self, lr_schedule: LearningRateSchedule = TimeDecayLR, beta=0.9):
        super().__init__(lr_schedule)
        self.beta = beta
        self.velocity = None

    def update_weights(self):
        X_train = self.model.X_train
        y_train = self.model.y_train

        gradients = self.model.compute_gradients(X_train, y_train)
        lr = self.lr_schedule.get_lr(self.iteration)

        if self.velocity is None:
            self.velocity = np.zeros_like(gradients)
        self.velocity = self.beta * self.velocity + lr * gradients
        weight_delta = -self.velocity
        self.model.w = self.model.w + weight_delta

        return weight_delta


class Adam(BaseDescent):
    def __init__(self, lr_schedule: LearningRateSchedule = TimeDecayLR, beta1=0.9, beta2=0.999, eps=1e-8):
        super().__init__(lr_schedule)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = None  # Momentum
        self.v = None  # RMSprop

    def update_weights(self):
        X_train = self.model.X_train
        y_train = self.model.y_train
        gradients = self.model.compute_gradients(X_train, y_train)
        lr = self.lr_schedule.get_lr(self.iteration)
        k = self.iteration + 1

        if self.m is None:
            self.m = np.zeros_like(gradients)
            self.v = np.zeros_like(gradients)

        self.m = self.beta1 * self.m + (1 - self.beta1) * gradients
        self.v = self.beta2 * self.v + (1 - self.beta2) * (gradients ** 2)

        m_hat = self.m / (1 - self.beta1 ** k)
        v_hat = self.v / (1 - self.beta2 ** k)

        weight_delta = (-lr * m_hat) / (np.sqrt(v_hat) + self.eps)
        self.model.w = self.model.w + weight_delta

        return weight_delta
