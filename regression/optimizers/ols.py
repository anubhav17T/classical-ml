from abc import abstractmethod, ABC
import numpy as np


class OrdinaryLeastSquare(ABC):
    @abstractmethod
    def slope(self):
        pass

    @abstractmethod
    def intercept(self):
        pass


class SingleOrdinaryLeastSquare(OrdinaryLeastSquare, ABC):
    def __init__(self, independent: np.ndarray, dependent: np.ndarray):
        self.independent = independent
        self.dependent = dependent

    def mean(self):
        return np.mean(self.independent), np.mean(self.dependent)

    def slope(self):
        x_mean, y_mean = self.mean()
        numerator = 0
        denominator = 0
        for i in range(0, len(self.independent)):
            numerator += (self.independent[i] - x_mean) * (self.dependent[i] - y_mean)
            denominator += (self.independent[i] - x_mean) ** 2
        return numerator / denominator

    def intercept(self):
        x_mean, y_mean = self.mean()
        return y_mean - self.slope() * x_mean


class MultiOrdinaryLeastSquare(OrdinaryLeastSquare, ABC):
    def __init__(self, independent: np.ndarray, dependent: np.ndarray):
        self.independent = independent
        self.dependent = dependent

    def intercept(self):
        pass

    def slope(self):
        XTX = self.independent.T @ self.independent
        XTY = self.independent.T @ self.dependent
        beta = np.linalg.pinv(XTX) @ XTY
        return beta




