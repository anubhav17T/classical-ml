import numpy as np
from optimizers.ols import SingleOrdinaryLeastSquare, MultiOrdinaryLeastSquare


def single_ols_data(points: int):
    x = np.linspace(0, 20, points)
    y = np.linspace(0, 20, points)
    return x, y


def multi_ols_data():
    X = np.array([[1, 2, 3],
                  [1, 4, 5],
                  [1, 6, 7],
                  ])
    y = np.array([[6], [9], [14]])
    return X, y


def fit_single():
    x, y = single_ols_data(points=30)
    sols = SingleOrdinaryLeastSquare(independent=x, dependent=y)
    slope = sols.slope()
    intercept = sols.intercept()
    for i in range(0, len(x)):
        prediction_y = slope * x[i] + intercept
        print(
            "Prediction is {} ------ Actual Is {} ------- Error Is {}".format(prediction_y, y[i], y[i] - prediction_y))


if __name__ == "__main__":
    fit_single()
    X, y = multi_ols_data()
    mols = MultiOrdinaryLeastSquare(independent=X, dependent=y)
    beta = mols.slope()
    print("Intercept is {}".format(beta[0][0]))
    print("Feature 1 Coeff is {}".format(beta[1][0]))
    print("Feature 2 Coeff is {}".format(beta[2][0]))
