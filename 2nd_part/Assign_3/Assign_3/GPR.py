import numpy as np
from matplotlib import pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
np.set_printoptions(suppress=True)
np.random.seed(1)


def f(x):
    """The function to predict."""
    return x ** 2

if __name__ == '__main__':
    X = np.atleast_2d([-1, 1, -2, 2]).T
    y = f(X).ravel()
    # Mesh the input space for evaluations of the real function, the prediction and
    # its MSE
    x = np.atleast_2d(np.linspace(-3, 3, 1000)).T

    # Instantiate a Gaussian Process model
    gp = GaussianProcessRegressor(kernel=RBF(1), n_restarts_optimizer=25)

    # Fit the GPR with the X data provided. 
    gp.fit(X, y)


    # predict from -3 to 3 to get the posterior mean.
    y_pred, sigma = gp.predict(x, return_std=True)
    # Plot the function, the prediction and the 95% confidence interval based on
    # predictive space. (x -3 to 3.)
    plt.figure()
    plt.plot(x, f(x), 'r:', label=r'$f(x) = x^{2}$')
    plt.plot(X, y, 'r.', markersize=10, label='Observations')
    plt.plot(x, y_pred, 'b-', label='Prediction')
    plt.fill(np.concatenate([x, x[::-1]]),
            np.concatenate([y_pred - 1.960 * sigma,
                            (y_pred + 1.960 * sigma)[::-1]]),
            alpha=.5, fc='b', ec='None', label='95% confidence interval')
    plt.xlabel('$x$')
    plt.ylabel('$f(x)$')
    plt.ylim(-5, 15)
    plt.legend(loc='upper left')
