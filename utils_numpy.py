import numpy as np
from numpy.random import randn, multivariate_normal
from scipy.linalg import norm, toeplitz
from lbfgs_numpy import LBFGS


def simulate_data(coefs, n, std=1.0, corr=0.5, log_reg=False):
    """Simulation for the least-squares problem.

    Parameters
    ----------
    coefs : ndarray, shape (d,)
        The coefficients of the model
    n : int
        Sample size
    std : float, default=1.
        Standard-deviation of the noise
    corr : float, default=0.5
        Correlation of the features matrix
    log_reg : bool, default=False
        If True, the targets are {-1, 1}.

    Returns
    -------
    A : ndarray, shape (n, d)
        The design matrix.
    b : ndarray, shape (n,)
        The targets.
    """
    d = coefs.shape[0]
    cov = toeplitz(corr ** np.arange(0, d))
    A = multivariate_normal(np.zeros(d), cov, size=n)
    noise = std * randn(n)
    b = A.dot(coefs) + noise

    if log_reg:
        b = np.sign(b)

    return A, b


def grad_i_linreg(i, x, A, b, lbda):
    """Gradient with respect to a sample"""
    a_i = A[i]
    return (a_i.dot(x) - b[i]) * a_i + lbda * x


def grad_linreg(x, A, b, lbda):
    """Full gradient"""
    g = np.zeros_like(x)
    n = A.shape[0]
    for i in range(n):
        g += grad_i_linreg(i, x, A, b, lbda)
    return g / n


def loss_linreg(x, A, b, lbda):
    n = A.shape[0]
    return norm(A.dot(x) - b) ** 2 / (2.0 * n) + lbda * norm(x) ** 2 / 2.0


def grad_i_logreg(i, x, A, b, lbda):
    """Gradient with respect to a sample"""
    a_i = A[i]
    b_i = b[i]
    return -a_i * b_i / (1.0 + np.exp(b_i * np.dot(a_i, x))) + lbda * x


def grad_logreg(x, A, b, lbda):
    """Full gradient"""
    g = np.zeros_like(x)
    n = A.shape[0]
    for i in range(n):
        g += grad_i_logreg(i, x, A, b, lbda)
    return g / n


def loss_logreg(x, A, b, lbda):
    bAx = b * np.dot(A, x)
    return np.mean(np.log(1.0 + np.exp(-bAx))) + lbda * norm(x) ** 2 / 2.0


def get_time_lbfgs(x, A, b, lbda, f, f_grad):
    optimizer = LBFGS(f, f_grad, vector_free=False)
    _, _, time = optimizer.fit(x, A, b, lbda)
    return time


def get_time_vlbfgs(x, A, b, lbda, f, f_grad):
    optimizer = LBFGS(f, f_grad, vector_free=True)
    _, _, time = optimizer.fit(x, A, b, lbda)
    return time
