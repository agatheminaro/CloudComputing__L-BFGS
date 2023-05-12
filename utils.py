import numpy as np
from numpy.random import randn, multivariate_normal
from scipy.linalg import norm, toeplitz
from lbfgs_numpy import LBFGS
import torch


def simulate_data_numpy(coefs, n, std=1.0, corr=0.5, log_reg=False):
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


def simulate_data_torch(coefs, n, noise_sd=1.0, log_reg=False):
    """Simulate data for linear regression.

    Parameters
    ----------
    coefs : ndarray, shape (d,)
        The coefficients of the model
    n : int
        Sample size
    noise_sd : float, default=1.0
        Standard deviation of the Gaussian noise.
    log_reg : bool, default=False
        If True, the targets are {-1, 1}.

    Returns
    -------
    A : ndarray, shape (n, d)
        The design matrix.
    b : ndarray, shape (n,)
        The response variable.
    """
    d = len(coefs)
    A = torch.randn(n, d)
    noise = noise_sd * torch.randn(n)
    b = A.matmul(coefs) + noise

    if log_reg:
        b = torch.sign(b)
    return A, b


def grad_i_linreg(i, x, A, b, lbda):
    """Gradient with respect to a sample"""
    a_i = A[i]
    return (a_i.dot(x) - b[i]) * a_i + lbda * x


def grad_linreg_numpy(x, A, b, lbda):
    """Full gradient"""
    g = np.zeros_like(x)
    n = A.shape[0]
    for i in range(n):
        g += grad_i_linreg(i, x, A, b, lbda)
    return g / n


def loss_linreg_numpy(x, A, b, lbda):
    n = A.shape[0]
    return norm(A.dot(x) - b) ** 2 / (2.0 * n) + lbda * norm(x) ** 2 / 2.0


def grad_i_logreg(i, x, A, b, lbda):
    """Gradient with respect to a sample"""
    a_i = A[i]
    b_i = b[i]
    return -a_i * b_i / (1.0 + np.exp(b_i * np.dot(a_i, x))) + lbda * x


def grad_logreg_numpy(x, A, b, lbda):
    """Full gradient"""
    g = np.zeros_like(x)
    n = A.shape[0]
    for i in range(n):
        g += grad_i_logreg(i, x, A, b, lbda)
    return g / n


def loss_logreg_numpy(x, A, b, lbda):
    bAx = b * np.dot(A, x)
    return np.mean(np.log(1.0 + np.exp(-bAx))) + lbda * norm(x) ** 2 / 2.0


def grad_linreg_torch(x, A, b, lbda):
    """Full gradient"""
    n = A.size(0)
    g = (-1.0 / n) * A.t().matmul(b - A.matmul(x)) + lbda * x
    return g


def loss_linreg_torch(x, A, b, lbda):
    n = A.size(0)
    X = A.matmul(x) - b
    return X.matmul(X) / (2.0 * n) + lbda * x.matmul(x) / 2.0


def grad_logreg_torch(x, A, b, lbda):
    """Full gradient"""
    n = A.size(0)
    bAx = b * A.matmul(x)
    return (-1.0 / n) * A.t().matmul(b / (1.0 + torch.exp(bAx))).view(-1, 1) + lbda * x


def loss_logreg_torch(x, A, b, lbda):
    bAx = b * A.matmul(x)
    return torch.mean(torch.log(1.0 + torch.exp(-bAx))) + lbda * x.matmul(x) / 2.0


def get_time_lbfgs(x, A, b, lbda, f, f_grad, torch=False, device="cpu"):
    optimizer = LBFGS(f, f_grad, vector_free=False, torch=False, device=device)
    _, _, time = optimizer.fit(x, A, b, lbda)
    return time


def get_time_vlbfgs(x, A, b, lbda, f, f_grad, torch=False, device="cpu"):
    optimizer = LBFGS(f, f_grad, vector_free=True, torch=False, device="cpu")
    _, _, time = optimizer.fit(x, A, b, lbda)
    return time
