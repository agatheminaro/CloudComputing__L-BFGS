import numpy as np
from numpy.random import randn, multivariate_normal
from scipy.linalg import norm, toeplitz
from lbfgs_torch import LBFGS
import torch


def simulate_data(coefs, n, noise_sd=1.0, log_reg=False):
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


def grad_linreg(x, A, b, lbda):
    """Full gradient"""
    n = A.size(0)
    g = (-1.0 / n) * A.t().matmul(b - A.matmul(x)) + lbda * x
    return g


def loss_linreg(x, A, b, lbda):
    n = A.size(0)
    X = A.matmul(x) - b
    return X.matmul(X) / (2.0 * n) + lbda * x.matmul(x) / 2.0


def grad_logreg(x, A, b, lbda):
    """Full gradient"""
    n = A.size(0)
    bAx = b * A.matmul(x)
    return (-1.0 / n) * A.t().matmul(b / (1.0 + torch.exp(bAx))).view(-1, 1) + lbda * x

def loss_logreg(x, A, b, lbda):
    bAx = b * A.matmul(x)
    return torch.mean(torch.log(1.0 + torch.exp(-bAx))) + lbda * x.matmul(x) / 2.0

def get_time_lbfgs(x, A, b, lbda, f, f_grad, device="cuda:0"):
    optimizer = LBFGS(f, f_grad, vector_free=False, device=device)
    _, _, time = optimizer.fit(x, A, b, lbda)
    return time

def get_time_vlbfgs(x, A, b, lbda, f, f_grad, device="cuda:0"):
    optimizer = LBFGS(f, f_grad, vector_free=True, device=device)
    _, _, time = optimizer.fit(x, A, b, lbda)
    return time