import numpy as np
from time import time
import torch


class LBFGS:
    def __init__(
        self,
        f,
        f_grad,
        default_step=0.01,
        c1=0.0001,
        c2=0.9,
        max_iter=100,
        m=2,
        verbose=False,
        vector_free=False,
        device="cpu",
        torch=False,
    ) -> None:
        self.f = f
        self.f_grad = f_grad
        self.default_step = default_step
        self.c1 = c1
        self.c2 = c2
        self.max_iter = max_iter
        self.m = m
        self.verbose = verbose
        self.vector_free = vector_free
        self.device = device
        self.torch = torch

    def _two_loops(self, grad_x, s_list, y_list):
        """
        Parameters
        ----------
        grad_x : ndarray, shape (n,)
            gradient at the current point

        m : int
            memory size

        s_list : list of length m
            the past m values of s

        y_list : list of length m
            the past m values of y

        Returns
        -------
        p :  ndarray, shape (n,)
            the L-BFGS direction
        """
        if self.torch:
            m = len(s_list)
            p = -grad_x.clone().cpu()
            alpha_list = []

            # First loop
            for i in reversed(range(m)):
                y_i = y_list[i]
                s_i = s_list[i]
                alpha_i = s_i.matmul(p) / (s_i.matmul(y_i))
                alpha_list.insert(0, alpha_i)
                p -= alpha_i * y_i

            if m != 0:
                gamma = y_list[-1].matmul(s_list[-1]) / y_list[-1].matmul(y_list[-1])
                p *= gamma

            # Second loop
            for i in range(m):
                y_i = y_list[i]
                s_i = s_list[i]
                beta = y_i.matmul(p) / s_i.matmul(y_i)
                p += (alpha_list[i] - beta) * s_i

            return p
        else:
            m = len(s_list)
            p = -grad_x.copy()
            alpha_list = []

            # First loop
            for i in reversed(range(m)):
                y_i = y_list[i]
                s_i = s_list[i]
                alpha_i = s_i.T.dot(p) / (s_i.dot(y_i))
                alpha_list.insert(0, alpha_i)
                p -= alpha_i * y_i

            if m != 0:
                gamma = y_list[-1].dot(s_list[-1]) / y_list[-1].dot(y_list[-1])
                p *= gamma

            # Second loop
            for i in range(m):
                y_i = y_list[i]
                s_i = s_list[i]
                beta = y_i.dot(p) / s_i.dot(y_i)
                p += (alpha_list[i] - beta) * s_i

            return p

    def _vector_free_two_loops(self, dot_product_matrix, b):
        """
        Parameters
        ----------
        dot_matrix : ndarray, shape (2m + 1, 2m + 1)
            the results of dot products between every
            two base vectors as a scalar matrix of
            (2m + 1) * (2m + 1) scalars

        b : ndarray, shape (2m + 1, n)
            all memory vectors and current gradient

        Returns
        -------
        r : ndarray, shape (n,)
            the L-BFGS direction
        """
        m = int((dot_product_matrix.shape[0] - 1) / 2)
        alpha_list = []

        if self.torch:
            delta = torch.zeros((2 * m + 1)).to(self.device)
            delta[2 * m] = -1
            b = b.to(self.device)
            dot_product_matrix = dot_product_matrix.to(self.device)

            # First loop
            for i in reversed(range(m)):
                alpha_i = (
                    torch.sum(delta * dot_product_matrix[i, :])
                    / dot_product_matrix[i, m + i]
                )
                alpha_list.insert(0, alpha_i)
                delta[m + i] -= alpha_i

            for i in range(2 * m + 1):
                delta[i] *= (
                    dot_product_matrix[m - 1, 2 * m - 1]
                    / dot_product_matrix[2 * m - 1, 2 * m - 1]
                )

            # Second loop
            for i in range(m):
                beta = (
                    torch.sum(delta * dot_product_matrix[m + i, :])
                    / dot_product_matrix[i, m + i]
                )
                delta[i] += alpha_list[i] - beta

            for i in range(2 * m + 1):
                b[i, :] *= delta[i]

            r = b.sum(dim=0)

            return r

        else:
            delta = np.zeros(2 * m + 1)
            delta[2 * m] = -1

            # First loop
            for i in reversed(range(m)):
                alpha_i = (
                    np.sum(delta * dot_product_matrix[i, :])
                    / dot_product_matrix[i, m + i]
                )
                alpha_list.insert(0, alpha_i)
                delta[m + i] -= alpha_i

            for i in range(2 * m + 1):
                delta[i] *= (
                    dot_product_matrix[m - 1, 2 * m - 1]
                    / dot_product_matrix[2 * m - 1, 2 * m - 1]
                )

            # Second loop
            for i in range(m):
                beta = (
                    np.sum(delta * dot_product_matrix[m + i, :])
                    / dot_product_matrix[i, m + i]
                )
                delta[i] += alpha_list[i] - beta

            for i in range(2 * m + 1):
                b[i, :] *= delta[i]

            r = b.sum(axis=0)

            return r

    def _dot_product(self, grad_x, s_list, y_list):
        """
        Parameters
        ----------
        grad_x : ndarray, shape (n,)
            gradient at the current point

        s_list : list of length m
            the past m values of s

        y_list : list of length m
            the past m values of y

        Returns
        -------
        dot_matrix : ndarray, shape (2m + 1, 2m + 1)
            the results of dot products between every
            two base vectors as a scalar matrix of
            (2m + 1) * (2m + 1) scalars

        b : ndarray, shape (2m + 1, n)
            all memory vectors and current gradient
        """
        m = len(s_list)
        n = grad_x.shape[0]

        if self.torch:
            b = torch.empty((2 * m + 1, n)).to(self.device)

            for i, tensor in enumerate(s_list):
                b[i, :] = tensor

            for i, tensor in enumerate(y_list):
                b[m + i, :] = tensor

            b[2 * m, :] = grad_x

            dot_matrix = b.matmul(b.transpose(0, 1))

            return dot_matrix, b

        else:
            b = np.zeros((2 * m + 1, n))

            for i, tensor in enumerate(s_list):
                b[i, :] = tensor

            for i, tensor in enumerate(y_list):
                b[m + i, :] = tensor

            b[2 * m, :] = grad_x

            dot_matrix = b.dot(b.T)

            return dot_matrix, b

    def _line_search(self, f, f_grad, current_f, grad_x, x, d, A, b, lbda, c1, c2):
        """
        Parameters
        ----------
        f : callable
            objective function

        f_grad : callable
            gradient of the objective function

        current_f : float
            objective function value at the current point

        grad_x : ndarray, shape (n,)
            gradient at the current point

        x : ndarray, shape (n,)
            current point

        d : ndarray, shape (n,)
            descent direction

        A : ndarray, shape (m, n)
            matrix of the linear constraint

        b : ndarray, shape (m,)
            vector of the linear constraint

        lbda : float
            regularization parameter

        c1 : float
            parameter for Armijo condition

        c2 : float
            parameter for Wolfe condition

        Returns
        -------
        step : float
            step size

        new_f : float
            objective function value at the new point

        new_grad : ndarray, shape (n,)
            gradient at the new point
        """

        alpha = 0
        beta = "inf"
        step = self.default_step
        if self.torch:
            for _ in range(10):
                new_f = f(x + step * d, A, b, lbda).item()
                f1 = (current_f + c1 * step * grad_x.dot(d)).item()

                new_grad = f_grad(x + step * d, A, b, lbda)

                f2 = new_grad.matmul(d).item()
                f3 = c2 * grad_x.matmul(d).item()

                if new_f > f1:  # Armijo condition
                    beta = step
                    step = (alpha + beta) / 2

                elif f2 < f3:  # Wolfe condition
                    alpha = step
                    if beta == "inf":
                        step = 2 * alpha
                    else:
                        step = (alpha + beta) / 2
                else:
                    break
            return step, new_f, new_grad

        else:
            for _ in range(10):
                new_f = f(x + step * d, A, b, lbda)
                f1 = current_f + c1 * step * grad_x.dot(d)

                new_grad = f_grad(x + step * d, A, b, lbda)

                f2 = new_grad.dot(d)
                f3 = c2 * grad_x.dot(d)

                if new_f > f1:  # Armijo condition
                    beta = step
                    step = (alpha + beta) / 2

                elif f2 < f3:  # Wolfe condition
                    alpha = step
                    if beta == "inf":
                        step = 2 * alpha
                    else:
                        step = (alpha + beta) / 2
                else:
                    break

            return step, new_f, new_grad

    def fit(self, x0, A, target, lbda):
        """
        Parameters
        ----------
        x0 : ndarray, shape (n,)
            initial point

        A : ndarray, shape (m, n)
            matrix of the linear constraint

        target : ndarray, shape (m,)
            vector of the linear constraint

        lbda : float
            regularization parameter
        """
        t0 = time()
        all_x_k, all_f_k = list(), list()

        if self.torch:
            x = x0.to(self.device)
            A = A.to(self.device)
            target = target.to(self.device)

            all_x_k.append(x.cpu())

            new_f = self.f(x, A, target, lbda).item()
            all_f_k.append(new_f)

            grad_x = self.f_grad(x, A, target, lbda)

            y_list, s_list = [], []

            for k in range(1, self.max_iter + 1):
                # Step 1: compute step the direction
                if self.vector_free:
                    dot_product_matrix, b = self._dot_product(grad_x, s_list, y_list)
                    dot_product_matrix = dot_product_matrix.cpu()
                    b = b.cpu()

                    d = self._vector_free_two_loops(dot_product_matrix, b)
                    d = d.to(self.device)

                else:
                    d = self._two_loops(grad_x, s_list, y_list)
                    d = d.to(self.device)

                # Step 2: line search
                step, new_f, new_grad = self._line_search(
                    self.f,
                    self.f_grad,
                    new_f,
                    grad_x,
                    x,
                    d,
                    A,
                    target,
                    lbda,
                    self.c1,
                    self.c2,
                )

                # Step 3: update x
                s = step * d
                x += s
                y = new_grad - grad_x

                # Step 4: update memory
                y_list.append(y.cpu())
                s_list.append(s.cpu())

                if len(y_list) > self.m:
                    y_list.pop(0)
                    s_list.pop(0)

                all_x_k.append(x.cpu())
                all_f_k.append(new_f)

                l_inf_norm_grad = torch.max(torch.abs(new_grad)).item()

                if self.verbose:
                    print(
                        "iter: {} | f: {:.5f} | step: {:.5f} | ||grad||_inf: {:.5f}".format(
                            k, new_f, step, l_inf_norm_grad
                        )
                    )

                if l_inf_norm_grad < 1e-5:
                    break

                # Step 5: update gradient
                grad_x = new_grad

            t1 = time()
            computation_time = t1 - t0

            return all_x_k, all_f_k, computation_time

        else:
            x = x0.copy()

            all_x_k.append(x.copy())

            new_f = self.f(x, A, target, lbda)
            all_f_k.append(new_f)

            grad_x = self.f_grad(x, A, target, lbda)

            y_list, s_list = [], []

            for k in range(1, self.max_iter + 1):
                # Step 1: compute step the direction
                if self.vector_free:
                    dot_product_matrix, b = self._dot_product(grad_x, s_list, y_list)
                    d = self._vector_free_two_loops(dot_product_matrix, b)

                else:
                    d = self._two_loops(grad_x, s_list, y_list)

                # Step 2: line search
                step, new_f, new_grad = self._line_search(
                    self.f,
                    self.f_grad,
                    new_f,
                    grad_x,
                    x,
                    d,
                    A,
                    target,
                    lbda,
                    self.c1,
                    self.c2,
                )

                # Step 3: update x
                s = step * d
                x += s
                y = new_grad - grad_x

                # Step 4: update memory
                y_list.append(y.copy())
                s_list.append(s.copy())

                if len(y_list) > self.m:
                    y_list.pop(0)
                    s_list.pop(0)

                all_x_k.append(x.copy())
                all_f_k.append(new_f)

                l_inf_norm_grad = np.max(np.abs(new_grad))

                if self.verbose:
                    print(
                        "iter: {} | f: {:.5f} | step: {:.5f} | ||grad||_inf: {:.5f}".format(
                            k, new_f, step, l_inf_norm_grad
                        )
                    )

                if l_inf_norm_grad < 1e-5:
                    break

                # Step 5: update gradient
                grad_x = new_grad

            t1 = time()
            computation_time = t1 - t0

            return np.array(all_x_k), np.array(all_f_k), computation_time
