import numpy as np
from scipy import optimize


class LBFGS:
    def __init__(
        self,
        default_step=0.01,
        c1=0.0001,
        c2=0.9,
        max_iter=100,
        m=2,
        verbose=False,
    ) -> None:
        self.default_step = default_step
        self.c1 = c1
        self.c2 = c2
        self.max_iter = max_iter
        self.m = m
        self.verbose = verbose

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

    def __call__(self, x0, f, f_grad, f_hessian=None):
        all_x_k, all_f_k = list(), list()
        x = x0

        all_x_k.append(x.copy())
        all_f_k.append(f(x))

        grad_x = f_grad(x)

        y_list, s_list = [], []

        for k in range(1, self.max_iter + 1):
            # Step 1: compute step the direction
            d = self._two_loops(grad_x, s_list, y_list)

            # Step 2: compute step size
            step, _, _, new_f, _, new_grad = optimize.line_search(
                f, f_grad, x, d, grad_x, c1=self.c1, c2=self.c2
            )

            if step is None:
                print("Line search did not converge at iteration {}".format(k))
                step = self.default_step

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

        return np.array(all_x_k), np.array(all_f_k)


class VLBFGS:
    def __init__(
        self,
        default_step=0.1,
        c1=0.0001,
        c2=0.9,
        max_iter=100,
        m=2,
        verbose=False,
    ) -> None:
        self.default_step = default_step
        self.c1 = c1
        self.c2 = c2
        self.max_iter = max_iter
        self.m = m
        self.verbose = verbose

    def _vector_free_two_loops(self, dot_product_matrix, b):
        """
        Parameters
        ----------
        dot_matrix : ndarray, shape (2m + 1, 2m + 1)
            the results of dot products between every
            two base vectors as a scalar matrix of
            (2m + 1) ∗ (2m + 1) scalars

        Returns
        -------
        delta :  ndarray, shape (m,)
            coefficients
        """
        m = int((dot_product_matrix.shape[0] - 1) / 2)

        alpha_list = []
        delta = np.zeros((2 * m + 1,))
        delta[2 * m] = -1

        # First loop
        for i in reversed(range(m)):
            alpha_i = (
                np.sum(delta * dot_product_matrix[i:,]) / dot_product_matrix[i, m + i]
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
        m = len(s_list)
        n_features = grad_x.shape[0]

        b = np.zeros((2 * m + 1, n_features))

        for i, tensor in enumerate(s_list):
            b[i, :] = tensor

        for i, tensor in enumerate(y_list):
            b[m + i, :] = tensor

        b[2 * m, :] = grad_x

        dot_matrix = b.dot(b.T)

        return dot_matrix, b

    def __call__(self, x0, f, f_grad, f_hessian=None):
        all_x_k, all_f_k = list(), list()
        x = x0

        all_x_k.append(x.copy())
        all_f_k.append(f(x))

        grad_x = f_grad(x)
        new_f = f(x)

        y_list, s_list = [], []

        for k in range(1, self.max_iter + 1):
            # Step 1: compute step the direction
            dot_product_matrix, b = self._dot_product(grad_x, s_list, y_list)
            d = self._vector_free_two_loops(dot_product_matrix, b)

            # Step 2: compute step size
            step, _, _, new_f, _, new_grad = optimize.line_search(
                f, f_grad, x, d, grad_x, c1=self.c1, c2=self.c2
            )

            if step is None:
                print("Line search did not converge at iteration {}".format(k))
                step = self.default_step
                continue

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

        return np.array(all_x_k), np.array(all_f_k)
