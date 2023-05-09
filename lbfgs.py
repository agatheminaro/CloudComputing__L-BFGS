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

    def _two_loops(self, grad_x, s_list, y_list, mu_list, B0):
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

        mu_list : list of length m
            the past m values of mu

        B0 : ndarray, shape (n, n)
            Initial inverse Hessian guess

        Returns
        -------
        r :  ndarray, shape (n,)
            the L-BFGS direction
        """
        q = grad_x.copy()
        alpha_list = []

        # First loop
        for i in range(len(s_list)):
            alpha_list.append(mu_list[i] * s_list[i].T @ q)
            q = q - alpha_list[-1] * y_list[i]

        r = np.dot(B0, q)

        # Second loop
        for i in range(len(s_list) - 1, -1, -1):
            beta = mu_list[i] * y_list[i].T @ r
            r = r + s_list[i] * (alpha_list[i] - beta)

        return -r

    def __call__(self, x0, f, f_grad, f_hessian=None):
        all_x_k, all_f_k = list(), list()
        x = x0

        all_x_k.append(x.copy())
        all_f_k.append(f(x))

        B0 = np.eye(len(x))  # Hessian approximation

        grad_x = f_grad(x)

        y_list, s_list, mu_list = [], [], []

        for k in range(1, self.max_iter + 1):
            # Step 1: compute step the direction
            d = self._two_loops(grad_x, s_list, y_list, mu_list, B0)

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
            mu = 1 / np.dot(y, s)

            # Step 4: update memory
            y_list.append(y.copy())
            s_list.append(s.copy())
            mu_list.append(mu)
            if len(y_list) > self.m:
                y_list.pop(0)
                s_list.pop(0)
                mu_list.pop(0)

            all_x_k.append(x.copy())
            all_f_k.append(new_f)

            l_inf_norm_grad = np.max(np.abs(new_grad))

            if self.verbose:
                print(
                    "iter: {} | f: {:.5f} | step: {:.5f} | ||grad||_inf: {:.5f}".format(
                        k, new_f, step, l_inf_norm_grad
                    )
                )

            if l_inf_norm_grad < 1e-6:
                break

            # Step 5: update gradient
            grad_x = new_grad

        return np.array(all_x_k), np.array(all_f_k)
