import time
import scipy as sp
import numpy as np

from sksparse.cholmod import cholesky


def ensure_positive_definite(mtx, tau_k_minus_1):
    """
    Check if the Jacobian is positive definite by attempting Cholesky decomposition.
    If this fails, add small multiples of the identity matrix until positive definite.
    """

    # Default IPOPT parameters
    kappa_plus = 8.0
    kappa_minus = 1 / 3
    kappa_bar_plus = 100
    tau = 0
    tau_bar = 1e-4
    tau_bar_min = 1e-20

    start = time.time()
    I = sp.sparse.identity(mtx.shape[0], format="csr")

    if not is_positive_definite(mtx + tau * I):
        print("The Jacobian is not positive definite.")

        if tau_k_minus_1 == 0:
            tau = tau_bar
        else:
            tau = max(tau_bar_min, kappa_minus * tau_k_minus_1)

        while not is_positive_definite(mtx + tau * I):
            print("Adding %e * I to the Jacobian to make it positive definite." % tau)

            if tau_k_minus_1 == 0:
                tau = kappa_bar_plus * tau
            else:
                tau = kappa_plus * tau

    print(f"The Jacobian is positive definite. ({time.time() - start:.2f}s, tau={tau})")
    return mtx + tau * I, tau


def is_positive_definite(A):
    try:
        # Attempt Cholesky decomposition
        factor = cholesky(A.tocsc())
        return True
    except:
        # If decomposition fails, the matrix is not positive definite
        return False


def generate_symmetric_csr(size, density=0.1):
    random_matrix = sp.sparse.random(size, size, density=density, format="csr")
    symmetric_matrix = (random_matrix + random_matrix.T) / 2
    return symmetric_matrix


def main():
    """
    Testing the inertia correction algorithm on a random symmetric matrix.
    Algorithm from https://doi.org/10.1016/j.cma.2021.114091
    """
    # Single test
    tau = 0
    size = 1_000
    density = 0.01
    mtx = generate_symmetric_csr(size, density)
    mtx, tau = ensure_positive_definite(mtx, tau)

    # Repeat tests
    for _ in range(100):
        mtx -= 0.1 * sp.sparse.random(size, size, density=0.1, format="csr")
        mtx, tau = ensure_positive_definite(mtx, tau)


if __name__ == "__main__":
    main()
