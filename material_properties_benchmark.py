import time
import numpy as np

from sfepy.mechanics.tensors import get_deviator, get_volumetric_tensor
from sfepy.mechanics.matcoefs import stiffness_from_youngpoisson

NE = 150_000  # 150_000
NQ = 9
ETA = 1e-15
CMAT = stiffness_from_youngpoisson(dim=2, young=210e3, poisson=0.3, plane="strain")
CDEV = get_deviator(CMAT)
CVOL = get_volumetric_tensor(CMAT)


def main():
    strain = np.random.randn(NE, NQ, 3, 1)
    damage = np.random.randn(NE, NQ, 1, 1)

    strain.resize(NE * NQ, 3, 1)
    damage.resize(NE * NQ, 1, 1)
    psi = np.random.randn(NE * NQ, 1, 1)

    start = time.time()
    psi_loop, c_mod_loop = standard_loop(strain.copy(), damage.copy(), psi.copy())
    loop_duration = time.time() - start

    start = time.time()
    psi_einsum, c_mod_einsum = fast_einsum(strain.copy(), damage.copy(), psi.copy())
    einsum_duration = time.time() - start

    print(f"Loop duration: {loop_duration}")
    print(f"Einsum duration: {einsum_duration}")

    assert np.allclose(psi_loop, psi_einsum)
    assert np.allclose(c_mod_loop, c_mod_einsum)


def standard_loop(strain, damage, psi):
    c_mod = np.zeros((NE * NQ, 3, 3))
    stress = np.matmul(CMAT, strain)

    for i in range(strain.shape[0]):
        trace = strain[i][0][0] + strain[i][1][0]

        if trace >= 0:
            psi[i] = max(psi[i], np.tensordot(stress[i], strain[i]))
            c_mod[i] = (damage[i] ** 2 + ETA) * CMAT
        else:
            stress_dev = get_deviator(stress[i].T).T
            strain_dev = get_deviator(strain[i].T).T
            psi[i] = max(psi[i], np.tensordot(stress_dev, strain_dev))
            c_mod[i] = (damage[i] ** 2 + ETA) * CDEV + CVOL

    return psi, c_mod


def fast_einsum(strain, damage, psi):
    dims = psi.shape

    # Trace for tension or compression
    trace = strain[:, 0, 0] + strain[:, 1, 0] >= 0

    # Elastic strain energy
    psi = np.maximum(
        psi,
        np.where(
            trace,
            np.einsum("ijk, jk, ikl -> i", strain, CMAT, strain),
            np.einsum("ijk, jk, ikl -> i", strain, CDEV, strain),
        ).reshape(dims),
    )

    # Modified elastic stiffness tensor
    damage **= 2
    c_mod = np.where(
        trace.reshape(dims),
        (damage + ETA) * CMAT,
        (damage + ETA) * CDEV + CVOL,
    )

    return psi, c_mod


if __name__ == "__main__":
    main()
