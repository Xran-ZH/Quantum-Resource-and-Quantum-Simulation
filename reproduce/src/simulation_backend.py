from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import expm


PAULI_MATRICES = {
    "I": np.array([[1, 0], [0, 1]], dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}


def expH(hamiltonian: np.ndarray, t: float) -> np.ndarray:
    """Return exp(-i H t), matching the convention used in the legacy scripts."""
    return expm(-1j * t * hamiltonian)


def pf(h_list: list[np.ndarray], t: float, r: int = 1, order: int = 1) -> np.ndarray:
    """Dense first- or second-order product formula.

    This local implementation is intentionally minimal. It covers the product
    formulas needed by the cleaned reproduction scripts and avoids depending on
    the external quantum_simulation_recipe package.
    """
    if r < 1:
        raise ValueError("r must be a positive integer")
    if order not in (1, 2):
        raise ValueError("Only first- and second-order product formulas are supported")

    dim = h_list[0].shape[0]
    step = t / r

    if order == 1:
        one_step = np.eye(dim, dtype=complex)
        for hamiltonian in h_list:
            one_step = expH(hamiltonian, step) @ one_step
    else:
        one_step = np.eye(dim, dtype=complex)
        for hamiltonian in h_list:
            one_step = expH(hamiltonian, step / 2) @ one_step
        for hamiltonian in reversed(h_list):
            one_step = expH(hamiltonian, step / 2) @ one_step

    result = np.eye(dim, dtype=complex)
    for _ in range(r):
        result = one_step @ result
    return result


def _single_pauli(n: int, pauli: str, qubit: int) -> np.ndarray:
    labels = ["I"] * n
    labels[qubit] = pauli
    return pauli_string(labels)


def _two_pauli(n: int, pauli: str, q0: int, q1: int) -> np.ndarray:
    labels = ["I"] * n
    labels[q0] = pauli
    labels[q1] = pauli
    return pauli_string(labels)


def pauli_string(labels: list[str]) -> np.ndarray:
    op = PAULI_MATRICES[labels[0]]
    for label in labels[1:]:
        op = np.kron(op, PAULI_MATRICES[label])
    return op


def _zero_hamiltonian(n: int) -> np.ndarray:
    return np.zeros((2**n, 2**n), dtype=complex)


@dataclass(frozen=True)
class NearestNeighbour1D:
    """Minimal dense nearest-neighbour spin-chain Hamiltonian.

    The cleaned scripts currently use the mixed-field Ising case with Jx, hx,
    and hy. Optional Jy, Jz, and hz terms are included so the same helper can
    cover the long-time Heisenberg checks later.
    """

    n: int
    Jx: float = 0.0
    Jy: float = 0.0
    Jz: float = 0.0
    hx: float = 0.0
    hy: float = 0.0
    hz: float = 0.0
    pbc: bool = True

    def __post_init__(self) -> None:
        if self.n < 1:
            raise ValueError("n must be positive")

        object.__setattr__(self, "x_terms", self._local_terms("X", self.hx))
        object.__setattr__(self, "y_terms", self._local_terms("Y", self.hy))
        object.__setattr__(self, "z_terms", self._local_terms("Z", self.hz))
        object.__setattr__(self, "xx_terms", self._coupling_terms("X", self.Jx))
        object.__setattr__(self, "yy_terms", self._coupling_terms("Y", self.Jy))
        object.__setattr__(self, "zz_terms", self._coupling_terms("Z", self.Jz))

        ham_par = [
            self.x_terms,
            self.y_terms,
            self.z_terms,
            self.xx_terms,
            self.yy_terms,
            self.zz_terms,
        ]
        ham_par = [term for term in ham_par if np.any(term)]
        object.__setattr__(self, "ham_par", ham_par)
        object.__setattr__(self, "ham", sum(ham_par, start=_zero_hamiltonian(self.n)))

    def _edges(self) -> list[tuple[int, int]]:
        edges = [(i, i + 1) for i in range(self.n - 1)]
        if self.pbc and self.n > 2:
            edges.append((self.n - 1, 0))
        return edges

    def _local_terms(self, pauli: str, coefficient: float) -> np.ndarray:
        if coefficient == 0:
            return _zero_hamiltonian(self.n)
        total = _zero_hamiltonian(self.n)
        for qubit in range(self.n):
            total += coefficient * _single_pauli(self.n, pauli, qubit)
        return total

    def _coupling_terms(self, pauli: str, coefficient: float) -> np.ndarray:
        if coefficient == 0:
            return _zero_hamiltonian(self.n)
        total = _zero_hamiltonian(self.n)
        for q0, q1 in self._edges():
            total += coefficient * _two_pauli(self.n, pauli, q0, q1)
        return total
