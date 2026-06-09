from __future__ import annotations

from itertools import product

import numpy as np
from qiskit.quantum_info import Pauli, Statevector, entropy, partial_trace, random_clifford


def child_seed(rng: np.random.Generator) -> int:
    return int(rng.integers(0, 2**32 - 1, dtype=np.uint32))


def local_random_clifford(n: int, rng: np.random.Generator):
    """Tensor product of independently seeded one-qubit random Cliffords."""
    clifford = random_clifford(1, seed=child_seed(rng))
    for _ in range(1, n):
        clifford = clifford.expand(random_clifford(1, seed=child_seed(rng)))
    return clifford


def pauli_group(n: int) -> list[str]:
    return ["".join(labels) for labels in product("IXYZ", repeat=n)]


def part_entropy(states: list[Statevector], subsystem_size: int) -> np.ndarray:
    values = []
    traced_qubits = list(range(subsystem_size))
    for state in states:
        reduced = partial_trace(state, traced_qubits)
        values.append(entropy(reduced, 2))
    return np.asarray(values, dtype=float)


def magic(state: Statevector, pauligroup: list[str]) -> float:
    values = []
    d = len(state)
    for paulistr in pauligroup:
        evolved = state.evolve(Pauli(paulistr))
        overlap = state.inner(evolved)
        values.append(np.sqrt(np.real(np.conj(overlap) * overlap)))
    return float(1 - d * np.average(np.power(values, 4)))
