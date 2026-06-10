from __future__ import annotations

from functools import lru_cache
from itertools import product

import numpy as np
from qiskit.quantum_info import Operator, Pauli, Statevector, entropy, partial_trace, random_clifford

try:
    import jax
    import jax.numpy as jnp

    _HAS_JAX = True
except ImportError:
    jax = None
    jnp = None
    _HAS_JAX = False


def child_seed(rng: np.random.Generator) -> int:
    return int(rng.integers(0, 2**32 - 1, dtype=np.uint32))


def local_random_clifford(n: int, rng: np.random.Generator):
    """Tensor product of independently seeded one-qubit random Cliffords."""
    clifford = random_clifford(1, seed=child_seed(rng))
    for _ in range(1, n):
        clifford = clifford.expand(random_clifford(1, seed=child_seed(rng)))
    return clifford


T_GATE = Operator(np.asarray([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex))


def generate_t_magic_state_family(n: int, seed: int = 1234) -> list[Statevector]:
    """Generate n+1 states with increasing numbers of T gates.

    For each i in 0, ..., n, the state is generated as

        |0...0> -> random global Clifford -> T on qubits 0,...,i-1
        -> random global Clifford.

    The two random global Clifford layers are independently seeded for every
    value of i.
    """
    if n < 1:
        raise ValueError("n must be positive")

    rng = np.random.default_rng(seed)
    states = []
    for t_count in range(n + 1):
        state = Statevector.from_int(0, 2**n)
        state = state.evolve(random_clifford(n, seed=child_seed(rng)))
        for qubit in range(t_count):
            state = state.evolve(T_GATE, [qubit])
        state = state.evolve(random_clifford(n, seed=child_seed(rng)))
        states.append(state)
    return states


def pauli_group(n: int) -> list[str]:
    return ["".join(labels) for labels in product("IXYZ", repeat=n)]


def part_entropy(states: list[Statevector], subsystem_size: int) -> np.ndarray:
    values = []
    traced_qubits = list(range(subsystem_size))
    for state in states:
        reduced = partial_trace(state, traced_qubits)
        values.append(entropy(reduced, 2))
    return np.asarray(values, dtype=float)


def _pauli_masks_from_group_uncached(pauligroup: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
    x_masks = np.zeros(len(pauligroup), dtype=np.int32)
    z_masks = np.zeros(len(pauligroup), dtype=np.int32)
    n = len(pauligroup[0]) if pauligroup else 0

    for row, paulistr in enumerate(pauligroup):
        if len(paulistr) != n:
            raise ValueError("All Pauli strings must have the same length")
        for pos, label in enumerate(paulistr):
            bit = 1 << (n - pos - 1)
            if label in ("X", "Y"):
                x_masks[row] |= bit
            if label in ("Y", "Z"):
                z_masks[row] |= bit
    return x_masks, z_masks


@lru_cache(maxsize=16)
def _pauli_masks_from_group(pauligroup: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_masks, z_masks = _pauli_masks_from_group_uncached(pauligroup)
    return x_masks, z_masks, _y_phases(x_masks, z_masks)


@lru_cache(maxsize=16)
def _all_pauli_masks(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    codes = np.arange(4**n, dtype=np.int64)
    x_masks = np.zeros_like(codes, dtype=np.int32)
    z_masks = np.zeros_like(codes, dtype=np.int32)

    for pos in range(n):
        digit = (codes // (4 ** (n - pos - 1))) % 4
        bit = np.int32(1 << (n - pos - 1))
        x_masks |= np.where((digit == 1) | (digit == 2), bit, 0).astype(np.int32)
        z_masks |= np.where((digit == 2) | (digit == 3), bit, 0).astype(np.int32)
    return x_masks, z_masks, _y_phases(x_masks, z_masks)


def _y_phases(x_masks: np.ndarray, z_masks: np.ndarray) -> np.ndarray:
    y_counts = np.asarray([int(mask).bit_count() for mask in (x_masks & z_masks)], dtype=np.int32)
    phase_lookup = np.asarray([1, 1j, -1, -1j], dtype=np.complex64)
    return phase_lookup[y_counts % 4]


def _magic_qiskit_loop(state: Statevector, pauligroup: list[str]) -> float:
    values = []
    d = len(state)
    for paulistr in pauligroup:
        evolved = state.evolve(Pauli(paulistr))
        overlap = state.inner(evolved)
        values.append(np.sqrt(np.real(np.conj(overlap) * overlap)))
    return float(1 - d * np.average(np.power(values, 4)))


if _HAS_JAX:

    def _parity(values):
        values = values ^ (values >> 16)
        values = values ^ (values >> 8)
        values = values ^ (values >> 4)
        values = values ^ (values >> 2)
        values = values ^ (values >> 1)
        return values & 1


    @jax.jit
    def _pauli_abs4_sum_batch(psi, x_masks, z_masks, y_phases):
        indices = jnp.arange(psi.shape[0], dtype=jnp.int32)
        shifted_indices = jnp.bitwise_xor(indices[None, :], x_masks[:, None])
        shifted_psi = psi[shifted_indices]

        parities = _parity(jnp.bitwise_and(indices[None, :], z_masks[:, None]))
        z_phases = jnp.where(parities == 0, 1.0 + 0.0j, -1.0 + 0.0j)
        phases = y_phases[:, None] * z_phases

        expectations = jnp.sum(jnp.conj(shifted_psi) * phases * psi[None, :], axis=1)
        return jnp.sum(jnp.abs(expectations) ** 4)


def magic(
    state: Statevector | np.ndarray,
    pauligroup: list[str] | None = None,
    batch_size: int = 4096,
) -> float:
    """Compute the stabilizer magic used in the legacy scripts.

    The quantity is

        M(|psi>) = 1 - d * average_P |<psi|P|psi>|^4,

    where the average is over the n-qubit Pauli group. When JAX is installed,
    the Pauli expectations are evaluated in batches using bit-mask arithmetic.
    """
    if isinstance(state, Statevector):
        psi_np = np.asarray(state.data, dtype=np.complex64)
    else:
        psi_np = np.asarray(state, dtype=np.complex64)

    d = len(psi_np)
    n = int(np.log2(d))
    if 2**n != d:
        raise ValueError("state length must be a power of two")

    if pauligroup is None:
        x_masks, z_masks, y_phases = _all_pauli_masks(n)
    else:
        x_masks, z_masks, y_phases = _pauli_masks_from_group(tuple(pauligroup))

    if not _HAS_JAX:
        if pauligroup is None:
            pauligroup = pauli_group(n)
        return _magic_qiskit_loop(Statevector(psi_np), pauligroup)

    psi = jnp.asarray(psi_np)
    total_abs4 = 0.0

    for start in range(0, len(x_masks), batch_size):
        stop = min(start + batch_size, len(x_masks))
        x_batch = jnp.asarray(x_masks[start:stop], dtype=jnp.int32)
        z_batch = jnp.asarray(z_masks[start:stop], dtype=jnp.int32)
        y_batch = jnp.asarray(y_phases[start:stop])
        total_abs4 += float(_pauli_abs4_sum_batch(psi, x_batch, z_batch, y_batch))

    return float(1 - d * total_abs4 / len(x_masks))
