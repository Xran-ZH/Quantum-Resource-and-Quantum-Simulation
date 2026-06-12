from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from qiskit.quantum_info import Operator, Statevector

from .resources import local_random_clifford, part_entropy
from .simulation_backend import NearestNeighbour1D, expH, pf


@dataclass(frozen=True)
class VarianceConfig:
    case: str
    n: int = 10
    depth: int = 40
    samples: int = 2000
    delta_t: float = 0.01
    Jx: float = 1.0
    hx: float = 0.8090
    hy: float = 0.9045
    pbc: bool = True
    trotter_order: int = 1
    trotter_steps: int = 1
    seed: int = 1234
    start_index: int = 0

    @property
    def prefix(self) -> str:
        if self.case == "typical":
            return "ent"
        if self.case == "atypical":
            return "antient"
        raise ValueError("case must be 'typical' or 'atypical'")


def default_variance_config(case: str, seed: int, samples: int, depth: int) -> VarianceConfig:
    if case == "typical":
        return VarianceConfig(case=case, seed=seed, samples=samples, depth=depth, hx=0.8090)
    if case == "atypical":
        return VarianceConfig(case=case, seed=seed, samples=samples, depth=depth, hx=0.0)
    raise ValueError("case must be 'typical' or 'atypical'")


def evolved_states(model: NearestNeighbour1D, n: int, depth: int) -> list[Statevector]:
    states = []
    state = Statevector.from_int(0, 2**n)
    states.append(state)
    for step in range(depth):
        state = state.evolve(Operator(expH(model.ham, step + 1)))
        states.append(state)
    return states


def sample_trotter_errors(
    state: Statevector,
    error_matrix: np.ndarray,
    n: int,
    samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    errors = []
    for _ in range(samples):
        sampled_state = state.evolve(local_random_clifford(n, rng).to_circuit())
        error_state = error_matrix @ sampled_state.data
        errors.append(np.sqrt(np.real(error_state.conj().T @ error_state)))
    return np.asarray(errors, dtype=float)


def run_variance_experiment(config: VarianceConfig, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(config.seed)

    model = NearestNeighbour1D(
        n=config.n,
        Jx=config.Jx,
        hx=config.hx,
        hy=config.hy,
        pbc=config.pbc,
    )
    exact_u = expH(model.ham, config.delta_t)
    trotter_u = pf(
        h_list=model.ham_par,
        t=config.delta_t,
        r=config.trotter_steps,
        order=config.trotter_order,
    )
    error_u = exact_u - trotter_u

    states = evolved_states(model, config.n, config.depth)
    np.save(output_dir / f"{config.prefix}states.npy", np.asarray([state.data for state in states]))
    for subsystem_size in (2, 3, 4):
        np.save(
            output_dir / f"part{subsystem_size}{config.case}.npy",
            part_entropy(states, subsystem_size),
        )

    for index, state in enumerate(states):
        if index < config.start_index:
            continue
        errors = sample_trotter_errors(
            state=state,
            error_matrix=error_u,
            n=config.n,
            samples=config.samples,
            rng=rng,
        )
        np.save(output_dir / f"{config.prefix}_{index}.npy", errors)

    metadata = {
        "case": config.case,
        "n": config.n,
        "depth": config.depth,
        "samples": config.samples,
        "delta_t": config.delta_t,
        "Jx": config.Jx,
        "hx": config.hx,
        "hy": config.hy,
        "pbc": config.pbc,
        "trotter_order": config.trotter_order,
        "trotter_steps": config.trotter_steps,
        "seed": config.seed,
        "start_index": config.start_index,
    }
    with (output_dir / "metadata.txt").open("w", encoding="utf-8") as handle:
        for key, value in metadata.items():
            handle.write(f"{key}: {value}\n")
    return output_dir
