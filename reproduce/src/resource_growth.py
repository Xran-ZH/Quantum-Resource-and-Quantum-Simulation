from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from qiskit.quantum_info import Operator, Statevector

from .resources import magic
from .simulation_backend import NearestNeighbour1D, expH


@dataclass(frozen=True)
class ResourceGrowthConfig:
    case: str = "typical"
    n: int = 10
    time_max: float = 4.0
    steps: int = 40
    subsystem_size: int = 4
    Jx: float = 1.0
    hx: float = 0.8090
    hy: float = 0.9045
    pbc: bool = True
    magic_batch_size: int = 4096

    @property
    def label(self) -> str:
        if self.case == "typical":
            return "typical"
        if self.case == "atypical":
            return "atypical"
        raise ValueError("case must be 'typical' or 'atypical'")


def default_resource_growth_config(
    case: str,
    n: int,
    time_max: float,
    steps: int,
    subsystem_size: int,
    magic_batch_size: int,
) -> ResourceGrowthConfig:
    if case == "typical":
        return ResourceGrowthConfig(
            case=case,
            n=n,
            time_max=time_max,
            steps=steps,
            subsystem_size=subsystem_size,
            hx=0.8090,
            magic_batch_size=magic_batch_size,
        )
    if case == "atypical":
        return ResourceGrowthConfig(
            case=case,
            n=n,
            time_max=time_max,
            steps=steps,
            subsystem_size=subsystem_size,
            hx=0.0,
            magic_batch_size=magic_batch_size,
        )
    raise ValueError("case must be 'typical' or 'atypical'")


def _entropy_from_state_vector(state: Statevector, subsystem_size: int) -> float:
    from qiskit.quantum_info import entropy, partial_trace

    reduced = partial_trace(state, list(range(subsystem_size)))
    return float(entropy(reduced, 2))


def _write_metadata(path: Path, metadata: dict[str, object]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for key, value in metadata.items():
            handle.write(f"{key}: {value}\n")


def run_resource_growth_case(config: ResourceGrowthConfig, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    model = NearestNeighbour1D(
        n=config.n,
        Jx=config.Jx,
        hx=config.hx,
        hy=config.hy,
        pbc=config.pbc,
    )
    initial_state = Statevector.from_int(0, 2**config.n)
    times = np.linspace(0.0, config.time_max, config.steps + 1)

    states = []
    entropies = np.empty(len(times), dtype=float)
    magics = np.empty(len(times), dtype=float)
    for index, time in enumerate(times):
        state = initial_state.evolve(Operator(expH(model.ham, float(time))))
        states.append(state)
        entropies[index] = _entropy_from_state_vector(state, config.subsystem_size)
        magics[index] = magic(state, batch_size=config.magic_batch_size)

    np.save(output_dir / "times.npy", times)
    np.save(output_dir / "states.npy", np.asarray([state.data for state in states]))
    np.save(output_dir / "entropies.npy", entropies)
    np.save(output_dir / "magics.npy", magics)

    _plot_resource_growth(
        times=times,
        entropies=entropies,
        magics=magics,
        title=config.label.title(),
        output_path=output_dir / "resource_growth.pdf",
    )

    metadata = {
        "case": config.case,
        "n": config.n,
        "time_max": config.time_max,
        "steps": config.steps,
        "subsystem_size": config.subsystem_size,
        "Jx": config.Jx,
        "hx": config.hx,
        "hy": config.hy,
        "pbc": config.pbc,
        "magic_batch_size": config.magic_batch_size,
        "state_generation": "Statevector.from_int(0, 2**n) evolved by exp(-i H t)",
    }
    _write_metadata(output_dir / "metadata.txt", metadata)
    return output_dir


def run_resource_growth(cases: list[ResourceGrowthConfig], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    case_dirs = []
    for config in cases:
        case_dir = output_dir / config.label
        run_resource_growth_case(config, case_dir)
        case_dirs.append(case_dir)

    if len(case_dirs) > 1:
        _plot_combined_growth(case_dirs, output_dir / "resource_growth_combined.pdf")
    return output_dir


def _plot_resource_growth(
    times: np.ndarray,
    entropies: np.ndarray,
    magics: np.ndarray,
    title: str,
    output_path: Path,
) -> None:
    fig, ax_entropy = plt.subplots(figsize=(8, 4.5), layout="constrained")
    ax_entropy.set_title(title)
    ax_entropy.set_xlabel("Time t")
    ax_entropy.set_ylabel("Entanglement")
    ax_entropy.plot(times, entropies, "o-", color="#E4A031", linewidth=2, markersize=4, label="Entanglement")

    ax_magic = ax_entropy.twinx()
    ax_magic.set_ylabel("Magic")
    ax_magic.plot(times, magics, "o-", color="#73A5A2", linewidth=2, markersize=4, label="Magic")

    h1, l1 = ax_entropy.get_legend_handles_labels()
    h2, l2 = ax_magic.get_legend_handles_labels()
    ax_entropy.legend(h1 + h2, l1 + l2, loc="best", framealpha=0)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.1, dpi=200, transparent=True)
    plt.close(fig)


def _plot_combined_growth(case_dirs: list[Path], output_path: Path) -> None:
    fig, axes = plt.subplots(len(case_dirs), 1, figsize=(8, 4 * len(case_dirs)), sharex=True, layout="constrained")
    if len(case_dirs) == 1:
        axes = [axes]

    for ax_entropy, case_dir in zip(axes, case_dirs):
        times = np.load(case_dir / "times.npy")
        entropies = np.load(case_dir / "entropies.npy")
        magics = np.load(case_dir / "magics.npy")

        ax_entropy.set_title(case_dir.name.title())
        ax_entropy.set_ylabel("Entanglement")
        ax_entropy.plot(times, entropies, "o-", color="#E4A031", linewidth=2, markersize=4, label="Entanglement")

        ax_magic = ax_entropy.twinx()
        ax_magic.set_ylabel("Magic")
        ax_magic.plot(times, magics, "o-", color="#73A5A2", linewidth=2, markersize=4, label="Magic")

        h1, l1 = ax_entropy.get_legend_handles_labels()
        h2, l2 = ax_magic.get_legend_handles_labels()
        ax_entropy.legend(h1 + h2, l1 + l2, loc="best", framealpha=0)

    axes[-1].set_xlabel("Time t")
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.1, dpi=200, transparent=True)
    plt.close(fig)
