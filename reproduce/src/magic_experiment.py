from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from qiskit.quantum_info import Statevector

from .bootstrap import bootstrap_ci
from .kurtosis import kurtosis_statistic, r_squared_linear_fit
from .resources import GLOBAL_CLIFFORD_SAMPLER, generate_t_magic_state_family, global_random_clifford, magic
from .simulation_backend import NearestNeighbour1D, expH, pf


@dataclass(frozen=True)
class MagicKurtosisConfig:
    n: int = 10
    samples: int = 2000
    seed: int = 1234
    state_seed: int = 4321
    bootstrap_samples: int = 10000
    bootstrap_seed: int = 2468
    delta_t: float = 0.1
    Jx: float = 1.0
    hx: float = 0.8090
    hy: float = 0.9045
    pbc: bool = False
    trotter_order: int = 1
    steps: int = 1
    magic_batch_size: int = 4096
    append: bool = False


def _state_error(state_data: np.ndarray, error_matrix: np.ndarray) -> float:
    error_state = error_matrix @ state_data
    return float(np.sqrt(np.real(error_state.conj().T @ error_state)))


def sample_global_clifford_errors(
    state: Statevector,
    error_matrix: np.ndarray,
    n: int,
    samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    errors = np.empty(samples, dtype=float)
    for index in range(samples):
        sampled_state = state.evolve(global_random_clifford(n, rng).to_circuit())
        errors[index] = _state_error(sampled_state.data, error_matrix)
    return errors


def _write_metadata(path: Path, metadata: dict[str, object]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for key, value in metadata.items():
            handle.write(f"{key}: {value}\n")


def _experiment_signature(config: MagicKurtosisConfig) -> dict[str, object]:
    return {
        "n": config.n,
        "state_seed": config.state_seed,
        "delta_t": config.delta_t,
        "Jx": config.Jx,
        "hx": config.hx,
        "hy": config.hy,
        "pbc": config.pbc,
        "trotter_order": config.trotter_order,
        "steps": config.steps,
        "magic_batch_size": config.magic_batch_size,
        "global_clifford_sampler": GLOBAL_CLIFFORD_SAMPLER,
        "state_family": "random global Clifford, i T gates for i=0..n, random global Clifford",
    }


def _read_metadata(path: Path) -> dict[str, str]:
    metadata = {}
    if not path.exists():
        return metadata
    for line in path.read_text(encoding="utf-8").splitlines():
        if ": " in line:
            key, value = line.split(": ", 1)
            metadata[key] = value
    return metadata


def _validate_append_compatibility(config: MagicKurtosisConfig, output_dir: Path) -> None:
    if not config.append:
        return

    metadata = _read_metadata(output_dir / "metadata.txt")
    if not metadata:
        return

    mismatches = []
    for key, value in _experiment_signature(config).items():
        old_value = metadata.get(key)
        if key == "global_clifford_sampler" and old_value is None:
            mismatches.append(f"{key}: existing=<missing>, requested={value}")
            continue
        if old_value is not None and old_value != str(value):
            if key == "steps" and metadata.get("trotter_steps") == str(value):
                continue
            mismatches.append(f"{key}: existing={old_value}, requested={value}")

    if mismatches:
        joined = "; ".join(mismatches)
        raise ValueError(f"Cannot append because experiment parameters differ: {joined}")


def _load_or_generate_states(
    config: MagicKurtosisConfig,
    states_dir: Path,
) -> tuple[list[Statevector], np.ndarray]:
    states_path = states_dir / "states.npy"
    magics_path = states_dir / "magics.npy"
    if config.append and states_path.exists() and magics_path.exists():
        state_data = np.load(states_path)
        states = [Statevector(data) for data in state_data]
        magics = np.load(magics_path)
        if len(states) != config.n + 1:
            raise ValueError("Existing states do not match n + 1")
        return states, magics

    states = generate_t_magic_state_family(config.n, seed=config.state_seed)
    magics = np.asarray([magic(state, batch_size=config.magic_batch_size) for state in states])
    np.save(states_path, np.asarray([state.data for state in states]))
    np.save(magics_path, magics)
    return states, magics


def run_magic_kurtosis_experiment(config: MagicKurtosisConfig, output_dir: Path) -> Path:
    if config.steps < 1:
        raise ValueError("steps must be a positive integer")

    output_dir.mkdir(parents=True, exist_ok=True)
    _validate_append_compatibility(config, output_dir)
    states_dir = output_dir / "states"
    errors_dir = output_dir / "errors"
    statistics_dir = output_dir / "statistics"
    figures_dir = output_dir / "figures"
    for directory in (states_dir, errors_dir, statistics_dir, figures_dir):
        directory.mkdir(parents=True, exist_ok=True)

    states, magics = _load_or_generate_states(config, states_dir)

    model = NearestNeighbour1D(
        n=config.n,
        Jx=config.Jx,
        hx=config.hx,
        hy=config.hy,
        pbc=config.pbc,
    )
    total_time = config.delta_t * config.steps
    exact_u = expH(model.ham, total_time)
    trotter_u = pf(
        h_list=model.ham_par,
        t=total_time,
        r=config.steps,
        order=config.trotter_order,
    )
    error_matrix = exact_u - trotter_u

    rng = np.random.default_rng(config.seed)
    bootstrap_rng = np.random.default_rng(config.bootstrap_seed)
    kurtosis_values = np.empty(len(states), dtype=float)
    kurtosis_ci = np.zeros((2, len(states)), dtype=float)
    scaled_error_series = []

    scale = config.delta_t ** (-(2 * config.trotter_order + 2))
    for index, state in enumerate(states):
        errors = sample_global_clifford_errors(
            state=state,
            error_matrix=error_matrix,
            n=config.n,
            samples=config.samples,
            rng=rng,
        )
        errors_path = errors_dir / f"magic_{config.n}bit{index + 1}.npy"
        if config.append and errors_path.exists():
            old_errors = np.load(errors_path)
            errors = np.concatenate([old_errors, errors])
        np.save(errors_path, errors)

        scaled_errors = errors**2 * scale
        scaled_error_series.append(scaled_errors)
        value, lower_err, upper_err = bootstrap_ci(
            scaled_errors,
            statistic=kurtosis_statistic,
            n_resamples=config.bootstrap_samples,
            rng=bootstrap_rng,
        )
        kurtosis_values[index] = value
        kurtosis_ci[0, index] = upper_err
        kurtosis_ci[1, index] = lower_err

    coefficients, r_squared = r_squared_linear_fit(magics, kurtosis_values)
    np.save(statistics_dir / "kurtosis.npy", kurtosis_values)
    np.save(statistics_dir / "kurtosis_ci.npy", kurtosis_ci)
    np.save(statistics_dir / "linear_fit_coefficients.npy", coefficients)
    np.save(statistics_dir / "r_squared.npy", np.asarray(r_squared))

    _plot_kurtosis(
        magics=magics,
        kurtosis_values=kurtosis_values,
        kurtosis_ci=kurtosis_ci,
        coefficients=coefficients,
        r_squared=r_squared,
        output_path=figures_dir / "kurtosis_vs_magic.pdf",
    )
    _plot_error_distributions(
        magics=magics,
        scaled_error_series=scaled_error_series,
        output_path=figures_dir / "error_distributions.pdf",
    )

    metadata = {
        **_experiment_signature(config),
        "samples": config.samples,
        "seed": config.seed,
        "bootstrap_samples": config.bootstrap_samples,
        "bootstrap_seed": config.bootstrap_seed,
        "append": config.append,
        "samples_added_per_state_this_run": config.samples,
        "total_samples_per_state": len(scaled_error_series[0]) if scaled_error_series else 0,
        "total_time": total_time,
        "error_rescaling": f"error**2 * delta_t^(-{2 * config.trotter_order + 2})",
    }
    _write_metadata(output_dir / "metadata.txt", metadata)
    _write_metadata(states_dir / "metadata.txt", metadata)
    _write_metadata(errors_dir / "metadata.txt", metadata)
    _write_metadata(statistics_dir / "metadata.txt", metadata)
    with (output_dir / "append_history.txt").open("a" if config.append else "w", encoding="utf-8") as handle:
        handle.write(
            f"seed={config.seed}, samples_added_per_state={config.samples}, "
            f"total_samples_per_state={metadata['total_samples_per_state']}, append={config.append}\n"
        )
    return output_dir


def _plot_kurtosis(
    magics: np.ndarray,
    kurtosis_values: np.ndarray,
    kurtosis_ci: np.ndarray,
    coefficients: np.ndarray,
    r_squared: float,
    output_path: Path,
) -> None:
    order = np.argsort(magics)
    sorted_magics = magics[order]
    fit = np.poly1d(coefficients)

    fig, ax = plt.subplots(figsize=(7, 5), layout="constrained")
    ax.errorbar(
        magics,
        kurtosis_values,
        yerr=kurtosis_ci,
        fmt="o",
        linestyle="None",
        color="#7C4D77",
        ecolor="#993A9C",
        capsize=4,
        label=r"Kur[$s_E(\psi)$]",
    )
    ax.plot(sorted_magics, fit(sorted_magics), "--", color="gray", linewidth=2, label="Linear fit")
    ax.set_xlabel("Magic M")
    ax.set_ylabel("Kurtosis")
    ax.grid(alpha=0.3)
    ax.text(0.05, 0.08, rf"$R^2$ = {r_squared:.4f}", transform=ax.transAxes)
    ax.legend(framealpha=0)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.1, dpi=200, transparent=True)
    plt.close(fig)


def _plot_error_distributions(
    magics: np.ndarray,
    scaled_error_series: list[np.ndarray],
    output_path: Path,
) -> None:
    colors = plt.cm.viridis(np.linspace(0.05, 0.95, len(scaled_error_series)))

    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
    for index, (magic_value, scaled_errors) in enumerate(zip(magics, scaled_error_series)):
        if np.allclose(scaled_errors, scaled_errors[0]):
            ax.axvline(
                scaled_errors[0],
                linewidth=2,
                color=colors[index],
                label=rf"$M={magic_value:.3f}$",
            )
        else:
            bins_count = min(200, max(10, len(scaled_errors) // 2))
            hist, bins = np.histogram(scaled_errors, bins=bins_count, density=True)
            ax.plot(
                bins[:-1],
                hist,
                linewidth=2,
                color=colors[index],
                label=rf"$M={magic_value:.3f}$",
            )

    ax.set_xlabel(r"Simulation error $s_E(\psi)$")
    ax.set_ylabel("Density")
    ax.ticklabel_format(style="sci", scilimits=(-2, 2), axis="both", useMathText=True)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, framealpha=0, ncol=2)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.1, dpi=200, transparent=True)
    plt.close(fig)
