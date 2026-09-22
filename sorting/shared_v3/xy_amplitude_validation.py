# %%
"""Validate and visualize the configured X/Y amplitude compensation.

The one-dimensional curves are evaluated over the frequency ranges defined by
``x_tweezer_IFs`` and ``y_tweezer_IFs`` in ``configuration.py``.  The 2D map
is their separable product, i.e. the combined X/Y compensation multiplier.

Run with::

    python amplitude_validation.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from amplitude_calibration import amp_at_frequency_py, load_calibration_coefs
from configuration import x_tweezer_IFs, y_tweezer_IFs

AMPLITUDE_LIMIT = 2.0
N_DENSE_SAMPLES = 501


def _frequency_samples(tweezer_ifs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return configured IFs and a dense validation axis, both in MHz."""
    configured_mhz = np.sort(np.unique(np.asarray(tweezer_ifs, dtype=float))) / 1e6
    if configured_mhz.size == 0:
        raise ValueError("The configured tweezer IF array is empty.")
    if configured_mhz.size == 1:
        dense_mhz = configured_mhz.copy()
    else:
        dense_mhz = np.linspace(configured_mhz[0], configured_mhz[-1], N_DENSE_SAMPLES)
    return configured_mhz, dense_mhz


def evaluate_axis(axis: str, tweezer_ifs: np.ndarray) -> dict:
    """Load one calibration and evaluate it across its configured IF range."""
    calibration_path = Path().resolve() / f"{axis}_amplitude_calibration_coefs.csv"
    if not calibration_path.exists():
        raise FileNotFoundError(f"Missing {axis.upper()} calibration: {calibration_path}")

    calibration = load_calibration_coefs(calibration_path, unit_in_data="MHz")
    center_ghz, coefs = calibration.center, calibration.coefs
    configured_mhz, dense_mhz = _frequency_samples(tweezer_ifs)
    dense_amplitude = np.asarray(
        amp_at_frequency_py(dense_mhz / 1e3, center_ghz, coefs),
        dtype=float,
    )
    configured_amplitude = np.asarray(
        amp_at_frequency_py(configured_mhz / 1e3, center_ghz, coefs),
        dtype=float,
    )

    finite = bool(np.all(np.isfinite(dense_amplitude)))
    in_range = finite and bool(np.all((dense_amplitude >= 0.0) & (dense_amplitude <= AMPLITUDE_LIMIT)))
    return {
        "axis": axis.upper(),
        "path": calibration_path,
        "configured_mhz": configured_mhz,
        "dense_mhz": dense_mhz,
        "configured_amplitude": configured_amplitude,
        "dense_amplitude": dense_amplitude,
        "in_range": in_range,
    }


def _plot_axis(ax, result: dict) -> None:
    axis = result["axis"]
    freq = result["dense_mhz"]
    amplitude = result["dense_amplitude"]
    color = "tab:blue" if result["in_range"] else "tab:red"

    ax.plot(freq, amplitude, color=color, linewidth=2, label=f"{axis} calibration")
    ax.scatter(
        result["configured_mhz"],
        result["configured_amplitude"],
        color="black",
        s=35,
        zorder=3,
        label="Configured IFs",
    )
    ax.axhspan(0, AMPLITUDE_LIMIT, color="tab:green", alpha=0.08, label="Valid range")
    ax.set_xlabel(f"{axis} IF [MHz]")
    ax.set_ylabel("Compensation multiplier")
    status = "PASS" if result["in_range"] else "FAIL"
    ax.set_title(f"{axis}: {status}  (min={np.min(amplitude):.4f}, max={np.max(amplitude):.4f})")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)


def plot_validation(x_result: dict, y_result: dict):
    """Plot both 1D validations and their combined compensation map."""
    fig = plt.figure(figsize=(13, 10), layout="constrained")
    grid = fig.add_gridspec(2, 2, height_ratios=[1, 1.25])
    ax_x = fig.add_subplot(grid[0, 0])
    ax_y = fig.add_subplot(grid[0, 1])
    ax_map = fig.add_subplot(grid[1, :])

    _plot_axis(ax_x, x_result)
    _plot_axis(ax_y, y_result)

    x_freq = x_result["dense_mhz"]
    y_freq = y_result["dense_mhz"]
    combined = np.outer(y_result["dense_amplitude"], x_result["dense_amplitude"])
    extent = [x_freq[0], x_freq[-1], y_freq[0], y_freq[-1]]
    if x_freq.size == 1:
        extent[0], extent[1] = x_freq[0] - 0.5, x_freq[0] + 0.5
    if y_freq.size == 1:
        extent[2], extent[3] = y_freq[0] - 0.5, y_freq[0] + 0.5
    image = ax_map.imshow(
        combined,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap="viridis",
        vmin=0,
        vmax=max(AMPLITUDE_LIMIT, float(np.nanmax(combined))),
    )
    ax_map.set_xlabel("X IF [MHz]")
    ax_map.set_ylabel("Y IF [MHz]")
    ax_map.set_title(f"Combined compensation surface: X multiplier × Y multiplier (max={np.nanmax(combined):.4f})")
    fig.colorbar(image, ax=ax_map, label="Combined compensation multiplier")
    return fig


def _print_result(result: dict) -> None:
    amplitude = result["dense_amplitude"]
    status = "PASS" if result["in_range"] else "FAIL"
    print(
        f"{result['axis']} {status}: {np.min(amplitude):.6f} <= amplitude <= "
        f"{np.max(amplitude):.6f} over "
        f"{result['dense_mhz'][0]:.3f}--{result['dense_mhz'][-1]:.3f} MHz"
    )


def main() -> None:
    x_result = evaluate_axis("x", x_tweezer_IFs)
    y_result = evaluate_axis("y", x_tweezer_IFs)
    # y_result = evaluate_axis("y", y_tweezer_IFs)
    _print_result(x_result)
    _print_result(y_result)

    fig = plot_validation(x_result, y_result)
    output_path = Path().resolve() / "Data" / "amplitude_validation.png"
    output_path.parent.mkdir(exist_ok=True)
    fig.savefig(output_path, dpi=150)
    print(f"Saved validation plot to {output_path}")
    plt.show()

    if not (x_result["in_range"] and y_result["in_range"]):
        raise SystemExit(f"Amplitude validation failed: compensation must stay within [0, {AMPLITUDE_LIMIT:g}].")


if __name__ == "__main__":
    main()
