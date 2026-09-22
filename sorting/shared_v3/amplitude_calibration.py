"""
Amplitude calibration utilities for AOD frequency-dependent compensation.

Typically called automatically at the end of 01_measure_amplitude_response.py
after collecting data. Can also be run standalone to re-fit from existing
datasets:

    python amplitude_calibration.py x
    python amplitude_calibration.py y

Running without an axis argument uses the legacy filenames
(amplitude_data.csv / amplitude_calibration_coefs.csv).
"""

from pathlib import Path
from typing import Iterable, Literal, NamedTuple

import matplotlib.pyplot as plt
import numpy as np
from qm.qua import *
from tqdm.auto import tqdm

MIN_WAIT = 4


class Calibration(NamedTuple):
    center: float
    coefs: list
    computation_latency_cc: int = MIN_WAIT
    latency_only: bool = False


def load_amplitude_data(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load frequency vs. amplitude measurements from CSV (header: freq, magnitude)."""
    data = np.genfromtxt(csv_path, delimiter=",", skip_header=1)
    if data.ndim == 1:
        data = np.expand_dims(data, axis=0)
    return data[:, 0], data[:, 1]


def computation_latency_key(file_name: str) -> str:
    """CSV row name for a script's processor gap, e.g. ``computation_latency_cc_02_arb_tone_MJ_manual``."""
    return f"computation_latency_cc_{Path(file_name).stem}"


def _latency_from_fields(fields: dict[str, str], file_name: str | None) -> int:
    if file_name:
        value = fields.get(computation_latency_key(file_name), "").strip()
        return int(float(value)) if value else MIN_WAIT
    value = fields.get("computation_latency_cc", "").strip()
    return int(float(value)) if value else MIN_WAIT


def load_calibration_coefs(
    csv_path: Path,
    unit_in_data: Literal["MHz", "GHz"] = "MHz",
    fixed_compatible: bool = True,
    file_name: str | None = None,
) -> Calibration:
    """Load saved polynomial coefficients from amplitude_calibration_coefs.csv.

    Returns a :class:`Calibration` ``(center_frequency_GHz, coefs, computation_latency_cc)``.
    *coefs* are ordered highest degree first.  When *fixed_compatible* is True,
    each coefficient is rewritten as ``(n, f)`` with ``coef == 2**n * f`` and
    ``1 <= abs(f) < 2`` (or zero), so *f* is a legal QUA fixed literal and the
    scale is applied with ``<< n`` / ``>> |n|``.

    When *file_name* is given, latency is read from ``computation_latency_cc_<stem>``.
    Missing rows yield :data:`MIN_WAIT`.
    """
    fields = _parse_calibration_csv(csv_path)
    if "center" not in fields or "coefs" not in fields:
        raise ValueError(f"{csv_path} must contain 'center' and 'coefs' rows")
    center = float(fields["center"])
    coefs = np.array([float(v) for v in fields["coefs"].split(",")])
    computation_latency_cc = _latency_from_fields(fields, file_name)

    if unit_in_data == "MHz":
        center /= 1e3
        coefs = coefs * (1e3 ** np.arange(len(coefs) - 1, -1, -1))

    if fixed_compatible:
        return Calibration(center, [_as_pow2_float(c) for c in coefs], computation_latency_cc)

    return Calibration(center, coefs.tolist(), computation_latency_cc)


def _as_pow2_float(coef: float) -> tuple[int, float]:
    """Return ``(n, f)`` such that ``coef == 2**n * f`` and ``1 <= abs(f) < 2``.

    Zero is ``(0, 0.0)``.  *f* always sits in the QUA fixed-point range [-8, 8).
    """
    coef = float(coef)
    if coef == 0.0:
        return 0, 0.0
    n = int(np.floor(np.log2(abs(coef))))
    return n, coef / (2.0**n)


def fixed_times(var, factor):
    """``var * factor`` without putting *factor* in a 4.28 literal.

    *factor* is a packed ``(n, f)`` with ``factor == 2**n * f`` and
    ``1 <= abs(f) < 2``, or a Python float (packed on the fly).  The product
    is ``(var << n) * f`` or ``(var >> -n) * f``.
    """
    n, f = factor if isinstance(factor, tuple) else _as_pow2_float(float(factor))
    if f == 0.0:
        return var * 0.0
    if n > 0:
        return (var << n) * f
    if n < 0:
        return (var >> -n) * f
    return var * f


def _coef_value(coef) -> float:
    """Reconstruct a packed ``(n, f)`` coefficient as a Python float."""
    n, f = coef
    return f * (2.0**n)


def _align_xy_coefs(x_coefs: list, y_coefs: list) -> tuple[list, list]:
    """Pad the high-degree side so X/Y polynomials have the same length.

    Missing terms are ``(0, 0.0)`` (i.e. ``2**0 * 0``) and do not change the
    polynomial.
    """
    n_terms = max(len(x_coefs), len(y_coefs))
    pad = (0, 0.0)
    x_coefs = [pad] * (n_terms - len(x_coefs)) + list(x_coefs)
    y_coefs = [pad] * (n_terms - len(y_coefs)) + list(y_coefs)
    return x_coefs, y_coefs


def load_xy_amp_calibration(
    directory: Path | None = None,
    latency_only: bool = False,
    unit_in_data: Literal["MHz", "GHz"] = "MHz",
    fixed_compatible: bool = True,
    file_name: str | None = None,
) -> tuple[Calibration, Calibration]:
    """Load per-axis amplitude calibration CSVs if present.

    Looks for ``x_amplitude_calibration_coefs.csv`` and
    ``y_amplitude_calibration_coefs.csv`` in *directory* (cwd by default).
    Missing files yield a latency-only :class:`Calibration` with
    ``computation_latency_cc=MIN_WAIT``.

    When *latency_only* is True, polynomial coefficients are not loaded and
    :func:`minimal_jerk.play_minimal_jerk_chirp` uses the constant-amplitude path.

    When both axes have coefficients, they are padded so they share the same
    polynomial length (missing high-degree terms are zero).

    Pass *file_name* (a script's ``SCRIPT_NAME``, since ``__file__`` is undefined
    under Jupyter) so each script loads its own ``computation_latency_cc_<stem>`` row.
    """
    directory = Path().resolve() if directory is None else Path(directory)
    latency_key = computation_latency_key(file_name) if file_name else "computation_latency_cc"
    calibrations: list[Calibration] = []
    for axis in ("x", "y"):
        csv_path = directory / f"{axis}_amplitude_calibration_coefs.csv"
        if csv_path.exists() and not latency_only:
            calibration = load_calibration_coefs(
                csv_path,
                unit_in_data=unit_in_data,
                fixed_compatible=fixed_compatible,
                file_name=file_name,
            )
            print(
                f"Loaded {axis.upper()} calibration "
                f"(center={calibration.center:.4f} GHz, "
                f"{latency_key}={calibration.computation_latency_cc})"
            )
        elif csv_path.exists():
            fields = _parse_calibration_csv(csv_path)
            latency = _latency_from_fields(fields, file_name)
            calibration = Calibration(0.0, [], latency, latency_only=True)
            print(f"Loaded {axis.upper()} {latency_key}={latency} (latency only)")
        else:
            print(f"No {csv_path.name} -- {axis.upper()} {latency_key}={MIN_WAIT}")
            calibration = Calibration(0.0, [], latency_only=True)
        calibrations.append(calibration)
    x_cal, y_cal = calibrations[0], calibrations[1]
    if not x_cal.latency_only and not y_cal.latency_only:
        x_coefs, y_coefs = _align_xy_coefs(x_cal.coefs, y_cal.coefs)
        x_cal = x_cal._replace(coefs=x_coefs)
        y_cal = y_cal._replace(coefs=y_coefs)
    return x_cal, y_cal


def _parse_calibration_csv(csv_path: Path) -> dict[str, str]:
    fields: dict[str, str] = {}
    for line in csv_path.read_text(encoding="utf-8").strip().splitlines():
        if not line.strip() or "," not in line:
            continue
        key, rest = line.split(",", 1)
        fields[key.strip()] = rest.strip()
    return fields


def upsert_computation_latency_cc(csv_path: Path, computation_latency_cc: int, file_name: str) -> None:
    """Insert or replace ``computation_latency_cc_<stem>``, leaving other rows intact."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    key = computation_latency_key(file_name)
    lines = csv_path.read_text(encoding="utf-8").strip().splitlines()
    row = f"{key},{int(computation_latency_cc)}"
    updated = False
    out = []
    for line in lines:
        line_key = line.split(",", 1)[0].strip() if "," in line else line.strip()
        if line_key == key:
            out.append(row)
            updated = True
        else:
            out.append(line)
    if not updated:
        out.append(row)
    csv_path.write_text("\n".join(out) + "\n", encoding="utf-8")


def _fetch_timestamp_values(job, stream_name: str) -> np.ndarray:
    try:
        handle = job.result_handles.get(stream_name)
    except Exception:
        return np.array([], dtype=np.int64)
    if handle is None:
        return np.array([], dtype=np.int64)
    try:
        data = handle.fetch_all()
    except Exception:
        return np.array([], dtype=np.int64)
    if data is None:
        return np.array([], dtype=np.int64)
    if isinstance(data, dict):
        data = data.get("value", data.get("data", []))
    elif getattr(data, "dtype", None) is not None and getattr(data.dtype, "names", None):
        if "value" in data.dtype.names:
            data = data["value"]
    values = np.asarray(data).reshape(-1)
    if values.size and isinstance(values.flat[0], dict):
        values = np.array([v.get("value", v) for v in values.tolist()])
    if values.size == 0:
        return np.array([], dtype=np.int64)
    return np.asarray(values, dtype=np.int64).reshape(-1)


def _mode_diff_cc(timestamps: np.ndarray) -> int | None:
    if timestamps.size < 2:
        return None
    diffs = np.diff(timestamps)
    diffs = diffs[diffs > 0]
    if diffs.size == 0:
        return None
    values, counts = np.unique(diffs, return_counts=True)
    return int(values[np.argmax(counts)])


def _plot_timestamp_dt_histograms(timestamps_by_axis: dict[str, dict[str, np.ndarray]]) -> None:
    """Histogram of ``np.diff(t)`` for the reference stream on each axis."""
    all_dt = []
    for axis in ("x", "y"):
        streams = timestamps_by_axis[axis]
        if not streams:
            continue
        reference = streams[next(iter(streams))]
        if reference.size >= 2:
            all_dt.append(np.diff(reference))
    if not all_dt:
        return

    all_dt = np.concatenate(all_dt)
    lo = float(all_dt.min())
    hi = float(all_dt.max())
    if lo == hi:
        lo -= 0.5
        hi += 0.5
    shared_bins = np.linspace(lo, hi, 101)

    fig, axs = plt.subplots(2, 1, figsize=(5, 5), sharex=True, constrained_layout=True)
    for ax, axis in zip(axs, ("x", "y")):
        streams = timestamps_by_axis[axis]
        if not streams:
            ax.set_visible(False)
            continue
        reference_name = next(iter(streams))
        reference = streams[reference_name]
        if reference.size < 2:
            ax.set_visible(False)
            continue
        dt = np.diff(reference)
        ax.hist(dt, bins=shared_bins)
        ax.set_yscale("log")
        mode_dt = _mode_diff_cc(reference)
        if mode_dt is not None:
            ax.axvline(mode_dt, ls="--", color="r", lw=1.5, label=f"segment = {mode_dt} cc = {mode_dt * 4} ns")
        ax.set_title(f"{axis.upper()} axis: {reference_name}")
        ax.set_ylabel("Count")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(fontsize=9)
    axs[-1].set_xlabel(r"$\Delta t$ [cc]")
    fig.suptitle("Segment Latency Check")
    plt.show()


def _element_axis(element: str) -> str | None:
    if element.startswith("x"):
        return "x"
    if element.startswith("y"):
        return "y"
    return None


def check_tweezer_computation_latency(
    job,
    elements: Iterable[str],
    segment_length_ns: int,
    x_calibration: Calibration | None = None,
    y_calibration: Calibration | None = None,
    tolerance_cc: int = 0,
    directory: Path | None = None,
    file_name: str | None = None,
) -> bool:
    """Verify per-tweezer chirp timing and update each axis's per-script latency.

    Pass the same per-axis calibrations the QUA program was built with, so the
    baseline is what actually ran rather than whatever is on disk now (a previous
    run may already have written a corrected value).  The residual timing error is
    ``delta = mode(np.diff(t)) - segment_length//4``; the corrected latency is the
    *sum* ``loaded_latency + delta`` and is written back as
    ``computation_latency_cc_<stem>`` on that axis's CSV.
    A ``delta`` of 0 means the latency is already correct (printed green);
    otherwise the corrected value is printed red.

    Empty streams (tweezer never played) are skipped.  Within an axis, non-empty
    equal-length streams must be identical and share the same
    ``mode(np.diff(t))``; otherwise computation time is not calibrated and a
    ``RuntimeError`` is raised.  A negative ``corrected_latency`` also raises.

    *tolerance_cc* applies only across axes: ``|latency_x - latency_y|`` may
    differ by that many clock cycles.  Tweezers on the same axis must match
    exactly.

    Returns True when every measured axis has ``delta == 0``.
    """
    GREEN = "\033[92m"
    RED = "\033[91m"
    RESET = "\033[0m"

    directory = Path().resolve() if directory is None else Path(directory)
    segment_cc = int(segment_length_ns) // 4
    calibration_by_axis = {"x": x_calibration, "y": y_calibration}

    timestamps_by_axis: dict[str, dict[str, np.ndarray]] = {"x": {}, "y": {}}
    for element in tqdm(elements, desc="Fetching timestamps"):
        axis = _element_axis(element)
        if axis is None:
            continue
        values = _fetch_timestamp_values(job, f"t_{element}")
        if values.size == 0:
            continue
        timestamps_by_axis[axis][element] = values

    corrected_by_axis: dict[str, int] = {}
    deltas_by_axis: dict[str, int] = {}
    mismatches: list[str] = []

    for axis in ("x", "y"):
        streams = timestamps_by_axis[axis]
        if not streams:
            continue

        names = list(streams)
        reference_name = names[0]
        reference = streams[reference_name]
        reference_mode = _mode_diff_cc(reference)
        for name, values in streams.items():
            mode_dt = _mode_diff_cc(values)
            if values.size == reference.size and not np.array_equal(values, reference):
                mismatches.append(f"{name} timestamps != {reference_name}")
            elif mode_dt != reference_mode:
                mismatches.append(f"{name} mode(diff)={mode_dt} != {reference_name} mode(diff)={reference_mode}")

        if reference_mode is None:
            print(f"{axis.upper()} axis: too few timestamps to measure computation_latency_cc")
            continue

        # Baseline is the latency this axis's chirps were actually compiled with.
        calibration = calibration_by_axis[axis]
        if calibration is None:
            print(f"{axis.upper()} axis: no loaded calibration -- skipping")
            continue
        loaded_latency = int(calibration.computation_latency_cc)

        csv_path = directory / f"{axis}_amplitude_calibration_coefs.csv"
        delta = reference_mode - segment_cc
        corrected_latency = loaded_latency + delta
        corrected_by_axis[axis] = corrected_latency
        deltas_by_axis[axis] = delta
        if corrected_latency < 0:
            mismatches.append(
                f"{axis} axis negative computation_latency: "
                f"loaded={loaded_latency}, mode(diff)={reference_mode}, "
                f"segment_cc={segment_cc}, delta={delta}, corrected={corrected_latency}"
            )

        latency_key = computation_latency_key(file_name) if file_name else "computation_latency_cc"
        if delta != 0 and csv_path.exists() and file_name and corrected_latency >= 0:
            upsert_computation_latency_cc(csv_path, corrected_latency, file_name)
            print(f"Updated {csv_path} with {latency_key}={corrected_latency}")

        color, status = (GREEN, "correct") if delta == 0 else (RED, "needs reload")
        print(
            f"{color}{axis.upper()} axis {latency_key} {status}: "
            f"loaded={loaded_latency}, mode(diff)={reference_mode}, segment_cc={segment_cc}, "
            f"delta={delta}, corrected={corrected_latency}{RESET}"
        )

    _plot_timestamp_dt_histograms(timestamps_by_axis)

    if not corrected_by_axis and not mismatches:
        print("No tweezer timestamp streams -- skipping computation_latency_cc check")
        return True

    if corrected_by_axis and max(corrected_by_axis.values()) - min(corrected_by_axis.values()) > tolerance_cc:
        raise RuntimeError(
            f"Each axis has a different computation_latency_cc: {corrected_by_axis}"
            + (f" (tolerance_cc={tolerance_cc})" if tolerance_cc else "")
        )

    if mismatches:
        raise RuntimeError("Computation time is not calibrated: " + "; ".join(mismatches))

    return all(d == 0 for d in deltas_by_axis.values())


# ---------------------------------------------------------------------------
# Python-side polynomial evaluation (for plotting / verification)
# ---------------------------------------------------------------------------


def amp_at_frequency_py(
    freq: float | np.ndarray,
    center: float,
    coefs: Iterable,
) -> float | np.ndarray:
    """Evaluate the compensation polynomial in Python.

    *freq* and *center* are in GHz.  *coefs* is ordered highest degree first
    and may contain ``(n, f)`` entries with ``coef == 2**n * f``.
    """
    x = freq - center
    result = 0.0
    coefs = list(coefs)
    for i, coef in enumerate(coefs):
        power = len(coefs) - 1 - i
        result = result + _coef_value(coef) * x**power
    return result


# ---------------------------------------------------------------------------
# QUA real-time polynomial evaluation
# ---------------------------------------------------------------------------


def amp_at_frequency_qua(freq, center, coefs, x_terms=None):
    """Generate QUA code that evaluates the compensation polynomial.

    *freq* is a QUA fixed variable (frequency in GHz).  *center* is a Python
    float (GHz).  *coefs* is the packed list from load_calibration_coefs, each
    entry ``(n, f)`` with ``coef == 2**n * f``.  Each term is ``x**i`` then
    :func:`fixed_times` with that pair (``n`` reduced by ``i * shift_bit`` so
    the x-scaling does not overflow 4.28).

    Returns a QUA expression (fixed) for the amplitude scaling factor.
    """
    # Typical |freq - center| is <= 0.1 GHz.  Left-shifting by *shift_bit*
    # scales that to O(1) so high powers of x do not underflow in 4.28.
    # The coefficient scale must then undo 2**(i * shift_bit), not 2**shift_bit.
    shift_bit = 4
    x = declare(fixed)
    assign(x, (freq - center) << shift_bit)

    if x_terms is None:
        x_terms = [declare(fixed) for _ in coefs]

    # coefs: highest degree first  ->  reverse to align with x^0, x^1, ...
    for i, (coef, x_term) in enumerate(zip(coefs[::-1], x_terms)):
        n, f = coef
        if i == 0:
            assign(x_term, 1.0)
        else:
            factors = ["x"] * i
            op_str = ") * ".join(factors)
            op_str = "(" * (len(factors) - 1) + op_str
            assign(x_term, eval(op_str))
        assign(x_term, fixed_times(x_term, (n - i * shift_bit, f)))

    op_str = " + ".join(f"x_terms[{i}]" for i in range(len(x_terms)))
    return eval(op_str)


# ---------------------------------------------------------------------------
# Interactive fitting (run this file directly)
# ---------------------------------------------------------------------------


def main(axis: str | None = None, max_order: int = 6, data_type: Literal["dBm", "V"] = "V") -> None:
    if axis:
        csv_path = Path().resolve() / f"{axis}_amplitude_data.csv"
        output_name = f"{axis}_amplitude_calibration_coefs.csv"
    else:
        csv_path = Path().resolve() / "amplitude_data.csv"
        output_name = "amplitude_calibration_coefs.csv"

    axis_label = f" ({axis.upper()} axis)" if axis else ""
    print(f"Loading{axis_label}: {csv_path}")
    x, y = load_amplitude_data(csv_path)

    x_sorted_idx = np.argsort(x)
    x = x[x_sorted_idx]
    y = y[x_sorted_idx]

    orders = list(range(1, max_order + 1))
    residual_sums = []
    coefs_by_order: dict[int, np.ndarray] = {}

    if data_type == "dBm":
        y = 10 ** (y / 20)

    center_idx = len(x) // 2
    center = x[center_idx]
    x_centered = x - center
    y_centered = y[center_idx]

    # If max exceeds y_centered by a factor of 2, shift center to the max
    # to stay within the amp() multiplier range [0, 2].
    max_idx = np.argmax(y)
    if y[max_idx] > y_centered * 2:
        center_idx = max_idx
        center = x[max_idx]
        x_centered = x - center
        y_centered = y[max_idx]

    print(f"center frequency: {center:.3f} MHz")
    print(f"min/max range ratio: {np.min(y) / y_centered:.4f}  {np.max(y) / y_centered:.4f}")

    y_relative = y_uncomp = y / y_centered
    y_relative = 1 / y_relative  # invert to get compensation curve

    fig, (ax_raw, ax_comp, ax_resid) = plt.subplots(3, 1, figsize=(8, 10), sharex=False)

    # --- Top: polynomial fit to the compensation (inverse) curve ---
    ax_raw.scatter(x, y_relative, label="1 / data (target)", color="k", zorder=5)

    for order in orders:
        coefs = np.polyfit(x_centered, y_relative, order)
        coefs_by_order[order] = coefs

        x_dense = np.linspace(x_centered[0], x_centered[-1], 200)
        y_fit_dense = np.polyval(coefs, x_dense)
        ax_raw.plot(x_dense + center, y_fit_dense, label=f"order {order}")

        y_fit = np.polyval(coefs, x_centered)
        y_comp = y_uncomp * y_fit
        residual_sum = np.sum((y_comp - 1) ** 2)
        residual_sums.append(residual_sum)

        ax_comp.plot(x, y_comp, label=f"order {order}")
        print(f"order {order}: SSE = {residual_sum:.6e}")

    ax_raw.set_xlabel("Frequency (MHz)")
    ax_raw.set_ylabel("Compensation multiplier")
    ax_raw.set_title(f"Polynomial fit to inverse response (centered at {center:.1f} MHz)")
    ax_raw.legend(ncols=2, fontsize=9)
    ax_raw.grid(True, linestyle="--", alpha=0.4)

    # --- Middle: compensated result per order ---
    ax_comp.scatter(x, y_uncomp, label="data (uncompensated)", color="k", zorder=5)
    ax_comp.axhline(1, color="gray", linestyle="--", label="ideal")
    ax_comp.set_xlabel("Frequency (MHz)")
    ax_comp.set_ylabel("Relative magnitude")
    ax_comp.set_title("Compensated magnitude (data * fit)")
    ax_comp.legend(ncols=2, fontsize=9)
    ax_comp.grid(True, linestyle="--", alpha=0.4)

    # --- Bottom: residual vs order ---
    ax_resid.plot(orders, residual_sums, marker="o", color="tab:purple")
    ax_resid.set_yscale("log")
    ax_resid.set_xlabel("Polynomial order")
    ax_resid.set_ylabel("Total residual (SSE)")
    ax_resid.set_title("Residual vs polynomial order")
    ax_resid.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.show()

    order_input = input("Enter polynomial order to save [press Enter to skip]: ").strip()
    if not order_input:
        return
    try:
        order_to_save = int(order_input)
    except ValueError:
        print(f"Invalid input: {order_input}")
        return

    if order_to_save not in coefs_by_order:
        print(f"Order {order_to_save} not available.")
        return

    coefs = coefs_by_order[order_to_save]
    output_path = Path().resolve() / output_name
    coefs_str = ",".join(str(c) for c in coefs)
    text = f"center,{center}\ncoefs,{coefs_str}"
    if output_path.exists():
        for key, value in _parse_calibration_csv(output_path).items():
            if key.startswith("computation_latency_cc") and value.strip():
                text += f"\n{key},{value}"
    output_path.write_text(text + "\n", encoding="utf-8")
    print(f"Saved order-{order_to_save} coefficients to {output_path}")


if __name__ == "__main__":
    import sys

    _axis = sys.argv[1] if len(sys.argv) > 1 else None
    main(_axis)
