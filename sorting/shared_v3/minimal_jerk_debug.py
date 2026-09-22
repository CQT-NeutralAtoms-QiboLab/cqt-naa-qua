# %%
#! %load_ext autoreload
#! %autoreload 2
"""Debug duplicate of ``minimal_jerk._play_compensated``.

Streams every QUA variable so it can be compared with a Python replay of
the same arithmetic. Run this file to simulate one X tweezer, fetch the
streams, and plot QUA vs Python.

    python minimal_jerk_debug.py
"""

from pathlib import Path

import numpy as np
from amplitude_calibration import amp_at_frequency_py, amp_at_frequency_qua, fixed_times
from qm.qua import *

STREAM_VARS = (
    "tau",
    "tau2",
    "chirp_rate",
    "cur_frequency",
    "freq_ghz",
    "amp_mod",
    "ramp_rate",
    "cumsum_v",
)


def play_compensated_debug(
    element,
    start_frequency,
    detuning,
    segment_length,
    number_of_segments,
    amp_calibration,
):
    """
    Minimal-jerk chirp with amplitude compensation.

    Hardware model assumed here:

        - The DAC starts each ``play(ramp)`` from a 16-bit code.
        - The value we track (``cumsum_v``) is that same 16-bit code.
        - The slope itself is Q4.28 and may be used at full precision
          during the ramp.
        - When the pulse ends, bits below 16-bit are forgotten.

    Therefore do **not** snap ``target - dac`` onto the DAC grid before
    forming the slope.  That turns a few µV of error into a full ±1 LSB
    bang-bang.  Use the 28-bit residual for the rate; forget the
    remainder only in the persistent DAC state:

        ramp_rate_k = Q28((target_k - dac_state_k) / play_ns)
        dac_state_{k+1} = Q16(dac_state_k + play_ns * ramp_rate_k)
    """

    from configuration import chirp_pulse_amplitude

    center_ghz = amp_calibration.center
    coefs = amp_calibration.coefs
    computation_latency_cc = amp_calibration.computation_latency_cc

    streams = {name: declare_stream() for name in STREAM_VARS}

    # ------------------------------------------------------------
    # Constants
    # ------------------------------------------------------------

    fixed_bits = 28
    dac_bits = 16

    # Q4.28 -> effective DAC16 amplitude grid
    dac_bitshift = fixed_bits - dac_bits  # 12

    inv_number_of_segments = 1.0 / (number_of_segments + 1)

    play_ns = segment_length - 4 * computation_latency_cc
    inv_play_ns = 1.0 / play_ns

    inv_amp = 1.0 / chirp_pulse_amplitude

    chirp_scaling = 30.0 * 1e3 / ((number_of_segments + 1) * segment_length)

    qua_Hz_to_GHz = lambda f: (2**fixed_bits / 1e9) * Cast.unsafe_cast_fixed(f)

    # ------------------------------------------------------------
    # Variables
    # ------------------------------------------------------------

    tau = declare(fixed)
    tau2 = declare(fixed)

    chirp_rate = declare(int)

    cur_frequency = declare(
        int,
        value=0,
    )

    assign(
        cur_frequency,
        start_frequency,
    )

    freq_ghz = declare(fixed)

    amp_mod = declare(fixed)

    # Ideal target amplitude for this segment.
    target_v = declare(fixed)

    # Difference between target and current DAC16 state.
    diff_v = declare(fixed)

    # Q4.28 ramp slope.
    ramp_rate = declare(fixed, value=0.0)

    # Persistent 16-bit DAC code, stored in a Q4.28 variable whose
    # lowest 12 fractional bits are kept at zero.
    cumsum_v = declare(fixed)

    # Optional: unquantized endpoint before applying DAC16 truncation.
    endpoint_v = declare(fixed)

    # ------------------------------------------------------------
    # Initial state
    # ------------------------------------------------------------

    assign(
        freq_ghz,
        qua_Hz_to_GHz(cur_frequency),
    )

    assign(
        amp_mod,
        amp_at_frequency_qua(
            freq_ghz,
            center_ghz,
            coefs,
        ),
    )

    assign(
        target_v,
        amp_mod * chirp_pulse_amplitude,
    )

    # Initial persistent DAC state:
    #
    #     Q16(target_v)
    #
    # cumsum_v remains represented as Q4.28, but its lowest 12
    # fractional bits are zero.
    assign(
        cumsum_v,
        (target_v >> dac_bitshift) << dac_bitshift,
    )

    _save_vars(
        streams,
        tau,
        tau2,
        chirp_rate,
        cur_frequency,
        freq_ghz,
        amp_mod,
        ramp_rate,
        cumsum_v,
    )

    # ------------------------------------------------------------
    # Initial output
    # ------------------------------------------------------------

    update_frequency(
        element,
        start_frequency,
    )

    play(
        "rampup" * amp(amp_mod),
        element,
    )

    # ------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------

    with for_(
        tau,
        inv_number_of_segments,
        tau < (1 - 0.5 * inv_number_of_segments),
        tau + inv_number_of_segments,
    ):
        # --------------------------------------------------------
        # Minimal-jerk chirp
        # --------------------------------------------------------

        assign(
            tau2,
            tau * tau,
        )

        assign(
            chirp_rate,
            Cast.mul_int_by_fixed(
                detuning,
                chirp_scaling * tau2 * (1 - 2 * tau + tau2),
            ),
        )

        assign(
            cur_frequency,
            cur_frequency
            + Cast.mul_int_by_fixed(
                chirp_rate,
                1e-3 * segment_length,
            ),
        )

        assign(
            freq_ghz,
            qua_Hz_to_GHz(cur_frequency),
        )

        assign(
            amp_mod,
            amp_at_frequency_qua(
                freq_ghz,
                center_ghz,
                coefs,
            ),
        )

        # --------------------------------------------------------
        # Ideal target amplitude
        # --------------------------------------------------------

        assign(
            target_v,
            amp_mod * chirp_pulse_amplitude,
        )

        # --------------------------------------------------------
        # Error relative to persistent DAC state
        # --------------------------------------------------------
        #
        # cumsum_v is already on the DAC16 grid here.
        # Keep the full Q4.28 residual for the slope.
        # --------------------------------------------------------

        assign(diff_v, target_v - cumsum_v)

        # --------------------------------------------------------
        # Ramp slope (Q4.28).  Negative diff_v ramps down directly.
        # --------------------------------------------------------

        assign(
            ramp_rate,
            diff_v * inv_play_ns,
        )

        # --------------------------------------------------------
        # Compute actual Q4.28 endpoint of this ramp
        # --------------------------------------------------------
        #
        # During the segment:
        #
        #     V[n] = cumsum_v + n * ramp_rate
        #
        # The transient accumulator may contain bits below DAC16.
        # --------------------------------------------------------

        assign(
            endpoint_v,
            cumsum_v
            + Cast.mul_fixed_by_int(
                ramp_rate,
                play_ns,
            ),
        )

        # --------------------------------------------------------
        # Persistent state for next segment
        # --------------------------------------------------------
        #
        # The hardware forgets the sub-DAC-LSB remainder when the
        # next ramp begins.
        #
        # Therefore:
        #
        #     cumsum_v = Q16(endpoint_v)
        # --------------------------------------------------------

        assign(
            cumsum_v,
            ((endpoint_v >> dac_bitshift) << dac_bitshift),
        )

        # --------------------------------------------------------
        # Output
        # --------------------------------------------------------

        play(
            ramp(ramp_rate),
            element,
            duration=play_ns // 4,
            chirp=(
                chirp_rate,
                "mHz/nsec",
            ),
            continue_chirp=True,
            timestamp_stream=f"t_{element}",
        )

        _save_vars(
            streams,
            tau,
            tau2,
            chirp_rate,
            cur_frequency,
            freq_ghz,
            amp_mod,
            ramp_rate,
            cumsum_v,
        )

    # ------------------------------------------------------------
    # Final rampdown
    # ------------------------------------------------------------

    play(
        "rampdown"
        * amp(
            fixed_times(
                cumsum_v,
                inv_amp,
            )
        ),
        element,
        continue_chirp=False,
    )

    ramp_to_zero(
        element,
    )

    return streams


def stream_save_debug(streams, element):
    """Call inside ``with stream_processing()``."""
    for name, stream in streams.items():
        stream.save_all(f"{name}_{element}")


def _save_vars(streams, tau, tau2, chirp_rate, cur_frequency, freq_ghz, amp_mod, ramp_rate, cumsum_v):
    save(tau, streams["tau"])
    save(tau2, streams["tau2"])
    save(chirp_rate, streams["chirp_rate"])
    save(cur_frequency, streams["cur_frequency"])
    save(freq_ghz, streams["freq_ghz"])
    save(amp_mod, streams["amp_mod"])
    save(ramp_rate, streams["ramp_rate"])
    save(cumsum_v, streams["cumsum_v"])


FIXED_BITS = 28
DAC_BITS = 16
DAC_SHIFT = FIXED_BITS - DAC_BITS
Q28 = 1 << FIXED_BITS


def _to_q28(x):
    return int(np.round(np.float64(x) * Q28))


def _from_q28(i):
    return float(i) / Q28


def _q28_mul(a, b):
    """QUA ``fixed * fixed``: both Q4.28, result Q4.28."""
    return _from_q28((_to_q28(a) * _to_q28(b)) >> FIXED_BITS)


def _q28_mul_int(a, n):
    """QUA ``Cast.mul_fixed_by_int``."""
    return _from_q28(_to_q28(a) * int(n))


def _q16_forget(x):
    """Drop bits below the 16-bit DAC (arithmetic ``>>``, same as QUA)."""
    i = _to_q28(x)
    return _from_q28((i >> DAC_SHIFT) << DAC_SHIFT)


def replay_compensated_python(
    start_frequency,
    detuning,
    segment_length,
    number_of_segments,
    amp_calibration,
    chirp_pulse_amplitude,
    wf_amplitude,
):
    """Python replay of the compensated loop, including the DAC 16/28 split."""
    center, coefs = amp_calibration.center, amp_calibration.coefs
    latency_cc = int(amp_calibration.computation_latency_cc)
    inv = 1 / (number_of_segments + 1)
    chirp_scaling = 30 * 1e3 / ((number_of_segments + 1) * segment_length)
    play_ns = int(segment_length - 4 * latency_cc)
    inv_play_ns = 1.0 / play_ns

    def poly(freq_hz):
        return float(amp_at_frequency_py(freq_hz / 1e9, center, coefs))

    def mul_int_by_fixed(integer, fixed_val):
        return int(np.round(integer * fixed_val))

    cur_frequency = int(start_frequency)
    freq_ghz = cur_frequency * 1e-9
    amp_mod = poly(cur_frequency)
    target_v = _q28_mul(amp_mod, chirp_pulse_amplitude)
    cumsum_v = _q16_forget(target_v)

    rows = {
        "tau": [0.0],
        "tau2": [0.0],
        "chirp_rate": [0],
        "cur_frequency": [cur_frequency],
        "freq_ghz": [freq_ghz],
        "amp_mod": [amp_mod],
        "ramp_rate": [0.0],
        "cumsum_v": [cumsum_v],
        "target_v": [target_v],
        "endpoint_v": [cumsum_v],
    }

    tau = inv
    while tau < (1 - 0.5 * inv):
        tau2 = tau * tau
        chirp_rate = mul_int_by_fixed(detuning, chirp_scaling * tau2 * (1 - 2 * tau + tau2))
        cur_frequency = cur_frequency + mul_int_by_fixed(chirp_rate, 1e-3 * segment_length)
        freq_ghz = cur_frequency * 1e-9
        amp_mod = poly(cur_frequency)
        target_v = _q28_mul(amp_mod, chirp_pulse_amplitude)
        diff_v = target_v - cumsum_v
        ramp_rate = _q28_mul(diff_v, inv_play_ns)
        endpoint_v = cumsum_v + _q28_mul_int(ramp_rate, play_ns)
        cumsum_v = _q16_forget(endpoint_v)
        rows["tau"].append(tau)
        rows["tau2"].append(tau2)
        rows["chirp_rate"].append(chirp_rate)
        rows["cur_frequency"].append(cur_frequency)
        rows["freq_ghz"].append(freq_ghz)
        rows["amp_mod"].append(amp_mod)
        rows["ramp_rate"].append(ramp_rate)
        rows["cumsum_v"].append(cumsum_v)
        rows["target_v"].append(target_v)
        rows["endpoint_v"].append(endpoint_v)
        tau = tau + inv

    out = {k: np.asarray(v) for k, v in rows.items()}
    # Analog voltage at the end of each play, before the DAC forgets the remainder.
    out["sticky_level"] = out["endpoint_v"]
    out["rampdown_target"] = wf_amplitude * out["amp_mod"]
    return out


def fetch_stream(job, name):
    handle = job.result_handles.get(name)
    data = handle.fetch_all()
    if data is None:
        return np.array([])
    if isinstance(data, dict):
        data = data.get("value", data.get("data", []))
    elif getattr(data, "dtype", None) is not None and getattr(data.dtype, "names", None):
        if "value" in data.dtype.names:
            data = data["value"]
    return np.asarray(data).reshape(-1)


def fetch_debug_streams(job, element):
    return {name: fetch_stream(job, f"{name}_{element}") for name in STREAM_VARS}


def plot_qua_vs_python(
    qua,
    py,
    wf_amplitude,
    save_path: Path,
    title: str,
    trace=None,
    rampup_ns=1000,
    segment_ns=1000,
    play_ns=792,
    rampdown_ns=1000,
):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(4, 2, figsize=(12, 11))
    fig.suptitle(title)

    def panel(ax, key, scale=1.0, ylabel=""):
        n = min(len(qua[key]), len(py[key]))
        x = np.arange(n)
        ax.plot(x, py[key][:n] * scale, label="python", lw=1.4)
        ax.plot(x, qua[key][:n] * scale, label="QUA", lw=0.9, alpha=0.85)
        ax.set_ylabel(ylabel or key)
        ax.grid(True, alpha=0.35)
        ax.set_xlabel("segment")

    panel(axes[0, 0], "tau")
    panel(axes[0, 1], "chirp_rate", ylabel="chirp_rate (mHz/ns)")
    panel(axes[1, 0], "cur_frequency", scale=1e-6, ylabel="cur_frequency (MHz)")
    panel(axes[1, 1], "freq_ghz", scale=1e3, ylabel="freq_ghz (MHz)")
    panel(axes[2, 0], "amp_mod")
    n_rr = min(len(qua["ramp_rate"]), len(py["ramp_rate"]))
    x_rr = np.arange(n_rr)
    axes[2, 1].plot(x_rr, py["ramp_rate"][:n_rr], label="python", lw=1.4)
    axes[2, 1].plot(x_rr, qua["ramp_rate"][:n_rr], label="QUA", lw=0.9, alpha=0.85)
    qua_rr_mean = _centered_rolling_mean(np.asarray(qua["ramp_rate"][:n_rr], dtype=float), 51)
    axes[2, 1].plot(x_rr, qua_rr_mean, label="QUA roll 51", lw=1.6, zorder=3)
    axes[2, 1].axhline(0.0, color="k", lw=0.5, alpha=0.4)
    axes[2, 1].set_ylabel("ramp_rate (V/ns)")
    axes[2, 1].set_xlabel("segment")
    axes[2, 1].grid(True, alpha=0.35)
    axes[2, 1].legend(loc="best", fontsize=8)

    t_py, v_py = _sticky_polyline(py["sticky_level"], rampup_ns, segment_ns, play_ns)
    t_qua, v_qua = _sticky_polyline(qua["cumsum_v"], rampup_ns, segment_ns, play_ns)
    t_tgt, v_tgt = _sticky_polyline(py["rampdown_target"], rampup_ns, segment_ns, play_ns)
    pulse_end_us = float(t_qua[-1]) * 1e-3

    if trace is not None:
        t_env, env, dc = _ac_envelope(trace)
        gain, offset = _envelope_affine(t_env, env, t_py, v_py, rampup_ns=rampup_ns, rampdown_ns=rampdown_ns)
        env_aligned = gain * env + offset
        dt_ns = float(t_env[1] - t_env[0]) if len(t_env) > 1 else 20.0
        roll_us = 1.0
        n_roll = max(3, int(round(roll_us * 1e3 / dt_ns)))
        # Leave a half-window before rampdown so the centered mean never mixes in the drop to 0.
        t_body_lo = float(rampup_ns)
        t_body_hi = float(t_qua[-1]) - float(rampdown_ns) - 0.5 * n_roll * dt_ns
        in_body = (t_env >= t_body_lo) & (t_env <= t_body_hi)
        env_roll = _centered_rolling_mean(np.where(in_body, env_aligned, np.nan), n_roll)
        roll_ok = np.isfinite(env_roll) & in_body
        v0 = _print_envelope_vs_sticky(t_env, env_roll, t_py, v_py, rampup_ns=rampup_ns, rampdown_ns=rampdown_ns)
        for ax in (axes[3, 0], axes[3, 1]):
            ax.plot(
                t_env[in_body] * 1e-3,
                env_aligned[in_body],
                label=f"envelope {gain:.3f}·x{offset * 1e3:+.1f} mV",
                lw=0.4,
                alpha=0.25,
                zorder=0,
            )
            ax.plot(
                t_env[roll_ok] * 1e-3,
                env_roll[roll_ok],
                label=f"envelope roll {roll_us:.0f} us",
                lw=1.6,
                zorder=2,
            )
            if v0 is not None:
                ax.axhline(v0, color="0.5", lw=0.8, ls=":", label="start level")
        print(
            f"  envelope DC removed = {dc * 1e3:+.2f} mV,  "
            f"align sticky = {gain:.4f} * env + {offset * 1e3:+.2f} mV"
        )

    t_py, v_py = _drop_idle_zero(t_py, v_py)
    t_qua, v_qua = _drop_idle_zero(t_qua, v_qua)
    t_tgt, v_tgt = _drop_idle_zero(t_tgt, v_tgt)

    for ax in (axes[3, 0], axes[3, 1]):
        ax.plot(t_tgt * 1e-3, v_tgt, label="python target", lw=1.0, alpha=0.7)
        ax.plot(t_py * 1e-3, v_py, label="python analog end", lw=1.4)
        ax.plot(t_qua * 1e-3, v_qua, label="QUA DAC16", lw=0.9, alpha=0.85)

    body = np.concatenate([v_py, v_qua, v_tgt])
    body = body[np.abs(body) > 0.5 * np.max(np.abs(body))]
    if trace is not None and np.any(roll_ok):
        body = np.concatenate([body, env_roll[roll_ok], env_aligned[in_body]])
    y_min, y_max = np.nanpercentile(body, [1.0, 99.0])
    y_pad = max(0.15 * (y_max - y_min), 1e-4)

    axes[3, 0].set_ylabel("envelope / sticky (V)")
    axes[3, 0].set_xlabel("time (us)")
    axes[3, 0].set_xlim(0, pulse_end_us + 4)
    axes[3, 0].set_ylim(y_min - y_pad, y_max + y_pad)
    axes[3, 0].grid(True, alpha=0.35)
    axes[3, 0].legend(loc="best", fontsize=8)

    axes[3, 1].set_ylabel("envelope / sticky (V)")
    axes[3, 1].set_xlabel("time (us)")
    axes[3, 1].set_xlim(pulse_end_us - 6, pulse_end_us + 4)
    axes[3, 1].set_ylim(y_min - y_pad, y_max + y_pad)
    axes[3, 1].grid(True, alpha=0.35)
    axes[3, 1].legend(loc="best", fontsize=8)

    axes[0, 0].legend(loc="best", fontsize=8)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    print(f"Saved {save_path}")
    return fig


def _sticky_polyline(sticky, rampup_ns, segment_ns, play_ns):
    """Commanded sticky voltage vs time.

    Covers the whole pulse: the opening rampup, the per-segment ramps held across
    the latency gap, then the closing rampdown back to zero.  Padding both ends
    means the curve can be compared against the measured envelope edge for edge.
    """
    sticky = np.asarray(sticky, dtype=float)
    t = [0.0]
    v = [0.0]
    t.append(float(rampup_ns))
    v.append(float(sticky[0]))
    t_play = float(rampup_ns)
    for i in range(1, len(sticky)):
        t.append(t_play + play_ns)
        v.append(float(sticky[i]))
        t_play += segment_ns
        t.append(t_play)
        v.append(float(sticky[i]))
    return np.asarray(t), np.asarray(v)


def _drop_idle_zero(t, v):
    """Trim leading/trailing zeros so autoscale is not pinned to 0 V."""
    t = np.asarray(t, dtype=float)
    v = np.asarray(v, dtype=float)
    peak = float(np.max(np.abs(v)))
    if peak == 0.0:
        return t, v
    keep = np.abs(v) > 0.05 * peak
    if not np.any(keep):
        return t, v
    first, last = int(np.argmax(keep)), int(len(keep) - np.argmax(keep[::-1]))
    return t[first:last], v[first:last]


def _centered_rolling_mean(y, window):
    """Nan-aware centered uniform moving average. Edges need a half-full window."""
    y = np.asarray(y, dtype=float)
    n = int(window)
    if n < 2 or y.size == 0:
        return y.copy()
    if n % 2 == 0:
        n += 1
    kernel = np.ones(n, dtype=float)
    finite = np.isfinite(y)
    num = np.convolve(np.where(finite, y, 0.0), kernel, mode="same")
    den = np.convolve(finite.astype(float), kernel, mode="same")
    out = np.full(y.shape, np.nan)
    ok = den >= 0.5 * n
    out[ok] = num[ok] / den[ok]
    return out


def _ac_envelope(trace, window=20, quiet_samples=500):
    """Hilbert envelope of the DC-removed trace, one sample per *window* ns."""
    trace = np.asarray(trace, dtype=float)
    dc = float(np.median(trace[-quiet_samples:]))
    env = _envelope(trace - dc, window=window)
    t_ns = (np.arange(len(env)) + 0.5) * window
    return t_ns, env, dc


def _envelope_affine(t_env, env, t_sticky, v_sticky, rampup_ns=0, rampdown_ns=0):
    """Least-squares ``sticky ≈ a * env + b`` over the chirp body.

    *a* is chain gain (DAC, RF, detector).  *b* is baseline (ADC DC is already
    removed; this is leftover mixer/detector/demod bias).

    Do not fit a start-only gain with *b* = 0: a constant bias then masquerades
    as extra AM.  Do not force *a* = 1 either; fit both on the body.
    """
    t_env = np.asarray(t_env, dtype=float)
    env = np.asarray(env, dtype=float)
    t_sticky = np.asarray(t_sticky, dtype=float)
    v_sticky = np.asarray(v_sticky, dtype=float)

    t_lo = float(t_sticky[0]) + float(rampup_ns)
    t_hi = float(t_sticky[-1]) - float(rampdown_ns)
    in_body = (t_env >= t_lo) & (t_env <= t_hi) & np.isfinite(env)
    if not np.any(in_body):
        return 1.0, 0.0

    sticky_on_env = np.interp(t_env, t_sticky, v_sticky)
    peak_s = float(np.max(np.abs(v_sticky)))
    peak_e = float(np.max(env[in_body]))
    mask = in_body & (sticky_on_env > 0.5 * peak_s) & (env > 0.3 * peak_e)
    if int(np.count_nonzero(mask)) < 8:
        return 1.0, 0.0

    e = env[mask]
    s = sticky_on_env[mask]
    design = np.column_stack((e, np.ones(e.size)))
    (gain, offset), *_ = np.linalg.lstsq(design, s, rcond=None)
    residual = s - (gain * e + offset)
    print(
        f"  affine residual RMS    = {float(np.sqrt(np.mean(residual**2))) * 1e3:.3f} mV  "
        f"(n={int(e.size)})"
    )
    return float(gain), float(offset)


def _print_envelope_vs_sticky(t_env, env_roll, t_sticky, v_sticky, rampup_ns=0, rampdown_ns=0, edge_us=30.0):
    """Print start-to-end drop of the rolling envelope vs sticky. Returns the start level."""
    t_env = np.asarray(t_env, dtype=float)
    env_roll = np.asarray(env_roll, dtype=float)
    t_sticky = np.asarray(t_sticky, dtype=float)
    v_sticky = np.asarray(v_sticky, dtype=float)

    t_lo = float(t_sticky[0]) + float(rampup_ns)
    t_hi = float(t_sticky[-1]) - float(rampdown_ns)
    edge = float(edge_us) * 1e3
    if t_hi - t_lo <= 2 * edge:
        return None

    start = np.isfinite(env_roll) & (t_env >= t_lo) & (t_env <= t_lo + edge)
    end = np.isfinite(env_roll) & (t_env >= t_hi - edge) & (t_env <= t_hi)
    if not np.any(start) or not np.any(end):
        return None

    e0 = float(np.mean(env_roll[start]))
    e1 = float(np.mean(env_roll[end]))
    s0 = float(np.mean(np.interp(t_env[start], t_sticky, v_sticky)))
    s1 = float(np.mean(np.interp(t_env[end], t_sticky, v_sticky)))
    drop_e = e0 - e1
    drop_s = s0 - s1
    ratio = drop_e / drop_s if drop_s != 0.0 else np.nan
    print(
        f"  sticky drop            = {drop_s * 1e3:+.3f} mV  ({drop_s / s0 * 100:+.2f} %)"
        if s0
        else f"  sticky drop            = {drop_s * 1e3:+.3f} mV"
    )
    print(
        f"  envelope roll drop     = {drop_e * 1e3:+.3f} mV  ({drop_e / e0 * 100:+.2f} %)"
        if e0
        else f"  envelope roll drop     = {drop_e * 1e3:+.3f} mV"
    )
    print(
        f"  drop ratio env/sticky  = {ratio:.3f}  "
        "(1.0 means the analog drop matches python; multiplicative start-gain is not used)"
    )
    return e0


def _envelope(x, window=40):
    """Phase-insensitive envelope from the analytic signal.

    A peak-per-window estimator is biased for a chirped sinusoid because the
    carrier phase sampled inside each fixed window changes as the instantaneous
    frequency changes.  The Hilbert envelope |x + i H{x}| removes that carrier
    phase dependence.  We then average within each window only to reduce the
    data rate and suppress residual ripple.
    """
    from scipy.signal import hilbert

    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return np.array([], dtype=float)

    analytic = hilbert(x)
    amplitude = np.abs(analytic)

    n = (len(amplitude) // window) * window
    if n == 0:
        return np.array([], dtype=float)
    return amplitude[:n].reshape(-1, window).mean(axis=1)


def measure_residual(
    trace,
    wf_amplitude,
    label,
    window=20,
    quiet_samples=500,
    save_path: Path | None = None,
):
    """Measure the plateau, the rampdown waist, and the bump that follows it.

    The tail has two distinct features. The *waist* is the level the rampdown
    fails to cancel (the ramp-accumulation residual). The *bump* after it grows
    again before the output is cut, so it is a separate effect. DC is taken from
    the settled end of the trace, which for the ADC sits well away from zero.
    """
    trace = np.asarray(trace, dtype=float)
    dc = float(np.median(trace[-quiet_samples:]))
    ac = trace - dc
    noise = float(np.std(ac[-quiet_samples:]))

    env = _envelope(ac, window=window)
    threshold = max(3 * noise, 1e-7)

    # Anchor on the plateau, then walk forward to where the output really stops.
    # A bare "last sample above threshold" picks up noise spikes when the chirp
    # ends well before the end of the ADC window.
    strong = np.flatnonzero(env > 0.3 * env.max())
    if strong.size == 0:
        raise RuntimeError(f"{label}: trace never exceeds the noise floor")
    plateau_last = int(strong[-1])
    quiet_run = max(200 // window, 1)
    search_end = min(plateau_last + (4000 // window), len(env) - 1)
    last = plateau_last
    for i in range(plateau_last, search_end + 1):
        if env[i] > threshold and np.all(env[i + 1 : i + 1 + quiet_run] <= threshold):
            last = i
            break
        if env[i] > threshold:
            last = i

    plateau = float(env[max(last - (5000 // window), 0)])

    # Where the descending rampdown lands.  Taking argmin of the tail latches onto
    # whichever noise dip happens to be lowest in the already-silent stretch, so
    # follow the gradient instead: find the steepest fall, then walk forward to
    # where it flattens out.
    lo = max(last - (3000 // window), 0)
    grad = np.gradient(env[lo : last + 1])
    fall = int(np.argmin(grad))
    flat = 0.02 * abs(float(grad[fall]))
    waist_i = fall
    while waist_i < len(grad) - 1 and grad[waist_i] < -flat:
        waist_i += 1
    waist_abs = lo + waist_i

    # Everything past the waist is uncancelled output, and what matters is how
    # far it climbs back before the cut, so take the peak rather than a mean.
    bump = env[waist_abs : last + 1]
    residual_i = waist_abs + int(np.argmax(bump))
    residual = float(env[residual_i])

    print(f"\n--- {label} ---")
    print(f"  DC offset removed        = {dc * 1e3:+.3f} mV")
    print(f"  noise (1 sigma, settled) = {noise * 1e3:.3f} mV")
    print(f"  plateau before rampdown  = {plateau:.6f} V")
    print(f"  zero crossing (waist)    = {float(env[waist_abs]) * 1e3:.3f} mV  at {waist_abs * window} ns")
    print(
        f"  residual (max past waist)= {residual * 1e3:.3f} mV"
        f"  ({residual / plateau * 100:.3f} % of plateau)  at {residual_i * window} ns"
    )
    print(f"  output cut at            = {last * window} ns")

    if save_path is not None:
        import matplotlib.pyplot as plt

        fig, (ax_env, ax_raw) = plt.subplots(2, 1, figsize=(10, 7))
        fig.suptitle(f"{label}  (DC {dc * 1e3:+.2f} mV removed)")
        t_env = np.arange(len(env)) * window
        env_plot = env
        env_label = "envelope"
        residual_plot = residual
        ax_env.plot(t_env, env_plot, lw=0.9, label=env_label)
        x0 = (last - 4000 // window) * window
        x1 = (last + 1000 // window) * window
        ax_env.set_xlim(x0, x1)
        in_win = env_plot[max(lo, 0) : last + 1]
        ymax = float(np.max(in_win)) if in_win.size else residual_plot
        ax_env.set_ylim(0, max(ymax, residual_plot, noise) * 1.15)
        ax_env.axvline(waist_abs * window, color="0.4", lw=0.8, ls="-.", label="zero crossing")
        ax_env.axhline(residual_plot, color="r", lw=0.8, ls="--", label=f"residual {residual_plot * 1e3:.2f} mV")
        ax_env.plot(residual_i * window, residual_plot, "rv", ms=5)
        ax_env.axhline(noise, color="0.5", lw=0.8, ls=":", label=f"noise {noise * 1e3:.2f} mV")
        ax_env.set_ylabel("envelope (sticky units)")
        ax_env.legend(fontsize=8)
        ax_env.grid(True, alpha=0.35)

        raw_start = max(last * window - 4000, 0)
        raw_stop = min(last * window + 1000, len(ac))
        ax_raw.plot(np.arange(raw_start, raw_stop), ac[raw_start:raw_stop], lw=0.5)
        ax_raw.set_ylabel("V (DC removed)")
        ax_raw.set_xlabel("sample (ns)")
        ax_raw.grid(True, alpha=0.35)
        fig.tight_layout()
        fig.savefig(save_path, dpi=150)
        print(f"Saved {save_path}")

    return plateau, residual


def print_mismatch_summary(qua, py, wf_amplitude):
    n = min(len(qua["amp_mod"]), len(py["amp_mod"]))
    print(f"\n--- stream length QUA={len(qua['amp_mod'])} python={len(py['amp_mod'])} ---")
    for key in STREAM_VARS:
        n_key = min(len(qua[key]), len(py[key]))
        if n_key == 0:
            print(f"  {key}: EMPTY QUA stream")
            continue
        diff = np.asarray(qua[key][:n_key], dtype=float) - np.asarray(py[key][:n_key], dtype=float)
        print(
            f"  {key:16s}  max|QUA-py|={np.nanmax(np.abs(diff)):.6g}  "
            f"end QUA={qua[key][n_key - 1]:.6g}  py={py[key][n_key - 1]:.6g}"
        )
    dac_lsb = 2.0**-16
    py_cs = np.asarray(py["cumsum_v"][:n], dtype=float)
    py_ep = np.asarray(py["sticky_level"][:n], dtype=float)
    py_tgt = np.asarray(py["rampdown_target"][:n], dtype=float)
    qua_cs = np.asarray(qua["cumsum_v"][:n], dtype=float)
    print(
        f"\n  last QUA DAC16={qua_cs[-1]:.6f} V  python DAC16={py_cs[-1]:.6f} V  "
        f"python analog end={py_ep[-1]:.6f} V  target={py_tgt[-1]:.6f} V"
    )
    print(
        f"  python analog-target  max|err|={np.max(np.abs(py_ep - py_tgt)) * 1e6:.2f} uV  "
        f"python DAC16-target max|err|={np.max(np.abs(py_cs - py_tgt)) * 1e6:.2f} uV  "
        f"(1 DAC LSB = {dac_lsb * 1e6:.2f} uV)"
    )
    residual = qua_cs[-1] - wf_amplitude * qua["amp_mod"][n - 1]
    print(
        f"  QUA DAC16 vs amp_mod*wf residual={residual * 1e3:+.3f} mV  "
        "(should be < 1 DAC LSB if remainder-forget tracks the target)"
    )
    d_amp = wf_amplitude * (qua["amp_mod"][n - 1] - qua["amp_mod"][0])
    d_cs = float(qua_cs[-1] - qua_cs[0])
    d_py = float(py_ep[-1] - py_ep[0])
    print(f"  dV amp_mod*wf          = {d_amp * 1e3:+.3f} mV")
    print(f"  dV QUA DAC16           = {d_cs * 1e3:+.3f} mV")
    print(f"  dV python analog end   = {d_py * 1e3:+.3f} mV")


import matplotlib

# matplotlib.use("Agg")
from amplitude_calibration import check_tweezer_computation_latency, load_xy_amp_calibration
from configuration import (
    _wf_amplitude,
    chirp_pulse_amplitude,
    cluster_name,
    config,
    make_run_dir,
    number_of_segments,
    qop_ip,
    qop_port,
    rampdown_length,
    rampup_length,
    segment_length,
    x_tweezer_elements,
    x_tweezer_phases,
)
from qm.qua import program
from readout_analysis import (
    adc_traces_to_volts,
    measure_detector_adc,
    simulated_dac_trace,
    stream_save_raw_adc,
)

SCRIPT_NAME = "minimal_jerk_debug"
ELEMENT = x_tweezer_elements[0]
# Same chirp as the overlapping 20260915_082217 run.
START_HZ = int(40e6)
DETUNING_HZ = int(80e6)
NUM_SEGMENTS = int(number_of_segments)
SEGMENT_LENGTH = int(segment_length)
simulate = False


def build_debug_program(amp_calibration):
    with program() as arb_debug:
        raw_adc = declare_stream(adc_trace=True)
        frame_rotation_2pi(float(x_tweezer_phases[0]), ELEMENT)
        debug_streams = play_compensated_debug(
            ELEMENT, START_HZ, DETUNING_HZ, SEGMENT_LENGTH, NUM_SEGMENTS, amp_calibration=amp_calibration
        )
        measure_detector_adc(raw_adc)
        align(ELEMENT, "detector")
        with stream_processing():
            stream_save_debug(debug_streams, ELEMENT)
            stream_save_raw_adc(raw_adc)
    return arb_debug


def main():
    from qm import QuantumMachinesManager, SimulationConfig, generate_qua_script

    x_cal, _ = load_xy_amp_calibration(file_name=SCRIPT_NAME)
    play_ns = int(SEGMENT_LENGTH) - 4 * int(x_cal.computation_latency_cc)
    arb_debug = build_debug_program(x_cal)

    qmm_kwargs = {"host": qop_ip, "cluster_name": cluster_name, "timeout": 300}
    if qop_port is not None:
        qmm_kwargs["port"] = qop_port
    qmm = QuantumMachinesManager(**qmm_kwargs)

    run_dir = make_run_dir(SCRIPT_NAME)
    debug_path = run_dir / f"{SCRIPT_NAME}_qua.py"
    debug_path.write_text(generate_qua_script(arb_debug, config), encoding="utf-8")
    print(f"Saved QUA debug script to {debug_path}")

    chirp_label = (
        f"{ELEMENT}  {START_HZ / 1e6:.0f}->{(START_HZ + DETUNING_HZ) / 1e6:.0f} MHz  "
        f"({NUM_SEGMENTS} segments of {SEGMENT_LENGTH} ns)"
    )
    if simulate:
        duration_cc = int(2_000 + NUM_SEGMENTS * SEGMENT_LENGTH + 20_000) // 4
        print(f"Simulating {duration_cc} clock cycles ({chirp_label})")
        job = qmm.simulate(config, arb_debug, SimulationConfig(duration_cc))
        job.result_handles.wait_for_all_values()
        trace = simulated_dac_trace(job)
        trace_label = "simulated DAC samples"
        qm = None
    else:
        print(f"Executing on {qop_ip} ({chirp_label})")
        qm = qmm.open_qm(config, close_other_machines=True)
        job = qm.execute(arb_debug)
        job.result_handles.wait_for_all_values()
        trace = adc_traces_to_volts(job.result_handles.get("raw_data").fetch_all()["value"])[-1]
        trace_label = "measured detector ADC"

    qua = fetch_debug_streams(job, ELEMENT)
    py = replay_compensated_python(
        START_HZ,
        DETUNING_HZ,
        SEGMENT_LENGTH,
        NUM_SEGMENTS,
        x_cal,
        chirp_pulse_amplitude,
        _wf_amplitude,
    )
    print_mismatch_summary(qua, py, _wf_amplitude)
    measure_residual(
        trace,
        _wf_amplitude,
        trace_label,
        save_path=run_dir / "rampdown_tail.png",
    )
    plot_qua_vs_python(
        qua,
        py,
        _wf_amplitude,
        save_path=run_dir / "qua_vs_python.png",
        title=chirp_label,
        trace=trace,
        rampup_ns=int(rampup_length),
        segment_ns=SEGMENT_LENGTH,
        play_ns=play_ns,
        rampdown_ns=int(rampdown_length),
    )
    check_tweezer_computation_latency(
        job,
        [ELEMENT],
        SEGMENT_LENGTH,
        x_cal,
        None,
        file_name=SCRIPT_NAME,
    )
    np.savez(
        run_dir / "qua_vs_python.npz",
        **{f"qua_{k}": v for k, v in qua.items()},
        **{f"py_{k}": v for k, v in py.items()},
        trace=trace,
    )
    print(f"Saved arrays to {run_dir / 'qua_vs_python.npz'}")
    if qm is not None:
        qm.close()


if __name__ == "__main__":
    main()
