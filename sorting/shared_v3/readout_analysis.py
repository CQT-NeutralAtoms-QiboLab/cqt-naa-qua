"""ADC / simulated-analog collection and spectrogram plotting.

Used by the sort scripts for both ``simulate=True`` (OPX analog outputs) and
real execution (detector ADC traces).

Copy either ``configuration_opx+.py`` or ``configuration_opx1k.py`` to
``configuration.py`` before running.  The active file must export
``AOD_ANALOG_OUTPUTS`` (or legacy ``X_PORTS`` / ``Y_PORTS`` on OPX+).
"""

import configuration as _cfg
import matplotlib.pyplot as plt
import numpy as np
from amplitude_calibration import amp_at_frequency_py
from configuration import (
    DETECTOR_INPUT_PORT,
    number_of_segments,
    segment_length,
)
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.ticker import FuncFormatter
from qm.exceptions import JobFailedError
from qm.qua import *
from qualang_tools.units import unit

u = unit(coerce_to_integer=True)

DEFAULT_TIME_SCALE = 1e6
# Display STFT: more overlap than this is invisible at typical figure sizes.
_MAX_SPEC_TIME_BINS = 900
_SAVEFIG_DPI = 150
# Spectrogram : max-hold column widths, matching the width of two of the three
# occupation panels below them.
_SPEC_HOLD_WIDTH_RATIOS = [2, 1]
# Occupied max-hold: bins at least this fraction of the peak (tweezer tones, not noise).
_MAX_HOLD_PEAK_FRACTION = 0.9

FONT_SIZE = 12
plt.rcParams.update(
    {
        "font.size": FONT_SIZE,
        "axes.titlesize": FONT_SIZE,
        "axes.labelsize": FONT_SIZE,
        "xtick.labelsize": FONT_SIZE,
        "ytick.labelsize": FONT_SIZE,
        "legend.fontsize": FONT_SIZE - 2,
        "figure.titlesize": FONT_SIZE,
    }
)


def _want_show(show):
    if show is False:
        return False
    return plt.get_backend().lower() != "agg"


_OCCUPATION_MATRIX_CMAP = ListedColormap(["red", "black", "white", "cyan"])
_OCCUPATION_MATRIX_NORM = BoundaryNorm([-1.5, -0.5, 0.5, 1.5, 2.5], _OCCUPATION_MATRIX_CMAP.N)
OCCUPATION_MATRIX_KINDS = ("initial", "target", "current")

# OPX/+ uses X_PORTS/Y_PORTS; OPX1000 uses X_FEMS/Y_FEMS. Prefer the unified export.
AOD_ANALOG_OUTPUTS = getattr(_cfg, "AOD_ANALOG_OUTPUTS", None)
if AOD_ANALOG_OUTPUTS is None:
    if hasattr(_cfg, "X_PORTS"):
        AOD_ANALOG_OUTPUTS = _cfg.X_PORTS + _cfg.Y_PORTS
    elif hasattr(_cfg, "X_FEMS"):
        AOD_ANALOG_OUTPUTS = _cfg.X_FEMS + _cfg.Y_FEMS
    else:
        raise RuntimeError(
            "configuration.py must define AOD_ANALOG_OUTPUTS (copy configuration_opx+.py or configuration_opx1k.py)"
        )


def measure_detector_adc(raw_adc):
    """Schedule a parallel detector ADC trace (do not align to tweezers first)."""
    measure("readout", "detector", adc_stream=raw_adc)


def stream_save_raw_adc(raw_adc, name="raw_data"):
    if DETECTOR_INPUT_PORT == 1:
        raw_adc.input1().save_all(name)
    else:
        raw_adc.input2().save_all(name)


def _analog_channel_key(channel):
    """Map a config channel entry to the simulated-analog dict key.

    OPX/+ entries are ``(controller, port)`` and keyed by ``str(port)``.
    OPX1000 entries are ``(controller, fem_slot, port)`` and keyed by
    ``(fem_slot, port)``.
    """
    if len(channel) == 2:
        return str(channel[1])
    if len(channel) == 3:
        return (channel[1], channel[2])
    raise ValueError(f"unexpected AOD channel entry {channel!r}")


def _lookup_analog_key(analog, channel):
    key = _analog_channel_key(channel)
    if key not in analog and isinstance(key, str) and key.isdigit() and int(key) in analog:
        key = int(key)
    return key


def first_x_aod_channel():
    """First X-axis AOD output entry from the active ``configuration.py``."""
    if hasattr(_cfg, "X_PORTS") and _cfg.X_PORTS:
        return _cfg.X_PORTS[0]
    if hasattr(_cfg, "X_FEMS") and _cfg.X_FEMS:
        return _cfg.X_FEMS[0]
    if not AOD_ANALOG_OUTPUTS:
        raise RuntimeError("AOD_ANALOG_OUTPUTS is empty in configuration.py")
    return AOD_ANALOG_OUTPUTS[0]


def simulated_dac_trace(job, channel=None, controller=None):
    """Simulated DAC waveform for one AOD output channel.

    Defaults to the first X-axis channel (``x_tweezer_1`` on OPX/+ or OPX1000).
    """
    channel = first_x_aod_channel() if channel is None else channel
    con = channel[0] if controller is None else controller
    analog = getattr(job.get_simulated_samples(), con).analog
    key = _lookup_analog_key(analog, channel)
    if key not in analog:
        raise KeyError(f"simulated analog channel {key!r} not found on {con}")
    return np.asarray(analog[key], dtype=float)


def summed_aod_output(samples, controller=None):
    """Sum simulated analog outputs for every AOD channel in the active config."""
    if not AOD_ANALOG_OUTPUTS:
        raise RuntimeError("AOD_ANALOG_OUTPUTS is empty in configuration.py")
    aod_output = None
    sampling_rate = None
    for channel in AOD_ANALOG_OUTPUTS:
        con = channel[0] if controller is None else controller
        analog = getattr(samples, con).analog
        rates = getattr(samples, con).analog_sampling_rate
        key = _lookup_analog_key(analog, channel)
        if key not in analog:
            continue
        channel_data = analog[key]
        aod_output = channel_data if aod_output is None else aod_output + channel_data
        sampling_rate = rates[key]
        print(f"Simulated AOD output for {key} at {sampling_rate / 1e9:.2f} GS/s")
    if aod_output is None:
        raise RuntimeError("No simulated AOD analog channels found.")
    return aod_output, sampling_rate


def _band_limits(Fs, start_freq, stop_freq):
    BW = max(abs(stop_freq - start_freq), 1.0)
    f0 = 0.5 * (start_freq + stop_freq)
    fmin = max(f0 - 0.51 * BW, 0.0)
    fmax = min(f0 + 0.51 * BW, Fs / 2)
    return fmin, fmax


def _specgram_params(Fs, start_freq, stop_freq, n_samples=None):
    fmin, fmax = _band_limits(Fs, start_freq, stop_freq)
    # Desired FFT-bin resolution. 0.1 MHz/bin resolves ~1 MHz-separated tones.
    target_df = 0.2e6
    nfft = int(2 ** np.ceil(np.log2(max(Fs / target_df, 128))))
    nfft = max(128, min(nfft, 8192))
    if n_samples is not None:
        nfft = min(nfft, max(int(n_samples), 128))

    hop = max(nfft // 4, 1)
    if n_samples is not None and n_samples > nfft:
        hop = max(hop, int(np.ceil((n_samples - nfft) / _MAX_SPEC_TIME_BINS)))
    noverlap = max(0, nfft - hop)
    return {
        "nfft": nfft,
        "window": np.hanning(nfft),
        "noverlap": noverlap,
        "fmin": fmin,
        "fmax": fmax,
    }


def _stft_magnitude(waveform, Fs, nfft, noverlap, window):
    """Magnitude STFT matching ``Axes.specgram(mode="magnitude")``.

    Returns ``(Pxx, freqs, times)`` shaped (frequency, time).  The window-sum
    normalization is what keeps the max-hold trace in volts.
    """
    x = np.asarray(waveform, dtype=float).reshape(-1)
    hop = max(nfft - noverlap, 1)
    if x.size < nfft:
        x = np.pad(x, (0, nfft - x.size))
    offsets = hop * np.arange(1 + (x.size - nfft) // hop)
    frames = np.take(x, offsets[:, None] + np.arange(nfft)[None, :]) * window
    pxx = np.abs(np.fft.rfft(frames, n=nfft, axis=1)).T / np.abs(window).sum()
    freqs = np.fft.rfftfreq(nfft, d=1.0 / Fs)
    times = (offsets + 0.5 * nfft) / Fs
    return pxx, freqs, times


def wait_for_simulation(job, timeout=120):
    """ADC traces often finish the simulator in Error; analog samples are still valid."""
    try:
        job.wait_until("Done", timeout)
    except JobFailedError as exc:
        print(f"Simulation finished with {exc}; continuing with analog samples.")


def _voltage_waveform_result(mode, waveform, Fs, start_freq, stop_freq):
    """Common metadata for simulated and measured voltage waveforms."""
    return {
        "mode": mode,
        "waveform": np.asarray(waveform, dtype=float),
        "Fs": float(Fs),
        "start_freq": float(start_freq),
        "stop_freq": float(stop_freq),
        "time_scale": DEFAULT_TIME_SCALE,
    }


def collect_simulation_results(job, start_freq, stop_freq, controller="con1"):
    wait_for_simulation(job)
    samples = job.get_simulated_samples()
    waveform, sampling_rate = summed_aod_output(samples, controller=controller)
    results = _voltage_waveform_result("simulation", waveform, sampling_rate, start_freq, stop_freq)
    results["samples"] = samples
    return results


def adc_traces_to_volts(raw):
    """One volt-scaled trace per saved ADC measurement.

    ``save_all`` returns a trace per ``measure`` call, so a row-by-row program
    yields one entry per row rather than a single concatenated waveform.
    """
    arr = np.asarray(raw)
    if arr.dtype == object:
        return [np.asarray(u.raw2volts(np.asarray(t).reshape(-1)), dtype=float) for t in arr.tolist()]
    if arr.ndim == 1:
        return [np.asarray(u.raw2volts(arr), dtype=float)]
    # (n_saves, n_inputs, n_samples) -> keep the first input of each save.
    while arr.ndim > 2:
        arr = arr[:, 0]
    return [np.asarray(u.raw2volts(trace), dtype=float) for trace in arr]


def _adc_to_volts(raw):
    return adc_traces_to_volts(raw)[0]


def collect_execution_results(job, start_freq, stop_freq, wait_for_all=True, n_values=1, timeout=60):
    res = job.result_handles
    if wait_for_all:
        res.wait_for_all_values(timeout)
    else:
        res.get("raw_data").wait_for_values(n_values, timeout)
    raw = res.get("raw_data").fetch_all()["value"]
    rows = adc_traces_to_volts(raw)
    # Latest acquisition is the default waveform for scripts that measure once.
    results = _voltage_waveform_result("execution", rows[-1], 1e9, start_freq, stop_freq)
    results.update({"rows": rows, "raw": raw})
    return results


def _max_hold(Pxx, search_bins=2, dbv=False):
    """Max-hold over time after summing *search_bins* neighbors on the frequency axis."""
    power = np.nan_to_num(np.asarray(Pxx, dtype=float), nan=0.0) ** 2
    if power.ndim == 1:
        power = power[:, None]
        squeeze = True
    else:
        squeeze = False
    pad = np.pad(power, ((search_bins, search_bins), (0, 0)))
    width = 2 * search_bins + 1
    csum = np.cumsum(pad, axis=0)
    csum = np.concatenate([np.zeros((1, csum.shape[1]), dtype=csum.dtype), csum], axis=0)
    summed_power = csum[width:] - csum[:-width]
    peak = np.sqrt(np.max(summed_power, axis=1))
    if squeeze:
        peak = peak.reshape(-1)
    if dbv:
        return 20.0 * np.log10(np.maximum(peak, 1e-20))
    return peak


def _occupied_max_hold_mask(max_hold, reference=None, peak_fraction=_MAX_HOLD_PEAK_FRACTION):
    """True on tone bins: finite, positive, and at least ``peak_fraction`` of the peak.

    A percentile of all positive bins is the noise floor (most frequencies are
    empty).  Half of the max-hold peak keeps the tweezer lines and drops zeros.
    """
    max_hold = np.asarray(max_hold, dtype=float)
    ok = np.isfinite(max_hold) & (max_hold > 0)
    if reference is not None:
        reference = np.asarray(reference, dtype=float)
        ok &= np.isfinite(reference) & (reference > 0)
    if not np.any(ok):
        return ok, 0.0
    floor = float(peak_fraction * np.max(max_hold[ok]))
    return ok & (max_hold >= floor), floor


def _rescale_reference_to_max_hold(reference, max_hold):
    """Affine ``max_hold ≈ a * reference + b`` on tweezer-tone bins.

    Same two-parameter model as the envelope overlay.  A chirp max-hold is
    still dwell-weighted and will not follow ``amp_mod(f)`` in the sweep body.
    """
    reference = np.asarray(reference, dtype=float)
    max_hold = np.asarray(max_hold, dtype=float)
    mask, floor = _occupied_max_hold_mask(max_hold, reference)
    n = int(np.count_nonzero(mask))
    if n == 0:
        return reference
    r = reference[mask]
    m = max_hold[mask]
    if n == 1:
        gain = float(m[0] / r[0]) if r[0] != 0.0 else 1.0
        offset = 0.0
    else:
        (gain, offset), *_ = np.linalg.lstsq(np.column_stack((r, np.ones(r.size))), m, rcond=None)
        gain, offset = float(gain), float(offset)
    print(
        f"  max-hold affine a={gain:.4f}  b={offset * 1e3:+.2f} mV  "
        f"floor={floor * 1e3:.2f} mV ({_MAX_HOLD_PEAK_FRACTION:.2f}*peak, n={n})"
    )
    return gain * reference + offset


def _draw_spectrogram_with_maxhold(
    ax_spec,
    ax_hold,
    waveform,
    *,
    Fs,
    nfft,
    window,
    noverlap,
    fmin,
    fmax,
    time_scale=1e9,
    max_hold_calibration=None,
    max_hold_base_amplitude=None,
    title=None,
):
    """Spectrogram on *ax_spec* and max-hold magnitude vs frequency on *ax_hold* (shared y)."""
    valid_time_scales = [1e9, 1e6, 1e3, 1]
    time_label_units = ["ns", "µs", "ms", "s"]
    assert time_scale in valid_time_scales, f"Invalid time scale: {time_scale}"
    time_label = f"Time [{time_label_units[valid_time_scales.index(time_scale)]}]"

    Pxx, freqs, times = _stft_magnitude(
        2 * waveform,  # doubled to account for the image (same as the old specgram path)
        Fs,
        nfft,
        noverlap,
        window,
    )
    visible = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(visible):
        visible = np.ones(freqs.size, dtype=bool)
    Pxx_v = Pxx[visible]
    freqs_v = freqs[visible]
    spec_db = 20.0 * np.log10(np.maximum(Pxx_v, 1e-20))
    t0 = float(times[0]) if times.size else 0.0
    t1 = float(times[-1]) if times.size else 1.0 / Fs
    f0 = float(freqs_v[0])
    f1 = float(freqs_v[-1]) if freqs_v.size > 1 else f0 + 1.0
    im = ax_spec.imshow(
        spec_db,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        extent=(t0, t1, f0, f1),
        cmap=plt.cm.gist_heat,
        rasterized=True,
    )
    ax_spec.set_ylim(fmin, fmax)
    ax_spec.set_xlabel(time_label)
    ax_spec.set_ylabel("Frequency [MHz]")
    if title is not None:
        ax_spec.set_title(title)
    ax_spec.xaxis.set_major_formatter(FuncFormatter(lambda value, _position: f"{value * time_scale:g}"))
    ax_spec.yaxis.set_major_formatter(FuncFormatter(lambda value, _position: f"{value / 1e6:g}"))

    max_hold = _max_hold(Pxx_v)
    ax_hold.plot(max_hold, freqs_v, color="C0", lw=1.4)
    if (
        max_hold_calibration is not None
        and not getattr(max_hold_calibration, "latency_only", False)
        and max_hold_calibration[1]
    ):
        if max_hold_base_amplitude is None:
            raise ValueError("max_hold_base_amplitude is required with max_hold_calibration")
        center_ghz, coefs = max_hold_calibration[0], max_hold_calibration[1]
        reference = max_hold_base_amplitude * np.asarray(
            amp_at_frequency_py(freqs_v / 1e9, center_ghz, coefs),
            dtype=float,
        )
        reference = _rescale_reference_to_max_hold(reference, max_hold)
        ax_hold.plot(
            reference,
            freqs_v,
            color="black",
            linestyle=":",
            lw=1.4,
            label="Calibration\nreference",
        )
        ax_hold.legend(loc="best")
    ax_hold.set_ylim(fmin, fmax)
    ax_hold.set_xlabel("Max hold [V]")
    ax_hold.grid(True, alpha=0.3)
    ax_hold.tick_params(labelleft=False)

    return Pxx_v, freqs_v, times, im


def _apply_common_max_hold_xlim(ax_holds, pad_fraction=0.05):
    """Apply one horizontal scale to a collection of max-hold axes."""
    finite_values = []
    for ax_hold in ax_holds:
        ymin, ymax = sorted(ax_hold.get_ylim())
        for line in ax_hold.lines:
            x_values = np.asarray(line.get_xdata(), dtype=float)
            y_values = np.asarray(line.get_ydata(), dtype=float)
            visible = np.isfinite(x_values) & np.isfinite(y_values) & (y_values >= ymin) & (y_values <= ymax)
            finite_values.extend(x_values[visible])

    if not finite_values:
        return None

    data_min = float(np.min(finite_values))
    data_max = float(np.max(finite_values))
    if data_min >= 0:
        common_xlim = (0.0, data_max * (1 + pad_fraction) if data_max > 0 else 1.0)
    else:
        span = max(data_max - data_min, 1e-12)
        padding = pad_fraction * span
        common_xlim = (data_min - padding, data_max + padding)

    for ax_hold in ax_holds:
        ax_hold.set_xlim(common_xlim)
    return common_xlim


def plot_results(
    results,
    title=None,
    save_path=None,
    show=True,
    *,
    ax_spec=None,
    ax_hold=None,
    add_colorbar=True,
    max_hold_calibration=None,
    max_hold_base_amplitude=None,
    return_plot_data=False,
):
    """Plot a collected waveform as a spectrogram and max-hold trace.

    With no axes supplied, this function creates and manages a standalone
    figure.  Pass both ``ax_spec`` and ``ax_hold`` to embed the same plotting
    path in a larger figure.  Embedded axes are never shown or closed here.
    """
    waveform = np.asarray(results["waveform"], dtype=float)
    Fs = float(results["Fs"])
    start_freq = results["start_freq"]
    stop_freq = results["stop_freq"]

    specgram_params = _specgram_params(
        Fs,
        start_freq,
        stop_freq,
        n_samples=len(waveform),
    )

    owns_figure = ax_spec is None and ax_hold is None
    if owns_figure:
        fig, (ax_spec, ax_hold) = plt.subplots(
            1,
            2,
            figsize=(9, 4),
            sharey=True,
            layout="constrained",
            gridspec_kw={"width_ratios": _SPEC_HOLD_WIDTH_RATIOS, "wspace": 0.08},
        )
    elif ax_spec is None or ax_hold is None:
        raise ValueError("ax_spec and ax_hold must be supplied together")
    elif ax_spec.figure is not ax_hold.figure:
        raise ValueError("ax_spec and ax_hold must belong to the same figure")
    else:
        fig = ax_spec.figure

    Pxx, freqs, times, im = _draw_spectrogram_with_maxhold(
        ax_spec,
        ax_hold,
        waveform,
        Fs=Fs,
        **specgram_params,
        time_scale=results["time_scale"],
        max_hold_calibration=max_hold_calibration,
        max_hold_base_amplitude=max_hold_base_amplitude,
        title=title or f"{results['mode']} spectrogram",
    )
    _apply_common_max_hold_xlim([ax_hold])
    if add_colorbar:
        fig.colorbar(im, ax=[ax_spec, ax_hold], pad=0.02, label="Spectral magnitude [dBV]")
    if save_path is not None:
        fig.savefig(str(save_path), dpi=_SAVEFIG_DPI)
        print(f"Saved spectrogram to {save_path}")
    if owns_figure:
        if _want_show(show):
            plt.show()
        else:
            plt.close(fig)
    if return_plot_data:
        return fig, {"Pxx": Pxx, "freqs": freqs, "times": times, "image": im}
    return fig


def plot_envelope_vs_compensation(
    waveform,
    start_hz,
    detuning_hz,
    amp_calibration,
    chirp_pulse_amplitude,
    save_path,
    *,
    rampup_ns=1000,
    segment_ns=None,
    rampdown_ns=1000,
    wf_amplitude=None,
    title=None,
):
    """Hilbert envelope vs the Q4.28/DAC16 sticky replay from minimal_jerk_debug.

    A spectrogram max-hold is biased by MJ dwell time and will not follow
    ``amp_mod(f)`` even when the analog does.  Align the envelope to the
    Same comparison as ``minimal_jerk_debug``.  Overlay uses
    ``sticky ≈ a * env + b`` fit on the chirp body (gain and offset).
    """
    from scipy.signal import hilbert

    if wf_amplitude is None:
        wf_amplitude = chirp_pulse_amplitude
    segment_ns = int(segment_length if segment_ns is None else segment_ns)
    n_seg = int(number_of_segments)
    latency_cc = int(amp_calibration.computation_latency_cc)
    play_ns = int(segment_ns) - 4 * latency_cc
    center, coefs = amp_calibration.center, amp_calibration.coefs

    fixed_bits = 28
    dac_shift = fixed_bits - 16
    q28 = 1 << fixed_bits

    def to_q28(x):
        return int(np.round(np.float64(x) * q28))

    def from_q28(i):
        return float(i) / q28

    def q28_mul(a, b):
        return from_q28((to_q28(a) * to_q28(b)) >> fixed_bits)

    def q28_mul_int(a, n):
        return from_q28(to_q28(a) * int(n))

    def q16_forget(x):
        i = to_q28(x)
        return from_q28((i >> dac_shift) << dac_shift)

    def poly(freq_hz):
        return float(amp_at_frequency_py(freq_hz / 1e9, center, coefs))

    inv = 1.0 / (n_seg + 1)
    chirp_scaling = 30.0 * 1e3 / ((n_seg + 1) * segment_ns)
    inv_play_ns = 1.0 / play_ns
    cur = int(start_hz)
    det = int(detuning_hz)
    amp_mod = poly(cur)
    target_v = q28_mul(amp_mod, chirp_pulse_amplitude)
    cumsum_v = q16_forget(target_v)
    amp_mods = [amp_mod]
    endpoints = [cumsum_v]
    tau = inv
    while tau < (1 - 0.5 * inv):
        tau2 = tau * tau
        chirp_rate = int(np.round(det * chirp_scaling * tau2 * (1 - 2 * tau + tau2)))
        cur = cur + int(np.round(chirp_rate * 1e-3 * segment_ns))
        amp_mod = poly(cur)
        target_v = q28_mul(amp_mod, chirp_pulse_amplitude)
        ramp_rate = q28_mul(target_v - cumsum_v, inv_play_ns)
        endpoint_v = cumsum_v + q28_mul_int(ramp_rate, play_ns)
        cumsum_v = q16_forget(endpoint_v)
        amp_mods.append(amp_mod)
        endpoints.append(endpoint_v)
        tau = tau + inv
    amp_mods = np.asarray(amp_mods, dtype=float)
    endpoints = np.asarray(endpoints, dtype=float)
    python_target = wf_amplitude * amp_mods

    def sticky_polyline(sticky):
        t = [0.0, float(rampup_ns)]
        v = [0.0, float(sticky[0])]
        t_play = float(rampup_ns)
        for i in range(1, len(sticky)):
            t.extend([t_play + play_ns, t_play + segment_ns])
            v.extend([float(sticky[i]), float(sticky[i])])
            t_play += segment_ns
        return np.asarray(t), np.asarray(v)

    t_ep, v_ep = sticky_polyline(endpoints)
    t_tgt, v_tgt = sticky_polyline(python_target)

    trace = np.asarray(waveform, dtype=float)
    dc = float(np.median(trace[-500:]))
    analytic = np.abs(hilbert(trace - dc))
    window = 20
    n = (len(analytic) // window) * window
    env = analytic[:n].reshape(-1, window).mean(axis=1)
    t_env = (np.arange(len(env)) + 0.5) * window

    t_body_lo = float(rampup_ns)
    t_body_hi = float(t_ep[-1]) - float(rampdown_ns)
    in_body = (t_env >= t_body_lo) & (t_env <= t_body_hi) & np.isfinite(env)
    sticky_on_env = np.interp(t_env, t_ep, v_ep)
    peak_sticky = float(np.max(np.abs(v_ep)))
    peak_env = float(np.max(env[in_body])) if np.any(in_body) else 0.0
    mask = in_body & (sticky_on_env > 0.5 * peak_sticky) & (env > 0.3 * peak_env)
    if int(np.count_nonzero(mask)) >= 8:
        e = env[mask]
        s = sticky_on_env[mask]
        (gain, offset), *_ = np.linalg.lstsq(np.column_stack((e, np.ones(e.size))), s, rcond=None)
        gain, offset = float(gain), float(offset)
    else:
        gain, offset = 1.0, 0.0
    env_aligned = gain * env + offset

    dt_ns = float(window)
    n_roll = max(3, int(round(1e3 / dt_ns)))
    if n_roll % 2 == 0:
        n_roll += 1
    t_body_lo = float(rampup_ns)
    t_body_hi = float(t_ep[-1]) - float(rampdown_ns) - 0.5 * n_roll * dt_ns
    in_body = (t_env >= t_body_lo) & (t_env <= t_body_hi)
    y = np.where(in_body, env_aligned, np.nan)
    kernel = np.ones(n_roll, dtype=float)
    finite = np.isfinite(y)
    num = np.convolve(np.where(finite, y, 0.0), kernel, mode="same")
    den = np.convolve(finite.astype(float), kernel, mode="same")
    env_roll = np.full_like(env_aligned, np.nan)
    roll_ok = den >= 0.5 * n_roll
    env_roll[roll_ok] = num[roll_ok] / den[roll_ok]
    roll_ok = roll_ok & in_body

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(
        t_env[in_body] * 1e-3,
        env_aligned[in_body],
        label=f"envelope {gain:.3f}·x{offset * 1e3:+.1f} mV",
        lw=0.4,
        alpha=0.25,
    )
    ax.plot(t_env[roll_ok] * 1e-3, env_roll[roll_ok], label="envelope roll 1 us", lw=1.6)
    ax.plot(t_tgt * 1e-3, v_tgt, label="python target", lw=1.0, alpha=0.7)
    ax.plot(t_ep * 1e-3, v_ep, label="python analog end", lw=1.4)
    ax.set_xlabel("time (us)")
    ax.set_ylabel("envelope / sticky (V)")
    pulse_end_us = float(t_ep[-1]) * 1e-3
    ax.set_xlim(0, pulse_end_us + 4)
    body_v = np.concatenate([v_ep, v_tgt])
    body_v = body_v[np.abs(body_v) > 0.5 * np.max(np.abs(body_v))]
    if np.any(roll_ok):
        body_v = np.concatenate([body_v, env_roll[roll_ok], env_aligned[in_body]])
    y_min, y_max = np.nanpercentile(body_v, [1.0, 99.0])
    y_pad = max(0.15 * (y_max - y_min), 1e-4)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)
    ax.grid(True, alpha=0.35)
    ax.legend(loc="best", fontsize=8)
    if title:
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(str(save_path), dpi=_SAVEFIG_DPI)
    plt.close(fig)
    print(f"Saved envelope overlay to {save_path}")

    edge = 30e3
    start = roll_ok & (t_env >= t_body_lo) & (t_env <= t_body_lo + edge)
    end = roll_ok & (t_env >= t_body_hi - edge) & (t_env <= t_body_hi)
    if np.any(start) and np.any(end):
        e0 = float(np.mean(env_roll[start]))
        e1 = float(np.mean(env_roll[end]))
        s0 = float(np.mean(np.interp(t_env[start], t_ep, v_ep)))
        s1 = float(np.mean(np.interp(t_env[end], t_ep, v_ep)))
        drop_e = e0 - e1
        drop_s = s0 - s1
        ratio = drop_e / drop_s if drop_s != 0.0 else np.nan
        print(f"  sticky drop            = {drop_s * 1e3:+.3f} mV")
        print(f"  envelope roll drop     = {drop_e * 1e3:+.3f} mV")
        print(f"  drop ratio env/sticky  = {ratio:.3f}")
    return fig


def plot_pulse_timeline(job, save_path=None, show=True):
    waveform_report = job.get_simulated_waveform_report()
    wfs = waveform_report.analog_waveforms
    elements_seen = sorted({w.element for w in wfs})
    elem_to_y = {e: i for i, e in enumerate(elements_seen)}

    fig_tl, ax_tl = plt.subplots(figsize=(14, max(4, len(elements_seen) * 0.6)))
    cmap = plt.cm.tab10

    print(
        f"\n{'Element':<16} {'Waveform':<14} {'t_start (ns)':>13} {'length (ns)':>12} "
        f"{'chirp_rate':>12} {'start_freq (MHz)':>18}"
    )
    print("-" * 90)

    for w in sorted(wfs, key=lambda w: (w.element, w.timestamp)):
        t_ns = w.timestamp
        len_ns = w.length
        rate_str = ""
        freq_str = ""
        if w.chirp_info and w.chirp_info.get("rate"):
            rate_str = f"{w.chirp_info['rate'][0]:.0f}"
            freq_str = f"{w.chirp_info['startFrequency'] / 1e6:.3f}"

        print(f"{w.element:<16} {w.waveform_name:<14} {t_ns:>13,} {len_ns:>12,} {rate_str:>12} {freq_str:>18}")

        y = elem_to_y[w.element]
        color = cmap(y % 10)
        ax_tl.barh(y, len_ns, left=t_ns, height=0.6, color=color, edgecolor="k", linewidth=0.4, alpha=0.8)

    ax_tl.set_yticks(range(len(elements_seen)))
    ax_tl.set_yticklabels(elements_seen)
    ax_tl.set_xlabel("Time (ns)")
    ax_tl.set_title("Pulse Timeline")
    ax_tl.invert_yaxis()
    fig_tl.tight_layout()
    if save_path is not None:
        fig_tl.savefig(str(save_path), dpi=150)
    if _want_show(show):
        plt.show()
    else:
        plt.close(fig_tl)
    return fig_tl, waveform_report


def if_axis_limits(ifs, pad_frac=0.1):
    """Frequency-axis limits from IF positions (no assumed uniform spacing)."""
    ifs = np.asarray(ifs, dtype=float)
    lo, hi = float(np.min(ifs)), float(np.max(ifs))
    span = max(hi - lo, 1e6)
    return lo - pad_frac * span, hi + pad_frac * span


def spectrogram_limits(x_freq, x_det, y_freq, y_det, *, pad_frac=0.1):
    """STFT frequency span from X/Y chirp start IFs and per-tweezer detunings.

    Idle tweezers with start and stop both at zero are ignored.  Returns
    ``(fmin, fmax)`` in Hz with *pad_frac* margin on each side.
    """
    starts = np.asarray(list(x_freq) + list(y_freq), dtype=float)
    detunings = np.asarray(list(x_det) + list(y_det), dtype=float)
    stops = starts + detunings
    used = ~((starts == 0) & (stops == 0))
    if np.any(used):
        starts = starts[used]
        stops = stops[used]
    lo = int(min(starts.min(), stops.min()))
    hi = int(max(starts.max(), stops.max()))
    padding = int(pad_frac * (hi - lo))
    limits = max(lo - padding, 0), hi + padding
    print(f"Spec limits: {limits}")
    return limits


def _coord_edges(centers):
    """Bin edges around possibly non-uniform IF centers, for pcolormesh."""
    centers = np.asarray(centers, dtype=float)
    if centers.size == 1:
        pad = abs(centers[0]) * 0.05 or 1e6
        return np.array([centers[0] - pad, centers[0] + pad])
    mids = 0.5 * (centers[:-1] + centers[1:])
    first = centers[0] - (mids[0] - centers[0])
    last = centers[-1] + (centers[-1] - mids[-1])
    return np.concatenate([[first], mids, [last]])


def atom_occupation_matrix(kind, *, atom_location_list, atom_target_list, atom_final_list):
    kind = kind.lower()
    if kind == "initial":
        return np.asarray(atom_location_list)
    if kind == "target":
        return np.asarray(atom_target_list)
    if kind == "current":
        return 2 * np.asarray(atom_final_list) - np.asarray(atom_target_list)
    raise ValueError(f"kind must be one of {OCCUPATION_MATRIX_KINDS!r}, got {kind!r}")


def plot_atom_occupation_matrix(mat, *, ax=None, title=None, fs=FONT_SIZE, column_IFs=None, row_IFs=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))
    mat = np.asarray(mat)
    if column_IFs is None or row_IFs is None:
        ax.pcolormesh(
            mat,
            edgecolor="gray",
            cmap=_OCCUPATION_MATRIX_CMAP,
            norm=_OCCUPATION_MATRIX_NORM,
            linewidth=0.6,
        )
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    else:
        x_edges = _coord_edges(column_IFs) / 1e6
        y_edges = _coord_edges(row_IFs) / 1e6
        ax.pcolormesh(
            x_edges,
            y_edges,
            mat,
            edgecolor="gray",
            cmap=_OCCUPATION_MATRIX_CMAP,
            norm=_OCCUPATION_MATRIX_NORM,
            linewidth=0.6,
        )
        ax.set_xlabel("Column IF [MHz]", fontsize=fs)
        ax.set_ylabel("Row IF [MHz]", fontsize=fs)
    ax.invert_yaxis()
    if title is not None:
        ax.set_title(title, fontsize=fs)
    return ax


def _annotate_sorting_row(ax, times, column_IFs, atom_locations, target_frequencies):
    """Add site guides and initial/target markers to a row spectrogram.

    Guides and markers are batched into one artist each; per-site ``axhline``
    and ``scatter`` calls dominated draw time once there were many rows.
    """
    column_IFs = np.asarray(column_IFs, dtype=float)
    x_min, x_max = ax.get_xlim()
    ax.hlines(column_IFs, x_min, x_max, colors="k", linewidths=1, linestyles=(0, (10, 5)))
    ax.set_xlim(x_min, x_max)
    if times is None or len(times) <= 6:
        return
    occupied = np.nonzero(np.asarray(atom_locations))[0]
    if occupied.size:
        ax.scatter(np.full(occupied.size, times[3]), column_IFs[occupied], s=10, color="g")
    targets = np.asarray([f for f in target_frequencies if f], dtype=float)
    if targets.size:
        ax.scatter(np.full(targets.size, times[-3]), targets, s=10, color="b")


def plot_row_spectrograms(
    raw_rows,
    column_IFs,
    atom_location_list,
    target_frequencies,
    *,
    Fs=1e9,
    start_freq=None,
    stop_freq=None,
    time_scale=DEFAULT_TIME_SCALE,
    save_path=None,
    show=True,
):
    """Plot rows using the same STFT path and parameters as :func:`plot_results`."""
    raw_rows = list(raw_rows)
    nb_of_rows = len(raw_rows)
    if start_freq is None or stop_freq is None:
        default_start, default_stop = if_axis_limits(column_IFs)
        start_freq = default_start if start_freq is None else start_freq
        stop_freq = default_stop if stop_freq is None else stop_freq
    fig, axes = plt.subplots(
        nb_of_rows,
        2,
        figsize=(9, 2.9 * nb_of_rows),
        squeeze=False,
        gridspec_kw={"width_ratios": _SPEC_HOLD_WIDTH_RATIOS, "wspace": 0.08},
    )

    plot_datas = []
    for i, raw in enumerate(raw_rows):
        row_results = {
            "mode": "row",
            "waveform": np.asarray(raw, dtype=float),
            "Fs": Fs,
            "start_freq": start_freq,
            "stop_freq": stop_freq,
            "time_scale": time_scale,
        }
        _row_fig, plot_data = plot_results(
            row_results,
            title=f"Row {i + 1}",
            show=False,
            ax_spec=axes[i, 0],
            ax_hold=axes[i, 1],
            add_colorbar=False,
            return_plot_data=True,
        )
        _annotate_sorting_row(
            axes[i, 0],
            plot_data["times"],
            column_IFs,
            atom_location_list[i],
            target_frequencies[i],
        )
        plot_datas.append(plot_data)

    _apply_common_max_hold_xlim(axes[:, 1])
    fig.subplots_adjust(left=0.09, right=0.99, top=0.97, bottom=0.03, hspace=0.55, wspace=0.1)

    if save_path is not None:
        fig.savefig(str(save_path), dpi=_SAVEFIG_DPI)
    if _want_show(show):
        plt.show()
    else:
        plt.close(fig)
    return fig, plot_datas


def plot_atom_sorting_results(
    results,
    *,
    column_IFs,
    atom_location_list,
    atom_target_list,
    atom_final_list,
    target_frequencies_full_python,
    row_IFs=None,
    max_hold_calibration=None,
    max_hold_base_amplitude=None,
    save_path=None,
    show=True,
):
    """Plot all sorting rows through :func:`plot_results` plus occupations."""
    if "rows" not in results:
        raise KeyError("sorting results must contain a 'rows' entry")
    raw_rows = list(results["rows"])
    nb_of_rows = len(atom_location_list)
    if len(raw_rows) != nb_of_rows:
        raise ValueError(f"expected {nb_of_rows} ADC traces, got {len(raw_rows)}")

    # Row block is two columns wide; only the occupation strip needs three.
    # A full 3-column grid would leave a third of the canvas blank and that
    # empty area still costs raster time on every save.
    fig = plt.figure(figsize=(9.5, 2.4 * (nb_of_rows + 1)))
    outer = fig.add_gridspec(
        nb_of_rows + 1,
        1,
        left=0.08,
        right=0.99,
        top=0.98,
        bottom=0.02,
        hspace=0.62,
    )
    row_axes = []
    for i in range(nb_of_rows):
        inner = outer[i].subgridspec(1, 2, width_ratios=_SPEC_HOLD_WIDTH_RATIOS, wspace=0.08)
        row_axes.append((fig.add_subplot(inner[0]), fig.add_subplot(inner[1])))

    occ_inner = outer[-1].subgridspec(1, 3, wspace=0.35)
    occ = dict(
        atom_location_list=atom_location_list,
        atom_target_list=atom_target_list,
        atom_final_list=atom_final_list,
    )
    for j, kind in enumerate(OCCUPATION_MATRIX_KINDS):
        plot_atom_occupation_matrix(
            atom_occupation_matrix(kind, **occ),
            ax=fig.add_subplot(occ_inner[j]),
            title=kind.capitalize(),
            column_IFs=column_IFs,
            row_IFs=row_IFs,
        )

    plot_data = None
    max_hold_axes = []
    for i, raw in enumerate(raw_rows):
        ax, ax_hold = row_axes[i]
        waveform = np.asarray(raw, dtype=float)
        if waveform.size < 32 or not np.any(np.abs(waveform) > 1e-12):
            ax.set_title(f"Row {i + 1} (truncated in simulation or empty)")
            ax_hold.set_visible(False)
            continue
        row_results = dict(results)
        row_results["waveform"] = waveform
        _row_fig, plot_data = plot_results(
            row_results,
            title=f"Row {i + 1}",
            show=False,
            ax_spec=ax,
            ax_hold=ax_hold,
            add_colorbar=False,
            max_hold_calibration=max_hold_calibration,
            max_hold_base_amplitude=max_hold_base_amplitude,
            return_plot_data=True,
        )
        _annotate_sorting_row(
            ax,
            plot_data["times"],
            column_IFs,
            atom_location_list[i],
            target_frequencies_full_python[i],
        )
        max_hold_axes.append(ax_hold)

    _apply_common_max_hold_xlim(max_hold_axes)

    if save_path is not None:
        fig.savefig(str(save_path), dpi=_SAVEFIG_DPI)
        print(f"Saved sorting figure to {save_path}")
    if _want_show(show):
        plt.show()
    else:
        plt.close(fig)
    return fig, plot_data


def save_atom_sorting_plot_data(
    path,
    raw_rows,
    *,
    column_IFs,
    row_IFs,
    atom_location_list,
    atom_target_list,
    atom_final_list,
    target_frequencies_full_python,
    extra=None,
):
    """Save traces and occupation matrices so the sorting figure can be replotted offline."""
    payload = dict(
        raw_data=np.asarray(raw_rows, dtype=object),
        column_IFs=np.asarray(column_IFs),
        row_IFs=np.asarray(row_IFs),
        atom_location_list=np.asarray(atom_location_list),
        atom_target_list=np.asarray(atom_target_list),
        atom_final_list=np.asarray(atom_final_list),
        target_frequencies_full_python=np.asarray(target_frequencies_full_python),
    )
    if extra:
        payload.update(extra)
    np.savez_compressed(path, **payload)
    print(f"Saved plot data to {path}")


def save_spectrogram_data(path, results):
    """Save waveform used by ``plot_results`` for offline replotting."""
    np.savez_compressed(
        path,
        waveform=np.asarray(results["waveform"]),
        Fs=np.float64(results["Fs"]),
        start_freq=np.float64(results["start_freq"]),
        stop_freq=np.float64(results["stop_freq"]),
        mode=np.array(results.get("mode", "")),
    )
    print(f"Saved spectrogram data to {path}")


def split_waveform_into_rows(waveform, nb_of_rows, row_duration_ns, Fs):
    """Slice a concatenated analog waveform into per-row traces."""
    samples_per_row = int(row_duration_ns * 1e-9 * Fs)
    rows = []
    for i in range(nb_of_rows):
        start = i * samples_per_row
        stop = start + samples_per_row
        if start >= len(waveform):
            rows.append(np.zeros(max(samples_per_row, 1)))
        else:
            rows.append(waveform[start:stop])
    return rows


def simulation_sequence_duration_ns():
    """Simulated window per sequence.

    Hardware programs still use the full ``readout_len`` measurement window.
    Simulation is capped so a 1 ms chirp does not generate a full 1 GS/s trace.
    """
    return min(int(readout_len_safe()), 128 * u.us)


def simulation_duration_clock_cycles(n_sequences=1, extra_ns=1_000):
    """Clock-cycle length for ``SimulationConfig``."""
    return int((n_sequences * simulation_sequence_duration_ns() + extra_ns) // 4)


def readout_len_safe():
    from configuration import readout_len

    return int(readout_len)
