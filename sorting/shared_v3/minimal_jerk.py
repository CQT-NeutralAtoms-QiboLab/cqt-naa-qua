from qm.qua import *

FIXED_DECIMAL_BIT = 28
ADC_BITSHIFT = FIXED_DECIMAL_BIT - 16


def play_minimal_jerk_chirp(
    element,
    start_frequency,
    detuning,
    segment_length=1000,
    number_of_segments=1000,
    amp_calibration=None,
):
    """Play a minimal-jerk chirp on a single element.

    Both paths use ``computation_latency_cc`` from *amp_calibration*.
    Amplitude compensation is used only when the calibration has polynomial
    coefficients (``latency_only`` is False).  Otherwise the constant-amplitude
    path runs with the same processor gap.
    """
    assert amp_calibration is not None
    if not amp_calibration.latency_only and amp_calibration.coefs:
        _play_compensated(
            element,
            start_frequency,
            detuning,
            segment_length,
            number_of_segments,
            amp_calibration,
        )
    else:
        _play_constant(
            element,
            start_frequency,
            detuning,
            segment_length,
            number_of_segments,
            amp_calibration.computation_latency_cc,
        )


# ---------------------------------------------------------------------------
# Original constant-amplitude path
# ---------------------------------------------------------------------------


def _play_constant(
    element,
    start_frequency,
    detuning,
    segment_length,
    number_of_segments,
    computation_latency_cc,
):
    number_of_steps = number_of_segments + 1
    inv_number_of_steps = 1 / number_of_steps
    chirp_scaling_mHz_nsec = 30 * 1e3 / (number_of_steps * segment_length)

    tau = declare(fixed, value=0.0)
    tau2 = declare(fixed)
    chirp_rate = declare(int)

    update_frequency(element, start_frequency)
    play("rampup", element)

    with for_(
        tau,
        inv_number_of_steps,
        tau < (1 - (0.5 * inv_number_of_steps)),
        tau + inv_number_of_steps,
    ):
        assign(tau2, tau * tau)
        assign(
            chirp_rate,
            Cast.mul_int_by_fixed(
                detuning,
                chirp_scaling_mHz_nsec * tau2 * (1 - 2 * tau + tau2),
            ),
        )
        play(
            "hold",
            element,
            duration=segment_length // 4 - computation_latency_cc,
            chirp=(chirp_rate, "mHz/nsec"),
            continue_chirp=True,
            timestamp_stream=f"t_{element}",
        )

    play("rampdown", element)
    ramp_to_zero(element)


# ---------------------------------------------------------------------------
# Compensated amplitude path
# ---------------------------------------------------------------------------


def _play_compensated(
    element,
    start_frequency,
    detuning,
    segment_length,
    number_of_segments,
    amp_calibration,
):
    from amplitude_calibration import amp_at_frequency_qua, fixed_times
    from configuration import chirp_pulse_amplitude

    center_ghz, coefs = amp_calibration.center, amp_calibration.coefs
    computation_latency_cc = amp_calibration.computation_latency_cc

    number_of_steps = number_of_segments + 1
    inv_number_of_steps = 1 / number_of_steps
    chirp_scaling_mHz_nsec = 30 * 1e3 / (number_of_steps * segment_length)
    play_ns = segment_length - (4 * computation_latency_cc)
    inv_play_ns = 1.0 / play_ns
    inv_amp = 1.0 / chirp_pulse_amplitude

    # Same conversion as minimal_jerk_debug.play_compensated_debug.
    qua_Hz_to_GHz = lambda f: (2**FIXED_DECIMAL_BIT / 1e9) * Cast.unsafe_cast_fixed(f)
    qua_chirp_rate_mHz_nsec_to_detuning_GHz = lambda cr: (
        (segment_length / (2**-FIXED_DECIMAL_BIT * 1e9) * 1e-3) * Cast.unsafe_cast_fixed(cr)
    )
    fixed_to_dac_bitshift = lambda f: (f >> ADC_BITSHIFT) << ADC_BITSHIFT

    tau = declare(fixed)
    tau2 = declare(fixed)
    chirp_rate = declare(int)
    cur_freq_ghz = declare(fixed)
    amp_mod = declare(fixed)
    ramp_rate = declare(fixed)
    target_v = declare(fixed)
    cumsum_v = declare(fixed)

    assign(cur_freq_ghz, qua_Hz_to_GHz(start_frequency))
    # Do not reuse polynomial temps across the loop — debug declares a fresh
    # set each evaluation.
    assign(amp_mod, amp_at_frequency_qua(cur_freq_ghz, center_ghz, coefs))
    assign(target_v, amp_mod * chirp_pulse_amplitude)
    assign(cumsum_v, fixed_to_dac_bitshift(target_v))

    update_frequency(element, start_frequency)
    play("rampup" * amp(amp_mod), element)

    with for_(
        tau,
        inv_number_of_steps,
        tau < (1 - (0.5 * inv_number_of_steps)),
        tau + inv_number_of_steps,
    ):
        assign(tau2, tau * tau)
        assign(
            chirp_rate,
            Cast.mul_int_by_fixed(
                detuning,
                chirp_scaling_mHz_nsec * tau2 * (1 - 2 * tau + tau2),
            ),
        )

        assign(
            cur_freq_ghz,
            cur_freq_ghz + qua_chirp_rate_mHz_nsec_to_detuning_GHz(chirp_rate),
        )
        assign(amp_mod, amp_at_frequency_qua(cur_freq_ghz, center_ghz, coefs))
        assign(target_v, amp_mod * chirp_pulse_amplitude)
        # 16-bit DAC start / tracked state; Q4.28 slope during the ramp;
        # remainder below 16-bit is forgotten at the pulse boundary.
        assign(ramp_rate, (target_v - cumsum_v) * inv_play_ns)
        assign(cumsum_v, fixed_to_dac_bitshift(cumsum_v + Cast.mul_fixed_by_int(ramp_rate, play_ns)))

        play(
            ramp(ramp_rate),
            element,
            duration=play_ns // 4,
            chirp=(chirp_rate, "mHz/nsec"),
            continue_chirp=True,
            timestamp_stream=f"t_{element}",
        )

    play("rampdown" * amp(fixed_times(cumsum_v, inv_amp)), element, continue_chirp=False)
    ramp_to_zero(element)
