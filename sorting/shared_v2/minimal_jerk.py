from qm.qua import *


def play_minimal_jerk_chirp(
    element,
    start_frequency,
    detuning,
    amplitude=1.0,
    segment_length=1000,
    number_of_segments=1000,
    computation_cycles=14,
):
    """Play one constant-amplitude minimal-jerk chirp on a single element."""
    tau = declare(fixed, value=0.0)
    tau2 = declare(fixed)
    chirp_rate = declare(int)

    inv_number_of_segments = 1 / (number_of_segments + 1)
    chirp_scaling = 30 * 1e3 / ((number_of_segments + 1) * segment_length)

    update_frequency(element, start_frequency)
    play("rampup" * amp(amplitude), element)

    with for_(
        tau,
        inv_number_of_segments,
        tau < (1 - 3 * inv_number_of_segments / 2),
        tau + inv_number_of_segments,
    ):
        assign(tau2, tau * tau)
        assign(
            chirp_rate,
            Cast.mul_int_by_fixed(
                detuning,
                chirp_scaling * tau2 * (1 - 2 * tau + tau2),
            ),
        )
        play(
            "hold",
            element,
            duration=segment_length // 4 - computation_cycles,
            chirp=(chirp_rate, "mHz/nsec"),
            continue_chirp=True,
        )
    wait(computation_cycles, element)
    assign(tau, 1 - inv_number_of_segments)
    assign(tau2, tau * tau)
    assign(
        chirp_rate,
        Cast.mul_int_by_fixed(
            detuning,
            chirp_scaling * tau2 * (1 - 2 * tau + tau2),
        ),
    )
    play(
        "hold",
        element,
        duration=segment_length // 4 - computation_cycles,
        chirp=(chirp_rate, "mHz/nsec"),
        continue_chirp=False,
    )

    play("rampdown" * amp(amplitude), element)
    ramp_to_zero(element)
