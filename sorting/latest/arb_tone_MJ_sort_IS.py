# %%
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from configuration import *
from minimal_jerk import play_minimal_jerk_chirp
from qm import QuantumMachinesManager, SimulationConfig, generate_qua_script
from qm.qua import *

DEFAULT_HARDWARE_DETUNING_HZ = 0
DEFAULT_START_FREQUENCY_HZ = int(60e6)
DEFAULT_START_SPACING_HZ = int(0.5e6)
DEFAULT_STOP_FREQUENCY_HZ = int(100e6)
DEFAULT_STOP_SPACING_HZ = int(2e6)
FIRST_TONE_DETUNING_HZ = DEFAULT_STOP_FREQUENCY_HZ - DEFAULT_START_FREQUENCY_HZ

simulate = False
write_qua_script = True

# Keep simulation short enough for quick compile/routing checks. Hardware runs
# still use all NUMBER_OF_TONES and the full chirp length from configuration.py.
SIMULATION_NUMBER_OF_TONES = min(NUMBER_OF_TONES, TONES_PER_OUTPUT + 1)
SIMULATION_NUMBER_OF_SEGMENTS = min(number_of_segments, 20)


def _program_tone_elements(use_input_streams):
    if use_input_streams:
        return tone_elements
    return tone_elements[:SIMULATION_NUMBER_OF_TONES]


def _program_number_of_segments(use_input_streams):
    if use_input_streams:
        return number_of_segments
    return SIMULATION_NUMBER_OF_SEGMENTS


def _default_start_frequencies(size):
    indices = np.arange(size, dtype=int)
    return (DEFAULT_START_FREQUENCY_HZ + DEFAULT_START_SPACING_HZ * indices).astype(int).tolist()


def _default_stop_frequencies(size):
    indices = np.arange(size, dtype=int)
    return (DEFAULT_STOP_FREQUENCY_HZ + DEFAULT_STOP_SPACING_HZ * indices).astype(int).tolist()


def _default_test_frequencies(size):
    return _default_start_frequencies(size)


def _default_sort_detunings(size):
    return _detunings_for_stop_grid(_default_start_frequencies(size))


def _detunings_for_stop_grid(start_frequencies):
    """Map each start tone to the fixed 200 MHz + 2 MHz stop comb."""
    starts = np.asarray(start_frequencies, dtype=int)
    stops = np.asarray(_default_stop_frequencies(starts.size), dtype=int)
    return (stops - starts).astype(int).tolist()


def _default_test_detunings(size):
    return _default_sort_detunings(size)


def _play_sort(elements, frequencies, detunings, chirp_segments):
    for index, element in enumerate(elements):
        play_minimal_jerk_chirp(
            element,
            frequencies[index],
            detunings[index],
            tone_amplitudes[index],
            segment_length,
            chirp_segments,
        )

    align(*elements)


def build_program(use_input_streams=True):
    program_elements = _program_tone_elements(use_input_streams)
    program_tone_count = len(program_elements)
    program_segments = _program_number_of_segments(use_input_streams)

    with program() as arb_tone_sort:
        if use_input_streams:
            frequencies = declare_input_stream(int, size=NUMBER_OF_TONES, name="frequencies")
            detunings = declare_input_stream(int, size=NUMBER_OF_TONES, name="detunings")
        else:
            frequencies = declare(int, value=_default_test_frequencies(program_tone_count))
            detunings = declare(int, value=_default_test_detunings(program_tone_count))

        for index, element in enumerate(program_elements):
            frame_rotation_2pi(float(tone_phases[index]), element)

        if use_input_streams:
            with infinite_loop_():
                advance_input_stream(frequencies)
                advance_input_stream(detunings)

                _play_sort(program_elements, frequencies, detunings, program_segments)
        else:
            _play_sort(program_elements, frequencies, detunings, program_segments)

    return arb_tone_sort


def _parse_integer_list(text, name):
    try:
        values = [int(value.strip()) for value in text.split(",")]
    except ValueError:
        print(f"{name} must contain only comma-separated integer Hz values.")
        return None

    if len(values) != NUMBER_OF_TONES:
        print(f"Expected {NUMBER_OF_TONES} {name}, got {len(values)}.")
        return None

    return values


def _print_sent_inputs(shot_number, frequencies, detunings):
    start_mhz = [frequency / 1e6 for frequency in frequencies]
    stop_mhz = [(frequency + detuning) / 1e6 for frequency, detuning in zip(frequencies, detunings)]
    print(f"Sent shot {shot_number}.")
    print(f"  start MHz: {np.round(start_mhz, 3).tolist()}")
    print(f"  stop MHz:  {np.round(stop_mhz, 3).tolist()}")


def _prompt_for_inputs(job):
    print(f"Enter {NUMBER_OF_TONES} comma-separated integer Hz values.")
    print(
        "Type default for 190 MHz + 0.5 MHz starts chirping to "
        f"200 MHz + 2 MHz stops (first tone +{FIRST_TONE_DETUNING_HZ / 1e6:.0f} MHz). "
        "Type q to quit. Press Enter at frequencies to resend the same starts "
        "with per-tone detunings that land on the 200 MHz + 2 MHz stop comb."
    )
    previous_frequencies = None
    shot_number = 0

    while True:
        frequency_text = input("frequencies: ").strip()
        if frequency_text.lower() in {"q", "quit", "exit"}:
            return

        if frequency_text.lower() in {"d", "default"}:
            frequencies = _default_start_frequencies(NUMBER_OF_TONES)
            detunings = _default_sort_detunings(NUMBER_OF_TONES)
            print("Sending 190 MHz + 0.5 MHz starts chirping to 200 MHz + 2 MHz stops.")
        elif frequency_text == "":
            if previous_frequencies is None:
                frequencies = _default_start_frequencies(NUMBER_OF_TONES)
                print("Sending 190 MHz + 0.5 MHz starts with per-tone detunings to the 200 MHz + 2 MHz stop comb.")
            else:
                frequencies = previous_frequencies
                print("Resending previous starts with per-tone detunings to the 200 MHz + 2 MHz stop comb.")
            detunings = _detunings_for_stop_grid(frequencies)
        else:
            detuning_text = input("detunings: ").strip()
            if detuning_text.lower() in {"q", "quit", "exit"}:
                return

            frequencies = _parse_integer_list(frequency_text, "frequencies")
            detunings = _parse_integer_list(detuning_text, "detunings")

            if frequencies is None or detunings is None:
                continue

        job.push_to_input_stream("frequencies", frequencies)
        job.push_to_input_stream("detunings", detunings)
        shot_number += 1
        previous_frequencies = frequencies
        _print_sent_inputs(shot_number, frequencies, detunings)


arb_tone_sort = build_program(use_input_streams=not simulate)

qmm_kwargs = {"host": qop_ip, "cluster_name": cluster_name, "timeout": 300}
if qop_port is not None:
    qmm_kwargs["port"] = qop_port
qmm = QuantumMachinesManager(**qmm_kwargs)

if write_qua_script:
    debug_script_path = Path(__file__).with_name("debug_arb_tone_sort.py")
    with open(debug_script_path, "w") as source_file:
        print(generate_qua_script(arb_tone_sort, config), file=source_file)
    print(f"Wrote {debug_script_path}")

if simulate:
    simulation_duration = SIMULATION_NUMBER_OF_TONES * (
        rampup_length + segment_length * (SIMULATION_NUMBER_OF_SEGMENTS + 2) + rampdown_length + 1_000
    )
    print(
        f"Simulating {SIMULATION_NUMBER_OF_TONES}/{NUMBER_OF_TONES} tones "
        f"for {SIMULATION_NUMBER_OF_SEGMENTS}/{number_of_segments} chirp segments"
    )
    job = qmm.simulate(config, arb_tone_sort, SimulationConfig(int(simulation_duration // 4)))
    job.wait_until("Done", timeout=120)

    try:
        samples = job.get_simulated_samples(include_analog=True, include_digital=False)
        samples.con1.plot()
        plt.tight_layout()
        simulated_samples_path = save_dir / "simulated_samples.png"
        plt.savefig(simulated_samples_path, dpi=150)
        print(f"Saved simulated samples plot to {simulated_samples_path}")
        plt.close()
    except Exception:
        print("Simulation status:", job.get_status())
        print("Simulation errors:", job.get_errors())
        print(job.execution_report())
        raise
else:
    qm = qmm.open_qm(config, close_other_machines=True)
    job = qm.execute(arb_tone_sort)
    try:
        _prompt_for_inputs(job)
    finally:
        job.halt()
