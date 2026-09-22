# %%
#! %load_ext autoreload
#! %autoreload 2
import numpy as np
from amplitude_calibration import check_tweezer_computation_latency, load_xy_amp_calibration
from configuration import *
from minimal_jerk import play_minimal_jerk_chirp
from qm import QuantumMachinesManager, SimulationConfig, generate_qua_script
from qm.qua import *
from readout_analysis import (
    collect_execution_results,
    collect_simulation_results,
    measure_detector_adc,
    plot_results,
    save_spectrogram_data,
    simulation_duration_clock_cycles,
    spectrogram_limits,
    stream_save_raw_adc,
)

# Names this script's run directory and its computation_latency_cc_<name> row.
# Hardcoded because __file__ is undefined when running as Jupyter cells.
SCRIPT_NAME = "03_arb_tone_MJ_sort_manual_IS"

simulate = False

x_amp_calibration, y_amp_calibration = load_xy_amp_calibration(file_name=SCRIPT_NAME)


# ---------------------------------------------------------------------------
# QUA program
# ---------------------------------------------------------------------------


def build_program(*, use_input_streams: bool, x_freq_init, x_det_init, y_freq_init, y_det_init):
    with program() as arb_tweezer_sort:
        raw_adc = declare_stream(adc_trace=True)
        if use_input_streams:
            x_frequencies = declare_input_stream(int, size=NUM_X_TWEEZERS, name="x_frequencies")
            x_detunings = declare_input_stream(int, size=NUM_X_TWEEZERS, name="x_detunings")
            y_frequencies = declare_input_stream(int, size=NUM_Y_TWEEZERS, name="y_frequencies")
            y_detunings = declare_input_stream(int, size=NUM_Y_TWEEZERS, name="y_detunings")
        else:
            x_frequencies = declare(int, value=list(x_freq_init))
            x_detunings = declare(int, value=list(x_det_init))
            y_frequencies = declare(int, value=list(y_freq_init))
            y_detunings = declare(int, value=list(y_det_init))

        for index, element in enumerate(x_tweezer_elements):
            frame_rotation_2pi(float(x_tweezer_phases[index]), element)
        for index, element in enumerate(y_tweezer_elements):
            frame_rotation_2pi(float(y_tweezer_phases[index]), element)

        def _play_chirps():
            for index, element in enumerate(x_tweezer_elements):
                play_minimal_jerk_chirp(
                    element,
                    x_frequencies[index],
                    x_detunings[index],
                    segment_length,
                    number_of_segments,
                    amp_calibration=x_amp_calibration,
                )
            for index, element in enumerate(y_tweezer_elements):
                play_minimal_jerk_chirp(
                    element,
                    y_frequencies[index],
                    y_detunings[index],
                    segment_length,
                    number_of_segments,
                    amp_calibration=y_amp_calibration,
                )
            measure_detector_adc(raw_adc)
            align(*all_tweezer_elements, "detector")

        if use_input_streams:
            with infinite_loop_():
                advance_input_stream(x_frequencies)
                advance_input_stream(x_detunings)
                advance_input_stream(y_frequencies)
                advance_input_stream(y_detunings)
                _play_chirps()
        else:
            _play_chirps()

        with stream_processing():
            stream_save_raw_adc(raw_adc)

    return arb_tweezer_sort


def _default_test_inputs():
    x_freq = x_tweezer_IFs.astype(int).tolist()
    x_det = np.linspace(-5e6, 5e6, NUM_X_TWEEZERS).astype(int).tolist() if NUM_X_TWEEZERS > 1 else [0]
    y_freq = y_tweezer_IFs.astype(int).tolist()
    y_det = [0] * NUM_Y_TWEEZERS
    return x_freq, x_det, y_freq, y_det


_QUIT_WORDS = {"q", "quit", "exit"}


class _QuitRequested(Exception):
    """Raised when the operator asks to leave the input loop."""


def _ask(prompt):
    """Prompt for one stream. Blank line, 'q', Ctrl+C or Ctrl+D quits."""
    try:
        text = input(prompt)
    except (EOFError, KeyboardInterrupt):
        raise _QuitRequested from None
    if not text.strip() or text.strip().lower() in _QUIT_WORDS:
        raise _QuitRequested
    return text


def _parse_stream_values(text, n_tweezers):
    values = [int(float(v.strip())) for v in text.split(",") if v.strip()]
    if len(values) > n_tweezers:
        raise ValueError(f"got {len(values)} values but only {n_tweezers} tweezers")
    # pad with zeros if the list is too short
    return values + [0] * (n_tweezers - len(values))


def _prompt_for_inputs(job):
    print(f"X: {NUM_X_TWEEZERS} tweezers  |  Y: {NUM_Y_TWEEZERS} tweezers")
    print("Enter comma-separated integer Hz values for each stream.")
    print("Blank line, 'q', Ctrl+C or Ctrl+D quits.")
    while True:
        try:
            x_freq = _parse_stream_values(_ask("x_frequencies: "), NUM_X_TWEEZERS)
            x_det = _parse_stream_values(_ask("x_detunings:   "), NUM_X_TWEEZERS)
            y_freq = _parse_stream_values(_ask("y_frequencies: "), NUM_Y_TWEEZERS)
            y_det = _parse_stream_values(_ask("y_detunings:   "), NUM_Y_TWEEZERS)
            # x_freq = _parse_stream_values("80e6", NUM_X_TWEEZERS)
            # x_det = _parse_stream_values("40e6", NUM_X_TWEEZERS)
            # y_freq = _parse_stream_values("80e6", NUM_Y_TWEEZERS)
            # y_det = _parse_stream_values("0", NUM_Y_TWEEZERS)
        except _QuitRequested:
            print("\nQuitting input loop.")
            return
        except ValueError as e:
            print(f"Invalid input ({e}) -- try again.")
            continue

        job.push_to_input_stream("x_frequencies", x_freq)
        job.push_to_input_stream("x_detunings", x_det)
        job.push_to_input_stream("y_frequencies", y_freq)
        job.push_to_input_stream("y_detunings", y_det)
        yield x_freq, x_det, y_freq, y_det

        raise _QuitRequested()


x_freq_init, x_det_init, y_freq_init, y_det_init = _default_test_inputs()
arb_tweezer_sort = build_program(
    use_input_streams=not simulate,
    x_freq_init=x_freq_init,
    x_det_init=x_det_init,
    y_freq_init=y_freq_init,
    y_det_init=y_det_init,
)

qmm_kwargs = {"host": qop_ip, "cluster_name": cluster_name, "timeout": 300}
if qop_port is not None:
    qmm_kwargs["port"] = qop_port
qmm = QuantumMachinesManager(**qmm_kwargs)

run_dir = make_run_dir(SCRIPT_NAME)
debug_path = run_dir / f"{SCRIPT_NAME}_debug.py"
with open(debug_path, "w", encoding="utf-8") as source_file:
    print(generate_qua_script(arb_tweezer_sort, config), file=source_file)
print(f"Saved QUA debug script to {debug_path}")


def _save_03_outputs(results, title, stem="03_spectrogram"):
    plot_results(
        results,
        title=title,
        save_path=run_dir / f"{stem}.png",
        max_hold_calibration=x_amp_calibration,
        max_hold_base_amplitude=x_tweezer_max_amplitude,
    )
    save_spectrogram_data(run_dir / f"{stem}.npz", results)


if simulate:
    job = qmm.simulate(config, arb_tweezer_sort, SimulationConfig(simulation_duration_clock_cycles()))
    spec_start, spec_stop = spectrogram_limits(x_freq_init, x_det_init, y_freq_init, y_det_init)
    results = collect_simulation_results(job, spec_start, spec_stop)
    _save_03_outputs(results, "03 simulated spectrogram")
    check_tweezer_computation_latency(
        job,
        all_tweezer_elements,
        segment_length,
        x_amp_calibration,
        y_amp_calibration,
        file_name=SCRIPT_NAME,
    )
else:
    qm = qmm.open_qm(config, close_other_machines=True)
    job = qm.execute(arb_tweezer_sort)
    try:
        for n, (x_freq, x_det, y_freq, y_det) in enumerate(_prompt_for_inputs(job), start=1):
            spec_start, spec_stop = spectrogram_limits(x_freq, x_det, y_freq, y_det)
            results = collect_execution_results(job, spec_start, spec_stop, wait_for_all=False, n_values=1)
            stem = "03_spectrogram" if n == 1 else f"03_spectrogram_{n:03d}"
            _save_03_outputs(results, "03 execution spectrogram", stem=stem)
            correct_latency = check_tweezer_computation_latency(
                job,
                all_tweezer_elements,
                segment_length,
                x_amp_calibration,
                y_amp_calibration,
                file_name=SCRIPT_NAME,
            )
            if not correct_latency:
                raise RuntimeError("Computation time is not calibrated")
    finally:
        # The QUA program sits in infinite_loop_ waiting on the input streams,
        # so it keeps holding the machine until it is halted.
        job.halt()
        qm.close()
        print("Job halted, quantum machine closed.")
