# %%
#! %load_ext autoreload
#! %autoreload 2
from amplitude_calibration import check_tweezer_computation_latency, load_xy_amp_calibration
from configuration import *
from minimal_jerk import play_minimal_jerk_chirp
from qm import QuantumMachinesManager, SimulationConfig, generate_qua_script
from qm.qua import *
from readout_analysis import (
    collect_execution_results,
    collect_simulation_results,
    measure_detector_adc,
    plot_pulse_timeline,
    plot_results,
    save_spectrogram_data,
    simulation_duration_clock_cycles,
    spectrogram_limits,
    stream_save_raw_adc,
)

# Names this script's run directory and its computation_latency_cc_<name> row.
# Hardcoded because __file__ is undefined when running as Jupyter cells.
SCRIPT_NAME = "02_arb_tone_MJ_manual"

simulate = False
set_x_detunings = int(40e6)
set_y_detunings = 0

x_amp_calibration, y_amp_calibration = load_xy_amp_calibration(file_name=SCRIPT_NAME)


# ---------------------------------------------------------------------------
# QUA program
# ---------------------------------------------------------------------------


def build_program():
    with program() as arb_tweezer_sort:
        x_frequencies = declare(int, value=list(x_tweezer_IFs))
        x_detunings = declare(int, value=[set_x_detunings] * NUM_X_TWEEZERS)
        y_frequencies = declare(int, value=list(y_tweezer_IFs))
        y_detunings = declare(int, value=[set_y_detunings] * NUM_Y_TWEEZERS)
        raw_adc = declare_stream(adc_trace=True)

        for index, element in enumerate(x_tweezer_elements):
            frame_rotation_2pi(float(x_tweezer_phases[index]), element)
        for index, element in enumerate(y_tweezer_elements):
            frame_rotation_2pi(float(y_tweezer_phases[index]), element)

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

        # Detector timeline starts at t=0 in parallel with the tweezers (do not align first).
        measure_detector_adc(raw_adc)
        align(*all_tweezer_elements, "detector")

        with stream_processing():
            stream_save_raw_adc(raw_adc)

    return arb_tweezer_sort


arb_tweezer_sort = build_program()

qmm_kwargs = {"host": qop_ip, "cluster_name": cluster_name, "timeout": 300}
if qop_port is not None:
    qmm_kwargs["port"] = qop_port
qmm = QuantumMachinesManager(**qmm_kwargs)

run_dir = make_run_dir(SCRIPT_NAME)
debug_path = run_dir / f"{SCRIPT_NAME}_debug.py"
with open(debug_path, "w", encoding="utf-8") as source_file:
    print(generate_qua_script(arb_tweezer_sort, config), file=source_file)
print(f"Saved QUA debug script to {debug_path}")

spec_start, spec_stop = spectrogram_limits(
    x_tweezer_IFs,
    [set_x_detunings] * NUM_X_TWEEZERS,
    y_tweezer_IFs,
    [set_y_detunings] * NUM_Y_TWEEZERS,
)


def _save_02_outputs(results, title):
    fig_path = run_dir / "02_spectrogram.png"
    data_path = run_dir / "02_spectrogram.npz"
    plot_results(
        results,
        title=title,
        save_path=fig_path,
        max_hold_calibration=x_amp_calibration,
        max_hold_base_amplitude=x_tweezer_max_amplitude,
    )
    save_spectrogram_data(data_path, results)


if simulate:
    job = qmm.simulate(config, arb_tweezer_sort, SimulationConfig(simulation_duration_clock_cycles()))
    results = collect_simulation_results(job, spec_start, spec_stop)
    plot_pulse_timeline(job, save_path=run_dir / "pulse_timeline.png", show=False)
    _save_02_outputs(results, "02 simulated spectrogram")
    try:
        job.get_simulated_waveform_report().create_plot(results["samples"], plot=True, save_path=str(run_dir.resolve()))
    except Exception as e:
        print(f"Error creating waveform plot: {e}")
else:
    qm = qmm.open_qm(config, close_other_machines=True)
    job = qm.execute(arb_tweezer_sort)
    results = collect_execution_results(job, spec_start, spec_stop)
    _save_02_outputs(results, "02 execution spectrogram")

check_tweezer_computation_latency(
    job,
    all_tweezer_elements,
    segment_length,
    x_amp_calibration,
    y_amp_calibration,
    file_name=SCRIPT_NAME,
)
