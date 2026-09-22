# %%
#! %load_ext autoreload
#! %autoreload 2
"""Row-by-row left-compression sorting using play_minimal_jerk_chirp."""

import numpy as np
from amplitude_calibration import check_tweezer_computation_latency, load_xy_amp_calibration
from array_sorting_macros import (
    assign_tweezers_to_atoms,
    find_number_of_tweezers,
    get_current_row,
    print_2d,
    python_row_compression_final,
)
from configuration import *
from minimal_jerk import play_minimal_jerk_chirp
from qm import QuantumMachinesManager, SimulationConfig, generate_qua_script
from qm.qua import *
from readout_analysis import (
    collect_execution_results,
    collect_simulation_results,
    if_axis_limits,
    measure_detector_adc,
    plot_atom_sorting_results,
    save_atom_sorting_plot_data,
    simulation_duration_clock_cycles,
    simulation_sequence_duration_ns,
    split_waveform_into_rows,
    stream_save_raw_adc,
)

# Names this script's run directory and its computation_latency_cc_<name> row.
# Hardcoded because __file__ is undefined when running as Jupyter cells.
SCRIPT_NAME = "04_arb_tone_MJ_row_sort"

simulate = False
CHECK_READOUT_WITH_LOOPBACK = True

total_number_of_tweezers = NUM_X_TWEEZERS
assert total_number_of_tweezers >= 1

# Site positions are the configured IFs (spacing may be non-uniform).
nb_of_cols = 16
nb_of_rows = 16
column_IFs = np.linspace(80e6, 120e6, nb_of_cols).astype(int)
row_IFs = np.linspace(80e6, 120e6, nb_of_rows).astype(int)

np.random.seed(42)
atom_location_list = [
    list(map(int, np.random.choice([0, 1], size=nb_of_cols, p=[0.4, 0.6]))) for _ in range(nb_of_rows)
]
# Left-compact target: fill the leftmost sites up to the tweezer budget.
atom_target_list = np.zeros((nb_of_rows, nb_of_cols), dtype=int)
target_col_offset = 3
max_fill = 5
n_fill = min(total_number_of_tweezers, nb_of_cols - target_col_offset, max_fill)
atom_target_list[:, target_col_offset : target_col_offset + n_fill] = 1
atom_target_list = atom_target_list.tolist()

atom_location_list_1d = [j for sub in atom_location_list for j in sub]
atom_target_1d_python = [j for sub in atom_target_list for j in sub]
target_frequencies_full_python = [
    [int(column_IFs[i]) if atom_target_list[j][i] else 0 for i in range(nb_of_cols)] for j in range(nb_of_rows)
]
target_frequencies_1d = [j for sub in target_frequencies_full_python for j in sub]
atom_final_list = python_row_compression_final(atom_location_list, atom_target_list, total_number_of_tweezers)

print("Initial occupation matrix:")
print_2d(atom_location_list)
print("Atom target matrix:")
print_2d(atom_target_list)
print("Expected final occupation matrix:")
print_2d(atom_final_list)

x_amp_calibration, y_amp_calibration = load_xy_amp_calibration(file_name=SCRIPT_NAME)


def build_program():
    with program() as atom_sorting:
        raw_adc = declare_stream(adc_trace=True)
        current_row = declare(int)
        atom_target_full_qua = declare(int, value=atom_target_1d_python)
        target_frequencies_full_qua = declare(int, value=target_frequencies_1d)
        row_frequencies_qua = declare(int, value=[int(x) for x in row_IFs])
        column_frequencies_qua = declare(int, value=[int(x) for x in column_IFs])
        atom_location_full_qua = declare(int, value=atom_location_list_1d)

        for index, element in enumerate(x_tweezer_elements):
            frame_rotation_2pi(float(x_tweezer_phases[index]), element)
        for index, element in enumerate(y_tweezer_elements):
            frame_rotation_2pi(float(y_tweezer_phases[index]), element)

        with for_(current_row, 0, current_row < nb_of_rows, current_row + 1):
            atom_location_qua, atom_target_qua, target_frequencies_qua = get_current_row(
                current_row,
                nb_of_cols,
                atom_location_full_qua,
                atom_target_full_qua,
                target_frequencies_full_qua,
            )
            number_of_tweezers = find_number_of_tweezers(atom_location_qua, atom_target_qua, total_number_of_tweezers)
            frequencies_qua, detunings_qua = assign_tweezers_to_atoms(
                number_of_tweezers,
                total_number_of_tweezers,
                atom_location_qua,
                column_frequencies_qua,
                target_frequencies_qua,
                nb_of_cols,
            )

            # force to wait until the calculation is completed
            align()

            for index, element in enumerate(x_tweezer_elements):
                with if_(number_of_tweezers > index):
                    play_minimal_jerk_chirp(
                        element,
                        frequencies_qua[index],
                        detunings_qua[index],
                        segment_length,
                        number_of_segments,
                        amp_calibration=x_amp_calibration,
                    )

                    # play the y-tweezer along if there is any x-tweezer assigned
                    if index == 0:
                        play_minimal_jerk_chirp(
                            y_tweezer_elements[0],
                            row_frequencies_qua[current_row],
                            0,
                            segment_length,
                            number_of_segments,
                            amp_calibration=y_amp_calibration,
                        )

            if CHECK_READOUT_WITH_LOOPBACK:
                measure_detector_adc(raw_adc)
            align(*all_tweezer_elements, "detector")

        with stream_processing():
            if CHECK_READOUT_WITH_LOOPBACK:
                stream_save_raw_adc(raw_adc)

    return atom_sorting


atom_sorting = build_program()

qmm_kwargs = {"host": qop_ip, "cluster_name": cluster_name, "timeout": 300}
if qop_port is not None:
    qmm_kwargs["port"] = qop_port
qmm = QuantumMachinesManager(**qmm_kwargs)

fig_dir = make_run_dir(SCRIPT_NAME)
debug_path = fig_dir / f"{SCRIPT_NAME}_debug.py"
fig_path = fig_dir / "04_sorting.png"
data_path = fig_dir / "04_sorting.npz"

with open(debug_path, "w", encoding="utf-8") as source_file:
    print(generate_qua_script(atom_sorting, config), file=source_file)
print(f"Saved QUA debug script to {debug_path}")

spec_start, spec_stop = if_axis_limits(column_IFs)

_plot_kwargs = dict(
    column_IFs=column_IFs,
    row_IFs=row_IFs,
    atom_location_list=atom_location_list,
    atom_target_list=atom_target_list,
    atom_final_list=atom_final_list,
    target_frequencies_full_python=target_frequencies_full_python,
)


def _save_sorting_outputs(results):
    if CHECK_READOUT_WITH_LOOPBACK:
        fig, plot_data = plot_atom_sorting_results(
            results,
            max_hold_calibration=x_amp_calibration,
            max_hold_base_amplitude=x_tweezer_max_amplitude,
            save_path=fig_path,
            **_plot_kwargs,
        )
        save_atom_sorting_plot_data(
            data_path,
            results["rows"],
            extra={"mode": results["mode"], "Fs": results["Fs"]},
            **_plot_kwargs,
        )


if simulate:
    job = qmm.simulate(
        config,
        atom_sorting,
        SimulationConfig(simulation_duration_clock_cycles(n_sequences=nb_of_rows)),
    )
    results = collect_simulation_results(job, spec_start, spec_stop)
    # The simulator returns one continuous analog trace covering every row.
    raw_rows = split_waveform_into_rows(
        results["waveform"], nb_of_rows, simulation_sequence_duration_ns(), results["Fs"]
    )
    results["rows"] = raw_rows
    _save_sorting_outputs(results)
else:
    qm = qmm.open_qm(config, close_other_machines=True)
    job = qm.execute(atom_sorting)
    results = collect_execution_results(job, spec_start, spec_stop)
    _save_sorting_outputs(results)

print("Done")

check_tweezer_computation_latency(
    job,
    all_tweezer_elements,
    segment_length,
    x_amp_calibration,
    y_amp_calibration,
    tolerance_cc=5,
    file_name=SCRIPT_NAME,
)

# %%
