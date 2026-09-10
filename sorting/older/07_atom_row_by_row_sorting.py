# %%
# !%load_ext autoreload
# !%autoreload 2
# %%
import matplotlib.pyplot as plt
import numpy as np
from array_sorting_macros import *
from configuration import *
from matplotlib.colors import BoundaryNorm, ListedColormap
from qm import LoopbackInterface, QuantumMachinesManager, SimulationConfig
from qm.qua import *

#####################################################################
# Configuration start

#############################
# Define run configurations #
#############################
# Enables the collision free sorting algorithm
collision_free = True
# Enables the piecewise chirp decomposition for minimal jerk trajectory
piecewise_chirp = True
# Reads the current occupation matrix via analog readout
analog_occupation_matrix = False
# Acquires chirp tones to plot spectrograms - output should be connected to OPX analog input
raw_adc_acquisition = True  # and MAX_NUMBER_OF_TWEEZERS_PER_FEM <= 10  # limit to 10 tweezers because readout consumes
# Runs the sorting only once, else is infinite loop
single_run = True
# Maximum constant chirp rate in Hz/ns, None uses default pulse length from config
maximum_chirp_rate = None
# Run simulation instead of real execution
simulation = False

# already defined in configuration.py
# nb_of_rows = 20
# nb_of_cols = 20
# total_number_of_tweezers = 10
nb_of_segments = 10
_column_IFs = column_IFs[:nb_of_cols]
_row_IFs = row_IFs[:nb_of_rows]


############################
# Target occupation matrix #
############################
# Defined pattern
# atom_target_list = [
#     [1, 0, 0, 0, 1],
#     [0, 1, 0, 1, 0],
#     [0, 0, 1, 0, 0],
#     [0, 1, 0, 1, 0],
#     [1, 0, 0, 0, 1],
# ]

# Random pattern
# np.random.seed(42)
# atom_target_list = [list(map(int, np.random.choice([0, 1], size=nb_of_cols, p=[0.4, 0.6]))) for i in range(nb_of_rows)]

# Centered pattern
n_border = (nb_of_cols - total_number_of_tweezers) // 2
# n_border = nb_of_cols // 4
atom_target_list = np.ones((nb_of_rows, nb_of_cols), dtype=int)
atom_target_list[:n_border, :] = 0  # top border
atom_target_list[-n_border:, :] = 0  # bottom border
atom_target_list[:, :n_border] = 0  # left border
atom_target_list[:, -n_border:] = 0  # right border

# Configuration end
#####################################################################

################
# Verification #
################
if maximum_chirp_rate is not None and piecewise_chirp:
    raise NotImplementedError("Dynamic pulse duration with piecewise chirps is not implemented.")
if not analog_occupation_matrix:
    # Initial occupation matrix in 2D
    atom_location_list = [
        list(map(int, np.random.choice([0, 1], size=nb_of_cols, p=[0.4, 0.6]))) for i in range(nb_of_rows)
    ]
else:
    raise NotImplementedError("Analog occupation matrix reading is not implemented.")
# Initial occupation matrix in 1D
atom_location_list_1d = [j for sub in atom_location_list for j in sub]

print("Initial occupation matrix:")
print_2d(atom_location_list)
print("Atom target matrix:")
print_2d(atom_target_list)

# %%

ATOM_SORTING_PLOT_DATA_FILE = save_dir / "atom_sorting_plot_data.npz"


def save_atom_sorting_plot_data(
    path,
    raw,
    *,
    column_IFs,
    column_spacing,
    atom_location_list,
    atom_target_list,
    atom_final_list,
    target_frequencies_full_python,
    nb_of_rows,
    nb_of_cols,
):
    """Save everything needed to reproduce the spectrogram / occupation-matrix figure."""
    np.savez_compressed(
        path,
        raw_data=np.asarray(raw, dtype=object),
        column_IFs=np.asarray(column_IFs),
        column_spacing=np.asarray(column_spacing),
        atom_location_list=np.asarray(atom_location_list),
        atom_target_list=np.asarray(atom_target_list),
        atom_final_list=np.asarray(atom_final_list),
        target_frequencies_full_python=np.asarray(target_frequencies_full_python),
        nb_of_rows=np.int32(nb_of_rows),
        nb_of_cols=np.int32(nb_of_cols),
        fs=np.int32(14),
        subplot_ncols=np.int32(4),
        specgram_nfft=np.int32(2**13),
        specgram_Fs=np.float64(1e9),
        specgram_noverlap=np.int32(100),
        specgram_mode=np.array("magnitude"),
    )
    print(f"Saved plot data to {path}")


_OCCUPATION_MATRIX_CMAP = ListedColormap(["red", "black", "white", "cyan"])  # -1, 0, 1, 2
_OCCUPATION_MATRIX_NORM = BoundaryNorm([-1.5, -0.5, 0.5, 1.5, 2.5], _OCCUPATION_MATRIX_CMAP.N)
OCCUPATION_MATRIX_KINDS = ("initial", "target", "current")


def atom_occupation_matrix(
    kind,
    *,
    atom_location_list,
    atom_target_list,
    atom_final_list,
):
    """Return the 2D occupation matrix for kind: 'initial', 'target', or 'current'."""
    kind = kind.lower()
    if kind == "initial":
        return np.asarray(atom_location_list)
    if kind == "target":
        return np.asarray(atom_target_list)
    if kind == "current":
        return 2 * np.asarray(atom_final_list) - np.asarray(atom_target_list)
    raise ValueError(f"kind must be one of {OCCUPATION_MATRIX_KINDS!r}, got {kind!r}")


def plot_atom_occupation_matrix(mat, *, ax=None, title=None, fs=14):
    """Plot one occupation matrix. Creates a new figure when ax is None."""
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))
    ax.pcolormesh(
        mat,
        edgecolor="gray",
        cmap=_OCCUPATION_MATRIX_CMAP,
        norm=_OCCUPATION_MATRIX_NORM,
        linewidth=3,
    )
    ax.invert_yaxis()
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    if title is not None:
        ax.set_title(title, fontsize=fs)
    return ax


def create_atom_sorting_axes(nb_of_rows, subplot_ncols=4, figsize_width=25, row_height=6):
    """Create one figure: first nb_of_rows axes for spectrograms, next 3 for occupation matrices."""
    n_occ = len(OCCUPATION_MATRIX_KINDS)
    subplot_nrows = nb_of_rows // subplot_ncols + n_occ
    fig, axes = plt.subplots(
        subplot_nrows,
        subplot_ncols,
        figsize=(figsize_width, row_height * subplot_nrows),
    )
    axes_flat = np.atleast_1d(axes).ravel()
    n_used = nb_of_rows + n_occ
    for j in range(n_used, len(axes_flat)):
        axes_flat[j].set_visible(False)
    return fig, axes_flat[:nb_of_rows], axes_flat[nb_of_rows:n_used]


def plot_atom_sorting_spectrogram_row(
    row_index,
    raw,
    *,
    column_IFs,
    column_spacing,
    atom_location_list,
    target_frequencies_full_python,
    ax=None,
    fs=14,
    specgram_nfft=2**13,
    specgram_Fs=1e9,
    specgram_noverlap=100,
    specgram_mode="magnitude",
):
    """Plot one row spectrogram with initial (green) and target (blue) markers. New figure if ax is None."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))
    for j in range(len(column_IFs)):
        ax.axhline(column_IFs[j], color="k", linewidth=1, linestyle="--")
    _, _, time, _ = ax.specgram(
        raw[row_index],
        NFFT=specgram_nfft,
        Fs=specgram_Fs,
        noverlap=specgram_noverlap,
        cmap=plt.cm.gist_heat,
        mode=specgram_mode,
    )

    for atom_loc in np.nonzero(atom_location_list[row_index])[0]:
        atom_freq = column_IFs[atom_loc]
        ax.scatter(time[3], atom_freq, s=100, color="g")

    for target_freq in target_frequencies_full_python[row_index]:
        ax.scatter(time[-3], target_freq, s=100, color="b")

    ax.set_ylabel("Frequency [MHz]", fontsize=fs)
    ax.set_xlabel("Time [µs]", fontsize=fs)
    freq_min = min(column_IFs) - 2 * abs(column_spacing)
    freq_max = max(column_IFs) + 2 * abs(column_spacing)
    ax.axis([0, time[-1], freq_min, freq_max])
    ax.set_title(f"Row {row_index + 1}", fontsize=fs)
    xticks = ax.get_xticks()
    yticks = ax.get_yticks()
    ax.set_xticks(xticks, [t * 1e6 for t in xticks], fontsize=fs)
    ax.set_yticks(yticks, [t / 1e6 for t in yticks], fontsize=fs)
    return ax


def plot_atom_sorting_spectrograms(
    raw,
    *,
    column_IFs,
    column_spacing,
    atom_location_list,
    target_frequencies_full_python,
    nb_of_rows,
    axes=None,
    fig=None,
    fs=14,
    subplot_ncols=4,
    specgram_nfft=2**13,
    specgram_Fs=1e9,
    specgram_noverlap=100,
    specgram_mode="magnitude",
):
    """All row spectrograms. Pass axes from create_atom_sorting_axes, or a new figure is created."""
    if axes is None:
        fig, axes, _ = create_atom_sorting_axes(nb_of_rows, subplot_ncols=subplot_ncols)
    elif fig is None:
        fig = axes[0].figure
    if len(axes) != nb_of_rows:
        raise ValueError(f"expected {nb_of_rows} spectrogram axes, got {len(axes)}")
    for i, ax in enumerate(axes):
        plot_atom_sorting_spectrogram_row(
            i,
            raw,
            column_IFs=column_IFs,
            column_spacing=column_spacing,
            atom_location_list=atom_location_list,
            target_frequencies_full_python=target_frequencies_full_python,
            ax=ax,
            fs=fs,
            specgram_nfft=specgram_nfft,
            specgram_Fs=specgram_Fs,
            specgram_noverlap=specgram_noverlap,
            specgram_mode=specgram_mode,
        )
    return fig


def plot_atom_sorting_results(
    raw,
    *,
    column_IFs,
    column_spacing,
    atom_location_list,
    atom_target_list,
    atom_final_list,
    target_frequencies_full_python,
    nb_of_rows,
    nb_of_cols,
    fs=14,
    subplot_ncols=4,
    specgram_nfft=2**13,
    specgram_Fs=1e9,
    specgram_noverlap=100,
    specgram_mode="magnitude",
):
    """One figure: occupation matrices on reserved axes, then spectrograms on the rest."""
    fig, spectrogram_axes, occupation_axes = create_atom_sorting_axes(nb_of_rows, subplot_ncols=subplot_ncols)
    occ = dict(
        atom_location_list=atom_location_list,
        atom_target_list=atom_target_list,
        atom_final_list=atom_final_list,
    )
    for ax, kind in zip(occupation_axes, OCCUPATION_MATRIX_KINDS):
        plot_atom_occupation_matrix(
            atom_occupation_matrix(kind, **occ),
            ax=ax,
            title=kind.capitalize(),
            fs=fs,
        )
    plot_atom_sorting_spectrograms(
        raw,
        column_IFs=column_IFs,
        column_spacing=column_spacing,
        atom_location_list=atom_location_list,
        target_frequencies_full_python=target_frequencies_full_python,
        nb_of_rows=nb_of_rows,
        axes=spectrogram_axes,
        fig=fig,
        fs=fs,
        subplot_ncols=subplot_ncols,
        specgram_nfft=specgram_nfft,
        specgram_Fs=specgram_Fs,
        specgram_noverlap=specgram_noverlap,
        specgram_mode=specgram_mode,
    )
    fig.tight_layout()
    return fig


def load_atom_sorting_plot_data(path):
    data = np.load(path, allow_pickle=True)
    return {
        "raw": data["raw_data"],
        "column_IFs": data["column_IFs"],
        "column_spacing": float(data["column_spacing"]),
        "atom_location_list": data["atom_location_list"],
        "atom_target_list": data["atom_target_list"],
        "atom_final_list": data["atom_final_list"],
        "target_frequencies_full_python": data["target_frequencies_full_python"],
        "nb_of_rows": int(data["nb_of_rows"]),
        "nb_of_cols": int(data["nb_of_cols"]),
        "fs": int(data["fs"]),
        "subplot_ncols": int(data["subplot_ncols"]),
        "specgram_nfft": int(data["specgram_nfft"]),
        "specgram_Fs": float(data["specgram_Fs"]),
        "specgram_noverlap": int(data["specgram_noverlap"]),
        "specgram_mode": str(data["specgram_mode"]),
    }


# %%

# Target occupation matrix in 1D
atom_target_1d_python = [j for sub in atom_target_list for j in sub]
# Target column frequencies in 1D and of size sum of 1s in target
target_frequencies_full_python = [
    [int(_column_IFs[i]) if atom_target_list[j][i] else 0 for i in range(len(atom_target_list[0]))]
    for j in range(nb_of_rows)
]
target_frequencies_1d = [j for sub in target_frequencies_full_python for j in sub]


# Get all relevant elements in a list for easy align
elements = list(config["elements"].keys())
elements = [e for e in elements if "chirp_AOD" in e]

###############
# QUA program #
###############

# --> Play single tweezer for frequency calibration
with program() as freq_calibration:
    update_frequency("row_chirp_AOD", _row_IFs[0])
    update_frequency("row_chirp_AOD_1", _column_IFs[0])
    # Keep playing row selector and column selector at constant amplitude, tone at 0,0, freq
    with infinite_loop_():
        play("const", "row_chirp_AOD")
        play("const", "row_chirp_AOD_1")

# --> Full atom rearrangment program
with program() as atom_sorting:
    # Variables that need resetting
    received_full_array = declare(bool, value=False)  # Flag indicating the end of occupation matrix readout

    # Debug variables
    raw_adc = declare_stream(adc_trace=True)  # Raw ADC trace for spectrograms or occupation matrix readout
    data_stream = declare_stream()  # stream used to extract variables for debug
    count_stream = declare_stream()  # stream used to extract variables for debug
    infinite_run = declare(bool, value=True)  # Flag used to perform the sorting only once instead of infinite_loop_()

    # QUA variable representing the current row
    current_row = declare(int)
    # QUA variable containing the full 1D target locations
    atom_target_full_qua = declare(int, value=atom_target_1d_python)
    # QUA variable containing the full 1D target frequencies
    target_frequencies_full_qua = declare(int, value=target_frequencies_1d)
    # QUA variable containing the row frequencies
    row_frequencies_qua = declare(int, value=[int(x) for x in _row_IFs])
    # QUA variable containing the column frequencies
    column_frequencies_qua = declare(int, value=[int(x) for x in _column_IFs])
    # QUA variable containing the tweezer phases
    tweezer_phases_qua = declare(fixed, value=phases_list)

    with while_(infinite_run):
        # Reset variables for new loop
        assign(received_full_array, False)

        ###############################################
        # Measure occupation matrix from analog input #
        ###############################################
        if analog_occupation_matrix:
            atom_location_full_qua, received_full_array = analog_readout(
                nb_of_rows, nb_of_cols, threshold, received_full_array
            )
        else:
            atom_location_full_qua = declare(int, value=atom_location_list_1d)
            assign(received_full_array, True)

        ################
        # Atom sorting #
        ################
        with if_(received_full_array):
            # Loop over the rows
            with for_(current_row, 0, current_row < nb_of_rows, current_row + 1):
                # Get the current and target locations and target frequencies of the current row
                (
                    atom_location_qua,
                    atom_target_qua,
                    target_frequencies_qua,
                ) = get_current_row(
                    current_row,
                    nb_of_cols,
                    atom_location_full_qua,
                    atom_target_full_qua,
                    target_frequencies_full_qua,
                )
                # Derive number of required tweezers
                number_of_tweezers = find_number_of_tweezers(
                    atom_location_qua, atom_target_qua, total_number_of_tweezers
                )
                # Assign the tweezers amplitude, initial frequency, phase and detuning using either a dummy logic that
                # will only avoid collisions on left compact targets, or a smarter collision-free algorithm
                if collision_free:
                    (
                        amplitude_qua,
                        frequency_qua,
                        phase_qua,
                        detuning_qua,
                    ) = assign_tweezers_to_atoms_collision_free(
                        number_of_tweezers,
                        total_number_of_tweezers,
                        atom_location_qua,
                        column_frequencies_qua,
                        target_frequencies_qua,
                        tweezer_phases_qua,
                        atom_target_qua,
                        nb_of_cols,
                        data_stream,
                        count_stream,
                    )
                else:
                    (
                        amplitude_qua,
                        frequency_qua,
                        phase_qua,
                        detuning_qua,
                    ) = assign_tweezers_to_atoms(
                        number_of_tweezers,
                        total_number_of_tweezers,
                        atom_location_qua,
                        column_frequencies_qua,
                        target_frequencies_qua,
                        tweezer_phases_qua,
                        data_stream,
                    )
                # Derive the chirp pulse duration from the default pulse length of the maximum chirp rate if not None.
                chirp_pulse_duration_qua = calculate_pulse_length(
                    detuning_qua, aod_const_len, max_rate=maximum_chirp_rate
                )
                # Derive the chirp rates defined as piecewise or constant
                if piecewise_chirp:
                    piecewise_chirp_rates_qua = calculate_piecewise_chirp_rates(
                        total_number_of_tweezers,
                        detuning_qua,
                        chirp_pulse_duration_qua,
                        nb_of_segments,
                    )
                else:
                    constant_chirp_rates_qua = calculate_chirp_rates(
                        total_number_of_tweezers, detuning_qua, chirp_pulse_duration_qua
                    )
                # Assign the frequencies and phases to the tweezers
                set_tweezers_frequencies_and_phases(
                    total_number_of_tweezers,
                    frequency_qua,
                    row_frequencies_qua[current_row],
                    phase_qua,
                )
                # Convert the chirp pulse duration in clock cycles
                assign(chirp_pulse_duration_qua, chirp_pulse_duration_qua >> 2)
                # align all tweezers/columns and row selector
                align(*elements)
                # Wait to calculate as much as possible before playing the pulses to minimize gaps
                if maximum_chirp_rate is not None:
                    wait(200)
                # ramp up power of occupied tweezers and row selector
                play("rampup", "row_AOD")
                for element_index in range(total_number_of_tweezers):
                    play(
                        "rampup" * amp(amplitude_qua[element_index]),
                        "col_AOD_{}".format(element_index + 1),
                    )
                # chirp tweezers
                play("const", "row_AOD")
                for element_index in range(total_number_of_tweezers):
                    if piecewise_chirp:
                        play(
                            "const" * amp(amplitude_qua[element_index]),
                            "col_AOD_{}".format(element_index + 1),
                            chirp=(
                                piecewise_chirp_rates_qua[element_index],
                                "mHz/nsec",
                            ),
                        )  # chirp is 1D vector
                    else:
                        if maximum_chirp_rate is None:
                            play(
                                "const" * amp(amplitude_qua[element_index]),
                                "col_AOD_{}".format(element_index + 1),
                                chirp=(
                                    constant_chirp_rates_qua[element_index],
                                    "mHz/nsec",
                                ),
                            )
                        else:
                            play(
                                "const" * amp(amplitude_qua[element_index]),
                                "col_AOD_{}".format(element_index + 1),
                                duration=chirp_pulse_duration_qua,
                                chirp=(
                                    constant_chirp_rates_qua[element_index],
                                    "mHz/nsec",
                                ),
                            )
                # ramp down power of occupied tweezers
                play("rampdown", "row_AOD")
                for element_index in range(total_number_of_tweezers):
                    play(
                        "rampdown" * amp(amplitude_qua[element_index]),
                        "col_AOD_{}".format(element_index + 1),
                    )
                # Measure raw adc trace for spectrograms
                if raw_adc_acquisition and not simulation:
                    measure("readout", "detector", adc_stream=raw_adc)
            # Exit the infinite loop in case just a single sorting sequence is needed
            if single_run:
                assign(infinite_run, False)

    with stream_processing():
        data_stream.save_all("data")
        count_stream.save_all("count")
        if raw_adc_acquisition and not simulation:
            if LF_RO[1] == 1:
                raw_adc.input1().save_all("raw_data")
            else:
                raw_adc.input2().save_all("raw_data")

#####################################
#  Open Communication with the QOP  #
#####################################

# qmm = QuantumMachinesManager(host=qop_ip, port=qop_port, cluster_name=cluster_name)
qmm = QuantumMachinesManager(host=qop_ip, cluster_name=cluster_name, timeout=1000)


if not simulation:
    # Open a quantum machine
    qm = qmm.open_qm(config, close_other_machines=True)

    job = qm.execute(atom_sorting)  # order_atoms
    # job = qm.execute(freq_calibration)
    res = job.result_handles
    res.wait_for_all_values()

    # Print atom displacement fo reach row: each pair is current --> target
    row_count = 0
    assign = res.get("data").fetch_all()["value"]

    # Get the final atom locations
    tot_atom_moved_list = res.get("count").fetch_all()["value"]
    atom_final_list = []
    cur_idx = 0
    for i, tot_atom_moved in enumerate(tot_atom_moved_list):
        # Check row by row and obtain the moved and unmoved atom locations
        print(f"Row {i + 1} - Total atoms moved: {tot_atom_moved}")
        row_assign = assign[cur_idx : cur_idx + tot_atom_moved * 2]
        initial_atom_row_idx = row_assign.reshape((-1, 2))[:, 0]
        final_atom_row_idx = row_assign.reshape((-1, 2))[:, 1]
        unmoved_atom_row_idx = set(np.nonzero(atom_location_list[i])[0]) - set(initial_atom_row_idx)
        final_atom_row_idx = np.concatenate([final_atom_row_idx, list(unmoved_atom_row_idx)], None).astype(int)
        final_atom_row_loc = np.zeros(nb_of_cols, dtype=int)
        final_atom_row_loc[final_atom_row_idx] = 1
        atom_final_list.append(list(final_atom_row_loc))
        cur_idx += tot_atom_moved * 2

    if raw_adc_acquisition:
        raw = res.get("raw_data").fetch_all()["value"]
        save_atom_sorting_plot_data(
            ATOM_SORTING_PLOT_DATA_FILE,
            raw,
            column_IFs=_column_IFs,
            column_spacing=column_spacing,
            atom_location_list=atom_location_list,
            atom_target_list=atom_target_list,
            atom_final_list=atom_final_list,
            target_frequencies_full_python=target_frequencies_full_python,
            nb_of_rows=nb_of_rows,
            nb_of_cols=nb_of_cols,
        )
        plot_atom_sorting_results(
            raw,
            column_IFs=_column_IFs,
            column_spacing=column_spacing,
            atom_location_list=atom_location_list,
            atom_target_list=atom_target_list,
            atom_final_list=atom_final_list,
            target_frequencies_full_python=target_frequencies_full_python,
            nb_of_rows=nb_of_rows,
            nb_of_cols=nb_of_cols,
        )

else:
    simulation_duration = min((nb_of_rows + 1) * readout_len + 500 * u.us, 10 * u.ms) // 4  # in clock cycles = 4ns
    # simulation_duration = 0.3 * u.ms // 4  # in clock cycles = 4ns
    job = qmm.simulate(
        config,
        atom_sorting,
        SimulationConfig(
            simulation_duration,
            # simulation_interface=LoopbackInterface([("con1", 1, "con1", 1), ("con1", 2, "con1", 2)]),
        ),
    )
    samples = job.get_simulated_samples()
    waveform_report = job.get_simulated_waveform_report()
    waveform_report.create_plot(samples, plot=True)
print("Done")

# %%
# Replot without QOP (after a run has saved atom_sorting_plot_data.npz):
# plot_kwargs = load_atom_sorting_plot_data(ATOM_SORTING_PLOT_DATA_FILE)
# raw = plot_kwargs.pop("raw")
# fig, spec_axes, occ_axes = create_atom_sorting_axes(plot_kwargs["nb_of_rows"], plot_kwargs["subplot_ncols"])
# occ = {k: plot_kwargs[k] for k in ("atom_location_list", "atom_target_list", "atom_final_list")}
# for ax, kind in zip(occ_axes, OCCUPATION_MATRIX_KINDS):
#     plot_atom_occupation_matrix(atom_occupation_matrix(kind, **occ), ax=ax, title=kind.capitalize())
# plot_atom_sorting_spectrograms(raw, axes=spec_axes, fig=fig, **plot_kwargs)
# fig.tight_layout()
# plot_atom_sorting_results(raw, **plot_kwargs)
# plt.show()

# %%
