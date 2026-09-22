# %%
#! %load_ext autoreload
#! %autoreload 2
"""
Sweep a single tweezer across the AOD frequency range and record the
photodetector response on an OPX ADC input.

Set SWEEP_AXIS to "x" or "y" to select which AOD to characterize.
Run the script once per axis to collect both calibration datasets.

After collecting the data, the script automatically launches the
polynomial fitting from amplitude_calibration.py and prompts you to
select a polynomial order to save.

Usage:
    1. Connect the photodetector output to the ADC at
       (DETECTOR_CON, DETECTOR_INPUT_PORT).
    2. Set SWEEP_AXIS and simulate = False, then run.
    3. Choose a polynomial order when prompted.
    4. Repeat for the other axis.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from configuration import *
from qm import QuantumMachinesManager, SimulationConfig, generate_qua_script
from qm.qua import *
from qualang_tools.results import fetching_tool

# ---------------------------------------------------------------------------
# Sweep parameters
# ---------------------------------------------------------------------------

SWEEP_AXIS = "x"  # "x" or "y"

SWEEP_START_HZ = 50e6
SWEEP_STOP_HZ = 150e6
SWEEP_POINTS = 50
INTEGRATION_TIME = 10 * u.us

SIGNAL_IS_DC = False

simulate = False

# ---------------------------------------------------------------------------
# Axis-dependent setup
# ---------------------------------------------------------------------------

if SWEEP_AXIS == "x":
    SWEEP_ELEMENT = x_tweezer_elements[0]
    OUTPUT_CSV = "x_amplitude_data.csv"
else:
    SWEEP_ELEMENT = y_tweezer_elements[0]
    OUTPUT_CSV = "y_amplitude_data.csv"

# ---------------------------------------------------------------------------
# Add only the sweep-specific detector readout duration
# ---------------------------------------------------------------------------

sweep_config = dict(config)

sweep_config["elements"] = dict(sweep_config["elements"])
detector_element = dict(sweep_config["elements"]["detector"])
detector_element["operations"] = dict(detector_element["operations"])
detector_element["operations"]["readout"] = "amplitude_sweep_readout_pulse"
sweep_config["elements"]["detector"] = detector_element

sweep_config["pulses"] = dict(sweep_config["pulses"])
sweep_config["pulses"]["amplitude_sweep_readout_pulse"] = {
    "operation": "measurement",
    "length": INTEGRATION_TIME,
    "waveforms": {"single": "zero_wf"},
    "digital_marker": "ON",
    "integration_weights": {
        "constant": "amplitude_sweep_weights",
    },
}

sweep_config["integration_weights"] = dict(sweep_config["integration_weights"])
sweep_config["integration_weights"]["amplitude_sweep_weights"] = {
    "cosine": [(1.0, INTEGRATION_TIME)],
    "sine": [(0.0, INTEGRATION_TIME)],
}

# ---------------------------------------------------------------------------
# QUA program
# ---------------------------------------------------------------------------

sweep_frequencies = np.linspace(SWEEP_START_HZ, SWEEP_STOP_HZ, SWEEP_POINTS).astype(int)

with program() as measure_aod:
    freq = declare(int)
    I = declare(fixed)
    I_st = declare_stream()
    f_st = declare_stream()

    with for_each_(freq, sweep_frequencies.tolist()):
        update_frequency(SWEEP_ELEMENT, freq)
        wait(4)
        align(SWEEP_ELEMENT, "detector")
        play("const", SWEEP_ELEMENT, duration=INTEGRATION_TIME // 4)

        if SIGNAL_IS_DC:
            measure("readout", "detector", integration.full("constant", I))
        else:
            measure("readout", "detector", demod.full("constant", I))

        save(I, I_st)
        save(freq, f_st)

        align(SWEEP_ELEMENT, "detector")
        ramp_to_zero(SWEEP_ELEMENT)
        wait(1000 // 4, SWEEP_ELEMENT)

    with stream_processing():
        I_st.save_all("amplitudes")
        f_st.save_all("frequencies")

# ---------------------------------------------------------------------------
# Execute
# ---------------------------------------------------------------------------

qmm_kwargs = {"host": qop_ip, "cluster_name": cluster_name, "timeout": 300}
if qop_port is not None:
    qmm_kwargs["port"] = qop_port
qmm = QuantumMachinesManager(**qmm_kwargs)

run_dir = make_run_dir(f"01_measure_amplitude_response_{SWEEP_AXIS}")
debug_path = run_dir / "01_measure_amplitude_response_debug.py"
with open(debug_path, "w", encoding="utf-8") as source_file:
    print(generate_qua_script(measure_aod, sweep_config), file=source_file)
print(f"Saved QUA debug script to {debug_path}")

if simulate:
    MAX_SIM_TIME = 100 * u.s
    total_time = min(SWEEP_POINTS * (INTEGRATION_TIME * 2 + 2000), MAX_SIM_TIME)
    job = qmm.simulate(sweep_config, measure_aod, SimulationConfig(int(total_time // 4)))
    # job.get_simulated_samples().con1.plot()
    # plt.title(f"Simulated {SWEEP_AXIS.upper()} AOD sweep")
    # fig_path = run_dir / "01_simulated_sweep.png"
    # plt.savefig(fig_path, dpi=150)
    # print(f"Saved figure to {fig_path}")
    # plt.show()
    samples = job.get_simulated_samples()
    job.get_simulated_waveform_report().create_plot(samples, plot=True, save_path=str(run_dir.resolve()))
    print("Simulation complete. Run with simulate = False for real data.")
else:
    qm = qmm.open_qm(sweep_config, close_other_machines=True)
    job = qm.execute(measure_aod)

    results = fetching_tool(job, data_list=["amplitudes", "frequencies"], mode="wait_for_all")
    amplitudes, frequencies = results.fetch_all()
    amplitudes = amplitudes["value"]
    frequencies = frequencies["value"]

    magnitudes = np.abs(amplitudes.astype(float))

    header = "freq, magnitude"
    freq_mhz = frequencies / 1e6
    data = np.column_stack([freq_mhz, magnitudes])
    output_path = run_dir / OUTPUT_CSV
    np.savetxt(output_path, data, delimiter=", ", header=header, comments="", fmt="%.3f")
    # Keep a copy next to the script so amplitude_calibration.py can load it.
    local_csv = Path().resolve() / OUTPUT_CSV
    np.savetxt(local_csv, data, delimiter=", ", header=header, comments="", fmt="%.3f")
    print(f"Saved {len(frequencies)} points to {output_path}")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(freq_mhz, magnitudes, "o-")
    ax.set_xlabel("Frequency (MHz)")
    ax.set_ylabel("Measured amplitude (V)")
    ax.set_title(f"{SWEEP_AXIS.upper()} AOD frequency response")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig_path = run_dir / f"01_{SWEEP_AXIS}_amplitude_response.png"
    fig.savefig(fig_path, dpi=150)
    print(f"Saved figure to {fig_path}")
    plt.show()

    from amplitude_calibration import main as fit_calibration

    fit_calibration(SWEEP_AXIS)
