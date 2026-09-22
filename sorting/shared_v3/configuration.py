import itertools
from pathlib import Path

import numpy as np
from qualang_tools.units import unit

u = unit(coerce_to_integer=True)
rng = np.random.default_rng()

######################
# Network parameters #
######################

# qop_ip = "172.16.33.114"
# cluster_name = "CS_4"
# qop_port = None

# ######################
# # Detector / readout #
# ######################

# # Photodetector (or loopback) ADC.
# DETECTOR_CON = "con1"
# DETECTOR_FEM_SLOT = 4
# DETECTOR_INPUT_PORT = 1
# # Dummy analog output used only to attach the measurement pulse (zero waveform).
# DETECTOR_OUTPUT_PORT = 1
# DETECTOR_FEMS = [(DETECTOR_CON, DETECTOR_FEM_SLOT, DETECTOR_OUTPUT_PORT)]
# DETECTOR_ANALOG_INPUTS = [(DETECTOR_CON, DETECTOR_FEM_SLOT, DETECTOR_INPUT_PORT)]

# ######################
# # Tweezer geometry   #
# ######################

# MAX_TWEEZERS_PER_FEM = 16

# # Total tweezers per axis -- distributed across the available FEMs/ports.
# X_TWEEZERS = MAX_TWEEZERS_PER_FEM // 2
# Y_TWEEZERS = 1

# # X-axis (column) AOD -- list of (controller, fem_slot, output_port)
# X_FEMS = [("con1", 4, 4)]
# # Y-axis (row) AOD -- list of (controller, fem_slot, output_port)
# Y_FEMS = [("con1", 4, 2)]

qop_ip = "172.16.33.115"
cluster_name = "CS_3"
qop_port = None

######################
# Detector / readout #
######################

# Photodetector (or loopback) ADC.
DETECTOR_CON = "con1"
DETECTOR_FEM_SLOT = 5
DETECTOR_INPUT_PORT = 1
# Dummy analog output used only to attach the measurement pulse (zero waveform).
DETECTOR_OUTPUT_PORT = 7
DETECTOR_FEMS = [(DETECTOR_CON, DETECTOR_FEM_SLOT, DETECTOR_OUTPUT_PORT)]
DETECTOR_ANALOG_INPUTS = [(DETECTOR_CON, DETECTOR_FEM_SLOT, DETECTOR_INPUT_PORT)]

######################
# Tweezer geometry   #
######################

MAX_TWEEZERS_PER_FEM = 16

# Total tweezers per axis -- distributed across the available FEMs/ports.
X_TWEEZERS = MAX_TWEEZERS_PER_FEM - 2
Y_TWEEZERS = 1

# X-axis (column) AOD -- list of (controller, fem_slot, output_port)
X_FEMS = [("con1", 5, 7)]
# Y-axis (row) AOD -- list of (controller, fem_slot, output_port)
Y_FEMS = [("con1", 5, 8)]

# Normalized list for readout / simulation: one entry per AOD output channel.
AOD_ANALOG_OUTPUTS = X_FEMS + Y_FEMS


def _distribute_tweezers(count, num_fems, max_per_fem):
    """Spread *count* tweezers evenly across *num_fems* FEMs."""
    assert count <= num_fems * max_per_fem, (
        f"Cannot fit {count} tweezers across {num_fems} FEMs "
        f"(max {max_per_fem} each, capacity {num_fems * max_per_fem})"
    )
    base, remainder = divmod(count, num_fems)
    return [base + (1 if i < remainder else 0) for i in range(num_fems)]


x_per_fem = _distribute_tweezers(X_TWEEZERS, len(X_FEMS), MAX_TWEEZERS_PER_FEM)
y_per_fem = _distribute_tweezers(Y_TWEEZERS, len(Y_FEMS), MAX_TWEEZERS_PER_FEM)

NUM_X_TWEEZERS = X_TWEEZERS
NUM_Y_TWEEZERS = Y_TWEEZERS
newman_phases_2pi = lambda N: np.arange(N) ** 2 / (2 * N)
x_tweezer_elements = [f"x_tweezer_{i + 1}" for i in range(NUM_X_TWEEZERS)]
# sample IFs (experiment should have more sites than the tweezers)
x_tweezer_IFs = np.linspace(80e6, 120e6, NUM_X_TWEEZERS).astype(int)
x_tweezer_phases = newman_phases_2pi(NUM_X_TWEEZERS)
y_tweezer_elements = [f"y_tweezer_{i + 1}" for i in range(NUM_Y_TWEEZERS)]
# sample IFs (experiment should have more sites than the tweezers)
y_tweezer_IFs = np.linspace(80e6, 120e6, NUM_Y_TWEEZERS).astype(int)
y_tweezer_phases = newman_phases_2pi(NUM_Y_TWEEZERS)

all_tweezer_elements = x_tweezer_elements + y_tweezer_elements

# Amplitude budget: all tweezers on one port sum together, so the per-tweezer
# amplitude is limited by the most-loaded FEM on each axis.
x_tweezer_max_amplitude = np.floor(0.4 / max(x_per_fem) * 2**15) / 2**15
y_tweezer_max_amplitude = np.floor(0.4 / max(y_per_fem) * 2**15) / 2**15
x_tweezer_amplitudes = [x_tweezer_max_amplitude] * NUM_X_TWEEZERS
y_tweezer_amplitudes = [y_tweezer_max_amplitude] * NUM_Y_TWEEZERS

# Referenced by the compensated chirp path in minimal_jerk.py.
chirp_pulse_amplitude = x_tweezer_max_amplitude

##################
# Pulse settings #
##################

chirp_pulse_length = 1 * u.ms
segment_length = 1 * u.us
rampup_length = 1 * u.us
rampdown_length = rampup_length
ramp_sampling_rate = 1e9

assert chirp_pulse_length % segment_length == 0
number_of_segments = chirp_pulse_length // segment_length

# ADC spectrogram window: covers ramp-up + chirp + ramp-down on the detector timeline.
readout_len = chirp_pulse_length + (rampup_length + rampdown_length) * 2

save_dir = Path().resolve() / "Data"
save_dir.mkdir(exist_ok=True)


def make_run_dir(label: str) -> Path:
    """Create ``Data/<YYYYMMDD_HHMMSS>_<label>/`` for one collection."""
    from datetime import datetime

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = save_dir / f"{stamp}_{label}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving collection to {run_dir}")
    return run_dir


#################
# Configuration #
#################

_port_cfg = {
    "offset": 0.0,
    "sampling_rate": 1e9,
    "output_mode": "direct",
    "upsampling_mode": "mw",
}


def _build_controller_config(fem_lists, analog_inputs=None):
    """Build controller config by aggregating all (controller, slot, port) entries."""
    controllers = {}
    for con, slot, port in itertools.chain(*fem_lists):
        controllers.setdefault(con, {})
        controllers[con].setdefault(slot, {"type": "LF", "analog_outputs": {}})
        controllers[con][slot]["analog_outputs"][port] = dict(_port_cfg)
    for con, slot, port in analog_inputs or []:
        controllers.setdefault(con, {})
        controllers[con].setdefault(slot, {"type": "LF", "analog_outputs": {}})
        controllers[con][slot].setdefault("analog_inputs", {})
        controllers[con][slot]["analog_inputs"][port] = {
            "offset": 0.0,
            "sampling_rate": 1e9,
        }
    return {"controllers": {con: {"type": "opx1000", "fems": fems} for con, fems in controllers.items()}}


controller_config = _build_controller_config(
    [X_FEMS, Y_FEMS, DETECTOR_FEMS],
    analog_inputs=DETECTOR_ANALOG_INPUTS,
)


def _blackman(duration, v_start, v_end):
    time_vector = np.arange(int(duration), dtype=float)
    return v_start + (
        time_vector / duration
        - (25 / (42 * np.pi)) * np.sin(2 * np.pi * time_vector / duration)
        + (1 / (21 * np.pi)) * np.sin(4 * np.pi * time_vector / duration)
    ) * (v_end - v_start)


def _fem_for_index(index, per_fem):
    """Return the FEM list index for a given tweezer index."""
    cumulative = 0
    for fem_idx, count in enumerate(per_fem):
        if index < cumulative + count:
            return fem_idx
        cumulative += count
    return len(per_fem) - 1


def _tweezer_element(index, fem_list, per_fem, if_array, core):
    con, slot, port = fem_list[_fem_for_index(index, per_fem)]
    return {
        "singleInput": {"port": (con, slot, port)},
        "intermediate_frequency": int(if_array[index]),
        "operations": {
            "hold": "zero_pulse",
            "const": "const_pulse",
            "rampup": "rampup_pulse",
            "rampdown": "rampdown_pulse",
        },
        "sticky": {"analog": True},
        "core": core,
    }


_elements = {}
for _i, _name in enumerate(x_tweezer_elements):
    _elements[_name] = _tweezer_element(_i, X_FEMS, x_per_fem, x_tweezer_IFs, _name)
for _i, _name in enumerate(y_tweezer_elements):
    _elements[_name] = _tweezer_element(_i, Y_FEMS, y_per_fem, y_tweezer_IFs, y_tweezer_elements[0])
_elements["detector"] = {
    "singleInput": {"port": (DETECTOR_CON, DETECTOR_FEM_SLOT, DETECTOR_OUTPUT_PORT)},
    "intermediate_frequency": 0,
    "operations": {"readout": "readout_zero_pulse"},
    "outputs": {"out1": (DETECTOR_CON, DETECTOR_FEM_SLOT, DETECTOR_INPUT_PORT)},
    "time_of_flight": 28,
    "smearing": 0,
}

# Use the larger of the two min amplitudes for waveform definitions.
_wf_amplitude = min(x_tweezer_max_amplitude, y_tweezer_max_amplitude)

config = controller_config | {
    "elements": _elements,
    "pulses": {
        "const_pulse": {
            "operation": "control",
            "length": segment_length,
            "waveforms": {"single": "const_wf"},
        },
        "rampup_pulse": {
            "operation": "control",
            "length": rampup_length,
            "waveforms": {"single": "rampup_wf"},
        },
        "rampdown_pulse": {
            "operation": "control",
            "length": rampdown_length,
            "waveforms": {"single": "rampdown_wf"},
        },
        "zero_pulse": {
            "operation": "control",
            "length": segment_length,
            "waveforms": {"single": "zero_wf"},
        },
        "readout_zero_pulse": {
            "operation": "measurement",
            "length": readout_len,
            "waveforms": {"single": "zero_wf"},
            "digital_marker": "ON",
            "integration_weights": {
                "constant": "constant_weights",
            },
        },
    },
    "waveforms": {
        "const_wf": {"type": "constant", "sample": _wf_amplitude},
        "zero_wf": {"type": "constant", "sample": 0.0},
        "rampup_wf": {
            "type": "arbitrary",
            "samples": _blackman(rampup_length, 0.0, _wf_amplitude),
            "sampling_rate": ramp_sampling_rate,
        },
        "rampdown_wf": {
            "type": "arbitrary",
            "samples": _blackman(rampdown_length, 0.0, -_wf_amplitude),
            "sampling_rate": ramp_sampling_rate,
        },
    },
    "digital_waveforms": {"ON": {"samples": [(1, 0)]}},
    "integration_weights": {
        "constant_weights": {
            "cosine": [(1.0, readout_len)],
            "sine": [(0.0, readout_len)],
        },
    },
}
