# %%

import math
from pathlib import Path
from pprint import pprint

import numpy as np
import plotly.io as pio
from qm import *
from qm.qua import *
from qualang_tools.config.waveform_tools import drag_gaussian_pulse_waveforms
from qualang_tools.units import unit

pio.renderers.default = "browser"
u = unit(coerce_to_integer=True)


# Slot numbers for FEMs
LF_ROW = {4: 1}
LF_COL = {4: 2}
LF_TRIG = {4: [1, 8]}

# Readout
LF_RO = (4, 2)
# Loopback connections for testing. Format: [(output_slot, output_port), (input_slot, input_port)]
LF_LOOPBACK = [(4, 2), LF_RO]

# # Slot numbers for FEMs
# LF_ROW = {1: 8}
# LF_COL = {1: 4}
# LF_TRIG = {1: [1, 8]}

# # Readout
# LF_RO = (1, 1)
# # Loopback connections for testing. Format: [(output_slot, output_port), (input_slot, input_port)]
# _LF_COL = list(LF_COL.items())[0]
# LF_LOOPBACK = [(_LF_COL[0], _LF_COL[1]), LF_RO]
# # LF_LOOPBACK = [(1, 1), LF_RO]


nb_of_rows = 20
nb_of_cols = nb_of_rows
total_sites = nb_of_rows * nb_of_cols
nb_of_segments = 1000

######################
# Network parameters #
######################

# qop_ip = "192.168.88.254"
# cluster_name = "YaqumoQM"
qop_ip = "172.16.33.114"
cluster_name = "CS_4"
# qop_ip = "192.168.88.252"
# cluster_name = "Cluster_1"
qop_port = None

# Path to save data
save_dir = Path(__file__).parent.resolve() / "Data"
save_dir.mkdir(exist_ok=True)

default_additional_files = {
    "configuration.py": "configuration.py",
}

##########################
# Dynamic array geometry #
##########################
assert len(LF_ROW) == 1
# Maximum set by number of cores in each LF-FEM (16), 14 if using readout.
MAX_NUMBER_OF_TWEEZERS_PER_FEM = int(min(nb_of_cols / len(LF_COL), 14))
total_number_of_tweezers = MAX_NUMBER_OF_TWEEZERS_PER_FEM * len(LF_COL)
combiner_rescaling = np.ones(len(LF_COL))
# Row/Column frequencies and frequency spacing for setting tweezer separation.
# column_min_IF = 100e6
# column_spacing = 10e6
# column_IFs = np.array([column_min_IF + i * column_spacing for i in range(total_number_of_tweezers)])
column_IFs = np.linspace(60e6, 130e6, nb_of_cols)
column_spacing = column_IFs[1] - column_IFs[0]
row_min_IF = 100e6
row_spacing = 20e6
row_IFs = np.array([row_min_IF + i * row_spacing for i in range(nb_of_rows)])
phases_list = np.random.uniform(0, 1, size=total_number_of_tweezers)
# phases_list = np.zeros(total_number_of_tweezers)
# duration of AOD constant pulse
aod_const_len = 1 * u.ms
rampup_length = 1 * u.us
rampdown_length = rampup_length
ramp_sampling_rate = 1e9

assert aod_const_len % nb_of_segments == 0, f"{aod_const_len=} must be divisible by {nb_of_segments=}"
overhead_per_segment = 60 * u.ns
segment_length = aod_const_len // nb_of_segments  # segment length in ns
compensated_segment_length = segment_length - overhead_per_segment
assert compensated_segment_length > 0, f"{compensated_segment_length=} must be positive"

# WARNING: total output cannot exceed 0.5V,
# must be < 0.49/MAX_NUMBER_OF_TWEEZERS_PER_FEM
chirp_pulse_amplitude = 0.4
# chirp_pulse_amplitude = 0.5 / MAX_NUMBER_OF_TWEEZERS_PER_FEM

# readout length
readout_len = aod_const_len + (rampup_length + rampdown_length) * 2

# trigger length
trig_length = 1 * u.us

threshold = -0.2290

shareable = False

#################
# Configuration #
#################

############################
# Define controller config
_port_cfg = {
    "offset": 0.0,
    "sampling_rate": 1e9,
    "output_mode": "direct",
    "upsampling_mode": "mw",
    "shareable": shareable,
}


def _lf_analog_outputs(ports):
    if isinstance(ports, int):
        ports = [ports]
    return {p: dict(_port_cfg) for p in ports}


row_fem = {
    slot: {
        "type": "LF",
        "analog_outputs": _lf_analog_outputs(ports),
    }
    for slot, ports in LF_ROW.items()
}
col_fem = {
    slot: {
        "type": "LF",
        "analog_outputs": _lf_analog_outputs(ports),
    }
    for slot, ports in LF_COL.items()
}
trig_fem = {
    slot: {
        "type": "LF",
        "digital_outputs": {
            p: {
                "shareable": shareable,
            }
            for p in ports
        },
    }
    for slot, ports in LF_TRIG.items()
}
readout_fem = {}
for slot, port in [LF_RO, LF_LOOPBACK[1]]:
    readout_fem.setdefault(slot, {"type": "LF", "analog_inputs": {}})
    readout_fem[slot]["analog_inputs"][port] = {
        "offset": 0.0,
        "sampling_rate": 1e9,
        "shareable": shareable,
    }
slot, port = LF_LOOPBACK[0]
readout_fem[slot]["analog_outputs"] = _lf_analog_outputs(port)


def _merge_fem(*args):
    merged = {"type": "LF", "analog_outputs": {}, "analog_inputs": {}, "digital_outputs": {}}
    for fem in args:
        if not fem:
            continue
        for key in ["analog_outputs", "analog_inputs", "digital_outputs"]:
            merged[key] |= fem.get(key, {})
    return merged


fems = {}
for slot in row_fem.keys() | col_fem.keys() | trig_fem.keys() | readout_fem.keys():
    r, c, t, ro = row_fem.get(slot), col_fem.get(slot), trig_fem.get(slot), readout_fem.get(slot)
    fems[slot] = _merge_fem(r, c, t, ro)

controller_config = {
    "controllers": {
        "con1": {
            "type": "opx1000",
            "fems": fems,
        }
    }
}


def _aod_elem(slot, port, ops, IF=200.0 * u.MHz, core=None):
    elem = {
        "singleInput": {"port": ("con1", slot, port)},
        "intermediate_frequency": IF,
        "operations": ops | {"const": "const_pulse"},
    }
    if core:
        elem["core"] = core
    return elem


#########################
# Define logical config

# Elements
row_elems = {}
for slot, port in LF_ROW.items():
    core = "row_AOD"
    row_elems |= {
        core: _aod_elem(
            slot,
            port,
            {
                "rampup": "row_rampup_pulse",
                "rampdown": "row_rampdown_pulse",
            },
            core=core,
        ),
    }

col_elems = {}
for i, (slot, port) in enumerate(LF_COL.items()):
    for j in range(MAX_NUMBER_OF_TWEEZERS_PER_FEM):
        tweezer_index = i * MAX_NUMBER_OF_TWEEZERS_PER_FEM + j
        IF = column_IFs[tweezer_index]
        core = f"col_AOD_{tweezer_index + 1}"
        col_elems |= {
            core: _aod_elem(
                slot,
                port,
                {
                    "rampup": "col_rampup_pulse",
                    "rampdown": "col_rampdown_pulse",
                },
                IF=IF,
                core=core,
            ),
        }

trig_elems = {}
for slot, ports in LF_TRIG.items():
    for i, port in enumerate(ports):
        trig_elems |= {
            f"trig_{i + 1}": {
                "digitalInputs": {
                    "trig": {
                        "port": ("con1", slot, port),
                        "delay": 0,
                        "buffer": 0,
                    }
                },
                "operations": {"trig": "trig_pulse"},
            }
        }

readout_elem = {
    "detector": {
        "singleInput": {"port": ("con1", LF_LOOPBACK[0][0], LF_LOOPBACK[0][1])},
        # same as loopback since it is already defined as an analog input in the readout_fem
        "intermediate_frequency": 0,
        "operations": {"readout": "readout_zero_pulse"},
        "outputs": {"out1": ("con1", LF_RO[0], LF_RO[1])},
        "time_of_flight": 28,
        "smearing": 0,
    },
    "loopback": {
        "singleInput": {"port": ("con1", LF_LOOPBACK[0][0], LF_LOOPBACK[0][1])},
        "intermediate_frequency": 0,
        "operations": {"readout": "readout_zero_pulse"},
        "outputs": {"out1": ("con1", LF_LOOPBACK[1][0], LF_LOOPBACK[1][1])},
        "time_of_flight": 28,
        "smearing": 0,
    },
}


# Waveforms
def blackman(t, v_start, v_end):
    """
    Amplitude waveform that minimizes the amount of side lobes in the Fourier domain.
    :param t: pulse duration [ns] (int)
    :param v_start: start amplitude [V] (float)
    :param v_end: end amplitude [V] (float)
    :return:
    """
    time_vector = np.asarray([x * 1.0 for x in range(int(t))])
    black = v_start + (
        time_vector / t
        - (25 / (42 * np.pi)) * np.sin(2 * np.pi * time_vector / t)
        + (1 / (21 * np.pi)) * np.sin(4 * np.pi * time_vector / t)
    ) * (v_end - v_start)
    return black


logical_config = {
    "elements": {
        **row_elems,
        **col_elems,
        **trig_elems,
        **readout_elem,
    },
    "pulses": {
        "const_pulse": {"operation": "control", "length": aod_const_len, "waveforms": {"single": "const_wf"}},
        # large-scale atom transport arbitrary waveform pulses
        "row_rampup_pulse": {
            "operation": "control",
            "length": rampup_length,
            "waveforms": {"single": "row_rampup_wf"},
        },
        "row_rampdown_pulse": {
            "operation": "control",
            "length": rampdown_length,
            "waveforms": {"single": "row_rampdown_wf"},
        },
        "col_rampup_pulse": {
            "operation": "control",
            "length": rampup_length,
            "waveforms": {"single": "col_rampup_wf"},
        },
        "col_rampdown_pulse": {
            "operation": "control",
            "length": rampdown_length,
            "waveforms": {"single": "col_rampdown_wf"},
        },
        "trig_pulse": {"operation": "control", "length": trig_length, "digital_marker": "ON"},
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
        "zero_wf": {"type": "constant", "sample": 0.0},
        "const_wf": {"type": "constant", "sample": chirp_pulse_amplitude},
        "row_rampup_wf": {
            "type": "arbitrary",
            "samples": blackman(
                rampup_length / (1e9 / ramp_sampling_rate),
                0.0,
                chirp_pulse_amplitude,
            ),
            "sampling_rate": ramp_sampling_rate,
        },
        "row_rampdown_wf": {
            "type": "arbitrary",
            "samples": blackman(
                rampdown_length / (1e9 / ramp_sampling_rate),
                chirp_pulse_amplitude,
                0.0,
            ),
            "sampling_rate": ramp_sampling_rate,
        },
        "col_rampup_wf": {
            "type": "arbitrary",
            "samples": blackman(
                rampup_length / (1e9 / ramp_sampling_rate),
                0.0,
                chirp_pulse_amplitude,
            ),
            "sampling_rate": ramp_sampling_rate,
        },
        "col_rampdown_wf": {
            "type": "arbitrary",
            "samples": blackman(
                rampdown_length / (1e9 / ramp_sampling_rate),
                chirp_pulse_amplitude,
                0.0,
            ),
            "sampling_rate": ramp_sampling_rate,
        },
    },
    "digital_waveforms": {"ON": {"samples": [(1, 0)]}},
    "integration_weights": {
        "constant_weights": {
            "cosine": [(1.0, readout_len)],
            "sine": [(0.0, readout_len)],
        },
        "cosine_weights": {
            "cosine": [(1.0, readout_len)],
            "sine": [(0.0, readout_len)],
        },
        "sine_weights": {
            "cosine": [(0.0, readout_len)],
            "sine": [(1.0, readout_len)],
        },
    },
}

config = controller_config | logical_config

# %%
