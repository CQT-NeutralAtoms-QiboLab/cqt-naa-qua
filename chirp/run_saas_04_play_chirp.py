# %%
from qm import QuantumMachinesManager
from qm.qua import *
from qm import SimulationConfig
import matplotlib.pyplot as plt
from qm_saas import QOPVersion, QmSaas

from configuration_opx1000_mwfem_lffem import *

# Chirp design
# T        = 8000      # ns — pulse duration, must be multiple of 4
f_start  = 20e6    # Hz
f_end    = 280e6    # Hz
rate     = int((f_end - f_start) / const_pulse_len)  # Hz/ns = 250_000 Hz/ns

# Patch config pulse length to match T
# config["pulses"]["const_pulse"]["length"] = T
# config["elements"]["col_selector_01"]["intermediate_frequency"] = int(100e6)
# Initialize QOP simulator client
client = QmSaas(email=EMAIL, password=PWD, host=HOST)

with program() as PROG:
    # Use just a few selectors so we stay within resource limits
    active_cols = [1, 2, 3]
    active_rows = [1, 2, 3]

    # # ===== STEP 1: Turn ON =====
    # for i in active_cols:
    #     play("blackman_up", f"col_selector_{i:02d}")
    # for i in active_rows:
    #     play("blackman_up", f"row_selector_{i:02d}")

    # align()

    # ===== STEP 2: Chirp =====
    for i in active_cols:
        play("const", f"col_selector_{i:02d}", chirp=(rate, "Hz/nsec"))
    for i in active_rows:
        play("const", f"row_selector_{i:02d}", chirp=(rate, "Hz/nsec"))

    align()

    # # ===== STEP 3: Turn OFF =====
    # for i in active_cols:
    #     play("blackman_down", f"col_selector_{i:02d}")
    # for i in active_rows:
    #     play("blackman_down", f"row_selector_{i:02d}")

    # align()
    # play("multitone", "col_multitone")

    # align()

    # ===== STEP 4: Trigger =====
    # play("on", "trigger_artiq")
    # play("cw", "qubit")

sim_duration = int(const_pulse_len) // 4 + 400 # ns, slightly longer than the pulse to capture the full

with client.simulator(QOPVersion(QOP_VER), make_cluster_config()) as instance:
    qmm = QuantumMachinesManager(
        host=instance.host,
        port=instance.port,
        connection_headers=instance.default_connection_headers,
    )
    job = qmm.simulate(config, PROG, SimulationConfig(sim_duration))
    samples = job.get_simulated_samples()
    print("Available keys:", list(samples.con1.analog.keys()))   # confirm format

# samples.con1.plot()
# plt.title("Full tweezer sequence: on → chirp → off → trigger")
# plt.show()

# ─────────────────────────────────────────────────────────────
# OPTIONAL: spectrogram of the col AOD (port 1) to SEE the chirp
# ─────────────────────────────────────────────────────────────

x = samples.con1.analog["1"]

Fs = 1e9
Pxx, freqs, bins, im = plt.specgram(
    x, NFFT=2**10, Fs=Fs, noverlap=1000, cmap=plt.cm.gist_heat
)
plt.close()
plt.figure(figsize=(12, 5))
ax = plt.subplot(111)
# Convert axes to ns and MHz at the DATA level
t_ns   = bins * 1e9        # seconds → ns
f_MHz  = freqs / 1e6       # Hz → MHz

mesh = ax.pcolormesh(
    t_ns, f_MHz, 10 * np.log10(Pxx),   # dB scale
    cmap=plt.cm.gist_heat, shading="auto"
)
plt.xlabel("t [ns]")
plt.ylabel("f [MHz]")
plt.title("Col AOD (port 1) — frequency vs time")
plt.colorbar(im, label="Power")
plt.tight_layout()
plt.show()
# %%