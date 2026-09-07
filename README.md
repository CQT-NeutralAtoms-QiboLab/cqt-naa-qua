# cqt-naa-qua

QUA (Quantum Machines' pulse-level language) code for the CQT neutral-atom project.

This repo is the home for **all QUA-language pieces of machinery** — the low-level
Quantum Machines control programs (tweezer chirps, ramps, triggers, readout, …) that
drive the neutral-atom setup. It is intentionally separate from the Qibo platform
(`qibolab_platforms_naa`); this is the raw QUA layer those pieces are translated from.

Current contents:

```
cqt-naa-qua/
├── chirp/
│   ├── configuration_opx1000_mwfem_lffem.py   # QM hardware config
│   └── run_saas_04_play_chirp.py              # QUA chirp program
├── sorting/
│   ├── array_sorting.py                       # QUA collision-free atom sorting program
│   └── 07_atom_row_by_row_sorting.py          # row-by-row sorting variant, piecewise chirps
├── .env_sample                                # template for credentials
└── requirements.txt                           # pinned dependencies
```

The chirp runs against the **Quantum Machines cloud simulator** (`qm-saas`) — no local
hardware required, just QM cloud credentials.

The sorting scripts drive **atom realignment**: rearranging an initial occupation
matrix of trapped atoms into a target pattern using chirped tweezer moves. They
depend on companion `configuration.py` / `config_array_sorting.py` and
`array_sorting_macros.py` modules not yet checked into this repo.

## Setup

Create and activate a virtual environment, then install the pinned dependencies
(the venv name `.venv` is already gitignored):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Credentials

The code loads QM cloud credentials from a `.env` file at the repo root (gitignored —
never commit it). Copy the sample and fill in your own:

```bash
cp .env_sample .env
# then edit .env:
#   QM_SAAS_EMAIL=your.email@example.com
#   QM_SAAS_PWD=your-password
```

If either is missing, the code raises a clear error naming the `.env` path.
(The non-secret `HOST` and `QOP_VER` are set in the config file, not the `.env`.)

## Run

```bash
cd chirp
python run_saas_04_play_chirp.py
```

This connects to the QM cloud simulator, plays the frequency chirp on the AOD column/row
tones, and opens a **spectrogram** (frequency vs. time) so you can see the chirp sweep.
