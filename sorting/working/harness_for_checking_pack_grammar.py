"""Throwaway, single-use grammar check for the sorting/working pack.

Builds the QUA program from working/07 (executing only the program-building
portion, i.e. everything BEFORE the 'Open Communication with the QOP' section so
no cluster connection is attempted) and serializes it with generate_qua_script.
Success => the pack assembles cleanly (grammar OK). Not a correctness/waveform check.
"""
import sys
import matplotlib

matplotlib.use("Agg")  # no GUI

WORKING = "/Users/elis/Desktop/Work/PROJECTS/Quantum/forks-CQT-NAA/cqt-naa-qua/sorting/working"
sys.path.insert(0, WORKING)

src_path = f"{WORKING}/07_atom_row_by_row_sorting.py"
with open(src_path) as f:
    src = f.read()

# Cut off before the cluster-connection section so nothing tries to connect.
marker = "#  Open Communication with the QOP  #"
assert marker in src, "cut marker not found; check 07 layout"
src_build_only = src.split(marker)[0]

ns = {"__name__": "__grammar_check__", "__file__": src_path}
try:
    exec(compile(src_build_only, src_path, "exec"), ns)
except Exception as e:
    print(f"BUILD FAILED: {type(e).__name__}: {e}")
    raise

from qm import generate_qua_script  # noqa: E402

config = ns["config"]
for prog_name in ("atom_sorting", "freq_calibration"):
    prog = ns[prog_name]
    text = generate_qua_script(prog, config)
    print(f"  {prog_name}: generate_qua_script OK ({len(text.splitlines())} lines)")

print("GRAMMAR OK: pack assembles cleanly.")
