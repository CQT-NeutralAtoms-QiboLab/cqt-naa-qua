# Grammar-check output — sorting/working pack

Produced by `harness_for_checking_pack_grammar.py` (run in the `cqt-naa-qua/.venv`),
after applying fixes #1 (align filter naming), #2 (freq_calibration naming), and
#3 (macro out-of-bounds) to the pack.

**Verdict: `GRAMMAR OK: pack assembles cleanly.`** — both QUA programs build and
serialize via `generate_qua_script`. This is an *assembles-cleanly* (grammar) check
only, not a correctness/waveform check (a real compile still needs the QOP/cluster).

Note: the "Initial occupation matrix" printed at the top is randomized per run
(`np.random.choice` with no seed in `07`), so those 0/1 values differ each run; the
`GRAMMAR OK` verdict and the `generate_qua_script OK` lines are the meaningful result.
The `qm - WARNING - Could not generate a loaded config ...` lines are expected when
running offline (no QOP capabilities) and do not affect the grammar verdict.

## Captured run output

```
1	0	0	1	0	1	0	0	1	1	1	0	0	1	1	1	1	0	0	0	
1	0	1	1	1	0	0	0	1	0	0	0	1	0	1	1	0	0	1	0	
1	0	0	0	1	0	1	0	1	1	1	1	1	1	1	1	1	0	0	1	
1	1	1	1	0	1	1	0	1	1	1	0	0	1	1	1	1	1	1	1	
Atom target matrix:
0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	
0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	
0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	
0	0	0	1	1	1	1	1	1	1	1	1	1	1	1	1	1	0	0	0	
0	0	0	1	1	1	1	1	1	1	1	1	1	1	1	1	1	0	0	0	
0	0	0	1	1	1	1	1	1	1	1	1	1	1	1	1	1	0	0	0	
0	0	0	1	1	1	1	1	1	1	1	1	1	1	1	1	1	0	0	0	
0	0	0	1	1	1	1	1	1	1	1	1	1	1	1	1	1	0	0	0	
0	0	0	1	1	1	1	1	1	1	1	1	1	1	1	1	1	0	0	0	
0	0	0	1	1	1	1	1	1	1	1	1	1	1	1	1	1	0	0	0	
0	0	0	1	1	1	1	1	1	1	1	1	1	1	1	1	1	0	0	0	
0	0	0	1	1	1	1	1	1	1	1	1	1	1	1	1	1	0	0	0	
0	0	0	1	1	1	1	1	1	1	1	1	1	1	1	1	1	0	0	0	
0	0	0	1	1	1	1	1	1	1	1	1	1	1	1	1	1	0	0	0	
0	0	0	1	1	1	1	1	1	1	1	1	1	1	1	1	1	0	0	0	
0	0	0	1	1	1	1	1	1	1	1	1	1	1	1	1	1	0	0	0	
0	0	0	1	1	1	1	1	1	1	1	1	1	1	1	1	1	0	0	0	
0	0	0	1	1	1	1	1	1	1	1	1	1	1	1	1	1	0	0	0	
0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	
0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	
0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	
2026-09-15 16:02:16,937 - qm - WARNING  - Could not generate a loaded config. Capabilities are required but not initialized. Please use QuantumMachinesManager to connect to a QOP server or manually set the capabilities using the `QuantumMachinesManager.set_capabilities_offline()` function. Please see the function documentation on how to set the capabilities you need.
  atom_sorting: generate_qua_script OK (640 lines)
2026-09-15 16:02:17,243 - qm - WARNING  - Could not generate a loaded config. Capabilities are required but not initialized. Please use QuantumMachinesManager to connect to a QOP server or manually set the capabilities using the `QuantumMachinesManager.set_capabilities_offline()` function. Please see the function documentation on how to set the capabilities you need.
  freq_calibration: generate_qua_script OK (397 lines)
GRAMMAR OK: pack assembles cleanly.
```
