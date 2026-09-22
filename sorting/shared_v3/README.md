# Flex Sort

Minimal-jerk chirps for rearranging atoms in a 2-D tweezer array, with
optional real-time amplitude compensation for AOD frequency response.

Scripts `02`–`04` are written so they can be run as Jupyter cells. Use the
hardcoded `SCRIPT_NAME` in each file (not `__file__`) for run-directory
names and per-script computation-latency CSV rows.

## File overview

| File | Role |
|---|---|
| `configuration.py` | OPX / OPX+ hardware config, tweezer geometry, pulses, `make_run_dir` |
| `configuration_opx1k.py` | OPX1000 variant (`type: "opx1000"` + `fems`) |
| `minimal_jerk.py` | QUA macro: plays a single compensated or constant-amplitude chirp |
| `amplitude_calibration.py` | Load/fit AOD polynomials, per-script `computation_latency_cc`, latency check |
| `readout_analysis.py` | ADC collection, spectrograms, sorting plots, save helpers |
| `array_sorting_macros.py` | Row-compression assignment used by `04` |
| `xy_amplitude_validation.py` | Plot X/Y compensation curves and their 2-D product |
| `01_measure_amplitude_response.py` | Sweep one AOD, record photodetector response, then fit calibration |
| `02_arb_tone_MJ_manual.py` | Fixed-parameter chirps (quick tests and simulation) |
| `03_arb_tone_MJ_sort_manual_IS.py` | Interactive input-stream chirps |
| `04_arb_tone_MJ_row_sort.py` | Row-by-row left-compression sort |

Outputs land in `Data/<YYYYMMDD_HHMMSS>_<SCRIPT_NAME>/` (QUA debug dump,
spectrograms, NPZ). Amplitude CSVs stay in this folder.

## Tweezer geometry

`configuration.py` defines two independent axes of tweezers, one per AOD:

- **X tweezers** (column AOD): `x_tweezer_1`, `x_tweezer_2`, ...
- **Y tweezers** (row AOD): `y_tweezer_1`, `y_tweezer_2`, ...

You declare the **total number of tweezers** per axis and the analog output
ports available. The tweezers are then distributed evenly across those
ports. `configuration.py` is the OPX / OPX+ layout (`type: "opx1"`, ports
as `(controller, port)`, no `fems`). OPX1000 stays in
`configuration_opx1k.py`.

```python
MAX_TWEEZERS_PER_PORT = 18

X_TWEEZERS = 32
Y_TWEEZERS = 32

X_PORTS = [("con1", 10), ("con1", 8)]
Y_PORTS = [("con1", 9)]
```

`_distribute_tweezers()` spreads the requested count across the available
ports, filling them as evenly as possible (e.g., 32 tweezers across 2 ports
gives 16 each; 20 across 3 gives 7, 7, 6). It asserts if the total exceeds
`len(PORTS) * MAX_TWEEZERS_PER_PORT`.

The per-tweezer amplitude budget is set by the most-loaded port on each axis,
since all tweezers sharing a port sum together:

```python
x_tweezer_max_amplitude = floor(0.4 / max(x_per_port) * 2**15) / 2**15
```

All elements use `"sticky": {"analog": True}` so that the output level and
NCO phase persist between pulses. The controller config is built
automatically from the union of all port entries via `_build_controller_config`,
so adding a new output only requires appending a tuple to `X_PORTS`
or `Y_PORTS`.

## Minimal-jerk chirp (`minimal_jerk.py`)

`play_minimal_jerk_chirp()` drives a single element through a smooth
frequency chirp from `start_frequency` to `start_frequency + detuning`.
The chirp follows a minimal-jerk (5th-order polynomial) velocity profile:

```
freq(tau) = start + detuning * (10*tau^3 - 15*tau^4 + 6*tau^5)
```

where `tau` runs from 0 to 1 over the total chirp duration.

An `amp_calibration` (`Calibration` named tuple) is required. Both the
compensated and constant-amplitude paths use
`amp_calibration.computation_latency_cc` as the processor gap.

### Segmented execution and computation latency

The chirp is divided into `number_of_segments` segments, each nominally
`segment_length` ns long. Each segment recalculates the chirp rate from
the minimal-jerk formula and issues a `play()` with `chirp=(rate, "mHz/nsec")`.

Because the FPGA needs time to compute the next segment's parameters, the
`play()` duration is shorter than the full segment:

```python
duration = segment_length // 4 - computation_latency_cc   # clock cycles (4 ns each)
```

`computation_latency_cc` is **per script and per axis**, loaded from the
calibration CSVs (see below). During this gap:

- **Frequency continues chirping** -- sticky elements with `continue_chirp=True`
  keep the NCO sweeping at the last chirp rate, so there is no frequency
  discontinuity. The effective frequency segment is the full `segment_length` ns.
- **Amplitude ramp flattens** -- the `ramp()` waveform only plays during the
  `play()` duration, so the output amplitude holds flat at whatever level the
  ramp reached when the pulse ended. This is why `ramp_scaling` and the
  `prev_amp_mod` tracking use the actual play duration
  `segment_length - 4 * computation_latency_cc`, not the full `segment_length`.

### Constant vs. compensated paths

- **`_play_compensated`** -- used when the calibration has polynomial
  coefficients (`latency_only` is False and `coefs` is non-empty). Each
  segment:
  1. Computes `chirp_rate` from the minimal-jerk formula (same as constant path).
  2. Tracks `cur_frequency` by accumulating the frequency change per segment.
  3. Evaluates the compensation polynomial at `cur_frequency` to get `amp_mod`.
  4. Computes `ramp_rate = (amp_mod - prev_amp_mod) * ramp_scaling` to smoothly
     transition amplitude over the play duration.
  5. Updates `prev_amp_mod` to the amplitude the ramp actually reaches at the
     end of the play window:
     `amp_mod + ramp_rate * (segment_length - 4 * computation_latency_cc)`.
  6. Plays `ramp(ramp_rate)` with the chirp.

- **`_play_constant`** -- used when there is no polynomial (missing CSV, or
  `load_xy_amp_calibration(..., latency_only=True)`). Each segment plays a
  `"hold"` pulse with a chirp; sticky amplitude stays at the initial
  `"rampup"`. The same `computation_latency_cc` still shortens `play()`.

## Sorting scripts (`02`–`04`)

All three call `load_xy_amp_calibration(file_name=SCRIPT_NAME)` and, after
the job, `check_tweezer_computation_latency(..., file_name=SCRIPT_NAME)`
with the in-memory X/Y calibrations (not a second CSV read).

### `02_arb_tone_MJ_manual.py`

Fixed frequencies/detunings compiled into the QUA program (`set_x_detunings`,
`set_y_detunings`). Good for simulation and a single hardware shot.

### `03_arb_tone_MJ_sort_manual_IS.py`

Hardware mode uses input streams and `infinite_loop_()`:

- `x_frequencies` / `x_detunings` — starting IF and shift per X tweezer
- `y_frequencies` / `y_detunings` — starting IF and shift per Y tweezer

At each prompt, enter comma-separated integer Hz values. Short lists are
padded with zeros. To quit: blank line, `q` / `quit` / `exit`, Ctrl+C, or
Ctrl+D at any prompt. A parse error re-prompts; it does not exit.

On quit (and on any exception after `qm.execute`), `job.halt()` and
`qm.close()` run so the OPX is not left blocked on `advance_input_stream`.

Simulation mode skips streams and plays the default test vectors once.

### `04_arb_tone_MJ_row_sort.py`

Row-by-row left-compression using `array_sorting_macros.py`. Each row
assigns tweezers to occupied sites and plays compensated (or
latency-only) chirps. Occupancy plots and spectrograms are saved via
`readout_analysis.py`.

## Amplitude calibration

AODs have frequency-dependent diffraction efficiency. Each axis passes
through a different physical AOD, so each needs its own calibration.

`load_xy_amp_calibration()` always returns two `Calibration` objects:

```python
Calibration(center, coefs, computation_latency_cc=MIN_WAIT, latency_only=False)
```

`MIN_WAIT` is 4 clock cycles. Missing CSVs, or `latency_only=True`, yield
`latency_only=True` with empty `coefs` (constant-amplitude path) and
`computation_latency_cc=MIN_WAIT` unless a latency row is already in the CSV.

When both axes have polynomials, coefficient trees are padded so they share
the same outer length and nested factor lengths (new terms get `[0.0]`,
factor lists pad with `1.0`) so QUA evaluation stays aligned.

### Workflow (once per axis)

1. **Measure and fit** -- Set `SWEEP_AXIS = "x"` (or `"y"`) and
   `simulate = False` in `01_measure_amplitude_response.py`, then run.
   The script sweeps a single tweezer across the AOD frequency range,
   records the photodetector response, saves the raw data to
   `x_amplitude_data.csv`, then immediately launches the fitting
   analysis. Select a polynomial order to save when prompted. Existing
   `computation_latency_cc_*` rows are preserved.

2. **Re-fit from existing data** (optional) -- Run
   `python amplitude_calibration.py x` (or `y`) to re-analyze a
   previously collected dataset without re-running the hardware sweep.
   Running without an axis argument uses legacy filenames
   (`amplitude_data.csv` / `amplitude_calibration_coefs.csv`).

3. **Inspect** -- `python xy_amplitude_validation.py` plots the X and Y
   curves over the configured IF ranges and their separable 2-D product.

4. **Run** -- `02`–`04` auto-load `{x,y}_amplitude_calibration_coefs.csv`.
   An axis with no CSV still runs, using `MIN_WAIT` latency and no
   amplitude compensation.

### Computation latency (`computation_latency_cc`)

Each sort script has its own processor gap because different programs take
a different number of cycles between segments. CSV keys look like:

```
computation_latency_cc_02_arb_tone_MJ_manual
computation_latency_cc_03_arb_tone_MJ_sort_manual_IS
computation_latency_cc_04_arb_tone_MJ_row_sort
```

`check_tweezer_computation_latency` reads timestamp streams `t_<element>`
after a job:

- Empty streams (tweezer never played) are skipped.
- Non-empty streams on the same axis must match (same samples and
  `mode(np.diff(t))`); otherwise it raises that computation time is not
  calibrated.
- Residual `delta = mode(np.diff(t)) - segment_length // 4`.
- Corrected latency is `loaded_latency + delta`, written back to that
  axis CSV under the script's key.
- **Green** print: `delta == 0` (already correct). **Red**: reload the
  notebook/script so the next compile uses the updated value.

Pass the `Calibration` objects the program was built with, not a fresh
CSV load: a previous run may already have written a corrected row that
does not match what the live job used.

### How the polynomial works

The fitting inverts the measured AOD response and fits a polynomial to the
inverse. At runtime, QUA evaluates this polynomial at the current frequency
to get a multiplier near 1.0 that flattens the output power. Coefficients
whose absolute value exceeds the QUA fixed-point range [-8, 8) are
automatically decomposed into nth-root factors.

### Calibration files

| File | Purpose |
|---|---|
| `01_measure_amplitude_response.py` | Sweep frequencies, record response, fit calibration |
| `amplitude_calibration.py` | Library + standalone re-fitting script |
| `xy_amplitude_validation.py` | Plot loaded X/Y compensation |
| `{x,y}_amplitude_data.csv` | Raw freq vs. amplitude measurements (generated) |
| `{x,y}_amplitude_calibration_coefs.csv` | Polynomial + per-script latency rows (generated) |

### Author
Jacob Warshauer, Soon Teh