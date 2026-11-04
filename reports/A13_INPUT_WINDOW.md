# A13 — Giving the encoder a temporal window: negative

Date: 2026-11-04. FSeval, w ∈ {2, 4, 8} × 3 seeds, 9 training runs.
Script `reproductions/tssl/scripts/run_a13_input_window.sh`, W&B tag `A13`.
Baseline arm is `A01_random_FSeval_s{0,1,2}`, identical config.

**Verdict: the prediction did not transfer from probe to training.** The window
makes the trained embedding more phase-decodable at w = 4, but MoF falls 11
points, purity is flat, and calibration gets worse. A12's reasoning was that the
single-frame input is the binding constraint; that is true of what a *probe* can
read and false of what *this objective* does with it.

## 1. What was predicted, before the runs

A12 measured within-action phase decodability of 0.103 from one raw frame and
0.314 from a ±4 window, and concluded the input was the binding constraint. The
pre-registered mechanism check, written into the run script before any run, was
that the phase decodability of the **trained embedding** should rise from 0.055
toward 0.314, with purity and ECE reported beside MoF because A07 showed MoF
alone cannot settle a representation question.

## 2. MoF

| arm | s0 | s1 | s2 | mean | Δ vs baseline |
|---|---:|---:|---:|---:|---:|
| baseline | 0.4510 | 0.4945 | 0.5266 | 0.4907 | — |
| w = ±2 | 0.3930 | 0.3804 | 0.3748 | 0.3827 | **−0.108** |
| w = ±4 | 0.3888 | 0.3707 | 0.3930 | 0.3842 | **−0.107** |
| w = ±8 | 0.3453 | 0.3020 | 0.3593 | 0.3355 | **−0.155** |

Consistent across all nine runs, far outside the ~5 pp resolution floor.

## 3. The mechanism check, and the representation-side metrics

| checkpoint | phase R² ridge | phase R² MLP |
|---|---:|---:|
| baseline s0 | 0.0262 | 0.0809 |
| w = 2 s0 | 0.0452 | 0.0341 |
| **w = 4 s0** | **0.0456** | **0.1524** |
| w = 8 s0 | 0.0291 | 0.0326 |

| | baseline (3 seeds) | w = 4 (3 seeds) |
|---|---:|---:|
| MoF | 0.4907 | 0.3842 |
| one-to-one accuracy | 0.468 | 0.410 |
| **many-to-one purity** | **0.6155** | **0.6057** |
| **ECE** | **0.4477** | **0.5624** |
| mean posterior confidence | 0.916 | 0.973 |

At w = 4 the embedding does become roughly twice as phase-decodable
(0.081 → 0.152 with a nonlinear readout), and w = 4 is exactly the window A12
identified as optimal. But that is the only thing that improved. Purity moved
−0.010, inside the baseline's own seed spread of 0.586–0.643. Calibration got
**worse**: ECE 0.448 → 0.562 and mean confidence 0.916 → 0.973, so the posterior
became more confident while no more correct.

Against the pre-registered criteria this is a fail. The mechanism check passed
only partially — a rise to 0.152 of a 0.314 target, at one of three window
sizes, on one seed — and the purity requirement did not pass at all.

## 4. Two explanations, and I cannot separate them with these runs

**(a) No incentive.** The window gives the encoder the *capability* to encode
phase but the objective gives it no *reason* to. A07 measured that this
benchmark and this loss reward temporal position: a position-only baseline
reaches 82% of VASOT per-video. A representation that spends capacity on phase
is not rewarded, so the extra information sits in the embedding (visible to a
probe, R² 0.152) without being used by the assignment.

**(b) Capacity and optimisation.** Concatenating 2w+1 frames multiplies the
input dimension by 9 at w = 4 and the first layer from roughly 8K to 74K
parameters, on the same data and the same 30 epochs. w = 8 is worse on *every*
metric including phase decodability (0.033), which is the signature of a model
that is simply not fitting, and that supports (b) at least for w = 8.

The decisive control is a **parameter-matched** window: give the encoder the same
temporal context without widening the input, for example a fixed temporal
average or a small depthwise temporal convolution over the raw features that
keeps the input dimension at 64. If a parameter-matched window still loses MoF
while raising phase decodability, (a) is the explanation and the next step is an
objective that rewards phase. If it recovers MoF, (b) was the explanation and the
concatenation was the wrong way to spend the capacity. This control has **not**
been run.

## 5. What this does to the A11/A12 recommendation

A12 ranked "give the encoder a temporal window" as the first intervention, on
the strength of a probe measurement. That ranking is **withdrawn** as stated: it
is not a drop-in improvement. What survives is narrower and still useful — a ±4
window does roughly double the phase content of the trained representation,
which is the first intervention in this project measured to do so at all. It
just needs an objective that uses it, and a control that rules out the capacity
explanation.

The broader lesson is about the inference, not the result: a probe measures what
is *available* in a representation, not what an objective will *do* with it. A12
treated the first as a proxy for the second. Nothing in the probe measurement
was wrong; the step from it to a training prediction was.

## 6. Limitations

FSeval only, 3 seeds, 30 epochs, one architecture family. The phase probe in §3
is reported for seed 0 at each window and for three seeds only implicitly
through the purity table; the 0.152 figure should be read as one seed. The
concatenation design was chosen to match exactly what the probe measured, which
makes the prediction testable but confounds context with capacity, as §4 says.
