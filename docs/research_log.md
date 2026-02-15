# Research Log

## Orientation Selectivity Index (OSI) and Preferred Orientation Metrics

**Date:** 2026-02-13

### OSI Definition (doubled-angle vector method)

We use the classic circular-variance-based OSI:

    OSI = |sum_k r(theta_k) * exp(i * 2 * theta_k)| / sum_k r(theta_k)

where `r(theta_k)` is the firing rate at orientation `theta_k` and the factor of 2 accounts
for the 180-degree periodicity of orientation (orientations theta and theta+180 are identical).

**Range:** [0, 1]. OSI=0 means no selectivity (flat tuning curve); OSI=1 means perfect selectivity
(responds to only one orientation).

**References:**
- Swindale, N. V. (1998). "Orientation tuning curves: empirical description and estimation
  of parameters." *Biological Cybernetics*, 78(1), 45-56. doi:10.1007/s004220050411
- Ringach, D. L., Shapley, R. M., & Hawken, M. J. (2002). "Orientation selectivity in macaque
  V1: diversity and laminar dependence." *Journal of Neuroscience*, 22(13), 5639-5651.
  doi:10.1523/JNEUROSCI.22-13-05639.2002

### Why both argmax-based and vector-sum preferred orientation are tracked

We compute two measures of preferred orientation per ensemble:

1. **`pref_deg_vec`** (vector-sum method): `0.5 * arg(sum r(theta) * exp(i*2*theta))` in [0, 180).
   - Smooth, circular mean that weights all tested orientations by their response magnitude.
   - Robust to noise when responses are broadly tuned.
   - Standard in the literature (Swindale 1998; Ringach et al. 2002).

2. **`pref_deg_peak`** (argmax method): orientation from the sampled grid with maximum firing rate.
   - Directly interpretable: "which orientation drives this ensemble hardest?"
   - Stable for monitoring drift across training epochs (discrete, no circular interpolation artifacts).
   - Useful for sorting heatmaps and building discrete orientation maps.

**Rationale for tracking both:**
- The vector-sum pref is more principled for statistics (circular mean, proper weighting),
  but can shift substantially for weakly tuned or bimodal cells where the vector average
  lands between two response modes.
- The argmax pref is coarser (grid-limited) but unambiguous for monitoring developmental
  drift during training: if the argmax jumps by more than one bin between checkpoints,
  that signals a real change in selectivity.
- Tracking both allows validation that they converge for well-tuned cells (a useful
  sanity check on tuning quality).

### Biological mapping

Orientation selectivity in primary visual cortex (V1) arises from the spatial arrangement of
thalamocortical (LGN->V1) synaptic weights, combined with intracortical inhibition. In this
model, STDP shapes the feedforward weights to develop oriented receptive fields, while
PV/SOM interneuron circuits provide contrast-invariant gain control and surround suppression.

**Key assumption:** The OSI computed from spike-count tuning curves reflects the same
selectivity metric used in extracellular recording studies (Ringach et al. 2002), allowing
direct comparison to empirical distributions (median OSI ~ 0.3-0.5 in macaque V1 layer 4).

---

## E→E Weight vs Orientation Distance Analysis

**Date:** 2026-02-13

### Motivation

Lateral excitatory (E→E) connections in V1 are known to preferentially link neurons with
similar orientation preferences ("like-to-like" connectivity). This has been demonstrated
anatomically via optical imaging + tracer injection and functionally via cross-correlation
of spike trains.

This analysis tracks whether STDP-driven lateral plasticity in the model reproduces this
biological pattern: after training, do E→E weights become a decreasing function of
orientation distance between pre and post ensembles?

### Method

For every directed off-diagonal synapse `W_e_e[post, pre]`:
1. Compute `d_ori = circ_diff_180(pref[post], pref[pre])` yielding degrees in [0, 90].
2. Bin distances into 10° bins: [0-10), [10-20), ..., [80-90).
3. Compute mean weight and SEM per bin.
4. Optionally filter to only include synapses where both pre and post have `OSI >= osi_min`
   (default 0.2), since orientation distance is meaningless for untuned ensembles.

Error bars use standard error of the mean (SEM = std / sqrt(n)), which is appropriate here
because we are estimating the population mean weight per distance bin, not making predictions
about individual synapses.

### References

- Bosking, W. H., Zhang, Y., Schofield, B., & Fitzpatrick, D. (1997). "Orientation
  selectivity and the arrangement of horizontal connections in tree shrew striate cortex."
  *Journal of Neuroscience*, 17(6), 2112-2127. doi:10.1523/JNEUROSCI.17-06-02112.1997
- Ko, H., Hofer, S. B., Pichler, B., Buchanan, K. A., Sjostrom, P. J., & Mrsic-Flogel, T. D.
  (2011). "Functional specificity of local synaptic connections in neocortical networks."
  *Nature*, 473(7345), 87-91. doi:10.1038/nature09880

---

## Heterogeneous E→E Conduction Delays

**Date:** 2026-02-13

### Biological motivation

Horizontal (lateral) connections in V1 are unmyelinated or thinly myelinated axons with
conduction velocities of ~0.1–0.3 m/s (Bringuier et al. 1999; Girard et al. 2001). For
cortical distances of 0.5–6 mm, this yields conduction delays of ~1–20 ms, with distance
being the primary determinant.

These delays are functionally significant: they shape the temporal dynamics of recurrent
amplification, synchrony, and surround modulation. Models that use instantaneous lateral
connections miss these effects.

### Implementation

We implement heterogeneous E→E conduction delays via:
1. **D_ee matrix** (M×M, int16): per-synapse delay in simulation timesteps.
   - Distance-dependent component: linear mapping from cortical distance to delay range
     `[ee_delay_ms_min, ee_delay_ms_max]`, controlled by `ee_delay_distance_scale` (0=random, 1=pure distance).
   - Gaussian jitter: additive noise with std `ee_delay_jitter_ms` (models axon diameter variability).
2. **delay_buf_ee** ring buffer (L_ee × M, uint8): stores recent V1 E spike history.
   - `ptr_ee` advances each timestep; reading at `(ptr_ee - D_ee[post,pre]) % L_ee`.
3. Diagonal entries of D_ee are zero (no self-connections).

### Default parameters

- `ee_delay_ms_min = 1.0` ms (nearest-neighbor, ~0.5 mm at 0.3 m/s)
- `ee_delay_ms_max = 6.0` ms (furthest cortical distance, ~1.8 mm at 0.3 m/s)
- `ee_delay_distance_scale = 1.0` (purely distance-dependent)
- `ee_delay_jitter_ms = 0.5` ms (biological variability)

### Key equations

    D_ee[i,j] = round(delay_min + dist_frac * delay_range * (d_ij / d_max) + N(0, jitter))
    delay_min = ee_delay_ms_min / dt_ms
    delay_max = ee_delay_ms_max / dt_ms
    delay_range = delay_max - delay_min

### References

- Bringuier, V., Chavane, F., Glaeser, L., & Frégnac, Y. (1999). "Horizontal propagation
  of visual activity in the synaptic integration field of area 17 neurons."
  *Science*, 283(5402), 695-699. doi:10.1126/science.283.5402.695
- Girard, P., Hupé, J. M., & Bullier, J. (2001). "Feedforward and feedback connections
  between areas V1 and V2 of the macaque monkey." *Experimental Brain Research*, 136(3),
  340-355. doi:10.1007/s002210000588
- González-Burgos, G., Barrionuevo, G., & Lewis, D. A. (2000). "Horizontal synaptic
  connections in monkey prefrontal cortex: an in vitro electrophysiological study."
  *Cerebral Cortex*, 10(1), 82-92. doi:10.1093/cercor/10.1.82

---

## E→E Drive Fraction (Feedforward vs Recurrent)

**Date:** 2026-02-13

### Biological motivation

In primary visual cortex, the response of a neuron to a visual stimulus is shaped by both
feedforward thalamocortical drive (LGN→V1) and recurrent intracortical excitation (E→E).
Estimates from intracellular recordings and optogenetic silencing experiments suggest that
recurrent excitation contributes approximately 50–80% of the total excitatory conductance
in V1 layer 4 during sensory stimulation (Douglas & Martin 2004; Li et al. 2013;
Lien & Scanziani 2013).

Tracking the feedforward vs recurrent drive fraction during training allows us to:
1. Verify that the model reaches a biologically plausible operating regime.
2. Monitor how the balance shifts as STDP shapes lateral connections.
3. Tune E→E connectivity parameters to match empirical drive ratios.

### Implementation

We split `g_v1_exc` into two components:
- `g_exc_ff` (float32, M): feedforward excitatory conductance from LGN→V1 drive.
- `g_exc_ee` (float32, M): recurrent excitatory conductance from E→E lateral connections.
- Total: `g_v1_exc = g_exc_ff + g_exc_ee` (used in membrane equation as before).

Both conductances decay with `tau_ampa` and receive input each timestep. The drive fraction
is computed from time-integrated conductances:

    drive_frac_ee = sum_t(g_exc_ee) / (sum_t(g_exc_ff) + sum_t(g_exc_ee))

Accumulators (`_drive_acc_ff`, `_drive_acc_ee`, `_drive_acc_steps`) track the running sum
and are reset at segment boundaries.

### Key methods

- `get_drive_fraction()` → (mean_frac, per_ensemble): read current accumulated fraction.
- `reset_drive_accumulators()`: zero accumulators for next measurement window.
- `measure_drive_fraction(theta_deg, ...)`: state-preserving measurement helper.

### References

- Douglas, R. J., & Martin, K. A. (2004). "Neuronal circuits of the neocortex."
  *Annual Review of Neuroscience*, 27, 419-451. doi:10.1146/annurev.neuro.27.070203.144152
- Li, Y. T., Ibrahim, L. A., Liu, B. H., Zhang, L. I., & Tao, H. W. (2013). "Linear
  transformation of thalamocortical input by intracortical excitation." *Nature Neuroscience*,
  16(9), 1324-1330. doi:10.1038/nn.3494
- Lien, A. D., & Scanziani, M. (2013). "Tuned thalamic excitation is amplified by visual
  cortical circuits." *Nature Neuroscience*, 16(9), 1315-1323. doi:10.1038/nn.3488

---

## Delay-Aware E→E STDP (Pair-Based, Weight-Dependent)

**Date:** 2026-02-13

### Biological motivation

The legacy E→E STDP rule used per-neuron traces and a 1-step lag (`prev_v1_spk`), ignoring
the actual per-synapse conduction delays stored in `D_ee`. With heterogeneous delays of 1–6 ms,
the effective pre-spike arrival time at the postsynaptic terminal varies across synapses. A
biologically plausible STDP rule must use the *actual delayed arrival* for each synapse to
determine spike timing differences.

### Rule

Pair-based STDP with exponential traces and weight-dependent bounds:

    Pre trace (per-synapse, M×M):
        x_pre[post,pre] *= exp(-dt / tau_pre)
        x_pre[post,pre] += ee_arrivals[post,pre]   (delayed spike arrival)

    Post trace (per-neuron, M):
        x_post[i] *= exp(-dt / tau_post)
        x_post[i] += post_spikes[i]

    LTD (on pre arrival, using OLD post trace):
        dW -= A_minus * ee_arrivals * x_post[:, None] * (W - w_min)

    LTP (on post spike, using NEW pre trace):
        dW += A_plus * post_spikes[:, None] * x_pre * (w_max - W)

The weight-dependent terms `(w_max - W)` for LTP and `(W - w_min)` for LTD provide
synapse-local stability: weights saturate soft-boundedly near limits without requiring
global normalization (Song et al. 2000).

### Key design choices

1. **Per-synapse pre-traces**: Because conduction delays are synapse-specific (from `D_ee`),
   a pre spike from neuron j arrives at neuron i at a different time than at neuron k. The
   pre-trace must therefore be (M×M) to capture per-synapse timing.

2. **Per-neuron post-traces**: Post-synaptic spikes are local events observed at the soma.
   All synapses on a given post-neuron see the same post-spike timing → (M,) trace suffices.

3. **Weight-dependent bounds**: LTP ∝ (w_max - w), LTD ∝ (w - w_min). This eliminates
   the need for weight clipping as a stability mechanism (though hard clips are still applied
   as a safety net) and produces a natural equilibrium distribution of weights (bimodal for
   non-weight-dependent, unimodal for weight-dependent; see van Rossum et al. 2000).

4. **LTD bias**: A_minus slightly > A_plus (default 0.00012 vs 0.0001) provides a mild
   heterosynaptic depression bias that prevents all-to-all strengthening.

### Two-phase training

To separate feedforward receptive field development from recurrent structure formation:

- **Phase A** (segments 1 to `phase_b_start_segment - 1`): Only feedforward STDP is active.
  V1 ensembles develop oriented receptive fields through thalamocortical STDP.
- **Phase B** (segments `phase_b_start_segment` onwards): Feedforward STDP is frozen,
  delay-aware E→E STDP activates. Lateral connections self-organize based on correlated
  activity patterns established by the now-fixed receptive fields.

Optional ramp: `ee_stdp_ramp_segments` linearly ramps A_plus and A_minus from 0 to their
full values over the specified number of segments at Phase B onset, preventing sudden
perturbations.

### Default parameters

- `ee_stdp_A_plus = 0.0001` (LTP rate)
- `ee_stdp_A_minus = 0.00012` (LTD rate)
- `ee_stdp_tau_pre_ms = 20.0` ms
- `ee_stdp_tau_post_ms = 20.0` ms
- `w_e_e_min = 0.0`, `w_e_e_max = 0.2`
- `ee_stdp_weight_dep = True`

### References

- Markram, H., Lübke, J., Frotscher, M., & Sakmann, B. (1997). "Regulation of synaptic
  efficacy by coincidence of postsynaptic APs and EPSPs." *Science*, 275(5297), 213-215.
  doi:10.1126/science.275.5297.213
- Bi, G. Q., & Poo, M. M. (1998). "Synaptic modifications in cultured hippocampal neurons:
  dependence on spike timing, synaptic strength, and postsynaptic cell type." *Journal of
  Neuroscience*, 18(24), 10464-10472. doi:10.1523/JNEUROSCI.18-24-10464.1998
- Song, S., Miller, K. D., & Abbott, L. F. (2000). "Competitive Hebbian learning through
  spike-timing-dependent synaptic plasticity." *Nature Neuroscience*, 3(9), 919-926.
  doi:10.1038/nn0900_919
- Sjöström, P. J., Turrigiano, G. G., & Nelson, S. B. (2001). "Rate, timing, and
  cooperativity jointly determine cortical synaptic plasticity." *Neuron*, 32(6), 1149-1164.
  doi:10.1016/S0896-6273(01)00542-6
- van Rossum, M. C. W., Bi, G. Q., & Turrigiano, G. G. (2000). "Stable Hebbian learning
  from spike timing-dependent plasticity." *Journal of Neuroscience*, 20(23), 8812-8821.
  doi:10.1523/JNEUROSCI.20-23-08812.2000

---

## Sequence Learning Experiment (Gavornik & Bear 2014)

**Date:** 2026-02-13

### Biological background

Gavornik & Bear (2014) demonstrated that mouse V1 can learn spatiotemporal sequences of
oriented gratings through repeated exposure. Key findings:

1. **Sequence-specific potentiation**: After days of training with a specific sequence (ABCD),
   neural responses (measured via visually-evoked potentials, VEPs) to the trained sequence
   increase relative to a novel-order control (DCBA).
2. **Timing specificity**: The potentiation is specific to the temporal structure — presenting
   the same orientations with altered timing (longer element durations) produces weaker
   responses than the trained timing.
3. **Prediction/omission responses**: When an expected element is omitted (A_CD), there is
   neural activity at the time the missing element would have occurred, consistent with a
   learned prediction. This predictive response is absent when the preceding context is
   changed (E_CD), confirming it depends on sequence-specific associations.

### Protocol (implemented)

**Training (Phase B):**
- Present sequence (e.g., 0°→45°→90°→135°) as drifting gratings.
- Each element: 150 ms at full contrast. Inter-trial interval: 1500 ms blank.
- 200 presentations per day, 4 days.
- Delay-aware E→E STDP active; feedforward STDP frozen (two-phase gating).

**Evaluation conditions (non-plastic):**
1. **Trained order**: ABCD at trained timing.
2. **Novel order**: DCBA (reverse) or random permutation.
3. **Timing-changed**: ABCD but each element at 300 ms (2x duration).
4. **Omission**: A_CD (blank replaces B), with sequence-context intact.
5. **Omission control**: E_CD (different pre-omission context) — isolates prediction
   from mere temporal expectation.

**Metrics:**
- **Potentiation index**: mean_rate(trained) / mean_rate(novel). Values > 1 indicate
  sequence-specific potentiation.
- **Timing specificity index**: mean_rate(trained) / mean_rate(timing-changed). Values > 1
  indicate temporal specificity.
- **Prediction index**: mean_rate(omission_window in A_CD) - mean_rate(omission_window in E_CD).
  Values > 0 indicate a learned prediction response.

### VEP-like readout

We use population E spike count (summed across all M ensembles) per simulation timestep as
a "VEP proxy." This is smoothed with a 10 ms causal boxcar filter for visualization.

**Caveat:** This is a spike-rate readout, not a field potential. True VEPs reflect summed
postsynaptic currents (mainly excitatory). A population spike count is a reasonable proxy
for the dominant component of the VEP signal in layer 4, but it lacks the precise
biophysics of extracellular field generation.

### Mechanism: how STDP supports sequence learning

The delay-aware E→E STDP naturally supports directional sequence learning:

1. During element A, ensemble neurons with preference near A's orientation fire.
2. After the conduction delay (1–6 ms), their spikes arrive at other ensembles.
3. When element B begins, ensemble neurons preferring B fire.
4. The pre-before-post timing (A arrival → B spike) produces LTP on the A→B synapse.
5. The reverse (B arrival → A spike from previous element) decays quickly because post-traces
   from A have already decayed, and B spikes arrive too late for A's post-spike window.

Over many presentations, this creates an asymmetric forward chain: W(B,A) > W(A,B),
W(C,B) > W(B,C), etc. During replay, the trained forward chain amplifies the sequence-specific
response beyond what the novel order (which lacks these directional biases) can produce.

### Cholinergic gating and scopolamine control

Gavornik & Bear (2014) showed that the learning requires intact cholinergic input (muscarinic
receptor blockade prevents sequence potentiation). In this model, the two-phase gating
(Phase A: feedforward development, Phase B: recurrent STDP) serves an analogous role:
cholinergic modulation may gate the transition from a development-dominated to a
plasticity-active recurrent regime. We do not explicitly model acetylcholine receptors,
but the Phase B gate is a functional stand-in.

**Scopolamine control implementation:** The `--seq-scopolamine-phase` flag disables the
delay-aware E→E STDP (`ee_stdp_active=False`) during the specified phase(s):
- `training`: blocks E→E STDP during Phase B training → prevents sequence-specific weight
  changes → potentiation index remains ~1.0 (no learning).
- `test`: blocks E→E STDP during evaluation only.
- `both`: blocks during both training and test.
- `none` (default): normal operation.

This matches the experimental finding that systemic scopolamine (muscarinic antagonist)
blocks acquisition of sequence-specific VEP potentiation (Chubykin et al. 2013).

### VEP signal modes and paper-matched scoring

**VEP proxy signal:** The model provides three readout modes (`--seq-vep-mode`):
- `spikes`: population spike count per timestep (default).
- `i_exc`: sum of excitatory current (I_exc) across all V1 neurons — approximates the
  local field potential (LFP) which reflects synaptic currents.
- `g_exc`: sum of excitatory conductance (g_exc) across all V1 neurons.

The `i_exc` mode most closely approximates the VEP signal in Gavornik & Bear (2014),
since VEPs are dominated by synaptic currents rather than spikes.

**Peak-to-peak scoring:** Following the paper's "sequence magnitude" metric:
1. Each element trace is smoothed (5 ms boxcar, edge-padded to avoid artifacts).
2. Baseline is subtracted (mean of the last 50 ms of the preceding element/ITI).
3. Peak-to-peak amplitude = max(signal) - min(signal) after baseline subtraction.
4. Sequence magnitude = mean peak-to-peak across all elements.
5. Potentiation index = trained_magnitude / novel_magnitude.

This replaces the earlier spike-rate-ratio metric with a signal that better matches
the experimental measurement procedure.

### Multi-seed robustness harness

The `--seq-seeds` flag enables running the full experiment across multiple random seeds
(e.g., `--seq-seeds 1,2,3,4,5`). For each seed:
- A fresh network is constructed and trained independently.
- All metrics are collected per-day.
After all seeds complete, mean±SEM of all metrics is computed and plotted with error bars.
This enables statistical assessment of learning robustness (e.g., potentiation index
significantly >1 across seeds).

### References

- Gavornik, J. P., & Bear, M. F. (2014). "Learned spatiotemporal sequence recognition
  and prediction in primary visual cortex." *Nature Neuroscience*, 17(5), 732-737.
  doi:10.1038/nn.3683
- Xu, S., Jiang, W., Poo, M. M., & Dan, Y. (2012). "Activity recall in a visual cortical
  ensemble." *Nature Neuroscience*, 15(3), 449-455. doi:10.1038/nn.3036
- Chubykin, A. A., Roach, E. B., Bear, M. F., & Bhatt, D. (2013). "A cholinergic mechanism
  for reward timing within primary visual cortex." *Neuron*, 77(4), 723-735.
  doi:10.1016/j.neuron.2012.12.039

---

## Sequence Learning: Weight-Dependent vs Additive STDP

**Date:** 2026-02-13

### Problem

The initial sequence learning implementation used weight-dependent STDP with default parameters
(A+=0.0001, A-=0.00012). After calibration, effective learning rates were too low for robust
forward > reverse asymmetry within a realistic number of training presentations (~200-600 per day).

### Weight-dependent STDP after calibration

The calibration step (`calibrate_ee_drive`) scales all E→E weights by a large factor (~300-400×)
to achieve a target drive fraction (default 5%). After calibration:
- W_e_e mean ≈ 0.5-0.7 (depending on seed and network selectivity)
- w_e_e_max = max(cal_mean × 2.0, 0.2) ≈ 1.0-1.4

With weight-dependent STDP, effective LTP per coincidence = A+ × (w_max - w) ≈ A+ × cal_mean.
At the default A+=0.0001, this gives only ~7e-5 per coincidence — far too slow for 200
presentations per day. Cross-element coincidence probability is ~9% per synapse pair per
presentation, giving ΔW ≈ 6e-6 per presentation for forward connections.

### Additive STDP attempt (failed)

Setting `ee_stdp_weight_dep=False` removes the (w_max - w) and (w - w_min) modulators.
Results with A+=0.002, A-=0.0024:
- Day 1: potentiation_index=1.818 (strong forward > reverse)
- Day 2: potentiation_index=0.687 (**novel > trained** — catastrophic)

**Root cause**: Positive feedback loop. With additive STDP, stronger synapses don't self-limit.
Forward connections strengthen → postsynaptic rates increase → more coincidences → even stronger
connections. W_e_e values hit the hard ceiling (w_max=1.418). Within-orientation-group weights
also saturate, making ALL orientations fire maximally → loss of selectivity → no trained/novel
distinction.

This is consistent with Song, Miller & Abbott (2000): additive STDP produces bimodal weight
distributions (0 or w_max) with unstable competition. With short training sequences, the initial
potentiation of forward connections quickly triggers a runaway cascade.

### Solution: Weight-dependent STDP with increased learning rates

Increased base learning rates to A+=0.005, A-=0.006 (25× original, 2.5× previous attempt).
Kept weight-dependent mode (`ee_stdp_weight_dep=True`).

Added **post-calibration learning rate normalization** to handle seed-dependent calibration:
```
target_eff_lr = 0.003  # target effective LTP per coincidence
lr_scale = target_eff_lr / (A+ × cal_mean)
A+ *= lr_scale; A- *= lr_scale
```

This ensures effective learning rate ≈ 0.003 per coincidence regardless of the calibrated weight
level, which varies across seeds (e.g., seed 1: cal_mean=0.709, seed 42: cal_mean=0.462).

### Validated results

**Seed 1 (3 days, 200 presentations/day):**
- pot = 1.940 → 1.089 → 2.702 (Day 1-3; non-monotonic Day 2 dip, strong Day 3)
- rate_ratio = 8.13 → 1.37 → 32.04 (trained_rate/novel_rate)
- Day 3: trained_rate=8.33 Hz, novel_rate=0.26 Hz → clear forward > reverse
- OSI maintained: 0.447 → 0.609 (actually improves due to lateral structure)

**Seed 42 (5 days, 200 presentations/day):**
- pot = 0.958 → 1.041 → 1.246 → 0.919 → 1.425 (slower convergence)
- rate_ratio = 0.62 → 1.14 → 4.33 → 0.30 → 3.25 (positive trend)
- Overall upward trajectory, requires more training days for robust effect
- Higher OSI (0.826) → fewer neurons per orientation group → fewer coincidences → slower

### Key insight: OSI-coincidence rate bottleneck

The rate of sequence learning depends on coincidence counts across sequential orientation
groups. High-OSI networks have sharp tuning → fewer neurons respond to each orientation →
fewer cross-element spike coincidences per presentation → slower STDP accumulation.

Seed 42 (mean OSI=0.826) has ~4 neurons per group, while seed 1 (mean OSI=0.447) has ~8.
The number of coincidences scales approximately as n_pre × n_post × p_coincidence, so
networks with broader tuning learn forward chains ~4× faster.

This is biologically consistent: V1 neurons have diverse selectivity levels (Ringach et al.
2002), and sequence learning in superficial layers (where lateral connections are denser)
may rely on populations with moderate selectivity.

### Non-monotonic Day-2/Day-4 dips

Both seeds show transient dips in potentiation index (seed 1: Day 2, seed 42: Day 4).
This reflects a phase where:
1. Within-group weights strengthen equally for all orientations (no directional bias)
2. PV homeostatic inhibition adapts to increased drive
3. The forward/reverse asymmetry is temporarily obscured by uniformly increased excitability

The effect is transient: continued STDP differentiates forward from reverse connections
because only forward pairs receive consistent pre-before-post timing.

### References

- Song, S., Miller, K. D., & Abbott, L. F. (2000). "Competitive Hebbian learning through
  spike-timing-dependent synaptic plasticity." *Nature Neuroscience*, 3(9), 919-926.
  doi:10.1038/nn0900_919
- van Rossum, M. C. W., Bi, G. Q., & Turrigiano, G. G. (2000). "Stable Hebbian learning
  from spike timing-dependent plasticity." *Journal of Neuroscience*, 20(23), 8812-8821.
  doi:10.1523/JNEUROSCI.20-23-08812.2000
- Ringach, D. L., Shapley, R. M., & Hawken, M. J. (2002). "Orientation selectivity in macaque
  V1: diversity and laminar dependence." *Journal of Neuroscience*, 22(13), 5639-5651.
  doi:10.1523/JNEUROSCI.22-13-05639.2002

---

## E→E Drive Fraction Calibration Target: 0.05 → 0.15

**Date:** 2026-02-14

### Change

Increased the default calibration target for E→E drive fraction from 5% to 15%.

### Biological justification

The previous 5% target was too conservative. At 5% recurrent contribution, a 20% forward/reverse
weight asymmetry produces only ~1% differential response per element — buried in noise.

Empirical estimates of recurrent excitation in V1 layer 4:

- Douglas & Martin (2004) estimate 70-80% of excitatory synapses in L4 are intracortical.
  Even accounting for lower per-synapse efficacy of lateral vs thalamocortical connections,
  recurrent excitation contributes substantially to total drive.
- Lien & Scanziani (2013) used optogenetic silencing to show that intracortical amplification
  roughly doubles thalamocortical responses, consistent with ~50% recurrent contribution.
- Li et al. (2013) found linear transformation of thalamocortical input by intracortical
  excitation, with recurrent gain of ~2-3x.

A target of 15% is conservative relative to biology (50-80%) but sufficient for the model:
at 15%, a 20% forward/reverse weight asymmetry produces ~3% differential per element,
compounding across 4 elements to ~12% total — reliably detectable above noise.

### Additional changes

- Removed post-calibration STDP learning rate normalization (`target_eff_lr` / `lr_scale`).
  This was a global hack that defeated weight-dependent STDP's self-regulating property.
  With the higher calibration target, effective learning rates from weight-dependent STDP
  alone (A+ x (w_max - w)) are sufficient without artificial normalization.
- Replaced day-based evaluation loop with presentation-based checkpoints, giving cleaner
  metrics-vs-training-amount curves.
- Added forward chain weight asymmetry instrumentation and E→E ablation test.

### References

- Douglas, R. J., & Martin, K. A. (2004). "Neuronal circuits of the neocortex."
  *Annual Review of Neuroscience*, 27, 419-451. doi:10.1146/annurev.neuro.27.070203.144152
- Lien, A. D., & Scanziani, M. (2013). "Tuned thalamic excitation is amplified by visual
  cortical circuits." *Nature Neuroscience*, 16(9), 1315-1323. doi:10.1038/nn.3488
- Li, Y. T., Ibrahim, L. A., Liu, B. H., Zhang, L. I., & Tao, H. W. (2013). "Linear
  transformation of thalamocortical input by intracortical excitation." *Nature Neuroscience*,
  16(9), 1324-1330. doi:10.1038/nn.3494
