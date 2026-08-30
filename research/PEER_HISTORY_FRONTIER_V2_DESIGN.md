# Peer-History Comparability–Diagnostic Frontier v2

## Working title

**When Does Performance History Become a Bad Peer Signal?**  
**The Comparability–Diagnostic Validity Frontier in Retail Peer Benchmarking**

한국어 작업 제목: **과거 성과는 언제 좋은 비교 신호에서 나쁜 진단 신호로 바뀌는가? 리테일 Peer Benchmarking의 비교가능성–진단타당성 경계**

## 1. Research question

The study does **not** ask merely whether using historical outcomes in peer construction can attenuate a persistent performance gap. A constant performance-level shift embedded in the history can make attenuation partly mechanical.

The primary question is therefore:

> **How much and what kind of performance history can be used before gains in peer comparability become losses in diagnostic validity?**

The target object is a **comparability–diagnostic validity frontier**, not a single best peer algorithm.

## 2. Mechanism decomposition

For store `i` and historical period `t`, decompose performance history conceptually as

`H_it = L_i + S_it`,

where:

- `L_i`: historical performance **level** (store-specific mean),
- `S_it`: mean-centered historical **shape** or trajectory.

Peer distance for the decomposed designs is

`D = (1 - alpha) D_context + alpha [beta D_level + (1 - beta) D_shape]`,

where:

- `alpha` controls the total weight assigned to performance history,
- `beta` controls how much of that history weight is assigned to level rather than shape.

A raw-trajectory distance is retained as a reference arm because it represents the common, direct use of historical outcomes for similarity.

## 3. Peer-construction arms

1. **Context only** — no historical outcome information.
2. **Level only** — historical mean performance is the outcome-derived signal.
3. **Shape only** — row-centered history; constant level shifts are removed.
4. **Level–shape hybrid** — explicit decomposition with varying `beta`.
5. **Raw trajectory reference** — historical trajectory used without the level/shape decomposition.

Primary grid:

- `alpha ∈ {0.00, 0.25, 0.50, 0.75, 1.00}`
- `beta ∈ {0.00, 0.25, 0.50, 0.75, 1.00}` for decomposed-history designs
- peer count `K = 15`

## 4. Data and temporal split

Primary empirical dataset: **Rossmann weekly data (`rossmann_1W`) from the FEV dataset collection**.

Primary split:

- peer-history window: **24 weeks**
- evaluation window: **8 weeks immediately after the history window**
- all peer-construction information must precede the evaluation period

Context variables are restricted to non-outcome store or operational descriptors available before evaluation. Historical sales are used only in the explicitly labeled history components.

## 5. Known-ground-truth intervention

For each eligible focal store, inject a known persistent negative performance shock.

Evaluation-period log outcome:

`Y*_i = Y_i + log(1 - delta)`.

The same shock is embedded into the final fraction `q` of the history window when testing contamination.

Primary contamination grid:

- `q ∈ {0.00, 0.25, 0.50, 0.75, 1.00}`

Shock magnitudes:

- **primary:** `delta = 0.10`
- robustness: `delta ∈ {0.05, 0.15, 0.20}`

The injected shock is applied to the focal store used to query the peer system; donor-store histories and donor evaluation outcomes remain unmodified.

## 6. Metrics

### 6.1 Clean peer comparability

For each configuration, construct peers using uncontaminated history and calculate the error between the peer benchmark and the focal store's evaluation-period outcome.

Primary clean metric:

`CleanMAE = mean_i |Benchmark_i - Y_i|`.

Relative comparability gain over context-only:

`ComparabilityGain = 100 × (MAE_context - MAE_config) / MAE_context`.

Higher is better.

### 6.2 Shock recovery

Let the known true log-shock magnitude be `T = -log(1 - delta)`.

For each focal store, compare the change in its benchmark gap before and after the injected shock:

`Recovery = DetectedEffect / T`.

Interpretation:

- `Recovery = 1`: correct diagnosis,
- `Recovery < 1`: attenuation / hidden underperformance,
- `Recovery > 1`: amplification / over-diagnosis.

### 6.3 Diagnostic Error AUC

Because both attenuation and amplification are diagnostic failures, the primary diagnostic metric is

`DiagnosticErrorAUC = normalized integral_q |Recovery(q) - 1| dq`.

Lower is better; zero is ideal.

`DiagnosticRetentionAUC = integral Recovery(q) dq` is retained only as a secondary descriptive metric because values above 1 can otherwise look artificially favorable.

### 6.4 Pareto frontier

A peer configuration is Pareto-dominated if another configuration has:

- equal or greater `ComparabilityGain`, and
- equal or lower `DiagnosticErrorAUC`,
- with at least one strict improvement.

The empirical object of interest is the nondominated **comparability–diagnostic validity frontier**.

## 7. Primary hypotheses / falsifiable expectations

### H1 — History can provide genuine comparability value

At least one history-using configuration should improve clean benchmark comparability relative to context-only.

### H2 — Outcome level is the principal contamination channel for persistent level deterioration

As historical contamination increases, designs assigning more weight to performance **level** should generally suffer larger diagnostic error than otherwise comparable shape-heavy designs.

This is an empirical expectation, not an identity: partial contamination can alter trajectory shape, so shape-only designs are not assumed to be immune.

### H3 — There is no universally dominant information design

If the trade-off is substantive, increasing historical-outcome information should create configurations with better clean comparability but worse diagnostic validity, producing a nontrivial Pareto frontier.

### H4 — The trade-off exists at operationally plausible shock sizes

The effect should be visible at **5–15%** deterioration, not only at the 20% robustness condition.

## 8. Pre-declared screening gates

The primary `H=24, K=15, delta=10%` experiment is judged before any sensitivity expansion.

### KILL / REDESIGN 1 — No material comparability benefit

If the maximum clean comparability improvement from adding history is **< 5%**, the intended trade-off is weak. The study risks reducing to 'a bad variable hurts diagnosis.'

### KILL / REDESIGN 2 — No material diagnostic cost

Among configurations with at least 5% comparability gain, if contamination does not create at least one material diagnostic deviation (approximately **10 percentage points** in recovery or `DiagnosticErrorAUC >= 0.10`), the practical trade-off is weak.

### KILL / REDESIGN 3 — Frontier collapses

If a single information design essentially dominates the alternatives, there is little evidence for a meaningful comparability–diagnostic frontier.

### KILL / REDESIGN 4 — Only extreme shocks work

If the trade-off appears only at `delta = 20%` but not at **5%, 10%, or 15%**, the empirical story is considered too fragile for the proposed claim.

### KILL / REDESIGN 5 — Sensitivity collapse

Only after the primary experiment survives, repeat across selected history lengths and K values. If the qualitative frontier disappears under reasonable alternatives, the claim must be narrowed or killed.

## 9. Stage-gated robustness plan

Robustness is intentionally **not** run before the primary gate to avoid searching a large specification space for a favorable result.

If the primary gate survives:

### Stage 2A — Design sensitivity

- history length: `H ∈ {12, 24, 52}` weeks
- peer count: `K ∈ {5, 15, 30}`

### Stage 2B — Deterioration shape

Primary injection is a step shock. Add a gradual/ramp deterioration condition because real store deterioration may emerge progressively within the history window.

### Stage 2C — Data-construction replication

Replicate the core result on the daily Rossmann source, if execution infrastructure permits, to verify that the weekly FEV representation is not driving the conclusion.

## 10. Interpretation constraints

1. Synthetic experiments are **design validation only**, not empirical evidence for Rossmann.
2. A decreasing recovery curve under raw level matching is not by itself a novel result; part of it can follow mechanically from matching on a contaminated outcome level.
3. The contribution must come from characterizing the **value–harm frontier**, identifying which historical components drive it, and showing its practical magnitude under realistic conditions.
4. Do not claim that historical outcomes should never be used for peer construction.
5. Do not claim that shape-only history is universally safe.
6. Do not claim that peer selection, performance-based clustering, or outcome-guided similarity is itself novel.
7. The empirical topic is not locked until the primary real-data frontier survives the pre-declared gates.

## 11. Decision labels

- `FRONTIER_SURVIVE`: primary real-data evidence meets the preregistered screening gates.
- `KILL_OR_REDESIGN`: one or more primary gates fail.
- `TOPIC_LOCK`: **not implied** by `FRONTIER_SURVIVE`; requires robustness and literature re-attack after the empirical result is known.
