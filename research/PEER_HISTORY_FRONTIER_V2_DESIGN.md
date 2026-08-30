# Peer-History Objective Misalignment v3

## Working title

**When Better Peer Matching Makes Worse Diagnoses**  
**Dynamic Peer Re-selection and Objective Misalignment in Retail Benchmarking**

한국어 작업 제목: **더 잘 맞는 유사점포가 더 나쁜 진단을 만들 때: 리테일 벤치마킹의 동적 Peer 재선정과 목적함수 불일치**

## 0. What v3 kills from v2

v3 does NOT claim that:

1. historical performance improves *structural comparability* merely because it lowers next-period benchmark MAE;
2. historical performance contamination mechanically lowering shock recovery is itself novel;
3. better predictive fit failing to imply better downstream decisions is novel;
4. adaptive baselines absorbing persistent change is novel;
5. peer selection or endogenous reference groups are novel.

Those claims are either not identified by the proposed data or have strong ancestors in retail benchmarking, reference-group selection, decision-focused learning, and drift-aware anomaly detection.

The remaining candidate is narrower: an algorithmic peer benchmark can endogenously move after persistent store deterioration because the focal store's contaminated history changes the peer set. Standard predictive validation can then prefer precisely those peer designs that are most diagnostically fragile.

## 1. Exact mechanism: dynamic peer re-selection

For focal store i, let P_i(0) be the clean peer set and B_i(0) its clean peer benchmark. Inject a known persistent negative log-shock of magnitude d > 0, so the evaluation outcome changes from Y_i to Y_i - d.

Let P_i(c) be the refreshed peer set after fraction c of the peer-construction history has been contaminated by the same deterioration, and let B_i(c) be its benchmark.

Detected_i(c) = [B_i(c) - (Y_i - d)] - [B_i(0) - Y_i]
              = d + B_i(c) - B_i(0).

Therefore

R_i(c) = Detected_i(c) / d
       = 1 + [B_i(c) - B_i(0)] / d.

Define Benchmark Absorption Ratio:

BAR_i(c) = 1 - R_i(c)
         = [B_i(0) - B_i(c)] / d.

Mandatory falsifier:

- If peers are frozen, P_i(c) = P_i(0), hence B_i(c) = B_i(0), R_i(c) = 1, BAR_i(c) = 0.
- Any attenuation in the proposed KNN-style design must therefore arise through benchmark movement induced by peer re-selection, not from the focal outcome shock by itself.

If the frozen-peer identity fails beyond numerical tolerance, the experiment is wrong.

## 2. Peer-turnover decomposition

For equal-weight K-nearest-neighbor benchmarks,

B_i(c) - B_i(0)
= (1/K) [sum_{j in New_i(c)} Y_j - sum_{j in Dropped_i(c)} Y_j],

where New_i(c)=P_i(c)\\P_i(0) and Dropped_i(c)=P_i(0)\\P_i(c).

Define

Turnover_i(c) = 1 - |P_i(c) intersection P_i(0)| / K.

The empirical mechanism requires:

contaminated history -> peer turnover -> downward benchmark movement -> lower recovery.

The algebra makes benchmark movement sufficient for recovery loss; turnover is the selection mechanism that explains where the benchmark movement came from.

## 3. Replace 'structural comparability' with predictive alignment

The v2 clean metric uses future outcomes:

CleanMAE_m = mean_i |B_i^m(0) - Y_i,future|.

This measures out-of-sample predictive benchmark alignment, NOT latent structural comparability. The paper must use that language consistently.

PredictiveAlignmentGain_m
= 100 * (MAE_context - MAE_m) / MAE_context.

A structural-comparability claim would require external or semi-synthetic latent ground truth and is not part of the primary empirical claim.

## 4. Objective misalignment

Let m index candidate peer-information designs.

Standard predictive selection chooses

m_pred = argmin_m CleanMAE_m.

Diagnostic risk is

D_m = integral_c |R_m(c)-1| dc,

for pre-declared deterioration magnitudes 5%, 10%, and 15% (20% only robustness).

Define useful designs:

M_useful = {m: PredictiveAlignmentGain_m >= max(5 percentage points, 0.5 * max_j PredictiveAlignmentGain_j)}.

Let

m_safe = argmin_{m in M_useful} D_m.

Selection Diagnostic Penalty:

SDP = D_{m_pred} - D_{m_safe}.

The central question is whether ordinary predictive validation of a retail peer benchmark selects a design with materially larger diagnostic risk than another design retaining substantial predictive value.

## 5. Primary falsifiable expectations

H1 — Predictive value: At least one history-using design produces >=5 percentage-point out-of-sample predictive alignment gain relative to context-only.

H2 — Frozen-peer falsifier: frozen-peer recovery equals 1 within numerical tolerance for every shock and contamination level.

H3 — Dynamic re-selection: contamination increases peer turnover and produces downward benchmark movement for the predictive-selected design; benchmark movement exactly accounts for recovery loss.

H4 — Objective misalignment: the predictive winner has materially larger diagnostic error than a useful safer design, with SDP >= 0.05 AUC as primary screening threshold.

H5 — Operational magnitude: H3-H4 appear at 5%, 10%, and/or 15% deterioration, not only 20%.

H6 — Decision consequence: dynamic re-selection changes false-negative underperformance flags and/or intervention-priority rankings relative to frozen peers or the safer useful design.

## 6. Primary empirical design

Primary dataset: Rossmann weekly retail panel from the public FEV collection.

- history H=24 weeks
- evaluation E=8 weeks
- K=15 peers
- shock delta = 5%, 10%, 15%; 20% robustness only
- contamination c = 0%, 25%, 50%, 75%, 100%
- history weights alpha = 0, .25, .50, .75, 1
- level/shape shares beta = 0, .25, .50, .75, 1
- raw-trajectory arm retained only as a reference

All donor histories and donor evaluation outcomes remain clean. Only the focal query store receives the known shock. This isolates how focal history changes its reference set.

## 7. Rolling-origin requirement

A single terminal split is insufficient. Primary evidence should aggregate at least four pre-declared rolling cutoffs separated by approximately one quarter where data allow.

For each cutoff, construct peers using only information available before its evaluation window. Report pooled results and cutoff-level dispersion.

KILL if objective misalignment is driven by one cutoff or flips direction in most other cutoffs.

## 8. Stronger primary gates

### KILL 1 — No predictive value
Max history-based PredictiveAlignmentGain < 5 percentage points across rolling origins.

### KILL 2 — Frozen control fails
Mean absolute frozen-peer recovery error > 1e-8 or the benchmark-movement decomposition fails numerical closure.

### KILL 3 — No dynamic diagnostic harm
Among useful history designs, 10% shock DiagnosticErrorAUC < 0.10 and no >=10 percentage-point recovery deviation at c=50% or 100%.

### KILL 4 — No objective misalignment
SDP < 0.05, or the safer design retains <50% of maximum predictive alignment gain.

### KILL 5 — No selection mechanism
Recovery falls but peer turnover/benchmark movement do not move in the predicted direction.

### KILL 6 — Only extreme shock
Effects appear only at 20% deterioration, not 5-15%.

### KILL 7 — Rolling-origin collapse
The central pattern is not directionally stable across pre-declared cutoffs.

### KILL 8 — Design sensitivity collapse
After primary survival, the qualitative result disappears for reasonable H in {12,24,52} or K in {5,15,30} rather than merely changing magnitude.

### KILL 9 — No decision consequence
Realistic underperformance thresholds and top-N intervention rankings are essentially unchanged. Managerial contribution must then be sharply downgraded.

## 9. Mandatory controls

1. Context-only peer design.
2. Frozen-peer version of every dynamic design.
3. History-level-only arm.
4. Centered-shape-only arm.
5. Level/shape hybrids.
6. Raw-history reference arm.
7. Multiple rolling origins.
8. Fixed donor pool; focal-only intervention.
9. Threshold/ranking decision analysis.

## 10. Interpretation constraints

- Do not call lower future MAE 'structural comparability'.
- Do not claim historical outcomes should never be used.
- Do not claim dynamic peers are inherently bad.
- Do not claim objective misalignment itself is novel; decision-focused learning is an explicit ancestor.
- Do not claim adaptive-reference contamination itself is novel; concept-drift/anomaly-detection work is an explicit ancestor.
- Do not claim endogenous reference selection is novel; organizational reference-group work is an explicit ancestor.
- The contribution, if it survives, is the retail peer-benchmarking mechanism connecting predictive model selection, focal-history contamination, algorithmic peer re-selection, benchmark absorption, and downstream underperformance decisions.

## 11. Decision labels

- KILL: any hard gate fails with no defensible narrower claim.
- REDESIGN: mechanism survives but objective-misalignment or decision-consequence claim fails.
- V3_SURVIVE: all primary gates survive on real retail data across rolling origins.
- TOPIC_LOCK: only after V3_SURVIVE, design sensitivity, and a post-result literature re-attack.
