# The Conditions for Kriging Performance: A Systematic Analysis of When Spatial Prediction Succeeds and When It Fails

## From Mathematical Guarantees to Practical Diagnostics

---

**Preface.** Ordinary kriging carries the label "best linear unbiased predictor" (BLUP), a designation that suggests unconditional optimality. The BLUP property is, however, strictly conditional: it holds only under intrinsic stationarity with a correctly specified variogram model. When these conditions degrade, kriging predictions can underperform even the sample mean. This lecture examines the full performance spectrum, from conditions that produce R² exceeding 0.85 to regimes where kriging fails entirely, and provides a diagnostic framework for distinguishing among them before committing to a model. The analysis draws on the foundational geostatistical literature (Isaaks & Srivastava, 1989; Cressie, 1993; Chilès & Delfiner, 2012), validated through controlled benchmarks against the Walker Lake, Meuse River, and California earthquake datasets, and incorporates practical insights from the GeostatsGuy lecture series (Pyrcz, 2024) on spatial data configuration effects.

---

## 1. The Kriging Equations and the Source of the BLUP Guarantee

### 1.1 The Weighted Average Formulation

Given n observations Z(s₁), ..., Z(sₙ) at spatial locations s₁, ..., sₙ, ordinary kriging predicts the value at an unobserved location s₀ as a weighted linear combination:

```
    Ẑ(s₀) = Σᵢ λᵢ Z(sᵢ)    subject to    Σᵢ λᵢ = 1
```

The weights λᵢ minimize the expected squared prediction error E[(Ẑ(s₀) − Z(s₀))²] under the unbiasedness constraint Σᵢ λᵢ = 1. The solution is the kriging system:

```
    ┌                             ┐ ┌      ┐   ┌          ┐
    │ γ(s₁,s₁) ... γ(s₁,sₙ)   1  │ │ λ₁   │   │ γ(s₁,s₀) │
    │    ⋮     ⋱     ⋮       ⋮  │ │  ⋮   │ = │    ⋮     │
    │ γ(sₙ,s₁) ... γ(sₙ,sₙ)   1  │ │ λₙ   │   │ γ(sₙ,s₀) │
    │   1     ...    1        0  │ │ −μ   │   │    1     │
    └                             ┘ └      ┘   └          ┘

    Γ λ = γ₀      →      λ = Γ⁻¹ γ₀
```

where γ(sᵢ, sⱼ) is the semivariogram for the separation vector between points i and j, and μ is the Lagrange multiplier. The minimized error variance is:

```
    σ²_K(s₀) = Σᵢ λᵢ γ(sᵢ, s₀) − μ
```

### 1.2 Decomposing the Kriging Variance

The kriging variance can be partitioned into three terms that reveal the fundamental trade-off governing kriging performance (Pyrcz, 2024; Markvoort & Deutsch, 2024):

```
    σ²_K = Σᵢ Σⱼ λᵢ λⱼ C(sᵢ, sⱼ)   −   2 Σᵢ λᵢ C(sᵢ, s₀)   +   C(0)
           ──────────────────────      ──────────────────      ────
           Data redundancy penalty     Data proximity benefit   Population
           (higher = worse)           (higher = better)        variance
```

The first term, a quadratic form in the weights involving data-to-data covariances, penalizes redundancy: if two data points are close together (highly correlated), assigning large weight to both inflates the variance. The second term, involving data-to-unknown covariances, rewards proximity: data points near the prediction location have higher covariance with the unknown and receive larger weights. The third term, the population variance C(0) = σ²_total, represents the irreducible error when no spatial data are available.

This decomposition explains the three fundamental factors governing kriging weights (Pyrcz, 2024):

**Closeness.** Data points nearer to the prediction location s₀ have greater spatial correlation C(sᵢ, s₀) with the unknown, which increases the weight λᵢ through the second term. This is the intuitive "nearby points matter more" behavior shared by all spatial interpolators.

**Redundancy.** Data points that are close to each other share information, which increases the penalty term λᵢ λⱼ C(sᵢ, sⱼ). Kriging responds by distributing weight across the cluster rather than concentrating it on any single point. This is the property that distinguishes kriging from inverse-distance weighting (IDW): IDW assigns weight solely by distance, ignoring clustering, while kriging automatically down-weights redundant clustered data.

**Spatial continuity.** The variogram model determines how quickly spatial correlation decays with distance and direction. A long range means data influence extends further, spreading weights across more points. A short range means only the nearest few data points receive meaningful weight. Anisotropy directs weight preferentially along the direction of greater continuity.

```
    ┌──────────────────────────────────────────────────────────────────┐
    │         THREE-PART TRADE-OFF IN KRIGING WEIGHT ALLOCATION          │
    │                                                                    │
    │                    ┌──────────────┐                                │
    │                    │  Prediction   │                               │
    │                    │  location s₀  │                               │
    │                    └──────┬───────┘                                │
    │                           │                                        │
    │              CLOSENESS    │    CLOSENESS                           │
    │           ┌───────────────┼───────────────┐                        │
    │           ▼               │               ▼                        │
    │      ┌─────────┐         │         ┌─────────┐                     │
    │      │  λ₁=0.4 │         │         │  λ₂=0.4 │                     │
    │      │  s₁     │         │         │  s₂     │                     │
    │      └────┬────┘         │         └────┬────┘                     │
    │           │              │              │                          │
    │           └──REDUNDANCY──┼──────────────┘                          │
    │                          │                                         │
    │  If s₁ and s₂ are close together: λ₁ + λ₂ ≈ 0.6, not 0.8          │
    │  because kriging recognizes they carry shared information.         │
    │  IDW would assign 0.8, over-weighting the cluster.                 │
    └──────────────────────────────────────────────────────────────────┘
```

### 1.3 Three Properties With Practical Consequences

The kriging system possesses three properties that are central to understanding its behavior in practice:

**Property 1: Configuration dependence, not value dependence.** The weights λᵢ and the kriging variance σ²_K(s₀) depend exclusively on the spatial configuration of data points and the variogram model. The data values Z(sᵢ) affect only the prediction Ẑ(s₀), never the weights or the variance. This property is both a strength — it enables pre-survey optimization of sampling designs without requiring data values — and a limitation — it means kriging cannot adapt its weighting strategy to local features such as sharp gradients or discontinuities that differ from the global variogram.

**Property 2: Exact interpolation only without a nugget effect.** Without a nugget effect (C₀ = 0), kriging is an exact interpolator: as the prediction location s₀ approaches a data location sᵢ, the weight λᵢ → 1 and the kriging variance σ²_K(s₀) → 0. With a non-zero nugget, kriging deliberately does not honor the data exactly. This behavior is not a defect: when the nugget represents genuine measurement error, the original observation is not the true value, and a smoothed prediction at the data location improves accuracy over the noisy measurement (Krivoruchko, 2011).

```
    ┌──────────────────────────────────────────────────────────────────┐
    │       EXACT INTERPOLATION VERSUS SMOOTHED PREDICTION               │
    │                                                                    │
    │  Nugget = 0 (exact):             Nugget > 0 (smoothing):           │
    │                                                                    │
    │  Value                            Value                            │
    │   │   ●                            │   ●                           │
    │   │  ╱ ╲                           │  ╱ ✕                          │
    │   │ ╱   ╲    ← surface passes     │ ╱    ✕     ← surface does     │
    │   │╱     ╲     through every      │╱       ✕    not pass through  │
    │   ●───────●    observation        ●─────────●   observations       │
    │       Distance                         Distance                    │
    │                                                                    │
    │  Kriging variance at sᵢ = 0      Kriging variance at sᵢ = C₀     │
    │  Risk: overfitting               Benefit: noise filtering         │
    │  Prediction SE discontinuous     Prediction SE smooth and realistic│
    └──────────────────────────────────────────────────────────────────┘
```

The practical implication is important: **including a correctly estimated nugget effect produces prediction standard errors that are continuous across the domain, whereas forcing zero nugget creates artificial discontinuities where the error jumps to zero at each data location**. Paradoxically, a non-zero nugget can improve overall predictive accuracy because it prevents the model from overfitting individual noisy measurements.

**Property 3: Conditional bias toward the mean.** Kriging systematically underpredicts large values and overpredicts small values — the Krige smoothing effect. The magnitude of this conditional bias increases with the local kriging variance: in data-sparse regions, predictions regress more strongly toward the global mean. This is an inherent consequence of minimum-variance linear prediction and should be considered a property of the estimator rather than a defect. For applications where the distribution of predicted values matters (resource estimation, risk assessment), sequential Gaussian simulation rather than kriging should be used to reproduce the target histogram and variogram.

---

## 2. Data Configuration Effects: String Effect and Screening

### 2.1 The String Effect

When data points are arranged in a linear string — a common occurrence in borehole surveys where measurements are taken at regular depth intervals along a vertical well — kriging exhibits a counterintuitive behavior: the endpoint samples receive disproportionately large weights while interior samples receive near-zero weights (Deutsch, 1993; Markvoort & Deutsch, 2024).

```
    ┌──────────────────────────────────────────────────────────────────┐
    │                     THE STRING EFFECT                              │
    │                                                                    │
    │  Borehole samples (•)         Kriging weights for prediction       │
    │  at equal intervals:          at center of string:                 │
    │                                                                    │
    │      •  λ=0.40                                                     │
    │      •  λ=0.05                                                     │
    │      •  λ=0.02                Interior points are "screened"       │
    │      •  λ=0.03                by the endpoints                    │
    │      •  λ=0.05                                                     │
    │      •  λ=0.40               Endpoints receive 80% of weight       │
    │                                                                    │
    │  Cause: Under the infinite stationary domain assumption, endpoints │
    │  have data on only one side and are therefore "less redundant."    │
    │  Interior points, surrounded by neighbors on both sides, are       │
    │  considered information-rich and therefore redundant.              │
    │                                                                    │
    │  Risk: In non-stationary domains (e.g., grade enrichment at a      │
    │  geological contact), endpoint-dominated weights cause systematic  │
    │  overestimation or underestimation bias.                           │
    └──────────────────────────────────────────────────────────────────┘
```

The string effect is stronger in ordinary kriging than in simple kriging (because ordinary kriging must also estimate the mean) and is reduced by a non-zero nugget effect (which diminishes the perceived redundancy of interior points) and by anisotropy with a longer range perpendicular to the string. Practical mitigation includes limiting the number of samples per borehole to two or three and applying search neighborhood restrictions.

### 2.2 Negative Weights and the Screening Effect

A related phenomenon is the screening effect: a data point located behind a closer neighbor can receive a negative weight, even when the variogram is positive-definite and all covariances are positive. This occurs because the closer point renders the farther point redundant, and the negative weight effectively subtracts the redundant information to prevent double-counting.

Negative weights enable extrapolation of spatial trends beyond the range of the data — for example, predicting values outside the convex hull of observations — but they can produce physically impossible negative estimates for strictly positive variables such as grade or concentration. Standard mitigation strategies include resetting negative estimates to zero, limiting the search radius, and restricting the number of data points per octant.

### 2.3 The Screening Interpretation of the Nugget Effect

The nugget effect has a direct geometric interpretation in terms of screening behavior. As the nugget effect increases relative to the sill, the screening effect diminishes: nearby data points no longer dominate the prediction because the nugget acts as a spatially uncorrelated noise floor that weakens the apparent redundancy among proximate observations. In the limit of pure nugget effect, kriging assigns equal weight to all data points regardless of proximity, predicting the global mean everywhere — the extreme case of the smoothing behavior discussed in Section 1.3.

```
    ┌──────────────────────────────────────────────────────────────────┐
    │     NUGGET EFFECT AND SCREENING: THE WEIGHT DISTRIBUTION           │
    │                                                                    │
    │  Nugget = 0% of sill:            Nugget = 50% of sill:            │
    │  ┌─────────────────────┐         ┌─────────────────────┐          │
    │  │ λ₁ = 0.65           │         │ λ₁ = 0.35           │          │
    │  │  (nearest neighbor) │         │  (nearest neighbor) │          │
    │  │ λ₂ = 0.20           │         │ λ₂ = 0.25           │          │
    │  │ λ₃ = 0.10           │         │ λ₃ = 0.22           │          │
    │  │ λ₄ = 0.05           │         │ λ₄ = 0.18           │          │
    │  └─────────────────────┘         └─────────────────────┘          │
    │  Strong screening:               Weak screening:                   │
    │  nearest point dominates         weights are more uniform          │
    │                                                                    │
    │  Nugget = 100% of sill (pure nugget):                             │
    │  All λᵢ = 1/n → prediction equals global mean                     │
    └──────────────────────────────────────────────────────────────────┘
```

---

## 3. The Nugget Effect as the Fundamental Performance Ceiling

### 3.1 Physical Sources of the Nugget

The nugget effect C₀ = σ²_nugget in the variogram model:

```
    γ(h) = C₀ + C₁ · f(h; a)          where f(0) = 0, f(∞) = 1
           ───   ─────────────────
         nugget   structured component
```

aggregates three physically distinct sources of variance operating at distances shorter than the minimum data-pair separation (Chilès & Delfiner, 2012, Chapter 4):

**Measurement error.** Each observation Z(sᵢ) carries independent noise with variance σ²_me, originating from instrument precision limits, sampling handling variability, and analytical laboratory error. This component is irreducible by any interpolation method: the signal cannot be predicted more accurately than it was measured.

**Microscale variability.** Spatial structure at scales finer than the closest sample pair is aliased into the nugget. If the minimum sample separation is 10 m, any variation at 1–5 m scales contributes to the apparent discontinuity at the origin. The remedy is denser sampling, not a better model.

**Model misspecification.** A variogram model that poorly fits the short-lag behavior of the experimental variogram attributes genuine spatial structure to the nugget, artificially inflating C₀ and lowering the R² ceiling. This is the most dangerous contributor because it masquerades as measurement error or microscale variability, concealing recoverable spatial signal.

### 3.2 Derivation of the R² Ceiling

The total variance decomposes as:

```
    Var[Z(s)] = σ²_total = C₀ + C₁ = σ²_nugget + σ²_structured
```

Any linear predictor recovers at most the structured component C₁. The nugget component C₀ is, by definition, spatially uncorrelated and unpredictable from neighboring observations. The fundamental performance bound is:

```
    R²_max = C₁ / (C₀ + C₁) = 1 − C₀ / (C₀ + C₁) = 1 − (nugget fraction)
```

This R² ceiling represents the maximum proportion of variance achievable by any linear spatial predictor given the data's intrinsic noise level. The table below maps nugget fractions to expected performance bands, validated against both the geostatistical literature (Isaaks & Srivastava, 1989, Chapter 16; Webster & Oliver, 2007, Chapter 5) and the benchmark results reported here:

```
    ┌──────────────────────────────────────────────────────────────────┐
    │      NUGGET FRACTION AND EXPECTED Kriging PERFORMANCE              │
    │                                                                    │
    │  Nugget %    Ceiling    Realistic R²    Verdict       Example      │
    │  ────────    ───────    ────────────    ───────       ───────      │
    │                                                                    │
    │   < 10%      > 0.90     0.70 – 0.90    Excellent     SRTM DEM,    │
    │                                                      groundwater   │
    │  10–25%      0.75–0.90   0.50 – 0.75    Good          Soil carbon  │
    │  25–50%      0.50–0.75   0.25 – 0.50    Moderate      Walker Lake  │
    │  50–75%      0.25–0.50   0.10 – 0.25    Poor          Sparse soil  │
    │  > 75%        < 0.25      < 0.10        Very poor     Weather radar│
    │  > 90%        < 0.10      ≈ 0           Useless       CA quakes    │
    └──────────────────────────────────────────────────────────────────┘

    The "Realistic R²" column accounts for variogram estimation error and
    model misspecification, which typically consume 20–30% of the ceiling.
```

The benchmark results illustrate the diagnostic value of comparing observed R² against the ceiling:

```
    Dataset       Nugget%    Ceiling    Observed R²    Gap    Interpretation
    ────────      ───────    ───────    ───────────    ───    ──────────────
    Walker Lake    28%        0.72        0.23         0.49   Model misspec.;
                                                              consider GP

    Meuse Zinc     15%        0.85       −0.08         0.93   Model structure
                                                              wrong

    CA Quakes      89%        0.11       −0.11         0.22   No spatial
                                                              structure
```

The gap between observed R² and the ceiling is as diagnostic as the ceiling itself. A gap below 0.10 indicates the model is near-optimal; a gap of 0.10–0.30 is standard, reflecting variogram estimation uncertainty and limited sample size; a gap of 0.30–0.50 suggests significant model misspecification; and a gap exceeding 0.50 indicates the stationarity assumption is violated, requiring a switch to universal kriging, regression kriging, or a non-stationary Gaussian process.

---

## 4. The Six Conditions for Kriging Success

The conditions that govern kriging performance form a hierarchy: each depends on the preceding ones, and a failure at any level propagates downstream. The framework draws on the classical geostatistical literature (Journel & Huijbregts, 1978; Cressie, 1993; Goovaerts, 1997) and has been validated against the controlled benchmarks reported here.

### Condition 1: The Phenomenon Is a Continuous Spatial Random Function

Kriging assumes {Z(s) : s ∈ D} is a random function defined continuously over the spatial domain D. At every point s ∈ D, a random variable Z(s) exists conceptually. Nearby values Z(s) and Z(s + h) are correlated in a manner that depends only on the separation vector h, and the correlation decays with distance.

Phenomena that violate this condition include earthquake magnitudes (point events governed by fault geometry rather than a continuous field), species presence or absence at survey plots (zero-inflated, compositional data), and mineral grades crossing lithological boundaries (discontinuous at contacts). For these phenomena, kriging can produce numerically valid output — it will always produce a smooth surface — but the surface lacks physical meaning because the continuity assumption on which kriging rests does not hold.

### Condition 2: Second-Order Stationarity Holds Approximately

Intrinsic stationarity requires:

```
    E[Z(s + h) − Z(s)] = 0                    (constant mean)
    Var[Z(s + h) − Z(s)] = 2γ(h)              (variogram depends only on h)
```

Real data never satisfy these conditions exactly. The relevant question is whether the violation magnitude is sufficient to degrade predictions. Three common violation patterns and their detection thresholds:

**Large-scale trend.** If E[Z(s)] = β₀ + β₁x + β₂y + ..., ordinary kriging assumes a constant mean, and predictions become biased where the trend surface deviates from the global mean. Detection: fit a polynomial surface and compute the F-test. If R² > 0.05 and p < 0.05, detrend before kriging. Trends explaining 5–20% of variance cause modest bias correctable by detrending; trends exceeding 20% require universal kriging with covariates, as polynomial detrending cannot capture non-polynomial drift.

**Heteroscedasticity.** If Var[Z(s)] varies systematically across the domain, the variogram estimated from all data pairs represents an average that fails to characterize any sub-region adequately. Detection: plot absolute residuals against coordinates. A Normal-Score Transform partially mitigates heteroscedasticity by enforcing uniform variance in the transformed space.

**Zonal anisotropy.** If the variogram sill differs by direction, standard geometric anisotropy models (which assume constant sill) are misspecified. Detection: examine directional variograms for convergence to different sills.

### Condition 3: The Variogram Is Well-Estimated

A well-estimated variogram requires sufficient data at each lag distance. The rule of 30 point pairs per lag bin maps to minimum sample sizes (Webster & Oliver, 2007):

```
    ┌──────────────────────────────────────────────────────────────────┐
    │         MINIMUM SAMPLE SIZE FOR RELIABLE VARIOGRAPHY               │
    │                                                                    │
    │  n         Omnidirectional    Directional       Anisotropy         │
    │  ──        ────────────────    ───────────       ──────────        │
    │  < 30      Unreliable          Impossible         Impossible       │
    │  30–50     Rough               Unreliable         Impossible       │
    │  50–100    Adequate            Marginal           Unreliable       │
    │  100–200   Good                Adequate           Marginal         │
    │  200–500   Excellent           Good               Adequate         │
    │  > 500     Excellent           Excellent          Good             │
    └──────────────────────────────────────────────────────────────────┘
```

The behavior of the experimental variogram near the origin is disproportionately influential. The first two or three lag bins effectively determine the nugget estimate and the short-range behavior, which control the kriging weights for the nearest data points — the most heavily weighted points in any prediction. The most common failure mode in automated variography is insufficient close pairs to constrain the near-origin behavior, producing an unreliable nugget estimate that cascades into poor predictions at all locations.

### Condition 4: Spatial Coverage Avoids Clustered Gaps

The sampling design affects both variogram estimation quality and prediction accuracy. Three archetypal designs with their consequences:

```
    ┌──────────────────────────────────────────────────────────────────┐
    │              SAMPLING DESIGN AND Kriging PERFORMANCE               │
    │                                                                    │
    │  REGULAR GRID              RANDOM                  CLUSTERED       │
    │  ══════════════            ════════                ═════════       │
    │  •  •  •  •  •            •    •  •               •••             │
    │  •  •  •  •  •               •       •            •••  •          │
    │  •  •  •  •  •            •    •     •               •••          │
    │  •  •  •  •  •              •  •   •               •     •••      │
    │  •  •  •  •  •            •     •              •••          •••   │
    │                                                                    │
    │  Variogram: Excellent      Variogram: Good       Variogram: Poor   │
    │  (all lags represented)    (most lags ok)        (lag gaps)        │
    │                                                                    │
    │  Prediction: Uniform       Prediction: Slightly  Prediction: Good  │
    │  accuracy everywhere       worse in sparse       in clusters,       │
    │                            regions               poor between       │
    └──────────────────────────────────────────────────────────────────┘
```

Clustered sampling is dangerous because it can produce misleadingly optimistic cross-validation when folds are randomly constructed rather than spatially blocked. The engine addresses this via KMeans-based spatial folds: if cross-validation performance is acceptable on spatial folds, the predictions are genuinely reliable. If spatial CV performance collapses while random CV appears adequate, clustered sampling is masking poor spatial coverage.

A practical density guideline: at least three to five points should lie within one practical range of any prediction location. For Walker Lake (range approximately 59 grid units, domain approximately 260 × 300), the average point density of approximately 0.006 points per unit squared yields approximately 63 points within range — above the threshold. For Meuse (range approximately 4,543 m, domain approximately 2,800 × 3,900 m), the density yields approximately 910 points within range, confirming that data sparsity is not the limiting factor for Meuse.

### Condition 5: The Variogram Model Family Matches the Physical Process

With nine variogram models searched by Optuna, model selection is automated, but the optimizer can only choose among the families it is given. The critical property distinguishing model families is their behavior near the origin, which directly controls the weight assigned to the nearest data points:

```
    ┌──────────────────────────────────────────────────────────────────┐
    │        VARIOGRAM MODEL BEHAVIOR NEAR THE ORIGIN                    │
    │                                                                    │
    │  Model          γ(h) near h=0        Interpretation                │
    │  ─────          ──────────────        ──────────────               │
    │                                                                    │
    │  Nugget-only    γ(0⁺) = C₀           White noise + microscale     │
    │                                                                    │
    │  Exponential    γ(h) ~ C₀ + C₁h/a    Linear at origin             │
    │                                       Rough, continuous process    │
    │                                                                   │
    │  Spherical      γ(h) ~ C₀ + 3C₁h/2a  Linear at origin             │
    │                                       Same class as exponential   │
    │                                                                   │
    │  Gaussian       γ(h) ~ C₀ + C₁h²/a²  Parabolic at origin          │
    │                                       Very smooth process          │
    │                                                                   │
    │  Stable         γ(h) ~ C₀ + C₁h^α     α ∈ (0,2]                   │
    │                                       Interpolates between above  │
    └──────────────────────────────────────────────────────────────────┘
```

The recommended model families for common physical phenomena are:

| Phenomenon | Recommended families | Reason |
|------------|---------------------|--------|
| Elevation / topography | Gaussian, Stable (α > 1.5) | Smooth at short scales, differentiable |
| Groundwater head | Gaussian, Stable (α > 1.5) | Smooth, differentiable surface |
| Soil properties | Exponential, Spherical | Rough, heterogeneous at fine scales |
| Ore grades | Exponential, Spherical | Typically rough, irregular |
| Precipitation | Stable (α ≈ 1) | Intermediate roughness |
| Air pollution | Exponential, Rational-Quadratic | Rough, multiscale |
| Temperature | Gaussian | Smooth at synoptic scales |

### Condition 6: The Nugget Does Not Dominate

If the nugget fraction exceeds approximately 50%, kriging cannot produce useful predictions regardless of sample size or variogram quality — the ceiling is simply too low. However, a low nugget is necessary but not sufficient. Two scenarios with identical nugget fractions illustrate the difference:

```
    Scenario A:                               Scenario B:
    n = 500, regular grid                     n = 50, clustered
    True model: Gaussian                      True model: Exponential
    Fitted model: Gaussian (correct)          Fitted model: Gaussian (wrong)
    Nugget = 10%, Ceiling = 0.90              Nugget = 10%, Ceiling = 0.90

    Observed R² ≈ 0.85 (near ceiling)         Observed R² ≈ 0.40 (far below ceiling)
    → Model is correct                        → Ceiling misleading; model wrong
```

Both scenarios share the same theoretical ceiling, yet Scenario A achieves 85% of it while Scenario B achieves only 40%. The gap isolates the impact of sample size, spatial coverage, and model correctness, none of which the nugget fraction alone captures.

---

## 5. Quantitative Performance Expectations

Based on the six conditions, a predictive framework for expected kriging performance given data characteristics emerges:

```
┌──────────────────────────────────────────────────────────────────────────────┐
│              EXPECTED Kriging R² BY COMBINED CONDITIONS                         │
│                                                                                │
│  Regime  C1       C2       C3         C4        C5      C6      Expected      │
│          Field    Station- Variogram  Sampling  Model   Nugget  R²             │
│                   arity    Quality    Coverage  Match   < 50%                  │
│  ──────  ───────  ───────  ─────────  ────────  ──────  ─────  ────────       │
│  A       ✓        ✓        Excellent  Grid      Exact   ✓      0.85+          │
│  B       ✓        ✓        Good       Random    Close   ✓      0.60–0.80      │
│  C       ✓        ~        Adequate   Random    Close   ✓      0.40–0.65      │
│          (mild trend)                                                          │
│  D       ✓        ~        Marginal   Clustered Wrong   ✓      0.15–0.40      │
│          (strong trend)                                                        │
│  E       ✓        ✗        Good       Any       Any     ✗      < 0.15         │
│          (non-stationary)                                      (high nugget)   │
│  F       ✗        N/A      N/A        N/A       N/A     ✗      < 0.10         │
│          (point process)                                                       │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                                │
│  Benchmarks placed in this framework:                                          │
│                                                                                │
│  Walker Lake (n=448): Regime C → expected 0.40–0.65, observed 0.23             │
│    Underperformance attributable to stable model with α=0.41 not matching      │
│    the true variogram; exponential or spherical models would be more           │
│    appropriate for this roughness regime.                                       │
│                                                                                │
│  Meuse Zinc (n=155):  Regime D → expected 0.15–0.40, observed −0.08           │
│    Non-stationary mean due to river floodplain geometry pushes below            │
│    regime D. Correction: universal kriging with log(distance-to-river)          │
│    as external drift (Pebesma, 2004).                                           │
│                                                                                │
│  CA Quakes (n=1072):  Regime F → expected <0.10, observed −0.11               │
│    Earthquake magnitudes as a marked point process along fault systems          │
│    lack the continuous-field structure that kriging requires.                   │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 6. Optimal Sampling Design Using the Kriging Variance

### 6.1 Pre-Survey Optimization

A distinctive property of kriging enables a capability that few other predictive methods share: the kriging variance at any location depends exclusively on the data configuration and the variogram model, never on the data values. This means an optimal sampling design can be computed before any data are collected, simply by minimizing the expected kriging variance over the domain for a candidate set of survey locations.

Spatial Simulated Annealing (SSA) is the standard algorithm for this optimization (Brus & Heuvelink, 2007; Melles et al., 2011). Candidate survey locations are perturbed iteratively; designs that reduce the mean or maximum kriging variance are accepted, and worsening designs are accepted with a decreasing probability to escape local minima. The objective function is typically:

```
    Objective = minimize [ mean(σ²_K(s)) ]  or  minimize [ max(σ²_K(s)) ]
                       s ∈ D                               s ∈ D
```

The mean kriging variance criterion produces designs that minimize average prediction uncertainty across the domain, appropriate for mapping applications. The maximum kriging variance criterion (minimax) identifies locations that minimize the worst-case uncertainty, appropriate for regulatory compliance where no location may exceed a threshold.

### 6.2 The Universal Kriging Extension

When auxiliary variables are incorporated through universal kriging or kriging with external drift, the optimal design must balance two competing objectives: coverage in feature space (for accurate estimation of the regression coefficients relating the target variable to covariates) and coverage in geographic space (for interpolation of the spatially correlated residuals). The spatially averaged universal kriging variance incorporates both components:

```
    σ²_UK = σ²_trend + σ²_residual
```

where σ²_trend is the variance of the estimated regression coefficients propagated to prediction locations, and σ²_residual is the ordinary kriging variance applied to the regression residuals. When residual spatial autocorrelation is weak or the sample size is small, optimal distribution in feature space dominates, and the design should prioritize covering the range of covariate values. When residual spatial autocorrelation is strong, geographic regularity becomes more important, and the design approaches a spatial grid.

### 6.3 Practical Guidelines

For a fixed budget of n sampling locations, the following hierarchy maximizes kriging prediction accuracy. First, ensure at least 100–150 points for reliable omnidirectional variography; 200–500 if anisotropy detection is required. Second, distribute points to cover the spatial domain uniformly, avoiding large gaps that exceed the practical range. Third, allocate a subset of points (10–20%) as close pairs (separation well below the expected range) to constrain the short-lag variogram and produce a reliable nugget estimate. Fourth, if covariates are available, ensure the sampling design spans the full range of each covariate, not merely the geographic extent.

---

## 7. Diagnostic Decision Framework

The six conditions, the nugget ceiling, and the gap analysis combine into a decision framework for the practicing geostatistician:

```
                                ┌──────────────────┐
                                │ Is the phenomenon │
                                │ a continuous      │
                                │ spatial field?    │
                                └────────┬─────────┘
                                         │
                              ┌──────────┴──────────┐
                              │ NO                   │ YES
                              ▼                      ▼
                    ┌──────────────────┐  ┌──────────────────────┐
                    │ Kriging is not   │  │ Sample size n ≥ 50?  │
                    │ the correct tool.│  └──────────┬───────────┘
                    │ Use: KDE, point  │             │
                    │ process models,  │  ┌──────────┴──────────┐
                    │ or non-spatial   │  │ NO        │ YES     │
                    │ ML methods.      │  ▼           ▼         │
                    └──────────────────┘ ┌────────┐ ┌──────────┐│
                                         │Collect │ │Fit       ││
                                         │more    │ │variogram ││
                                         │data or │ │and       ││
                                         │use     │ │compute   ││
                                         │expert  │ │R² ceiling││
                                         │prior   │ └────┬─────┘│
                                         └────────┘      │      │
                                                  ┌──────┴──────┐│
                                                  │Ceiling>0.5?  ││
                                                  └──────┬──────┘│
                                                         │       │
                                              ┌──────────┴──────┐│
                                              │ NO    │ YES      ││
                                              ▼       ▼          ││
                                        ┌──────────┐┌──────────┐││
                                        │Kriging   ││Run krige │││
                                        │cannot    ││and check │││
                                        │recover   ││gap to    │││
                                        │meaningful││ceiling   │││
                                        │signal.   │└────┬─────┘││
                                        │Consider: │     │      ││
                                        │• denser  │     ▼      ││
                                        │  sampling│┌──────────┐││
                                        │• different││Gap<0.3?  │││
                                        │  variable│└────┬─────┘││
                                        └──────────┘     │      ││
                                                  ┌──────┴─────┐││
                                                  │ YES │ NO   │││
                                                  ▼     ▼      │││
                                            ┌────────┐┌───────┐│││
                                            │Kriging ││Consider│││
                                            │works   ││UK with │││
                                            │well.   ││drift,  │││
                                            │R²>0.5  ││GP with │││
                                            │expected││non-stat│││
                                            │        ││kernel, │││
                                            │        ││or more │││
                                            │        ││covari- │││
                                            │        ││ates.   │││
                                            └────────┘└───────┘│││
                                                                 │││
              ┌──────────────────────────────────────────────────┘││
              │ Benchmarks placed in this framework:               ││
              │                                                    ││
              │ Walker Lake: n=448 ✓ → ceiling 0.72 ✓ → gap 0.49 ✗││
              │  → "Consider different model family or GP"         ││
              │                                                    ││
              │ Meuse Zinc: n=155 ✓ → ceiling 0.85 ✓ → gap 0.94 ✗ ││
              │  → "Universal kriging with covariates required"    ││
              │                                                    ││
              │ CA Quakes: n=1072 ✓ → ceiling 0.11 ✗              ││
              │  → "Kriging cannot recover signal; change method"  ││
              └────────────────────────────────────────────────────┘
```

---

## 8. Summary Performance Table

```
┌──────────────────────────────────────────────────────────────────────────────┐
│          KRIGING PERFORMANCE EXPECTATIONS — QUICK REFERENCE                     │
│                                                                                 │
│  Scenario                                    Expected    Minimum    Example    │
│                                              R²          n                    │
│  ────────────────────────────────────────    ───────     ────────    ───────   │
│                                                                                 │
│  Smooth field, dense regular sampling,       > 0.85      > 200      SRTM DEM   │
│  Gaussian variogram, low nugget                                              │
│                                                                                 │
│  Moderate roughness, random sampling,        0.60–0.80    > 100      Walker    │
│  well-estimated exponential variogram                                  Lake    │
│                                                                                 │
│  Rough field, sparse but adequate sampling,  0.40–0.60    50–100     Soil      │
│  moderate nugget, adequate variography                                 survey   │
│                                                                                 │
│  Strong trend, clustered sampling,           0.15–0.40    50–150     Meuse     │
│  non-stationary mean (ordinary kriging)                                Zinc     │
│                                                                                 │
│  High nugget, noisy measurements,            0.10–0.25    > 100      Low-cost  │
│  weak spatial signal                                                   sensors  │
│                                                                                 │
│  Point process, no continuous field,         < 0.10       Any        Earth-    │
│  dominant nugget                                                       quakes   │
│                                                                                 │
│  Colinear or nearly degenerate geometry      < 0.05       Any        S11 test  │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 9. Conclusion

Kriging is a specialized tool designed for a specific class of problems: the prediction of continuous spatial random functions from point observations, under the assumption that spatial correlation decays with distance in a manner estimable from the data. When these conditions hold — when the phenomenon is a genuine continuous field, when stationarity is approximately satisfied, when the variogram is well-estimated from adequate data with proper coverage, and when the nugget effect does not dominate — kriging is the optimal linear predictor, and automated pipelines achieve R² values of 0.60–0.90 without expert intervention.

When the conditions degrade, kriging fails in diagnostically informative ways. A high nugget fraction signals weak or absent spatial structure. A large gap between observed R² and the R² ceiling signals that the model structure, not the parameter estimates, is incorrect. A negative R² signals that kriging is fundamentally inappropriate. The data configuration effects discussed here — the string effect where endpoints dominate linear arrays, the screening effect where nearby points render distant ones redundant, and the smoothing behavior that filters noise at the cost of conditional bias toward the mean — are not algorithmic defects but direct mathematical consequences of the minimum-variance objective. Understanding these effects enables the practitioner to recognize when kriging output is physically meaningful and when it is merely numerically valid.

The six conditions, nugget ceiling, and gap analysis presented here constitute a systematic framework for making that judgment. A pipeline that reports R² alongside the nugget-derived ceiling and the configuration-derived kriging variance provides the three quantities necessary to distinguish between a model that needs refinement and a problem that requires a fundamentally different approach.

---

## References

1. Brus, D.J. & Heuvelink, G.B.M. (2007). Optimization of sample patterns for universal kriging of environmental variables. *Geoderma*, 138(1–2), 86–95.

2. Chilès, J.P. & Delfiner, P. (2012). *Geostatistics: Modeling Spatial Uncertainty* (2nd ed.). Wiley. Chapter 4: The nugget effect and its physical interpretations.

3. Cressie, N. (1993). *Statistics for Spatial Data* (Revised ed.). Wiley. Chapters 2–3: Spatial stationarity and the kriging equations.

4. Deutsch, C.V. (1993). Kriging in a finite domain. *Mathematical Geology*, 25(1), 41–52. The original analysis of the string effect and its implications for mining geostatistics.

5. Goovaerts, P. (1997). *Geostatistics for Natural Resources Evaluation*. Oxford University Press.

6. Isaaks, E.H. & Srivastava, R.M. (1989). *Applied Geostatistics*. Oxford University Press. Chapters 16–17: The Walker Lake case study and practical variogram interpretation.

7. Journel, A.G. & Huijbregts, C.J. (1978). *Mining Geostatistics*. Academic Press. The original presentation of the normal-score transform.

8. Krivoruchko, K. (2011). *Spatial Statistical Data Analysis for GIS Users*. Esri Press.

9. Markvoort, J. & Deutsch, C.V. (2024). The bias caused by the string effect in ordinary kriging: risks and solutions. *Applied Earth Science*, 130(4). The most recent comprehensive analysis of the string effect and its mitigation.

10. Melles, S.J., Heuvelink, G.B.M., Twenhöfel, C.J.W., van Dijk, A., Hiemstra, P.H., Baume, O. & Stöhlker, U. (2011). Optimizing the spatial pattern of networks for monitoring radioactive releases. *Computers & Geosciences*, 37(3), 280–288.

11. Pebesma, E.J. (2004). Multivariable geostatistics in S: the gstat package. *Computers & Geosciences*, 30(7), 683–691. The Meuse zinc analysis demonstrating universal kriging with external drift.

12. Pyrcz, M.J. (2024). GeostatsGuy Lectures: Spatial Data Analytics and Geostatistics. University of Texas at Austin. Available at: https://geostatsguy.github.io/GeostatsPyDemos_Book/

13. Stein, M.L. (1999). *Interpolation of Spatial Data: Some Theory for Kriging*. Springer. Rigorous treatment of kriging optimality under model misspecification.

14. Webster, R. & Oliver, M.A. (2007). *Geostatistics for Environmental Scientists* (2nd ed.). Wiley. Chapter 5: Sample size requirements for reliable variography.

---

*Lecture prepared 2025-05-25 for the Spatial Interpolation Engine project. Enhanced with insights from the GeostatsGuy lecture series (Pyrcz, 2024), recent research on data configuration effects (Markvoort & Deutsch, 2024), and optimal sampling design literature (Brus & Heuvelink, 2007; Melles et al., 2011).*
