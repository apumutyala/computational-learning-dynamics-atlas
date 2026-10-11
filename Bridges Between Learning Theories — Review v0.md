# Bridges Between Learning Theories — Review v0

Independent adversarial review of `Bridges Between Learning Theories — Draft v0.md` and `Bridges Between Learning Theories — Open Problems and Tests.md`, written 2026-10-07. I edited neither file. All computations ran on CPU with numpy. The scripts are in this session's scratchpad: `rlct_mc_rerun.py`, `rlct_fiber.py` and `checks_b9_b11.py`.

**Verdict labels:** CORRECT / CORRECT-WITH-CAVEATS / WRONG / UNVERIFIABLE-AS-STATED / MISLABELED. A bridge can carry more than one label when its parts differ.

---

## 0. Summary table

| Bridge | Claim (short) | Verdict | Main issue |
|---|---|---|---|
| B1 | Optimal Catoni PAC-Bayes value = (F_n(t)+log 1/δ)/t + tC²/8n | **CORRECT-WITH-CAVEATS**, partly MISLABELED | The identity is exact only for a loss in [0,C] and a **data-independent t**. "With the NLL loss" contradicts boundedness. The identity is known prior work (Catoni 2007; Germain et al. 2016), not a new derivation. At fixed β the λ term is swamped by the O(1) slack βC²/8. |
| B2 | MI = PAC-Bayes complexity = MDL = λ log n for the Gibbs algorithm | **WRONG** (MI part); CORRECT-WITH-CAVEATS (KL/MDL part) | I(S;W) for a fixed data law is **O(1)**, not λ log n. KL(Q*‖φ) ≈ λ log n holds, but step 3 is wrong at fixed β: t·E[L_n−L_n(θ₀)] → λ − βν, not λ. |
| B3 | rank(H)/2 ≤ λ ≤ d/2, with equality iff Morse–Bott | **CORRECT** (upper bound loose) | Sharper and still elementary: **λ ≤ r/2 + (d−r)/4**. The bound requires the prior to be positive at θ*, which §4.1/4.2 do not consistently assume. |
| B4 | Finite-n LLC sits between rank(H)/2 and λ; toy slope 1.00 | **CORRECT-WITH-CAVEATS** (idea); numeric **NOT REPRODUCED** | The re-run gives slopes of 1.05–1.12, not 1.00 ± 0.01. "Between" is not a theorem: localization or log factors can push finite-n values below rank/2. The draft computes a volume slope, not an LLC estimator. |
| B5 | Degenerate minimum never exponentially stable under GD | **CORRECT** (for points); consequence **overstated** | "Never point-valued certificates" is false: by Łojasiewicz, analytic K with small-step GD converges to a single point. Morse–Bott gives exponential convergence to the set. |
| B6 | Fisher = parameter observability Gramian; non-identifiability = unobservability = degeneracy | **CORRECT-WITH-CAVEATS** | "Three names for one thing" fails both ways: x⁴ is degenerate but identifiable, and permutation copies are non-identifiable but nondegenerate. |
| B7 | Constant-LR SGD stationary Σ = (η/2B)A^{−α}; E3 ≠ E2 | **CORRECT under C = A** | The conclusion depends entirely on noise ∝ Hessian; isotropic noise gives a posterior-shaped Σ ∝ A⁻¹. The result is known (Mandt et al. 2017). The draft omits the known drift along W₀ (Blanc et al.; Li–Wang–Arora 2022) and the non-equilibrium probability currents (Kunin et al.). |
| B8 | Susceptibility → classical influence under Laplace | **CORRECT-WITH-CAVEATS** | "BIF can add info only via singular structure" overreaches: BIF can also beat *approximate* influence functions (EK-FAC/damping) in regular models. |
| B9 | Order effect = η²(H_j g_i − H_i g_j) = η²[g_i,g_j] | **CORRECT** (re-derived and checked numerically); reading **overstated** | For block curricula the small parameter is K·η, not η. The simplex-constrained mixture system is control-affine **with drift**, not driftless; one step reaches the convex hull, not the cone. |
| B10 | GDN update = one LMS/SGD step | **CORRECT**; MISLABELED as new | This is stated verbatim in the GDN paper (and in DeltaNet, Longhorn and TTT). Cite it as THEOREM/known. |
| B11 | ‖α(I−βkkᵀ)‖₂ = α for β ∈ [0,2] | **CORRECT** (checked to 1e-15) | It needs ‖k‖ = 1 exactly. Without normalization, β‖k‖² > 2 gives norm > α (checked: 1.16). |
| B12 | Input-conditioned Gramians, exact null edits, persistence functional | **CORRECT-WITH-CAVEATS** | The null-edit property is a tautology, and it *does* transfer to free-running generation, contrary to §7.4. The product W_c(T)·W_o(0) mixes reference times, and W_c weighted by ‖v‖² is not a reachability Gramian. |
| §4.6 | Global RLCT 5/4 for n=2, r=1; review 02 wrong | **CORRECT** | Confirmed independently by a fibre-minimization method that reaches ε = 1e-16: the slope converges to 1.250. **The draft is right and review 02's "do not lower λ; raise m … it is an equality" is wrong.** |

| Conjecture | Verdict on statement and test |
|---|---|
| C1 | Well-posed for (2,1), where it is already a derivation. "Attained when all unused poles coalesce" is unsupported for general (n,r). T-C1 cannot falsify cleanly because of the pre-asymptotic drift shown in §B4 below; use an algebraic (Newton-polyhedron / blow-up) computation of the moment ideal instead. |
| C2 | Weakly falsifiable as written. The numerical Hessian rank needs a threshold tied to nβ and γ. "Violation ⇒ estimator bias" is wrong, because finite-n λ_eff can legitimately fall below rank/2 under localization. |
| C3 | The K = a²b² test is **mis-specified**: every point of W₀ has λ = 1/2, and the origin differs only in **multiplicity** (m = 2). The predicted direction (posterior favours the singular stratum *more* than SGD) may have the wrong sign, given stochastic-collapse results. |
| C4 | The single-swap version is testable and will pass, because it is a Taylor identity. The interesting block-curriculum version has no small parameter. |
| C5 | (a) is a tautology. (b) is a fair conjecture. (c) is valid only at critical points: symmetry directions are null directions of H only where ∇K = 0. |

---

## 1. Per-bridge detail

### B1 — PAC-Bayes optimum = tempered free energy

**Re-derivation.** The draft's statement of Alquier's Theorem 2.1 is correct: for a loss in [0,C] and fixed λ, E_ρR ≤ E_ρr + λC²/(8n) + (KL + log 1/δ)/λ (checked in arXiv 2110.11216). By Donsker–Varadhan, min_Q {E_Q L_n + KL(Q‖φ)/t} = −(1/t) log ∫e^{−tL_n}dφ = F_n(t)/t. So (2.2) is exact.

**Errors and caveats.**

1. **Bounded loss vs NLL.** "With t = nβ and the NLL loss, F_n(t) is exactly the tempered Bayesian free energy … This identity is exact." The bound (2.1) needs ℓ ∈ [0,C], and the NLL is not bounded. The two halves of the sentence hold for different losses. The draft's item 3 concedes this later. The headline should be:
   - *exact for bounded losses*; or
   - *for the NLL, exact with a sub-Gaussian or sub-gamma slack replacing tC²/8n* (Alquier §5.1; Germain et al. 2016, who handle exactly this case).
2. **t must not depend on the data** (Alquier is explicit about this). "The best PAC-Bayes certificate as a functional of the data" holds only for a pre-chosen t, or over a grid at a cost of log|grid| in place of log(1/δ).
3. **Fixed β makes the λ statement vacuous in practice.** The slack βC²/8 is Θ(1), while λ log n/(nβ) → 0. "Degeneracy literally shrinks the certificate" is true only in an O(log n/n) term sitting next to an O(1) term.
   - The interesting regime is t ≍ √(n log n). That is not covered by Watanabe 2013, whose expansion is for fixed β or β = β₀/log n.
   - The draft's "CONJECTURE" (log t in place of log n) is very likely true, because the fluctuation term shrinks when t/n → 0. It still needs a proof.
4. **Label.** The exact part is a known theorem: Catoni 2007, Germain et al. 2016 and Alquier §2. Label it THEOREM [cite], not DERIVATION.

### B2 — MI = PAC-Bayes complexity = MDL = λ log n

**Steps 1 and 2 are exact** (golden formula; KL(Q*‖φ) = F_n(t) − tE_{Q*}L_n).

**Error 1 (substantive): the MI claim is wrong.** For a *fixed* data distribution q (Xu–Raginsky's setting), I(S;W) = E_S KL(Q_S ‖ E_S Q_S). The average posterior concentrates at the same scale as each Q_S, so I(S;W) = O(1).

- **Regular Gaussian example, β = 1.** Q_S = N(θ̂, Σ) with θ̂ ~ N(θ₀, Σ), so I(S;W) = (d/2) log 2 for every n.
- **Numeric (`checks_b9_b11.py`, d = 3).** I(S;W) = 1.03, 1.04, 1.04 at n = 10², 10⁴, 10⁶, while E KL(Q‖φ) = 5.6, 12.5, 19.4, which tracks (d/2) log n.

The inequality I ≤ E KL(Q‖φ) is therefore very loose. "The information the posterior extracts from the data … equals λ log n" is false. The (d/2) log n of Clarke–Barron is I(Θ;S) with Θ drawn from the prior, which is a different quantity.

**Correct statement.** The PAC-Bayes complexity KL(Q*‖φ) and the Bayes-mixture code length F_n − nL_n(θ₀) both scale as λ log n. The mutual information I(S;W) at a fixed truth is O(1), and λ log n is only an upper bound on it. The Xu–Raginsky consequence √(2σ²λ log n/n) is valid but loose.

**Error 2: step 3 at fixed β.** The O_p(1) remainder of Watanabe's expansion depends on β. Differentiating in t at fixed n picks up β·∂_β of that remainder, which is O(1). The known equations of state (Watanabe 2010; book §8) give E[nβ(E_w L_n − L_n(w₀))] = λ − βν, where ν is the singular fluctuation.

- Regular case at β = 1: λ − βν = d/2 − d/2 = **0, not λ**.
- The identity behind the Lau et al. estimator needs β → 0 (β* = 1/log n), or the empirical minimizer in place of w₀.
- The leading λ log n term of KL(Q*‖φ) survives, with KL ≈ λ log n − λ + βν + O_p(1).

**Verdict.** WRONG for MI; CORRECT-WITH-CAVEATS for KL and MDL. Re-label the conclusion.

### B3 — rank(H)/2 ≤ λ ≤ d/2

**Re-derivation.** The derivation is sound:

- splitting lemma, K = |u|² + g(v), with g ≥ 0 because K ≥ 0;
- RLCTs of functions in disjoint variables add;
- λ_g ≥ 0, and λ_g > 0 unless g ≡ 0 (a nonzero analytic g has a positive rational RLCT);
- so λ = r/2 iff g ≡ 0 iff W₀ is locally a manifold of dimension d − r (Morse–Bott).

The answer to "is λ ≥ r/2 always?" is **yes**, for any K ≥ 0 analytic with K(θ*) = 0 and prior density positive at θ*.

**Improvement the draft missed.** g has vanishing gradient and Hessian, and g ≥ 0. A nonzero cubic Taylor term would take negative values along some ray, so the cubic term vanishes and g = O(‖v‖⁴). Hence vol{g < ε} ≳ ε^{(d−r)/4}, and

    r/2 ≤ λ(θ*) ≤ r/2 + (d−r)/4,

which is strictly below d/2 whenever H is singular. The bound is tight: K = a²b² has r = 0, d = 2 and λ = 1/2.

**Caveats.**

1. **The upper bound needs φ(θ*) > 0.** If the prior vanishes, λ can exceed d/2 (for K = x² with φ ∝ x², λ = 3/2). §4.1 allows an analytic φ₁ ≥ 0 that may vanish, and §4.2 then asserts "always 0 < λ ≤ d/2". That is inconsistent.
2. **Multiplicity.** In the Morse–Bott case m = 1. Otherwise m comes from g alone.
3. **Invariance under the splitting coordinates.** The splitting coordinates are C^∞ (Gromoll–Meyer). Invariance of the volume exponent under C¹ diffeomorphisms is enough, so citing Lau et al. App. C is not even needed.

**Verdict:** CORRECT.

### B4 — finite-n effective exponent, and the toy numeric

**Re-run.** I re-ran the draft's MC (same K, T = 40; 2 × 10⁷ samples per case; seed 12345):

| Case | Window | Draft | Re-run |
|---|---|---|---|
| G generic | [1e-6, 1e-4] | 1.486 | **1.502** |
| S split | [1e-8, 1e-4] | 1.248–1.293 | **1.248** (1.20–1.26 across windows) |
| C coincident, box 0.1 | [1e-8, 1e-4] | **1.00 ± 0.01** | **1.079** ([1e-6, 1e-4]: 1.047; [1e-7, 1e-5]: 1.117) |
| C coincident, box 0.1/0.3 | [1e-6, 1e-4] | 1.13–1.18 | **1.193** |

**Independent method (`rlct_fiber.py`).** For each (p₂, l₂) on a log-refined 440² grid, I minimized K over (p₁, l₁) by Gauss–Newton. That gives the fibre minimum m(q,a) ≈ 4.21·q²a⁴, which confirms the q²a⁴ structure of §4.6(ii). Then V(ε) = Σ w·2π(ε − m)₊/√det H. This method reaches ε = 1e-16:

| Window | Box 0.1 | Box 0.3 |
|---|---|---|
| [1e-6, 1e-4] | 1.047 | 1.220 |
| [1e-8, 1e-4] | 1.127 | 1.232 |
| [1e-12, 1e-10] | 1.246 | 1.249 |
| [1e-16, 1e-14] | **1.250** | **1.250** |

**Conclusions.**

1. **λ = 5/4 at the coincident point is confirmed.** Two independent methods agree, and the second reaches the asymptotic regime.
2. **"1.00 ± 0.01 = rank(H)/2" is not reproduced.**
   - The fibre minimum in the 0.1 box reaches 8 × 10⁻⁶, so the window [1e-8, 1e-4] straddles the crossover, and the slope there is about 1.08–1.13.
   - Only for ε ≳ 1e-5 does the slope approach 1. The draft's own statement sup g ≈ 10⁻⁶ contradicts fitting down to 1e-8.
   - The draft's inlined code also cannot reproduce the 0.1-box C row: `halves["C"]` is hard-coded to (0.1, 0.1, 0.3, 0.3).
3. **The qualitative point stands, but the "between" claim is not a theorem.** Finite-ε slopes are pre-asymptotic and lie between rank/2 and λ *in this example*. In general:
   - when the localization box or γ truncates the quadratic directions (ε ≳ curvature·R²), the slope falls *below* rank/2, toward 0;
   - log factors (m > 1) pull finite-window slopes below λ;
   - boxes containing higher-λ strata pull slopes above λ at intermediate ε.

   Replace "measures something between rank(H)/2 and λ" with "is a scale-dependent effective exponent, typically between rank(H)/2 and λ when the localization scale exceeds the quadratic scale."
4. **The test measures the wrong object.** The numeric is a volume slope, not an SGLD-LLC estimate. The draft's bridge to "practical LLC estimates" is an argument, not evidence. λ̂ = t·E[K] at t = nβ is d log Z/d log t, a Laplace-smoothed slope, which mixes scales over about a decade.

**Verdict.** The idea is CORRECT-WITH-CAVEATS. The numeric claim is WRONG as stated (1.00 → about 1.05–1.13).

### B5 — degenerate minima are not exponentially stable

For a C¹ map, exponential stability of a fixed point requires spectral radius < 1. An eigenvalue of 1 on ker H rules that out. For non-isolated minima, points of W₀ are not even asymptotically stable. So the statement is **true as stated about points**. The Morse–Bott description (normally attracting manifold of fixed points when 0 < ηa_i < 2 on normal directions) is correct.

**Overreach.** "Training-control certificates can at best be set-valued … never point-valued" is false.

- For analytic K and small η, every bounded GD trajectory converges to a **single** point. This is the Łojasiewicz gradient inequality (Absil, Mahony & Andrews 2005).
- In the Morse–Bott case, convergence to that point is exponential.

What fails is *uniform, perturbation-robust* contraction to a *prescribed* point. Say that instead. Non-Morse–Bott degeneracy (quartic g) gives only polynomial convergence in those directions, which is worth adding.

**Verdict:** CORRECT; the consequence is overstated.

### B6 — unobservability = singularity

The Fisher-as-sensitivity-Gramian identity is standard: local identifiability ⇔ nonsingular information at regular points (Rothenberg 1971).

"Three names for one thing" fails both ways:

- **Degenerate but identifiable.** K = x⁴ has H = 0 and W₀ = {0}.
- **Non-identifiable but nondegenerate.** Discrete symmetries such as permutations give isolated equivalent minima, each with full-rank H.

The draft's "measured at different orders" hedge covers the first case only. **Verdict:** CORRECT-WITH-CAVEATS.

### B7 — SGD is not a posterior sampler

**Re-derivation.** The exact discrete solution (commuting case, C = A, P = A^{−α}) is

    Σ = (η/2B)·A^{−α}·(I − (η/2)A^{1−α})^{−1}.

For α = 0 that predicts a relative error of η·a_max/2 = 0.05, which matches the reported 0.054. (5.1) and (5.2) are correct.

**Assumptions are decisive, and stated, but the consequences overreach.**

1. **The result hinges on C = A** (well-specified model, population sampling, at θ*). With isotropic noise C = σ²I, plain SGD gives Σ = (ησ²/2B)A^{−1}, which is posterior-shaped. "SGD fluctuations measure curvature, not λ" therefore holds *because* C ∝ A. Real nets have C ≠ A: misspecification, label noise, multi-epoch reuse, and the state-dependent noise of overparameterized interpolation, where C → 0 at zero loss.
2. **The result is known.** Mandt, Hoffman & Blei (2017) derive the optimal preconditioner ∝ (BBᵀ)⁻¹, i.e. natural gradient when C = A. Cite it as THEOREM/known rather than DERIVATION.
3. **Flat directions.** "SGD does not diffuse along W₀ at all to first order" is true of the *linear* OU picture. The draft omits the **known second-order drift along W₀**:
   - label-noise and minibatch SGD near a manifold of minima drift toward smaller tr(H) or tr(C-weighted H) (Blanc et al. 2020; Damian et al. 2021; Li, Wang & Arora 2022, ICLR);
   - this is a deterministic implicit regularizer on W₀ with a computable direction, and it directly bears on C3 and T-B7.
4. **Non-equilibrium beyond the quadratic case.** SGD's stationary law generally carries probability currents and is not of Gibbs form for any potential (Chaudhari & Soatto 2018; Kunin et al. 2021). This is a stronger and more general E3 ≠ E2 statement than §5.2, and it should be cited.
5. **Small inconsistency.** "Plain-SGD variance stays η/2B" as a_i → 0 and "SGD does not diffuse along W₀" are both limits, but the relaxation time 1/(ηa_i) diverges. Say that the stationary value is never reached.

**Verdict:** CORRECT under the stated assumptions. MISLABELED (known result). The consequences overreach unless C = A is foregrounded.

### B8 — susceptibility → classical influence

The computation checks: χ = −β·(nβ)⁻¹·G_O A⁻¹ G_ℓᵀ = −(1/n)·G_O A⁻¹ G_ℓᵀ.

The claim that "BIF can only add information through non-Gaussian, singular structure" is too strong. In practice, classical influence is computed approximately (EK-FAC, damping, truncated iterative solvers). A posterior-sampling estimator can beat *those approximations* in perfectly regular models. Rephrase as: BIF can differ from the **exact** classical influence only through non-Laplace structure and O(1/n) corrections.

**Verdict:** CORRECT-WITH-CAVEATS.

### B9 — curriculum Lie bracket

**Re-derivation.** θ_ij = θ − η(g_i + g_j) + η²H_j g_i + O(η³), so θ_ij − θ_ji = η²(H_j g_i − H_i g_j) = η²[g_i, g_j] with [X,Y] = (DY)X − (DX)Y. The signs are correct, since brackets of −g_i and −g_j equal brackets of g_i and g_j.

**Numeric check** (5-D non-quadratic losses): relative error of the prediction 2.7e-1, 8.1e-2, 2.7e-2, 8.0e-3 at η = 1e-1, 3e-2, 1e-2, 3e-3. That is O(η) relative, as expected.

**Overreach in the reading.**

1. **Mixture weights.** With π ∈ Δ^J the vector field is f₁ + Σ_{j≥2}π_j(f_j − f₁). That is **control-affine with drift**, not driftless. One step reaches the *convex hull* of {−∇L_j}, not a cone.
   - The Lie-algebra rank condition then gives *accessibility* (nonempty interior of the reachable set), not small-time local controllability.
   - Chow–Rashevskii needs symmetric controls.
   - "Local reachable directions = span of brackets" is therefore too strong.
2. **Real curricula use blocks of K steps.** By BCH, the order effect is ≈ K_iK_j η²[g_i, g_j] + O((Kη)³), so the expansion parameter is Kη·λ_max, which is ≫ 1 for realistic blocks. "Order matters at second order in η" holds only for single-step interleaving.
3. **T-B9** must reuse identical minibatches in both orders. Otherwise the O(√η)-scale noise differences swamp an O(η²) signal.

**Verdict:** CORRECT; the reading is overstated.

### B10 — GDN = online LMS

The GDN paper (arXiv 2412.06464, Eq. 10, checked) gives S_t = S_{t−1}(α_t(I − β_t k_t k_tᵀ)) + β_t v_t k_tᵀ, with α, β ∈ (0,1) and L2-normalized q and k. It says verbatim that the delta rule optimizes ½‖S_t k_t − v_t‖² "via test-time stochastic gradient descent".

The draft's α < 1 decomposition, α(S − β∇ℓ) + (1−α)βvkᵀ, checks algebraically.

**Verdict:** CORRECT, but MISLABELED as a derivation. It is the cited paper's own framing; see also DeltaNet (2406.06484), Longhorn (Liu et al. 2024), TTT (Sun et al. 2024) and Titans.

### B11 — non-expansive transitions

The bound ‖α(I − βkkᵀ)‖₂ = α for β ∈ [0,2] and ‖k‖ = 1 checks numerically: max error 1e-15 over 10⁴ random draws. The incremental bound ‖ΔS_t‖_F ≤ Πα‖ΔS₀‖_F holds because the state multiplies from the left by a fixed matrix and from the right by A_t.

**Caveats.**

- **Exact unit norm.** The result needs ‖k‖ = 1 exactly, which needs ε-free L2 normalization (most kernels use x/√(‖x‖² + ε)). With β‖k‖² > 2 the norm exceeds α; with ‖k‖ = 1.2 and β = 1.5 it is 1.16.
- **Unbounded state at α ≡ 1 is real, not just unproven.** Kaczmarz-type writes S k_t = v_t with nearly parallel keys and differing values force ‖S‖ ~ 1/angle. That is unbounded over adversarial sequences. Say so.

**Verdict:** CORRECT.

### B12 — Gramians, null edits, persistence functional

The algebra checks: the output perturbation is ΔS Φ(0,t) q_t, the energy is tr(ΔS W_o ΔSᵀ), and the write contribution is β_s v_s · k_sᵀΦ(s,t)q_t with |·| ≤ Πα.

**Issues.**

1. **The null-edit property is a tautology.** An edit that leaves outputs unchanged leaves outputs unchanged. Its value is as a *computable* null space, which is fine, but "perfect free negative control" oversells it.
   - It also **does** transfer to free-running generation: if the layer's outputs are unchanged on the realized trajectory, the sampled tokens (same seed) are unchanged, so the closed loop is unchanged.
   - That contradicts the blanket "no statement above transfers to free-running generation" in §7.4 and §9.
2. **Mixed reference times.** W_c(T) is a reachability Gramian *at time T*, while W_o is defined *from time 0*. "λ_i(W_c W_o) is invariant" and "Hankel-style" require both at the same time, with W_o(T) taken over [T, T+H].
   - Weighting W_c by β²‖v‖² gives an *actual-excitation* Gramian, not an input-normalized reachability Gramian.
   - For the matrix state, each row of S is a separate LTV system with scalar input v_i. The Gramian in the draft sums over rows only implicitly.
3. **Persistence functional.** The per-sequence formula is exact, and labelling it a diagnostic is honest.

**Verdict:** CORRECT-WITH-CAVEATS.

### §4.6 — the RLCT of a diagonal LTI SSM, and "review 02 was wrong"

I read the original repo text:

- `State-Space-Decision-Model/reviews/02-technical-pressure-test.md` line 202: "Doc B's λ ≤ r + (n−r)/2 — SOUND for real modes, **and it is an equality** … Additional loci (λ_i = λ_j coincidence, λ_i → 0) **do not lower λ; they raise m**."
- `plans/claude/PLAN.md` line 78: "corrected learning coefficient, λ = r + (n−r)/2 … Monte-Carlo-checked".
- `analysis/CODEX_CRITICAL_COMPARISON.md` lines 78–88: a split-stratum local computation u² + v² + d⁴ ⇒ 5/4.

**Independent derivation.** Write c_j = p₁b^j + q a^j for the coefficients on the independent sequences C(k,j)λ*^{k−j}. With v = c₁ (a valid coordinate since p₁ ≠ 0):

- c₂ ≡ q a²(p₁ + q)/p₁ mod v;
- c₃ ≡ q a³(1 − q²/p₁²) mod v, which lies in (q a²).

So the ideal is (u, v, q a²) and λ = 1 + RLCT(q²a⁴) = 1 + 1/4 = 5/4, with m = 1. The same holds with the (b_i, c_i) parameterization in place of p_i: RLCT((bc)²a⁴) = 1/4.

The fibre-minimization computation measured m(q,a)/(q²a⁴) ≈ 4.21, roughly constant, and a volume exponent of **1.250** at ε ∈ [1e-16, 1e-12].

**Who is right.**

- **The draft and Codex are right.** The global RLCT for n = 2, r = 1, with prior positive near the coincident or split strata, is 5/4, not 3/2.
- **Review 02 is wrong on two specific phrases:** "it is an equality" and "do not lower λ; they raise m".
- **Review 02's inequality λ ≤ r + (n−r)/2 is literally still true**, since 5/4 ≤ 3/2. Its MC numbers are correct generic-point local values.
- The same caveat applies to its complex-mode "λ = n + r", which is generic-local only.

**Small caveats on the draft's version.**

1. **Prior support.** "Global" presupposes prior mass at coincident or split configurations. Ordered or gapped pole parameterizations put that locus on the boundary, which leaves the exponent the same, but an initialization law that excludes it changes the global value.
2. **Truncation.** The draft's numerics truncate at T = 40, which matters at ε ≲ 1e-8 for |l| ≈ 0.8.
3. **Stratum labels.** The split stratum is what Codex computed. The draft's "coincident" point (q = 0, a = 0) and the split points share the 5/4 value, as the draft notes.

---

## 2. Conjectures C1–C5 (tests)

- **C1.**
  - For (2,1) this is a DERIVATION, so it should not be listed as a conjecture.
  - For general (n,r), "attained when all unused poles coalesce onto active poles" is a guess. Coalescing *two* unused poles onto *one* active pole and splitting them across active poles give different moment ideals.
  - The complex-pair claim "strictly below n + r" is plausible but unproven.
  - T-C1 (thermodynamic integration over t ∈ [10², 10⁸]) can be fooled by the slow pre-asymptotic approach shown in §B4. In the 0.1 box the slope is still 1.13 at ε ~ 1e-6, i.e. t ~ 10⁶. "Falsified if the slope converges to r + (n−r)/2" may never be decidable in range.
  - Prefer an exact algebraic computation: generate the moment ideal, then use a Newton polyhedron in good coordinates or a blow-up. Run the fibre method in parallel; it reaches ε = 1e-16 cheaply.
- **C2.**
  - Needs a rank threshold tied to the scale, rank_τ(H) = #{h_i > τ} with τ ~ 1/(nβR²). Otherwise "rank(H)" at finite precision is itself scale-dependent and the test is circular.
  - "A sandwich violation means estimator bias" is wrong: λ_eff < rank/2 occurs legitimately under strong localization.
- **C3 / T-B7.**
  - For K = a²b², λ = 1/2 at *every* point of W₀; the origin differs only in multiplicity (m = 2 vs 1). So the posterior's preference for the origin is a log n factor, not an n^{Δλ} factor.
  - The SGD endpoint law depends on the initialization law. A KS test will reject almost surely, so "falsified if they agree" is nearly unfalsifiable in the interesting direction.
  - Stochastic collapse (Chen, Kunin et al. 2023) predicts SGD is *attracted* to low-noise invariant sets such as the origin. The predicted sign ("posterior puts more mass at the singular stratum than SGD") may be wrong. Make the sign the falsifiable prediction.
- **C4.** As noted in §B9, the single-step version is a Taylor identity and cannot fail at small η if implemented correctly. The scientifically interesting version uses blocks with Kη ~ O(1) and needs BCH or a full solve.
- **C5.**
  - (a) is a tautology.
  - (b) is a reasonable rank-correlation conjecture.
  - (c) has three problems:
    - At a non-converged checkpoint, H·ξ = −(Dg-action)ᵀ∇K ≠ 0 for symmetry generators ξ, so symmetry directions are not exactly in ker H.
    - Data-dependent degeneracies add nullity.
    - Lanczos nullity needs a threshold.
  - Report Hessian nullity restricted to symmetry tangents, and the gradient norm.
- **C6.** Fine as stated for KL(Q*‖φ). It does not test MI (see B2).

---

## 3. Other errors and overreaches

1. **§4.2: "Always 0 < λ ≤ d/2"** needs a positive prior at the optimum, which conflicts with §4.1's allowance of an analytic φ₁ ≥ 0.
2. **§3.3 NTK.**
   - "λ = rank(G)/2 in the lazy regime" is a property of the *linearized model's* K, not of the network's K.
   - Lazy training means the trajectory stays near the linearization; the population loss of the nonlinear network keeps its higher-order structure.
   - "SLT is trivial there" should read "SLT's extra structure is invisible to the trajectory".
3. **§3.4 and §9: "Edge of stability: empirical only."** Outdated. Self-stabilization theory exists (Damian, Nichani & Lee 2022, ICLR 2023), as does an analysis of sharpness reduction at EoS (Arora, Li & Panigrahi 2022).
4. **§3.3 and §8: the lazy vs mean-field dichotomy omits μP / Tensor Programs** (Yang & Hu 2021). μP is a feature-learning infinite-width limit with fixed hyperparameter transfer, and it is the regime in which C-lazy's "rich" side actually lives at scale. Missing.
5. **§4.2 WBIC.** The statement is consistent with Watanabe 2013. Watanabe's WAIC (2010) and the equations of state, the λ − βν relations, are cited only in passing. They matter for B2, and they are the theorem behind the LLC estimator's choice β = 1/log n.
6. **Missing SLT literature directly relevant to §4.6:**
   - Aoyagi's exact RLCTs for deep linear networks (2024);
   - Wei, Murfet et al., "Deep learning is singular, and that's good" (2022);
   - Watanabe's book (2009) for the volume/zeta equivalence and the addition rule used in B3.
7. **§8 "Fixed-n LLC estimates … at practical n they often sit near rank(H)/2."** No evidence is given beyond the toy, and the toy re-run shows about 1.05–1.13, not rank/2.

---

## 4. Three strongest and three weakest parts

**Strongest**

1. **§4.6 / C1 correction of review 02.** The algebra is right, the multiplicity is right, and it is now confirmed independently to slope 1.250 at ε = 1e-16. It is a genuine and useful correction to the repo's plan wording.
2. **B3 sandwich.** The derivation is clean and correct, cheaply testable, and correctly identifies the Morse–Bott case as the only one where second-order geometry determines λ. It can be tightened to r/2 + (d−r)/4.
3. **B11/B12 LTV analysis of delta-rule layers.** The non-expansiveness is exact, the persistence functional is exact per sequence, and the scope and limits are honestly stated ("diagnostic, not certificate", RWKV-7 flagged as open).

**Weakest**

1. **B2's "MI = λ log n".** It is false: I(S;W) is O(1) at a fixed truth. Step 3's derivative argument also drops a −βν term, which at β = 1 in the regular case cancels λ entirely.
2. **The B4 numerical evidence.** "1.00 ± 0.01 = rank(H)/2" does not reproduce (1.05–1.13), and the inlined code cannot produce that row. The "between rank/2 and λ" claim is presented as general when it is example-specific.
3. **B1 headline "exact … with NLL", and B7's E3 ≠ E2 consequences.**
   - B1 mixes a bounded-loss theorem with an unbounded loss and a data-dependent temperature, and its λ term is O(log n/n) next to an O(1) slack.
   - B7's sweeping consequences ("any complexity proxy from SGD fluctuations is a curvature trace") rest entirely on C = A. They ignore the known second-order drift along W₀ and the non-equilibrium (current-carrying) stationary laws that give a more general E3 ≠ E2.

---

## 5. Label corrections

| Item | Draft label | Should be |
|---|---|---|
| B1 exact part | DERIVATION | THEOREM [Catoni 2007; Germain et al. 2016; Alquier Thm 2.1 + DV], bounded loss, fixed t |
| B2 conclusion | DERIVATION | DERIVATION for KL/MDL (with the λ − βν correction); WRONG for MI |
| B4 numeric | NUMERICAL, 1.00 | NUMERICAL, 1.05–1.13 (window-dependent) |
| B7 (5.2) | DERIVATION | THEOREM-level known (Mandt et al. 2017), conditional on C = A |
| B10 | DERIVATION | THEOREM [Yang, Kautz & Hatamizadeh 2024, stated there] |
| C1 for (n,r) = (2,1) | CONJECTURE | DERIVATION + NUMERICAL (confirmed) |
| C5(a) | conjecture/test | tautology (implementation check only) |
