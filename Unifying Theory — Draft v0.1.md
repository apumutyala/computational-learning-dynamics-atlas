# Unifying Theory — Draft v0.1

*From statistical learning to deep, latent and recurrent learning, with training control, developmental interpretability and singular learning theory used only where they apply.*

Written 2026-10-07. This revises `Unifying Theory — Draft v0.md` in response to the independent mathematical review `Unifying Theory — Review v0.md`. Every finding of the review is applied. **§11 (Changes from v0)** maps each change to the review item it answers. The v0 files, the Review and the five source documents were not modified. The companion file is `Unifying Theory — Open Problems and Tests v0.1.md`.

---

## 0. Read this first

### 0.1 The thesis in one paragraph

No single theorem unifies these fields, and this draft does not claim one. What does unify them is a **shared object**: a learning run is a *controlled stochastic dynamical system on parameter space*. It has three parts:

- an **objective**, which comes from statistics;
- a **geometry**, which comes from the loss landscape and the parameterization;
- **actuators**, the training choices an engineer controls.

Each theory considered here is a statement about one *functional of the law of this process*, valid in one *limit or regime*. Unifying them means two things:

1. one notation in which every theory's object can be written down;
2. a **regime map** (§9) that says when two theories' objects coincide, and what the error is when they don't.

The useful content lies in the bridges, and each is labelled with its status. §8 lists what does **not** unify.

### 0.2 Status labels

| Label | Meaning |
|---|---|
| **THEOREM [cite]** | A published result, with its assumptions as I understand them. "(checked)" marks a primary source fetched this session. |
| **KNOWN** | A result that v0 presented as a derivation but that is already in the literature. Cited; the derivation is kept only for checking. |
| **DERIVATION** | Derived here, from stated assumptions. Short enough to check. |
| **NUMERICAL** | A CPU computation, with code in the companion file. Evidence, not proof. |
| **HYPOTHESIS / CONJECTURE** | Plausible, with a falsifiable test. Not established. |
| **ANALOGY-ONLY** | A shared word or shape with no mathematical identity. |

### 0.3 Commitments inherited from earlier reviews (unchanged from v0)

The user's project reviews (`State-Space-Decision-Model/reviews/02`, `/03`, `PLAN.md`) rejected three things. This draft re-asserts none of them:

- **No LLC-as-sensor or control loop.** Policy: *log, never gate*. The lab measured an SGLD bias proportional to εN/B·λ_max(A), invisible to R-hat.
- **No "injection half-life" certificate.** §7.3 gives an exact *per-sequence* persistence functional, explicitly as a diagnostic.
- **No universal "LLC tracks Hankel rank" law.** One correction to review 02 stands, now independently confirmed (§4.6):
  - For n = 2, r = 1 the global RLCT is **5/4**.
  - Review 02's "it is an equality" and "[coincidence loci] do not lower λ; they raise m" are wrong.
  - Its value r + (n−r)/2 is the **local coefficient at a generic true parameter only**.
  - Its inequality λ ≤ r + (n−r)/2 is literally still true.

---

## 1. The shared object

### 1.1 Variables and time axes

The four time axes from the source documents are kept separate:

- **t**: environment time;
- **j**: token position;
- **s**: generation time;
- **k**: training step.

Only **k** and **j** are used here.

| Symbol | Meaning |
|---|---|
| q | data distribution |
| p(y\|x,θ), θ ∈ W ⊂ ℝ^d | model |
| φ(θ) | prior (Bayesian side) or initialization law (optimization side) |
| ℓ(θ;z) | per-example loss, usually the NLL |
| L(θ) = E_q ℓ; L_n = (1/n) Σ_i ℓ(θ;z_i) | population and empirical loss |
| K(θ) = L(θ) − L(θ₀) | excess population loss |
| W₀ = {K = 0} | the optimal set; in singular models a variety, not a point |

### 1.2 Learning as a controlled stochastic system

DEFINITION.

    s_{k+1} = U(s_k, u_k, ξ_k),   s_k = (θ_k, m_k),   y_k = h(θ_k) + ν_k.

- **Control inputs** u_k: learning rate η_k, batch size B_k, mixture weights π_k ∈ Δ^J, regularizer and constraint multipliers.
- **Randomness** ξ_k: minibatch draws, injected noise.
- **Measurements** y_k: losses, evaluations, probes, logged LLC.

For SGD with mixture control:

    θ_{k+1} = θ_k − η_k Σ_j π_{k,j}∇L_j(θ_k) − η_k ζ_k,   E[ζ_k|θ_k] = 0,   Cov(ζ_k|θ_k) = C(θ_k)/B_k.      (1.1)

Three "endpoint" objects must never be confused:

| Endpoint | Object | Theories |
|---|---|---|
| **E1** trajectory or point | θ_K | optimization, NTK, mean-field, μP, implicit bias, stability bounds |
| **E2** Gibbs posterior | p_β(θ\|D_n) ∝ e^{−nβL_n(θ)}φ(θ) | Bayes, **SLT**, PAC-Bayes (optimal posterior) |
| **E3** stationary law of (1.1) | μ_∞ at constant u | SGD-as-SDE, Langevin, "Almost Bayesian" |

**The central regime question** is when E1 or E3 is close to E2. In general SGD's stationary law carries probability currents and is not of Gibbs form for any potential ([Chaudhari & Soatto 2018](https://arxiv.org/abs/1710.11029); [Kunin et al. 2021](https://arxiv.org/abs/2107.09133)). §5 makes the discrepancy explicit in the simplest regime, and also notes when it disappears.

### 1.3 Two notions of "state"

A recurrent model has two coupled dynamical systems:

- a **slow system**: training, θ_k;
- a **fast system**: the per-layer recurrent state S_j over tokens.

For delta-rule SSMs, the fast system is itself an online SGD learner. This is **KNOWN**: it is stated in the Gated DeltaNet paper (§7.1). So the tools of §§2–6 apply on both time scales.

### 1.4 Population vs empirical, asymptotic vs finite-n

Most SLT statements are asymptotic in n with d fixed. Modern networks have d ≫ n. §3.2 and §4.4 make the consequences explicit.

---

## 2. Layer (a): statistical learning theory, and its exact join to SLT

### 2.1 What the classical layer gives, and where it fails for overparameterized networks

| Result | Statement | Object | Status for modern networks |
|---|---|---|---|
| Uniform convergence | sup over the class of \|L − L_n\| is small | the hypothesis class | **Vacuous**: random labels are fit ([Zhang et al. 2016](https://arxiv.org/abs/1611.03530)). Even algorithm-dependent two-sided uniform convergence can be vacuous ([Nagarajan & Kolter 2019](https://arxiv.org/abs/1902.04742)). |
| Stability | E gen ≤ ε_stab; for convex, smooth SGD, ε_stab ≤ (2L²/n)Σα_t ([Hardt, Recht & Singer 2015](https://arxiv.org/abs/1509.01240)) | E1 | Theorem for convex losses; degrades with training length otherwise. |
| Information-theoretic | \|E gen\| ≤ √(2σ² I(S;W)/n) ([Xu & Raginsky 2017](https://arxiv.org/abs/1705.07809)) | joint law of data and output | Theorem. Needs noise; I(S;W) can be infinite. |
| PAC-Bayes | E_Q L ≤ E_Q L_n + complexity(KL(Q‖φ)) | a randomized Q | Theorem. Non-vacuous once Q is optimized around the SGD solution ([Dziugaite & Roy 2017](https://arxiv.org/abs/1703.11008)). |
| Singular Bayes generalization | E[G_n] = L(θ₀) + λ/n + o(1/n) at β = 1 under realizability; general β involves ν (§4.2) ([Watanabe 2010](https://jmlr.org/papers/v11/watanabe10a.html)) | E2 | Theorem under §4.1 conditions. |

The bounds that survive overparameterization concern **distributions over parameters**. That is why a Bayesian-geometric complexity layer is natural, *provided* the predictor really is posterior-like.

### 2.2 Bridge B1 (KNOWN): the optimal PAC-Bayes value is a tempered free energy

**THEOREM (Donsker–Varadhan).** log E_φ e^f = sup_Q {E_Q f − KL(Q‖φ)}, attained at dQ* ∝ e^f dφ.

**THEOREM (Catoni-type bound; [Alquier 2021, Thm 2.1](https://arxiv.org/abs/2110.11216), checked by the reviewer).** Assume:

- the loss takes values in **[0, C]**;
- t > 0 is **fixed before seeing the data**;
- δ ∈ (0,1).

Then, with probability ≥ 1 − δ, simultaneously for all Q,

    E_Q L ≤ E_Q L_n + tC²/(8n) + (KL(Q‖φ) + log(1/δ))/t.                (2.1)

**KNOWN consequence** (Catoni 2007; [Germain, Bach, Lacoste & Lacoste-Julien 2016](https://arxiv.org/abs/1605.08636), checked; Alquier §2). Minimizing the right side over Q gives the Gibbs posterior Q*_t ∝ e^{−tL_n}φ and the value

    B*(t) = [F_n(t) + log(1/δ)]/t + tC²/(8n),     F_n(t) := −log ∫ e^{−tL_n}dφ.            (2.2)

**Scope of exactness.**

- (2.2) is exact **for a bounded loss and a data-independent t**.
- Choosing t from a grid costs log|grid| in place of log(1/δ).
- For the **NLL**, which is unbounded, F_n(t) at t = nβ is the tempered Bayesian free energy, but (2.1) does not apply as stated. One needs either:
  - **(i)** truncation of the NLL to [0, C], in which case F_n is the free energy *of the truncated loss*; or
  - **(ii)** a sub-Gaussian or sub-gamma slack in place of tC²/8n (Alquier §5; Germain et al. 2016 treat exactly this case).

**Asymptotic reading (DERIVATION, under §4.1 conditions, fixed β, applied to the bounded or truncated loss).** By Watanabe's expansion ([2013, Thm 2](https://www.jmlr.org/papers/v14/watanabe13a.html), checked),

    B*(nβ) = L_n(θ₀) + λ log n/(nβ) − (m−1) log log n/(nβ) + βC²/8 + O_p(1/n).

**What this buys.** The complexity term of the Gibbs predictor's certificate carries λ, not d/2.

**What it does not buy.**

1. **At fixed β this is cosmetic.** The λ term is O(log n/n), sitting next to an **O(1) slack βC²/8**. The informative regime is t ≍ √(n log n), which Watanabe 2013 does not cover. Its expansion is for fixed β or β = β₀/log n.
   - **CONJECTURE (B1-t):** F_n(t) − tL_n(θ₀) = λ log t + o(log t) for t = o(n), t → ∞. Plausible because the fluctuation term shrinks as t/n → 0; unproven.
2. The certificate covers a draw from the Gibbs posterior, **not** the SGD endpoint (§5).

### 2.3 Bridge B2: description length scales as λ log n; mutual information does not

**Exact steps.**

1. *Golden formula.* I(S;W) = E_S KL(Q_S‖P_W) ≤ E_S KL(Q_S‖φ) for any data-independent φ.
2. *Gibbs posterior.* KL(Q*_t‖φ) = F_n(t) − t·E_{Q*_t}[L_n].

**Equations of state (THEOREM, [Watanabe 2010](https://jmlr.org/papers/v11/watanabe10a.html); Watanabe 2009, *Algebraic Geometry and Statistical Learning Theory*).** For the tempered posterior at fixed β,

    E[ nβ( E_w[L_n(w)] − L_n(w₀) ) ] → λ − βν,                                  (2.3)

where ν is the **singular fluctuation**.

*Regular-case check (DERIVATION).* The posterior is ≈ N(θ̂, (nβA)⁻¹), so E_w L_n − L_n(θ̂) = d/(2nβ). Also E[L_n(θ̂) − L_n(θ₀)] = −d/(2n). So the left side of (2.3) is d/2 − βd/2. That matches λ − βν with λ = ν = d/2, and it **vanishes at β = 1**.

This corrects v0's step 3, which claimed tE[L_n − L_n(θ₀)] ≈ λ at fixed β. The LLC estimator escapes the −βν term only because it uses β* = 1/log n → 0 ([Lau et al.](https://arxiv.org/abs/2308.12108)).

**Corrected conclusion (DERIVATION, plus THEOREM inputs).** Combine Watanabe's expansion with (2.3):

    KL(Q*_{nβ}‖φ) = λ log n − (m−1) log log n + O_p(1).

So both of the following scale as **λ log n**, which is (d/2) log n in the regular case:

- the PAC-Bayes complexity of the Gibbs posterior;
- the Bayes-mixture code length F_n − nβL_n(θ₀), i.e. **MDL**. This is consistent with [Urdshals et al. 2025](https://arxiv.org/abs/2510.12077).

**The mutual information is not λ log n.** v0 claimed it was; that was wrong. For a *fixed* data law q, I(S;W) = E_S KL(Q_S‖E_S Q_S) is **O(1)**. In the regular Gaussian case at β = 1 it equals (d/2) log 2 for every n. The reviewer measured 1.03–1.04 nats at n = 10², 10⁴, 10⁶ for d = 3, while E KL(Q‖φ) grew as (d/2) log n.

- So λ log n is only a **loose upper bound** on I(S;W).
- The Xu–Raginsky consequence √(2σ²λ log n/n) is valid but loose.
- The (d/2) log n of Clarke–Barron is I(Θ;S) with Θ drawn from the prior, which is a different quantity.

**Test T-B2 (companion):** thermodynamic integration of KL(Q*‖φ) vs log n, in reduced-rank regression and the LTI toy.

---

## 3. Layer (b): geometry and dynamics

### 3.1 Bridge B3: the Hessian rank sandwich, tightened

**Setting.**

- K is analytic and K ≥ 0.
- θ* ∈ W₀ is an interior point.
- **The prior density is positive at θ\*** (φ(θ*) > 0). This is needed: with φ ∝ x² and K = x², λ = 3/2 > d/2.
- H = ∇²K(θ*), r = rank H. At a realizable, well-specified point, H equals the Fisher information.

**DERIVATION.**

1. *Splitting lemma* (Gromoll–Meyer; the coordinates are C^∞). Locally, K = |u|² + g(v), with u ∈ ℝ^r, v ∈ ℝ^{d−r}, and g(0) = 0, ∇g(0) = 0, ∇²g(0) = 0, g ≥ 0.
2. *Additivity.* The volume exponent is invariant under C¹ diffeomorphisms, and RLCTs of functions in disjoint variables add. So λ(θ*) = r/2 + λ_g.
3. *Lower bound.* λ_g ≥ 0, and λ_g > 0 unless g ≡ 0, because a nonzero analytic g has a positive rational RLCT.
4. *Upper bound.* g ≥ 0 with vanishing quadratic part forces the **cubic Taylor term to vanish**: a nonzero cubic is odd, so it would make g negative along some ray. Hence g = O(‖v‖⁴), and vol{g < ε} ≳ ε^{(d−r)/4}.

Therefore

    r/2  ≤  λ(θ*)  ≤  r/2 + (d − r)/4,                                            (3.1)

with λ = r/2 **iff** g ≡ 0, i.e. iff W₀ is locally a manifold of dimension d − r (the Morse–Bott case, where m = 1).

- The upper bound is **attained**: K = a²b² has r = 0, d = 2 and λ = 1/2 = (d−r)/4.
- It is strictly below d/2 whenever H is singular. (v0 stated only λ ≤ d/2.)

**Reading.**

- Second-order (Fisher, NTK, Hessian) geometry determines λ **only in the Morse–Bott case**.
- SLT's genuinely higher-order content is λ_g ∈ (0, (d−r)/4].
- The test is cheap: compare λ̂ against r/2 and r/2 + (d−r)/4. The rank must be thresholded at the resolved scale (C2).

### 3.2 Bridge B4 (HYPOTHESIS): finite-n effective exponents are scale-dependent

**Argument (not a theorem).** In the coordinates of §3.1, suppose localization restricts v to a box of half-width R, and the resolved loss scale is ε ≈ 1/(nβ).

- If sup_box g ≪ ε, then {K < ε} ≈ {|u|² < ε} × box, so the effective exponent is ≈ r/2.
- As ε falls below sup_box g, the exponent rises toward λ.

**HYPOTHESIS B4.** When the localization scale exceeds the quadratic scale, finite-n LLC estimates *typically* lie between rank(H)/2 and λ. This is **not a general bound**. Effective exponents can fall *below* r/2 in three ways:

- when the localization box or γ truncates the quadratic directions (ε ≳ curvature·R²);
- through log factors when m > 1;
- through higher-λ strata in the box, which push them *above* λ at intermediate ε.

**NUMERICAL (corrected).** At the coincident point of the §4.6 toy (rank H = 2, λ = 5/4), with a box of half-width 0.1:

- the reviewer's Monte Carlo re-run gives volume slopes **1.05–1.12**, depending on the window;
- the reviewer's fibre-minimization method, which reaches ε = 10⁻¹⁶, gives
  - 1.047 on [1e-6, 1e-4],
  - 1.127 on [1e-8, 1e-4],
  - 1.246 on [1e-12, 1e-10],
  - **1.250** on [1e-16, 1e-14];
- with a 0.3 box, the fibre method gives 1.220 → 1.250.

**v0's "1.00 ± 0.01 = rank(H)/2" did not reproduce and is withdrawn.**

- It came from a small (4 × 10⁶-sample) run whose window straddled the crossover; the fibre minimum in the 0.1 box reaches about 8 × 10⁻⁶.
- v0's inlined code could not produce that row, because the box was hard-coded; this is fixed in the companion file.

**What the numerics show:** a slow, monotone pre-asymptotic approach from about rank/2 toward λ, in this example.

**Caveat.** These are *volume slopes*, not SGLD-LLC estimates. λ̂ = t·E[K] is d log Z/d log t, a Laplace-smoothed slope that mixes about a decade of scales. That B4 transfers to practical LLC estimates is an argument, consistent with "LLC estimates are insensitive to modes below a data-dependent threshold" ([Chen & Murfet 2025](https://arxiv.org/abs/2504.18048), checked). It is not evidence.

### 3.3 NTK, lazy training, μP and mean field

**DERIVATION.** For the *linearized model* f_lin(x;θ) = f(x;θ₀) + J(x)(θ − θ₀) with Gaussian noise, K_lin is exactly quadratic, so λ_lin = rank(E[JᵀJ])/2. This is a property of the **linearized model's** K, not the network's.

In the lazy regime ([Jacot et al. 2018](https://arxiv.org/abs/1806.07572); [Chizat, Oyallon & Bach 2018](https://arxiv.org/abs/1812.07956)), the *trajectory* stays where K ≈ K_lin. The nonlinear network's population loss keeps its higher-order structure, but **SLT's extra structure is invisible to the trajectory there**. (v0 said "SLT is trivial there"; that wording is corrected.)

**Feature-learning limits.** Lazy vs mean-field is not the whole picture.

- **μP / Tensor Programs** ([Yang & Hu 2020/ICML 2021, "Feature Learning in Infinite-Width Neural Networks"](https://arxiv.org/abs/2011.14522)) gives an infinite-width limit in which features evolve, with hyperparameter transfer across width. This is the regime where the "rich" side of C-lazy lives at scale.
- **Mean field** ([Mei, Montanari & Nguyen 2018](https://arxiv.org/abs/1804.06561)): two-layer, Wasserstein gradient flow.

SLT has fixed finite d. Its relation to either infinite-width limit is **ANALOGY-ONLY**, apart from symmetry-induced singularities, which appear in both.

**CONJECTURE (C-lazy).** The gap λ̂ − r_τ/2 is ≈ 0 along a lazy parameterization and positive along μP, at matched width. **Test:** a width sweep under the NTK vs μP parameterizations of a small MLP.

### 3.4 SGD as an SDE; edge of stability

**THEOREM-level facts.**

1. SGD is weakly approximated by an SDE on finite horizons as η → 0, under smoothness and moment conditions, and the approximation can fail at large noise scale ([Li, Malladi & Arora 2021](https://arxiv.org/abs/2102.12470)).
2. **Edge of stability** (EoS): full-batch GD drives λ_max(H) to ≈ 2/η ([Cohen et al. 2021](https://arxiv.org/abs/2103.00065)). This is **no longer purely empirical**:
   - self-stabilization theory explains the mechanism through a cubic term that reduces sharpness ([Damian, Nichani & Lee 2022/ICLR 2023](https://arxiv.org/abs/2209.15594));
   - sharpness-reduction flows at EoS are analyzed in [Arora, Li & Panigrahi 2022](https://arxiv.org/abs/2205.09745).

**DERIVATION (what EoS breaks).** The descent lemma makes L a Lyapunov function iff η < 2/L_s, and linear stability of a minimum requires ηλ_max < 2. At EoS both sit at the boundary. So the OU/Lyapunov picture of §5, the bracket expansion of §6.3, and contraction arguments are invalid there. Self-stabilization theory replaces them with a different, non-quadratic analysis. Every training-control claim must carry the measurable flag **ηλ_max(H) ≪ 1**.

### 3.5 Implicit bias

Separable logistic regression: GD converges in direction to the max-margin solution while ‖θ‖ → ∞ ([Soudry et al. 2017](https://arxiv.org/abs/1710.10345)). There is no minimizer in a compact W, so SLT does not apply.

### 3.6 Bridge B5: degenerate minima are not exponentially stable points; trajectories still converge to points

**DERIVATION.** The GD map has Jacobian I − ηH at θ*. On ker H the eigenvalue is 1, so a degenerate minimum is never an exponentially stable point. For non-isolated minima, points of W₀ are not even asymptotically stable. In the Morse–Bott case with 0 < ηa_i < 2 on the normal directions, W₀ is a normally attracting manifold of fixed points.

**THEOREM ([Absil, Mahony & Andrews 2005, SIAM J. Optim. 16(2):531–547](https://doi.org/10.1137/040605266)).** By the Łojasiewicz gradient inequality, for analytic K and small enough step size, every bounded GD trajectory converges to a **single point**.

- In the Morse–Bott case the convergence is exponential.
- With a quartic g it is only polynomial in those directions.

**Consequence (corrected from v0).** v0 said control certificates are "never point-valued"; that was wrong, because individual trajectories do converge to points. What fails is **uniform, perturbation-robust contraction to a prescribed point**:

- the limit point on W₀ depends on initialization and noise;
- perturbations along W₀ are not corrected.

Contraction-based training control therefore has no authority *along* W₀. Certificates about a *prescribed* limit must be set-valued (W₀, or a sublevel set), while convergence of each run to *some* point is guaranteed by Łojasiewicz.

---

## 4. Layer (c): singular learning theory, stated precisely

### 4.1 Assumptions (Watanabe 2013 "fundamental conditions", checked)

1. **Compact W.** The prior is φ = φ₁φ₂, with φ₁ ≥ 0 analytic and φ₂ > 0 smooth.
   - Where the bounds λ ≤ d/2 and (3.1) are used, we additionally require **φ(w*) > 0**. A vanishing prior can push λ above d/2.
2. **Analyticity.** w ↦ log p(x|w) is analytic, with moment conditions.
3. **Relatively finite variance.** E[f] ≥ c·E[f²] with f = log p(x|w₀)/p(x|w) (Eq. (12)).
   - Realizability is **not** required.
   - The condition holds automatically if the model is realizable or regular.
   - It can fail when distinct optimal parameters represent different distributions.
4. **iid data**, and n → ∞ with d fixed.
5. **The object is the tempered Bayesian posterior**, not an optimizer's output.

### 4.2 The objects

- **RLCT λ, multiplicity m.** −λ is the largest pole of ζ(z) = ∫K^zφ, and m is its order. Equivalently, vol{K < ε} ~ cε^λ(−log ε)^{m−1} (Watanabe 2009, book). When φ(w*) > 0, we have 0 < λ ≤ d/2, and locally (3.1).
- **Free energy** ([Watanabe 2013, Thm 2](https://www.jmlr.org/papers/v14/watanabe13a.html), checked):

      F_n = nL_n(w₀) + λ log n − (m−1) log log n + O_p(1).

- **WBIC** = posterior mean of nL_n at β = 1/log n, which equals nL_n(w₀) + λ log n + U_n√(λ log n/2) + O_p(1). It matches F_n **only through λ log n**.
- **Equations of state and singular fluctuation** ([Watanabe 2010](https://jmlr.org/papers/v11/watanabe10a.html)).
  - E[G_n] = L(w₀) + ((λ − ν)/β + ν)/n + o(1/n). At β = 1 this is λ/n. (The general-β form is quoted from memory and not re-checked this session.)
  - The relation (2.3) gives E[nβ(E_w L_n − L_n(w₀))] → λ − βν.
  - WAIC estimates G_n without knowing λ or ν.
  - These are the theorems behind the LLC estimator's choice β* = 1/log n, which sends βν → 0.
- **LLC** ([Lau et al. 2023/AISTATS 2025](https://arxiv.org/abs/2308.12108), checked): λ(w*) at a local minimum w* of K, estimated by

      λ̂ = nβ*(E_{w|w*,β*,γ}[L_n] − L_n(w*)),

  using SGLD. It is invariant under local diffeomorphisms (App. C). Its equality with λ(w*) rests on localization conditions.

### 4.3 Developmental interpretability: what a "phase transition" is

**DERIVATION (Bayesian phase transition).** Suppose regions W_α have local optimal losses L_α and coefficients λ_α, so that F_α(n) ≈ nL_α + λ_α log n. With L₁ > L₂ and λ₁ < λ₂, the posterior moves from region 1 to region 2 near the n_c solving

    n_c(L₁ − L₂) = (λ₂ − λ₁) log n_c.                                    (4.1)

The location is uncertain by O(1) free-energy terms. Multiplicities m_α add log log n corrections.

| Claim | Status |
|---|---|
| Posterior transitions governed by (4.1) | DERIVATION from THEOREM |
| SGD stages co-occur with LLC changes | Empirical ([Hoogland et al. 2024](https://arxiv.org/abs/2402.02364), checked) |
| SGD transitions ↔ Bayesian transitions | Toy-model evidence ([Chen et al. 2023](https://arxiv.org/abs/2310.06301)). [Lee et al. 2025](https://arxiv.org/abs/2510.12071) call the Bayesian–SGD link "the primary theoretical gap". |
| SGD steady state ≈ tempered posterior | Conditional theorem under local homogeneity and isotropic diffusion along degenerate directions ([Hennick & De Baerdemacker, ICLR 2026](https://arxiv.org/abs/2503.22478), checked) |

### 4.4 Where SLT applies, where it is a heuristic, where it does not apply

| Setting | Verdict |
|---|---|
| Tempered posterior, analytic model, compact W, φ > 0 at optima, iid, n → ∞ | **Applies (theorem)** |
| tanh, sigmoid, softplus, SiLU, GELU, softmax attention, SwiGLU, linear/diagonal SSMs, deep linear networks, reduced-rank regression | Analyticity holds (normalizations away from zero). Exact RLCTs are known for reduced-rank regression (Aoyagi & Watanabe 2005) and deep linear networks ([Aoyagi 2024, *Neural Networks* 172:106132](https://pubmed.ncbi.nlm.nih.gov/38278091/)). Why singularity matters for deep learning: [Murfet, Wei et al. 2020](https://arxiv.org/abs/2010.11560). |
| ReLU, ReLU² (RWKV FFN), max, top-k MoE, hard quantization | Theorem not established (non-analytic) |
| Separable logistic | Does not apply (no compact minimizer) |
| SGD/Adam endpoints read as posterior samples | Heuristic (§5). λ(w*) is still a well-defined geometric number if w* is a local min of K; report ‖∇L(w*)‖. |
| SGLD-estimated LLC | Biased estimator (εN/B·λ_max bias, lab result; regular model) |
| Finite n, d ≫ n | Scale-dependent effective exponents (§3.2, hypothesis) |
| Restricted (LoRA, layer) LLC | A valid sub-model; not "the network's λ" |
| Non-realizable distillation | Conditional on relatively finite variance (§7.5) |
| RL, non-iid streams | Does not apply as stated |
| Transformers or SSMs at scale | Conditional heuristic; logged covariate only |

### 4.5 Bridge B6 (narrowed): the Fisher is the parameter-observability Gramian; non-identifiability and degeneracy are related but not identical

**DERIVATION.** For y = f(x;θ) + ε with ε ~ N(0, σ²I),

    I(θ) = σ⁻² E_x[J_θᵀJ_θ],

which is the noise-weighted observability Gramian of the parameter from input–output data. At regular points, local identifiability ⇔ nonsingular information ([Rothenberg 1971, *Econometrica* 39(3):577–591](https://ideas.repec.org/a/ecm/emetrp/v39y1971i3p577-91.html)). At a realizable point, I = H, so ker I = ker H gives the first-order unobservable directions.

**v0 claimed "three names for one thing"; that equivalence is withdrawn.** It fails in both directions:

- **Degenerate but identifiable.** K = x⁴: H = 0, but W₀ = {0}, so the parameter is identifiable. Only *first-order* observability fails.
- **Non-identifiable but nondegenerate.** Discrete symmetries, such as permuting hidden units, give isolated equivalent minima, each with full-rank H.

**What survives:**

- **first-order unobservability ⇔ Hessian/Fisher degeneracy**, at realizable points;
- a **non-trivial continuous fibre** of θ ↦ p(·|θ) through θ* implies degeneracy;
- **degeneracy does not imply non-identifiability.**

### 4.6 Worked example and correction: the RLCT of a diagonal LTI SSM

**Setup.** A real diagonal SISO model with n modes, under white-noise input and Gaussian noise:

    K(w) ∝ Σ_k (g_k(w) − g*_k)²,    g_k = Σ_i p_i λ_i^k,    |λ_i| ≤ ρ < 1.

The truth has r distinct nonzero modes.

**(i) Generic true point** (unused p_i = 0, with poles distinct from the active ones). The point is Morse–Bott with rank H = 2r + (n−r), so λ_local = r + (n−r)/2. Numerically: draft 1.486; reviewer re-run 1.502.

**(ii) Coincident or split stratum, n = 2, r = 1 (DERIVATION, confirmed by the reviewer).**

- Write c_j = p₁b^j + q·a^j for the coefficients on the independent sequences C(k,j)λ*^{k−j}, and take v = c₁ as a coordinate (valid because p₁ ≠ 0).
- Modulo v: c₂ ≡ q·a²(p₁ + q)/p₁ and c₃ ≡ q·a³(1 − q²/p₁²) ∈ (q·a²).
- So the ideal is (u, v, q·a²), and

      λ = 1 + RLCT(q²a⁴) = 1 + 1/4 = 5/4,    m = 1.

- The same holds in the (b_i, c_i) parameterization.

**(iii) NUMERICAL confirmation.**

| Method | Slope |
|---|---|
| Draft Monte Carlo, split point | 1.248–1.293 |
| Reviewer re-run, split point | 1.248 |
| **Reviewer fibre minimization** (m(q,a) ≈ 4.21·q²a⁴, reaching ε = 10⁻¹⁶) | **1.250** on [1e-16, 1e-14], in both the 0.1 and 0.3 boxes |

**Global RLCT for (n, r) = (2, 1): λ = 5/4.** This assumes the prior puts mass near the coincident or split configurations.

- Ordered or gapped pole parameterizations put that locus on the boundary of W, which leaves the exponent the same.
- A prior or initialization law that excludes the locus changes the global value.

**Verdict on the repo.**

- Review 02's "it is an equality" and "do not lower λ; they raise m" are **wrong**.
- Its inequality λ ≤ r + (n−r)/2 is **literally true**, and its Monte Carlo values are correct **generic-point local** coefficients.
- The same caveat applies to its complex-mode "n + r".
- `plans/claude/PLAN.md` line 78 should say "**local LLC at a generic true parameter**".
- The Codex counterexample is right.

**Caveat on the numerics.** The impulse response is truncated at T = 40. That matters below ε ≈ 10⁻⁸ when |λ| ≈ 0.8. It does not matter for these cases (λ* = 0.5).

**Why it matters.**

1. A "staircase" of LLC against Hankel rank is a statement about *generic local* values.
2. Pole coalescence has lower free energy, so as n → ∞ the posterior concentrates there (mass ratio ~ n^{1/4} for (2,1)). Whether SGD goes there is an open E1-vs-E2 question (C1b).

---

## 5. Bridge B7 (KNOWN, conditional on C = A): when SGD's stationary law is not the posterior

### 5.1 Setting and result

Assume a quadratic basin δ = θ − θ*, a constant step η, batch size B, gradient-noise covariance C, and a preconditioner P. Then δ_{k+1} = (I − ηPA)δ_k − ηPζ_k, with Cov ζ = C/B.

**KNOWN** ([Mandt, Hoffman & Blei 2017](https://arxiv.org/abs/1704.04289): the OU approximation, and the optimal preconditioner ∝ C⁻¹, which is natural gradient when C = A). In the commuting case with **C = A** and P = A^{−α}, the exact discrete stationary covariance is

    Σ = (η/2B)·A^{−α}·(I − (η/2)A^{1−α})^{−1}  ≈  (η/2B)·A^{−α}.                 (5.2)

The case **C = A** describes a well-specified model sampled from the population at θ*.

| α | Σ | Posterior-shaped, ∝ A⁻¹? |
|---|---|---|
| 0 (plain SGD) | (η/2B)·I | no |
| 1/2 (idealized full-matrix √-preconditioning) | (η/2B)·A^{−1/2} | no |
| 1 (natural gradient) | (η/2B)·A^{−1} | **yes**, with β_eff = 2B/(ηn) |

**NUMERICAL (draft, re-confirmed by the reviewer's exact formula).** Relative errors 0.054, 0.013 and 0.037. The α = 0 residual equals the predicted ηa_max/2 = 0.05. The plain-SGD "LLC-like" quantity tr(AΣ)B/η is 5.88, close to tr A/2, against d/2 = 1. Under natural gradient it is 1.02.

### 5.2 The conclusion depends on C = A, and flips under isotropic noise

With **isotropic noise C = σ²I**, plain SGD gives Σ = (ησ²/2B)·A⁻¹, which **is posterior-shaped**.

So the statement "plain-SGD fluctuations measure total curvature, not λ" holds **because** C ∝ A. It is not a general property of SGD. Real networks have C ≠ A, for several reasons:

- misspecification;
- label noise;
- multi-epoch data reuse;
- interpolation, where C → 0 at zero loss.

Any use of (5.2) must first measure C against A (for example, their eigenbasis alignment), and should not presume either case.

### 5.3 What really happens along W₀ (beyond the linear OU picture)

1. **Relaxation time.** In the linear picture, directions with curvature a_i → 0 have relaxation time 1/(ηa_i) → ∞, so the stationary value is never reached. "No first-order diffusion along W₀" is a statement about this limit.
2. **Second-order drift along W₀ (THEOREM-level, cited).** Near a manifold of minima, label-noise and minibatch SGD drift *deterministically* along W₀ toward smaller tr(H), or a noise-weighted version of it. References:
   - [Blanc, Gupta, Valiant & Valiant 2019](https://arxiv.org/abs/1904.09080);
   - [Damian, Ma & Lee 2021](https://arxiv.org/abs/2106.06530);
   - [Li, Wang & Arora 2021/ICLR 2022](https://arxiv.org/abs/2110.06914).

   SGD's position on W₀ is therefore *not* arbitrary. It follows a computable implicit regularizer. This bears directly on C1b and C3.
3. **Non-equilibrium.** Beyond quadratics, SGD's stationary law generally carries probability currents and is not Gibbs for any potential ([Chaudhari & Soatto 2018](https://arxiv.org/abs/1710.11029); [Kunin et al. 2021](https://arxiv.org/abs/2107.09133)). This is the general E3 ≠ E2 statement. §5.1 is just its simplest instance.
4. **Attraction to invariant sets.** Stochastic collapse ([Chen, Kunin et al. 2023](https://arxiv.org/abs/2306.04251)): SGD noise attracts dynamics toward low-noise invariant sets, such as sparse or low-rank subnetworks.

### 5.4 The correct framing of "LLC at an SGD checkpoint"

λ(w*) is a geometric invariant of K at w*, well-defined if w* is a local minimum of K. What is heuristic is reading the posterior theorems into the SGD-trained predictor. Where SGD lands on W₀ is governed by §5.3 (drift toward flat, low-noise regions), not by posterior typicality.

---

## 6. Layer (d): training as a controlled system

### 6.1 Plant, actuators, sensors

| Element | Meaning |
|---|---|
| State | (θ, m) |
| Actuators | η, B, π ∈ Δ^J, regularizer weights, constraint multipliers |
| Disturbance | noise C/B |
| Sensors | losses, behavioral metrics, logged LLC |

The meaningful notion is **local behavioral controllability**: the rank of ∂b(θ_K)/∂u_{0:K−1}. Parameter controllability is irrelevant and generically fails along W₀.

### 6.2 Bridge B8: susceptibility vs influence

**DERIVATION.** Under a Laplace posterior with covariance (nβA)⁻¹, χ = d/dv E_v[O] ≈ −(1/n)·G_O A⁻¹ G_ℓᵀ. That is the *exact* classical influence function.

**Corrected reading.** The Bayesian influence function (BIF) can differ from the **exact** classical influence only through non-Laplace (singular or multimodal) structure and O(1/n) corrections. In practice classical influence is computed **approximately** (EK-FAC, damping, truncated solvers). A posterior-sampling estimator can beat *those approximations* in perfectly regular models.

**Reported comparison** ([Kreer et al. v2](https://arxiv.org/abs/2509.26544), App. C.3, checked): Pythia-14M LDS 0.110 ± 0.112 (BIF) vs 0.221 ± 0.125 (EK-FAC) vs 0.198 ± 0.109 (GradSim). These are within about one SD, so the ordering is not resolved.

### 6.3 Bridge B9: curriculum order as a Lie bracket, and what it does and does not give

**DERIVATION (re-derived and numerically checked by the reviewer).** One step on i then one on j, minus the reverse order:

    θ_ij − θ_ji = η²(H_j g_i − H_i g_j) + O(η³) = η²[g_i, g_j] + O(η³),     [X,Y] = (DY)X − (DX)Y.      (6.1)

The reviewer measured relative errors of 0.27, 0.081, 0.027 and 0.008 at η = 0.1, 0.03, 0.01 and 0.003, which is O(η), as expected. For a behavioral readout,

    Δb_order ≈ η²∇bᵀ(H_j g_i − H_i g_j).

**Single swap.** This is a Taylor identity. It cannot fail at small η if implemented correctly. Any test must reuse **identical minibatches** in both orders; otherwise O(√η) noise differences swamp the O(η²) signal.

**Block curricula.** K_i steps on i, then K_j steps on j. By the Baker–Campbell–Hausdorff formula, the order effect is ≈ K_iK_jη²[g_i, g_j] + O((Kη)³) for frozen fields. **The small parameter is Kη·λ_max**, not η, and it is ≫ 1 for realistic blocks. Then the expansion is unusable, and the order effect needs a full solve. That is the scientifically interesting case.

**Control-theoretic reading (corrected).** With π ∈ Δ^J, the vector field is f₁ + Σ_{j≥2}π_j(f_j − f₁) with f_j = −∇L_j. That is **control-affine with drift**, not driftless.

- One step reaches the **convex hull** of {−∇L_j}, not a cone.
- The Lie-algebra rank condition then gives **accessibility** (the reachable set has nonempty interior), **not** small-time local controllability.
- Chow–Rashevskii needs symmetric controls, which nonnegative mixture weights are not.

### 6.4 Constraints during training

- **KKT / primal–dual.** PAC constrained learning, with a near-zero duality gap for rich parameterizations ([Chamon & Ribeiro 2020](https://arxiv.org/abs/2006.05487)). THEOREM under their assumptions.
- **Safe gradient flow (DERIVATION).** For the constraint h(θ) ≥ 0:

      v* = −g + max(0, (∇hᵀg − αh)/‖∇h‖²)·∇h.

  - In continuous time this gives forward invariance under the CBF conditions ([Ames et al.](http://ames.caltech.edu/ames2019control.pdf)).
  - Discrete steps need a margin: ∇hᵀv ≥ −αh + (ηM_h/2)‖v‖², with M_h ≥ ‖∇²h‖.
  - Stochastic estimates of h make the guarantee probabilistic.
  - The guarantee covers only h measured on the data used to estimate h.

### 6.5 Hyperparameter control = backprop through training

**DERIVATION.** The discrete Pontryagin costate is p_k = (I − η_kH_k)ᵀp_{k+1}, and the hypergradient is ∂J/∂u_k = (∂T/∂u)ᵀp_{k+1}.

- At EoS, eigenvalues near −1 prevent the costate from decaying.
- Along ker H, the costate is preserved exactly.

So long-horizon hypergradients are ill-conditioned, by structure.

---

## 7. Layers (e)+(f): latent state, interpretability as observability, delta-rule SSMs

### 7.1 Bridge B10 (KNOWN): a gated delta-rule layer is an online learner

The **Gated DeltaNet** update is

    S_t = S_{t−1}·α_t(I − β_t k_tk_tᵀ) + β_t v_tk_tᵀ,    o_t = S_tq_t,

with α, β ∈ (0,1) and L2-normalized q and k (projection → short convolution → SiLU → L2 norm). This is Eq. 10 of [Yang, Kautz & Hatamizadeh 2024](https://arxiv.org/abs/2412.06464), checked.

That paper states that the delta rule optimizes ½‖S k_t − v_t‖² "via test-time stochastic gradient descent". The same framing appears in:

- [DeltaNet (Yang et al. 2024)](https://arxiv.org/abs/2406.06484);
- [Longhorn (Liu et al. 2024)](https://arxiv.org/abs/2407.14207);
- [TTT (Sun et al. 2024)](https://arxiv.org/abs/2407.04620).

For α < 1 the decomposition is S_t = α_t(S_{t−1} − β_t∇ℓ_t) + (1 − α_t)β_tv_tk_tᵀ (algebra checked by the reviewer). So **β is the inner learning rate and α the forgetting factor**, and persistent excitation of the keys governs inner convergence.

### 7.2 Bridge B11: non-expansive transitions, and their exact conditions

**DERIVATION (checked to 10⁻¹⁵ by the reviewer).** With **‖k_t‖ = 1 exactly** and β ∈ [0,2]:

    ‖α_t(I − β_tk_tk_tᵀ)‖₂ = α_t.

So for identical layer inputs, ‖ΔS_t‖_F ≤ (Π_u α_u)·‖ΔS_0‖_F.

**Conditions and failures.**

1. **Exact unit norm is required.** Most kernels compute x/√(‖x‖² + ε), so ‖k‖ < 1 slightly; that case is harmless. Without normalization, β‖k‖² > 2 gives a norm above α. Counterexample (checked): ‖k‖ = 1.2, β = 1.5 gives 1 − β‖k‖² = −1.16, so the norm is **1.16·α**.
2. **Bounded state needs ᾱ < 1.** Then sup_t‖S_t‖_F ≤ ‖S_0‖ + β̄v̄/(1 − ᾱ).
3. **At α ≡ 1 the state is unbounded over adversarial sequences.** This is not merely unproven. Kaczmarz-type writes S k_t ≈ v_t with nearly parallel keys and different values force ‖S‖ ~ 1/angle.
4. **Coverage.**
   - Covers GDN, DeltaNet (with normalized keys), and Mamba-2 (A_t = α_tI).
   - **RWKV-7** ([Peng et al. 2025](https://arxiv.org/abs/2503.14456)) has a non-symmetric transition. Whether its norm is ≤ 1 is **open**.

### 7.3 Bridge B12: Gramians, null edits and the persistence functional, with corrected wording

Fix one layer's realized input sequence. Each **row** s_i of S is a separate LTV system:

    s_{i,t} = s_{i,t−1}A_t + v_{i,t}·(β_tk_t)ᵀ,

with scalar input v_{i,t}, input vector b_t = β_tk_t, and transition A_t = α_t(I − β_tk_tk_tᵀ). Write Φ(s,t) = A_{s+1}⋯A_t.

**Gramians at a common reference time T (DERIVATION).**

    reachability (input-normalized):   W_c(T) = Σ_{s≤T} Φ(s,T)ᵀ b_s b_sᵀ Φ(s,T)
    actual-excitation (row i):         W_c^{(i)}(T) = Σ_{s≤T} v_{i,s}² Φ(s,T)ᵀ b_s b_sᵀ Φ(s,T)
    observability over [T, T+H]:       W_o(T) = Σ_{t=T+1}^{T+H} Φ(T,t) q_t q_tᵀ Φ(T,t)ᵀ

- v0 mixed W_c(T) with W_o from time 0, and called the v²-weighted matrix a reachability Gramian. Both are corrected here: the weighted matrix is an **excitation** Gramian.
- Products λ_i(W_c(T)W_o(T)) taken at the same T are invariant under consistent state-coordinate changes. They are conditional on the realized sequence, since the future A_t and q_t depend on future inputs.

**Null edits: a tautology with a computable null space.** An edit ΔS at time T with ΔS·W_o(T) = 0 leaves the layer's outputs over [T, T+H] unchanged. That is true by definition. Its value is that **ker W_o(T) is computable**, so it gives a cheap *implementation check* for intervention code, not evidence about concepts.

- **It does carry over to free-running generation.** If the layer's outputs are unchanged on the realized trajectory, then with the same sampling seed the emitted tokens are unchanged, so the closed loop is unchanged.
- v0 said no statement transfers to generation. That was wrong for this exact statement; §7.4 is corrected accordingly.

**Persistence functional (DERIVATION, exact per sequence).** The contribution of the write at s to the output at t is

    β_s v_s (k_sᵀΦ(s,t)q_t),     with   |k_sᵀΦ(s,t)q_t| ≤ Π_{u=s+1}^{t} α_u.

It decomposes into decay, overwrite and readout alignment. It is **a diagnostic, not a certificate**: input-dependent gates can be held open adversarially. This agrees with reviews 02 and 03.

### 7.4 The whole stack (corrected)

- **Prefill / teacher forcing.** The stack is a causal cascade with no feedback from higher layers to lower ones, so per-layer incremental bounds compose. This needs ᾱ < 1 and finite gate/write Lipschitz constants L_A, L_B:

      ‖ΔS_t‖ ≤ ᾱ‖ΔS_{t−1}‖ + (S̄·L_A + L_B)‖Δx_t‖.

- **Free-running generation.** The loop closes through a discontinuous map (argmax or sampling).
  - **Incremental-stability and gain bounds do not transfer.** A small output perturbation can flip a token.
  - **Exact identities on the realized trajectory do transfer**, given a fixed seed: the null edits of §7.3, and the per-sequence persistence functional evaluated on the generated sequence.

### 7.5 SLT for attention→GDN conversion and fine-tuning

1. **Non-realizability.** Fixed-state students cannot represent exact long-range retrieval, so the relatively finite variance condition (§4.1(3)) must be checked.
   - **Test:** compare the output distributions of several seeds that reach equal teacher-KL. Distinct distributions flag a possible failure of the condition.
2. **Symmetry counting (DERIVATION, at critical points only).** A continuous symmetry with s-dimensional orbits makes K constant on orbits.
   - At a **critical point** (∇K = 0), the orbit tangents lie in ker H.
   - At a non-converged checkpoint, H·ξ ≠ 0 for symmetry generators ξ in general.
   - Predicted counts:
     - attention q/k block: GL(d_k), so s = d_k²;
     - with RoPE on d_rot dimensions (distinct frequencies): s = d_rot + (d_k − d_rot)²;
     - GDN with SiLU before L2-norm: only per-channel rescaling between the projection and a linear, bias-free depthwise conv, so s ≈ d_k per q and k path.
   - Report Hessian nullity **restricted to symmetry tangents**, plus ‖∇K‖ and a stated eigenvalue threshold. Data-dependent degeneracies add further nullity.
3. **Fine-tuning transitions.** (4.1) applies to a LoRA-restricted sub-model's *posterior*. For SGD it is a heuristic, and SGD's own drift (§5.3) may select differently. Logged, never gated.

---

## 8. What does NOT unify, and why

| Pair | Reason |
|---|---|
| Hankel singular values ↔ LLC | Different objects; counterexamples in *Learning Geometry* and §4.6. **ANALOGY-ONLY**, beyond generic-point counting. |
| "Singular" (HSV) ↔ "singular" (SLT) | Shared word only. |
| Uniform contraction to a prescribed point ↔ SLT degeneracy | In tension along W₀ (§3.6). Convergence to *some* point still holds (Łojasiewicz). |
| Identifiability ↔ degeneracy | Related, not equivalent (§4.5: x⁴; permutation copies). |
| SLT ↔ implicit bias on separable data | No compact minimizer. |
| SLT ↔ NTK, mean field, μP limits | SLT has fixed d. Only symmetry-singularity analogies. |
| SLT ↔ ReLU, MoE, quantized networks | Analyticity fails; theorem not established. |
| SLT ↔ RL, non-iid streams | iid fails. |
| Posterior (E2) ↔ SGD (E1/E3) | Non-equilibrium currents in general. Under C = A, not posterior-shaped unless natural gradient. Under isotropic noise, posterior-shaped in the quadratic regime. Drift along W₀ toward flatter regions. |
| LLC ↔ safety, alignment, semantics | No relation. Log, never gate. |
| Incremental-stability bounds on a prefill cascade ↔ free-running generation | A discontinuous feedback loop. (Exact per-trajectory identities do transfer.) |
| Interpretability features ↔ identified latents | Defined only up to realization equivalence. Observability and controllability are necessary for behavioral relevance, not sufficient for semantics. |
| PAC-Bayes (Gibbs) ↔ the deterministic SGD network | Needs derandomization. |
| Mutual information I(S;W) ↔ λ log n | I(S;W) = O(1) at fixed truth (§2.3). |
| Finite-n LLC ↔ asymptotic λ | Scale-dependent (§3.2, hypothesis). |

---

## 9. The regime map (summary)

| Regime (checkable flag) | Valid descriptions | Invalid there |
|---|---|---|
| Lazy (NTK stable) | Linearized model with λ_lin = rank/2; susceptibility = exact influence | SLT higher-order effects on the trajectory |
| μP / feature learning at large width | Feature-learning limit, hyperparameter transfer | Lazy-regime claims |
| Small step, quadratic basin (ηλ_max ≪ 1) | OU/Lyapunov (5.2), **conditional on measured C vs A**; drift along W₀ (§5.3); single-step bracket (6.1) | Block-curriculum expansions with Kη ≳ 1 |
| Edge of stability (ηλ_max ≈ 2) | Self-stabilization theory (Damian, Nichani & Lee); sharpness-reduction analysis (Arora, Li & Panigrahi) | Descent-lemma Lyapunov, OU, contraction, bracket expansions |
| Bayesian posterior, analytic, compact, φ > 0, iid, n → ∞ | SLT: F_n, WBIC (to λ log n), equations of state (λ − βν), transition law (4.1); KL/MDL = λ log n | — |
| Finite n, d ≫ n | Scale-dependent effective exponents; SGLD bias ∝ εN/B·λ_max | Asymptotic formulas as quantitative predictions |
| Delta-rule layer, fixed realized inputs, ‖k‖ = 1 | Exact LTV analysis: non-expansiveness, same-time Gramians, computable null space, persistence functional | Sequence-independent certificates |
| Free-running generation | Exact per-trajectory identities (fixed seed) | Incremental-stability bounds |

---

## 10. References

Marked "checked" if verified this session (by me, the citation agents, or the reviewer). IDs for all newly added items were verified by a citation agent.

**SLT and learning coefficients**
- Watanabe 2013, WBIC, JMLR 14: https://www.jmlr.org/papers/v14/watanabe13a.html (checked)
- Watanabe 2010, WAIC, equations of state, JMLR 11: https://jmlr.org/papers/v11/watanabe10a.html (title checked)
- Watanabe 2009, *Algebraic Geometry and Statistical Learning Theory*, Cambridge University Press
- Lau et al., LLC: https://arxiv.org/abs/2308.12108 (checked)
- Furman & Lau: https://arxiv.org/abs/2402.03698
- Chen & Murfet, modes: https://arxiv.org/abs/2504.18048 (checked)
- Murfet, Wei et al. 2020, *Deep Learning is Singular, and That's Good*: https://arxiv.org/abs/2010.11560 (checked)
- Aoyagi & Watanabe 2005, reduced-rank regression, *Neural Networks* 18(7):924–933 (checked)
- Aoyagi 2024, deep linear networks, *Neural Networks* 172:106132: https://pubmed.ncbi.nlm.nih.gov/38278091/ (checked)

**Developmental interpretability and attribution**
- Hoogland et al.: https://arxiv.org/abs/2402.02364
- Chen, Lau, Mendel, Wei, Murfet: https://arxiv.org/abs/2310.06301
- Wang et al. (refined LLC): https://arxiv.org/abs/2410.02984
- Lee et al. (influence dynamics): https://arxiv.org/abs/2510.12071
- Kreer et al. (BIF): https://arxiv.org/abs/2509.26544
- Urdshals et al. (MDL): https://arxiv.org/abs/2510.12077

**SGD dynamics vs the posterior**
- Hennick & De Baerdemacker (ICLR 2026): https://arxiv.org/abs/2503.22478
- Mandt, Hoffman & Blei: https://arxiv.org/abs/1704.04289
- Chaudhari & Soatto: https://arxiv.org/abs/1710.11029
- Kunin et al. (limiting dynamics): https://arxiv.org/abs/2107.09133
- Blanc et al.: https://arxiv.org/abs/1904.09080
- Damian, Ma & Lee: https://arxiv.org/abs/2106.06530
- Li, Wang & Arora: https://arxiv.org/abs/2110.06914
- Chen, Kunin et al. (stochastic collapse): https://arxiv.org/abs/2306.04251
- Li, Malladi & Arora: https://arxiv.org/abs/2102.12470

**Edge of stability**
- Cohen et al.: https://arxiv.org/abs/2103.00065
- Damian, Nichani & Lee (self-stabilization): https://arxiv.org/abs/2209.15594
- Arora, Li & Panigrahi: https://arxiv.org/abs/2205.09745

**Convergence of descent methods**
- Absil, Mahony & Andrews 2005, SIAM J. Optim. 16(2):531–547: https://doi.org/10.1137/040605266

**Infinite-width limits and implicit bias**
- Yang & Hu (μP): https://arxiv.org/abs/2011.14522
- Jacot et al.: https://arxiv.org/abs/1806.07572
- Chizat, Oyallon & Bach: https://arxiv.org/abs/1812.07956
- Mei, Montanari & Nguyen: https://arxiv.org/abs/1804.06561
- Soudry et al.: https://arxiv.org/abs/1710.10345

**Generalization theory**
- Germain et al. 2016: https://arxiv.org/abs/1605.08636
- Alquier 2021: https://arxiv.org/abs/2110.11216
- Dziugaite & Roy: https://arxiv.org/abs/1703.11008
- Zhang et al.: https://arxiv.org/abs/1611.03530
- Nagarajan & Kolter: https://arxiv.org/abs/1902.04742
- Xu & Raginsky: https://arxiv.org/abs/1705.07809
- Hardt, Recht & Singer: https://arxiv.org/abs/1509.01240

**Identification**
- Rothenberg 1971, *Econometrica* 39(3):577–591

**Control of training**
- Chamon & Ribeiro: https://arxiv.org/abs/2006.05487
- Ames et al. (CBF): http://ames.caltech.edu/ames2019control.pdf
- Lessard, Recht & Packard: https://arxiv.org/abs/1408.3595

**Delta-rule and linear-recurrent architectures**
- Gated DeltaNet: https://arxiv.org/abs/2412.06464 (checked)
- DeltaNet: https://arxiv.org/abs/2406.06484
- Longhorn: https://arxiv.org/abs/2407.14207
- TTT: https://arxiv.org/abs/2407.04620
- RWKV-7: https://arxiv.org/abs/2503.14456
- Grazzi et al.: https://arxiv.org/abs/2411.12537
- Miller & Hardt: https://arxiv.org/abs/1805.10369

---

## 11. Changes from v0 (each mapped to the Review v0 item it answers)

| # | Section | Change | Review item |
|---|---|---|---|
| 1 | §2.3 B2 | **Error fixed.** Dropped "MI ≈ λ log n": I(S;W) = O(1) at fixed truth, and λ log n is only a loose upper bound. Kept KL(Q*‖φ) and MDL ≈ λ log n. Replaced step 3 with the equations of state, E[nβ(E_wL_n − L_n(w₀))] → λ − βν, with a regular-case check showing it vanishes at β = 1. | B2 (WRONG), §3 item 5 |
| 2 | §3.2 B4 | **Error fixed.** Withdrew "slope 1.00 ± 0.01". Reported the reviewer's 1.05–1.13 re-run and the fibre method converging to 1.250. Restated B4 as a hypothesis, not a bound, listing how exponents can fall below rank/2. Noted that volume slope ≠ SGLD-LLC. | B4 (NOT REPRODUCED), §3 item 7 |
| 3 | Companion D.1 | **Error fixed.** Hard-coded box removed; separate 0.1-box and 0.3-box cases (C1 and C3). | B4 point 2 |
| 4 | §2.2 B1 | Relabelled KNOWN (Catoni; Germain et al. 2016; Alquier Thm 2.1). Exact only for loss in [0,C] and a pre-chosen t. NLL needs truncation or a sub-Gaussian slack. The O(1) slack βC²/8 makes the fixed-β λ term cosmetic. Regime t ≍ √(n log n) stated as a conjecture. | B1 |
| 5 | §3.1 B3 | Tightened to r/2 ≤ λ ≤ r/2 + (d−r)/4 (cubic term vanishes since g ≥ 0); tight at K = a²b². Added the φ(θ*) > 0 condition, and fixed §4.1/§4.2 consistently. | B3; §3 item 1 |
| 6 | §3.6 B5 | Kept the point statement. Dropped "never point-valued". Added Łojasiewicz convergence to a single point (Absil–Mahony–Andrews 2005), exponential in the Morse–Bott case and polynomial with quartic g. Restated the limitation as no uniform robust contraction to a *prescribed* point. | B5 |
| 7 | §4.5 B6 | Removed the "three names for one thing" equivalence. Added both counterexamples (x⁴; permutation copies) and Rothenberg 1971. | B6 |
| 8 | §5 B7 | Relabelled KNOWN (Mandt et al. 2017). Exact discrete Σ formula. Stated that the result needs C = A, and that isotropic noise flips it to posterior-shaped. Added the drift along W₀ (Blanc 2019/20; Damian–Ma–Lee 2021; Li–Wang–Arora 2022), non-equilibrium currents (Chaudhari–Soatto 2018; Kunin et al. 2021), and the relaxation-time remark. | B7 |
| 9 | §6.2 B8 | BIF can beat *approximate* influence functions in regular models. The singular-structure claim is restricted to *exact* influence. | B8 |
| 10 | §6.3 B9 | Drift form f₁ + Σπ_j(f_j − f₁); convex hull, not cone; accessibility, not controllability; Kη·λ_max as the small parameter for block curricula (BCH); identical minibatches required in tests. | B9 |
| 11 | §7.1 B10 | Relabelled KNOWN, citing GDN Eq. 10 and its "test-time SGD" statement, plus DeltaNet, Longhorn and TTT. | B10 |
| 12 | §7.2 B11 | Unit-norm requirement, with the counterexample ‖k‖ = 1.2, β = 1.5 giving 1.16. The state is unbounded at α ≡ 1 (Kaczmarz with near-parallel keys). | B11 |
| 13 | §7.3–7.4, §8, §9 B12 | Null edits labelled a tautology with a computable null space, which **does** transfer to free-running generation (fixed seed). §7.4/§8/§9 corrected to say only incremental bounds fail there. Gramians now defined at a common reference time T, per row; input-normalized vs actual-excitation distinguished. | B12 |
| 14 | §7.5 | Symmetry-nullity prediction restricted to critical points; report restricted nullity, ‖∇K‖ and a threshold. | C5(c) |
| 15 | §3.3 | "SLT trivial in lazy regime" → λ_lin is a property of the linearized model; SLT structure is invisible to the trajectory. Added μP / Tensor Programs. | §3 items 2, 4 |
| 16 | §3.4, §9 | EoS no longer "empirical only": added self-stabilization (Damian–Nichani–Lee 2022) and Arora–Li–Panigrahi 2022. | §3 item 3 |
| 17 | §4.2, §4.4 | Equations of state and WAIC; Aoyagi 2024 deep-linear RLCTs; Murfet, Wei et al. 2020; Watanabe 2009 book. | §3 items 5, 6 |
| 18 | §4.6 | Kept and strengthened the correction to `reviews/02`: λ = 5/4 at (2,1), confirmed by the fibre method at ε = 10⁻¹⁶. "r + (n−r)/2" is a generic-point local coefficient only; review 02's inequality is literally true, but its "equality" and "raise m" claims are wrong. Added caveats on prior support and T = 40 truncation. | §4.6 (CORRECT) |
| 19 | Companion | C1 for (2,1) is now a derivation; the general coalescence claim is downgraded; the algebraic ideal computation is proposed. C2 has a scale-tied rank threshold. C3 test re-specified and its direction reconsidered. C4 now tests block curricula. C5(a) dropped. | Conjectures C1–C5 |

