# Unifying Theory — Draft v0

*From statistical learning to deep, latent and recurrent learning, with training control, developmental interpretability and singular learning theory used only where they apply.*

Written 2026-10-07. This builds on the five documents in this folder (the "source documents"). They were not modified. A companion file, `Unifying Theory — Open Problems and Tests.md`, holds the claim ledger, the experiments and the falsification criteria.

---

## 0. Read this first

### 0.1 The thesis in one paragraph

No single theorem unifies these fields, and this draft does not claim one. What does unify them is a **shared object**: a learning run is a *controlled stochastic dynamical system on parameter space*. It has three parts:

- an **objective**, which comes from statistics;
- a **geometry**, which comes from the loss landscape and the model's parameterization;
- **actuators**, the training choices an engineer controls.

Each of the theories considered here is a statement about one *functional of the law of this process*, valid in one *limit or regime*. Unifying them means two things:

1. one notation in which every theory's object can be written down;
2. a **regime map** that says exactly when two theories' objects coincide, and what the error is when they don't.

The useful content lies in the bridges, and every bridge below is labelled with its status. Section 8 lists what does **not** unify.

### 0.2 Status labels (used for every bridge)

| Label | Meaning |
|---|---|
| **THEOREM [cite]** | A published result. The assumptions are stated as I understand them from the source; where I checked the primary source this session, it is marked "(checked)". |
| **DERIVATION** | Derived here in full, from stated assumptions. Errors are mine, and the derivation is short enough to check. |
| **NUMERICAL** | A CPU computation run this session. The code is in the companion file. Evidence, not proof. |
| **CONJECTURE** | Plausible, and comes with a falsifiable test. Not established. |
| **ANALOGY-ONLY** | A shared word or shape with no mathematical identity. Kept only to warn against it. |

### 0.3 Commitments inherited from earlier reviews

The user's project reviews ([`State-Space-Decision-Model/reviews/02`, `/03`], `PLAN.md`) rejected three things. This draft does not re-assert any of them:

- **No LLC-as-sensor or control loop.** Lab results showed SGLD susceptibility estimates biased by about 1.3–3.5×, with the bias proportional to εN/B·λ_max(A) and invisible to loss-trace R-hat. Estimates at 1.7B-LoRA scale were not usable. Policy: *log, never gate*.
  - In this draft, LLC appears only as a *Bayesian* quantity with stated assumptions, and as a logged covariate in experiments.
- **No "injection half-life" safety certificate.** Section 7 gives an *exact per-sequence persistence functional* for gated delta-rule states. It is a diagnostic. It is explicitly *not* a certificate: the gates are input-dependent, so an adversary can hold them open.
- **No universal "LLC tracks Hankel rank" law.** On one point I **disagree with review 02**: it says coincident-pole loci "do not lower λ, they raise m". That is wrong for the *global* RLCT. Section 4.6 gives the derivation and numerics, which agree with the Codex counterexample in `analysis/CODEX_CRITICAL_COMPARISON.md`. Review 02's measured value of 3/2 is correct as the *local* coefficient at a *generic* true parameter, and nowhere else.

---

## 1. The shared object

### 1.1 Variables and time axes

The four time axes from the source documents are kept separate:

- **t** is environment time;
- **j** is token position;
- **s** is generation (flow/diffusion) time;
- **k** is the training step.

This draft needs only **k** (training) and **j** (token time inside a recurrent layer).

| Symbol | Meaning |
|---|---|
| q(x) or q(x,y) | data distribution (unknown) |
| p(y\|x,θ), θ ∈ W ⊂ ℝ^d | model; W is the parameter set |
| φ(θ) | prior (Bayesian side) or initialization law (optimization side) |
| ℓ(θ;z) | per-example loss, usually the negative log-likelihood −log p(y\|x,θ) |
| L(θ) = E_q ℓ(θ;z); L_n(θ) = (1/n) Σ_i ℓ(θ;z_i) | population and empirical loss |
| K(θ) = L(θ) − L(θ₀) | excess population loss (a KL divergence when ℓ is the NLL); θ₀ ∈ argmin L |
| W₀ = {θ : K(θ) = 0} | the optimal set; it is *a set*, and singular models make it a variety, not a point |

### 1.2 Learning as a controlled stochastic system

DEFINITION.

    s_{k+1} = U(s_k, u_k, ξ_k),     s_k = (θ_k, m_k),     y_k = h(θ_k) + ν_k

- m_k is the optimizer state (momentum, Adam moments).
- u_k is the **control input**: learning rate η_k, batch size B_k, data-mixture weights π_k ∈ Δ^J, regularizer and constraint multipliers, precision. A curriculum is a sequence u_{0:K}.
- ξ_k is the randomness: minibatch draw, dropout, injected noise.
- y_k are the **measurements**: losses, evaluations, probes, LLC estimates.

For plain SGD with mixture control:

    θ_{k+1} = θ_k − η_k Σ_j π_{k,j} ∇L_j(θ_k) − η_k ζ_k,     E[ζ_k | θ_k] = 0,  Cov(ζ_k | θ_k) = C(θ_k)/B_k      (1.1)

Here C(θ) is the per-example gradient covariance under the current sampling mixture.

The process (1.1) induces a sequence of laws μ_k over θ. Every theory below is about one of three "endpoint" objects. The first job of unification is to never confuse them:

| Endpoint | Object | Theories that speak about it |
|---|---|---|
| **E1: trajectory or point** | θ_K, a random variable | optimization theory, NTK/mean-field, implicit bias, stability-based generalization |
| **E2: Gibbs posterior** | p_β(θ \| D_n) ∝ exp(−nβ L_n(θ)) φ(θ) | Bayesian statistics, **SLT** (λ, WBIC, LLC), PAC-Bayes (as the optimal posterior) |
| **E3: stationary law of (1.1)**, if it exists | μ_∞ with constant u | SGD-as-SDE, Langevin theory, "Almost Bayesian" |

**The central regime question** is when E1 or E3 is close to E2. Section 5 shows that, in the simplest tractable case, E3 is **not** E2 (Derivation 5.2). That is the main reason SLT is a *heuristic*, not a theorem, when applied to SGD-trained networks.

### 1.3 Two notions of "state", and why both are needed

A trained recurrent model has two coupled dynamical systems:

- the **slow system**: training, θ_k over k;
- the **fast system**: the recurrent state S_j inside each layer over tokens j, with dynamics parameterized by θ.

For delta-rule SSMs (Gated DeltaNet, RWKV-7, DeltaNet), the fast system is *itself* an online learner. It runs a least-mean-squares (LMS) gradient step on an in-context regression loss (§7.1). So the tools of §§2–5 apply on *both* time scales. This is the strongest structural reason the user's SSM focus is a natural home for this theory.

### 1.4 Population and empirical quantities

Throughout, "**finite-n**" means a statement about L_n or about a fixed sample size. "**asymptotic**" means a limit n → ∞ with the model held fixed. Most SLT statements are asymptotic in n with d fixed. Modern networks have d ≫ n. That alone limits what SLT can say about them (§4.5).

---

## 2. Layer (a): statistical learning theory, and the one exact join to SLT

### 2.1 What the classical layer gives, and where it fails for overparameterized networks

| Result | Statement (informal) | Object | Status for modern networks |
|---|---|---|---|
| Uniform convergence (VC, Rademacher) | sup over the hypothesis class of \|L − L_n\| is small | the whole hypothesis class | **Vacuous.** The same networks fit random labels ([Zhang et al. 2016](https://arxiv.org/abs/1611.03530)). Even algorithm-dependent two-sided uniform convergence can be vacuous ([Nagarajan & Kolter 2019](https://arxiv.org/abs/1902.04742)). |
| Algorithmic stability | E[gen] ≤ ε_stab. For convex, smooth, Lipschitz losses, SGD has ε_stab ≤ (2L²/n) Σ_t α_t ([Hardt, Recht & Singer 2015](https://arxiv.org/abs/1509.01240)). | the algorithm (E1) | Theorem in the convex case. Non-convex bounds degrade with training length. |
| Information-theoretic | \|E gen\| ≤ √(2σ² I(S;W)/n) for σ-sub-Gaussian loss ([Xu & Raginsky 2017](https://arxiv.org/abs/1705.07809)) | the joint law of data and output | Theorem. I(S;W) can be infinite for a deterministic algorithm with continuous outputs, so it needs noise. |
| PAC-Bayes | E_Q L ≤ E_Q L_n + complexity(KL(Q‖φ)) | a randomized predictor Q | Theorem. **Non-vacuous** for SGD-trained networks once Q is optimized around the SGD solution ([Dziugaite & Roy 2017](https://arxiv.org/abs/1703.11008)). |
| Bayes generalization in singular models | E[G_n] = L(θ₀) + λ/n + o(1/n) for the Bayes predictive at β = 1 under realizability; the general-β form also involves the singular fluctuation ν ([Watanabe 2010](https://jmlr.org/papers/v11/watanabe10a.html)) | the posterior predictive (E2) | Theorem under SLT conditions (§4.1). Not about point estimates. |

The pattern is that **the bounds that survive overparameterization are about distributions over parameters (E2-like objects), not about the hypothesis class.** That is the first structural reason a Bayesian-geometric theory (SLT) is the right complexity layer, *provided* the predictor really is posterior-like.

### 2.2 Bridge B1: the optimal PAC-Bayes value is a tempered free energy

**THEOREM (Donsker–Varadhan variational formula)** For a measurable f with E_φ e^f < ∞:

    log E_φ[e^{f}] = sup_Q { E_Q f − KL(Q‖φ) },

and the supremum is attained by dQ* ∝ e^f dφ.

**THEOREM (Catoni-type bound; form as in [Alquier 2021](https://arxiv.org/abs/2110.11216), §2)** For a loss in [0, C], a fixed t > 0 and δ ∈ (0,1): with probability ≥ 1 − δ over D_n, simultaneously for every Q,

    E_Q L ≤ E_Q L_n + t C²/(8n) + (KL(Q‖φ) + log(1/δ))/t.                (2.1)

**DERIVATION (B1).** Minimize the right side of (2.1) over Q, using the formula with f = −t L_n.

- The minimizer is the Gibbs posterior Q*_t ∝ e^{−t L_n} φ.
- The minimum value is

      B*(t) = [ F_n(t) + log(1/δ) ] / t + t C²/(8n),     F_n(t) := −log ∫ e^{−t L_n(θ)} φ(θ) dθ.      (2.2)

With t = nβ and the NLL loss, F_n(t) is exactly the tempered Bayesian free energy of SLT. So **the best PAC-Bayes certificate, as a functional of the data, equals the free energy per unit temperature plus a fixed slack.** This identity is exact, with no asymptotics. The PAC-Bayes ↔ marginal-likelihood connection itself is known ([Germain, Bach, Lacoste & Lacoste-Julien 2016](https://arxiv.org/abs/1605.08636), checked).

**DERIVATION (B1, asymptotic part, under SLT conditions §4.1, fixed β > 0).** Watanabe's free-energy expansion F_n = nβ L_n(θ₀) + λ log n − (m−1) log log n + O_p(1) ([Watanabe 2013, Thm 2](https://www.jmlr.org/papers/v14/watanabe13a.html), checked) gives

    B*(nβ) = L_n(θ₀) + λ log n/(nβ) − (m−1) log log n/(nβ) + βC²/8 + O_p(1/n).

**What this buys.** In the bound for the Gibbs predictor, the complexity term carries **λ, not d/2**. Degeneracy literally shrinks the certificate.

**What it does not buy.**

1. At fixed β, the slack βC²/8 does not vanish. Taking β_n → 0, for example t = √n, makes the bound consistent but moves outside the fixed-β free-energy theorem. That extension is a **CONJECTURE**: that the expansion holds with log t in place of log n.
2. The bound covers a sample from the Gibbs posterior, **not** the SGD endpoint (see §5).
3. The NLL is unbounded, so a bounded surrogate or a sub-Gaussian variant is needed.

### 2.3 Bridge B2: MDL, mutual information and the learning coefficient are the same number for the Gibbs algorithm

**DERIVATION.** These are three steps, and only the last uses asymptotics.

1. *Golden formula.* For any data-independent φ, I(S;W) = E_S KL(Q_S‖P_W) ≤ E_S KL(Q_S‖φ), because P_W = E_S Q_S minimizes E_S KL(Q_S‖·).
2. *For the Gibbs posterior*, the formula gives KL(Q*_t‖φ) = F_n(t) − t E_{Q*_t}[L_n].
3. *Under SLT conditions*: F_n(t) − t L_n(θ₀) ≈ λ log t, and t·E_{Q*_t}[L_n − L_n(θ₀)] ≈ λ. The second statement is the derivative of the first, which is valid for these Mellin-type expansions; it is also the identity behind the LLC estimator of [Lau et al.](https://arxiv.org/abs/2308.12108). So

       KL(Q*_t‖φ) ≈ λ log t − λ + O_p(log log t).

**Conclusion.** For the Gibbs algorithm, the information the posterior extracts from the data, its PAC-Bayes complexity, and its description length all equal **λ log n** to leading order. In the regular case that is the classical (d/2) log n of BIC and MDL. Combined with Xu–Raginsky, this gives |E gen| ≲ √(2σ² λ log n / n) for the Gibbs algorithm. This matches, and slightly generalizes, the reported MDL ↔ LLC link of [Urdshals et al. 2025](https://arxiv.org/abs/2510.12077) (checked: "extends MDL to singular models").

**Status:** DERIVATION. Steps 1–2 are exact. Step 3 substitutes SLT asymptotics inside an expectation over S, which needs uniform integrability that I have not proved.

**Falsifiable test (companion file, T-B2):** in a model with a known λ (reduced-rank regression, or the LTI toy of §4.6), compute KL(Q*‖φ) by thermodynamic integration and check the slope in log n.

---

## 3. Layer (b): geometry and dynamics

### 3.1 Bridge B3: the Hessian/Fisher rank sandwich (the cleanest link from (b) to (c))

**Setting.** K is analytic, θ* ∈ W₀ is an interior local minimum with K(θ*) = 0, and the prior density is positive and smooth at θ*. H = ∇²K(θ*). For a well-specified NLL model, H equals the Fisher information I(θ*), a standard identity at a realizable point.

**DERIVATION.**

1. By the Morse splitting lemma (Gromoll–Meyer), there are local analytic coordinates (u, v) ∈ ℝ^r × ℝ^{d−r}, with r = rank H, such that K = Σ_{i≤r} u_i² + g(v), where g(0) = 0, ∇g(0) = 0 and ∇²g(0) = 0.
2. The local learning coefficient is invariant under local diffeomorphisms ([Lau et al. 2023, App. C](https://arxiv.org/abs/2308.12108), checked). The volume integral factorizes, so λ(θ*) = r/2 + λ_g, where λ_g is the local RLCT of g at 0.
3. λ_g ≥ 0. Also g(v) ≤ C‖v‖² near 0, so vol{g < ε} ≥ c·ε^{(d−r)/2}, which gives λ_g ≤ (d−r)/2.

Therefore:

    rank(H)/2  ≤  λ(θ*)  ≤  d/2,     with λ(θ*) = rank(H)/2 exactly when g ≡ 0 (the Morse–Bott case: W₀ is locally a manifold of dimension d − r).     (3.1)

**Reading.**

- The Fisher/NTK/Hessian layer determines λ **only in the Morse–Bott case**.
- Everything SLT adds beyond second-order geometry is λ_g, which comes from the higher-order terms. That is where "singular" means something more than "flat".
- The sandwich is cheap to test: compare an LLC estimate against Hessian-rank/2 computed by Lanczos.

### 3.2 Bridge B4: finite-n effective exponent; why practical LLC estimates tend to see rank(H)/2

**DERIVATION.** In the coordinates of §3.1, suppose the prior box has half-width R in v, and the resolution is ε ≈ 1/(nβ), the scale the posterior resolves. If sup_{‖v‖≤R} g(v) ≪ ε, then {K < ε} ≈ {Σu² < ε} × box, so V(ε) ∝ ε^{r/2}. The **effective** exponent is therefore

    λ_eff(n) ≈ rank(H)/2   for   nβ ≪ 1/sup_box g,     and   λ_eff → λ   only for   nβ ≫ 1/sup_box g.

So **a finite-n LLC estimate measures something between rank(H)/2 and λ**. Where it falls depends on n, β, the localization strength γ (which sets R) and the scale of g. This is consistent with the published finding that LLC estimates are "insensitive to modes below a data-dependent threshold" ([Chen & Murfet 2025](https://arxiv.org/abs/2504.18048), checked).

**NUMERICAL (this session, §4.6 toy).** At a coincident-pole true parameter of a 2-mode LTI model, rank H = 2 and λ = 5/4.

- With a local box of half-width 0.1, the volume-scaling slope is **1.00 ± 0.01** over ε ∈ [10⁻⁸, 10⁻⁴]. That is rank(H)/2, because sup_box g ≈ 10⁻⁶.
- Widening the box in the degenerate coordinates to 0.3 raises the slope to **1.13–1.18**, moving toward 5/4.

This is the predicted pre-asymptotic behavior, in a case where the true λ is known.

### 3.3 NTK and lazy training: SLT is trivial there

**DERIVATION.** Linearize the model at initialization: f_lin(x;θ) = f(x;θ₀) + J(x)(θ − θ₀), with Gaussian-noise regression. Then K is exactly quadratic, K = ½(θ−θ*)ᵀ G (θ−θ*) with G = E[J(x)ᵀJ(x)], so g ≡ 0 and λ = rank(G)/2: half the rank of the NTK feature covariance on the data distribution.

In the regime where training stays near the linearization ([Jacot et al. 2018](https://arxiv.org/abs/1806.07572); [Chizat, Oyallon & Bach 2018](https://arxiv.org/abs/1812.07956)), the geometry SLT adds beyond the Fisher is invisible to training.

**CONJECTURE (C-lazy).** Higher-order SLT structure (λ_g > 0) is a *feature-learning* phenomenon: it is relevant to the trajectory only when the dynamics leave the lazy regime.

**Test:** compare LLC estimates against rank(Hessian)/2 across a width or output-scale sweep that moves a small network from lazy to rich. The gap λ̂ − r_H/2 should be ≈ 0 in the lazy limit and positive in the rich regime.

**Mean-field limits** ([Mei, Montanari & Nguyen 2018](https://arxiv.org/abs/1804.06561)) replace θ by a measure over neurons, evolving by a Wasserstein gradient flow. SLT is defined for fixed, finite d. Its relationship to the mean-field limit is **ANALOGY-ONLY**: permutation symmetry is a singularity source in both, and no theorem connects them that I know of.

### 3.4 SGD as an SDE: validity conditions

**THEOREM-level facts used (cited):**

1. SGD is weakly approximated by an SDE dθ = −∇L dt + √(η/B)·C^{1/2} dW over *finite time horizons*, as η → 0, under smoothness and moment conditions. The approximation can fail when the noise scale is large relative to curvature. [Li, Malladi & Arora 2021](https://arxiv.org/abs/2102.12470) (checked title and scope) give conditions and a practical test for failure.
2. **Edge of stability.** Full-batch GD typically drives the top Hessian eigenvalue to ≈ 2/η and stays there, with non-monotone loss ([Cohen et al. 2021](https://arxiv.org/abs/2103.00065), checked).

**DERIVATION (what EoS breaks).** The descent lemma, L(θ − η∇L) ≤ L(θ) − η(1 − ηL_s/2)‖∇L‖², makes L a Lyapunov function **iff** η < 2/L_s. Linear stability of a minimum under GD likewise requires ηλ_max(H) < 2. At the edge of stability both are at the boundary. So the Lyapunov argument, the OU/Lyapunov-equation picture of §5, and contraction arguments for the training map are **all invalid exactly in the regime where large-learning-rate training runs**.

Any "controlled training" theory must therefore carry an explicit regime flag ηλ_max(H) ≪ 1, and that flag can be measured.

### 3.5 Implicit bias

For separable data with logistic loss, GD converges in direction to the max-margin solution while ‖θ‖ → ∞ ([Soudry et al. 2017](https://arxiv.org/abs/1710.10345), checked). **This does not unify with SLT:** the loss has no minimizer in any compact W, so the existence of θ₀ and compactness both fail. The implicit-bias layer describes E1 in regimes where E2 is not even defined without a prior that does the regularizing.

### 3.6 Bridge B5 (negative): singular minima are not contracting

**DERIVATION.** The GD map T(θ) = θ − η∇L(θ) has Jacobian I − ηH at θ*. On ker H the eigenvalue is exactly 1. So a degenerate minimum is **never** an exponentially stable *point* of training. In the Morse–Bott case W₀ is a normally hyperbolic invariant manifold: attractive in the u-directions, neutral along W₀. Where on W₀ training ends up is then decided by noise, higher-order terms and initialization, not by the local contraction rate.

**Consequence.** "Contraction of training" and "SLT degeneracy" are in direct tension. The directions SLT counts as degenerate are exactly those where contraction-based control has *no authority*. Training-control certificates (§6) can at best be **set-valued**, for example convergence to W₀ or to a sublevel set, never point-valued.

---

## 4. Layer (c): singular learning theory, stated precisely

### 4.1 Assumptions (the "fundamental conditions")

The following follows [Watanabe 2013, JMLR 14](https://www.jmlr.org/papers/v14/watanabe13a.html), checked.

1. **Compact parameter set W.** The prior factorizes as φ = φ₁φ₂, where φ₁ ≥ 0 is analytic and φ₂ > 0 is smooth.
2. **Analyticity.** w ↦ log p(x|w) is analytic, extendable to a complex neighborhood of W, with the required moment bounds in x.
3. **Relatively finite variance.** With f(x,w) = log[p(x|w₀)/p(x|w)], we need E[f] ≥ c·E[f²] for some c > 0 (Watanabe's Eq. (12); the name comes from his later book).
   - **Realizability is not required.** The condition holds automatically when the model is realizable or regular.
   - It *can fail* when several optimal parameters represent *different* distributions with the same loss. This matters for non-realizable distillation (§7.4).
4. **Data are iid** from q, and n → ∞ with the model and d fixed.
5. **The object is the Bayesian (tempered) posterior** p_β(w|D_n) ∝ φ(w) Π_i p(x_i|w)^β. It is not an optimizer's output.

### 4.2 The objects

- **RLCT λ and multiplicity m.** −λ is the largest pole of ζ(z) = ∫ K(w)^z φ(w) dw, and m is its order. Equivalently, the prior volume of {K < ε} scales as c·ε^λ(−log ε)^{m−1}. Always 0 < λ ≤ d/2, with λ = d/2 in the regular case.
- **Free energy** (Thm 2 of the 2013 paper, checked):

      F_n = nL_n(w₀) + λ log n − (m−1) log log n + O_p(1).

- **WBIC.** The posterior mean of nL_n at β = 1/log n equals nL_n(w₀) + λ log n + U_n√(λ log n/2) + O_p(1) (checked). So WBIC matches F_n **only through the λ log n term**, not the log log n term, and it carries an O_p(√log n) fluctuation.
- **Bayes generalization (β = 1).** E[G_n] = L(w₀) + λ/n + o(1/n) under realizability. With general β it also involves the singular fluctuation ν ([Watanabe 2010](https://jmlr.org/papers/v11/watanabe10a.html)).
- **LLC** ([Lau et al. 2023/AISTATS 2025](https://arxiv.org/abs/2308.12108), checked). This is the local version λ(w*) at a local minimum w* of K, estimated by

      λ̂ = nβ*·(E_{w|w*,β*,γ}[L_n(w)] − L_n(w*)),    β* = 1/log n,

  where the expectation is over a γ-localized tempered posterior sampled with SGLD. λ(w*) is invariant under local diffeomorphisms (App. C, checked). The estimator's equality with λ(w*) relies on localization conditions; it is not a general identity.

### 4.3 Developmental interpretability: what a "phase transition" is, mathematically

**DERIVATION (Bayesian phase transition).** Partition W into regions W_α, each with its own local optimal loss L_α and local learning coefficient λ_α. Under §4.1 conditions applied in each region, F_α(n) ≈ nL_α + λ_α log n, and the posterior concentrates on the region with the smallest F_α.

Take two regions with L₁ > L₂ (region 1 fits worse) and λ₁ < λ₂ (region 1 is simpler). Region 1 dominates for small n, and the posterior switches at n_c solving

    n_c (L₁ − L₂) = (λ₂ − λ₁) log n_c.                                    (4.1)

This is a theorem-level consequence *if* the O_p(1) terms do not overturn the comparison near n_c. Near the crossing they can, so the location of the transition is predicted only to within a factor set by those terms.

**What is and is not established.**

| Claim | Status |
|---|---|
| Bayesian posteriors undergo transitions governed by (4.1) | DERIVATION from THEOREM (above) |
| Training (SGD) shows stages that co-occur with changes in LLC estimates (2-layer LM transformer, in-context regression) | Empirical association ([Hoogland et al. 2024](https://arxiv.org/abs/2402.02364), checked) |
| SGD transitions correspond to Bayesian transitions | Evidence in a toy model of superposition ([Chen et al. 2023](https://arxiv.org/abs/2310.06301), checked existence). **Not a theorem** for SGD. [Lee et al. 2025](https://arxiv.org/abs/2510.12071) state "the primary theoretical gap remains the link between the Bayesian learning process of SLT and the non-equilibrium dynamics of SGD" (checked). |
| SGD's steady state is a tempered posterior | Conditional theorem under local homogeneity and isotropic diffusion along degenerate directions, plus approximate steady state and accessibility constraints ([Hennick & De Baerdemacker, ICLR 2026, "Almost Bayesian: Dynamics of SGD Through Singular Learning Theory", arXiv 2503.22478](https://arxiv.org/abs/2503.22478), checked). §5.2 shows that the isotropy assumption is exactly what fails in the plain quadratic regime. |

### 4.4 Where SLT applies, where it is a heuristic, and where it does not apply

| Setting | Verdict | Reason |
|---|---|---|
| Bayesian posterior of an analytic model, compact prior support, iid data, n → ∞ | **Applies (theorem)** | §4.1 |
| Analytic architectures: tanh, sigmoid, softplus, SiLU/Swish, GELU, softmax attention, SwiGLU FFN, linear and diagonal SSMs, deep linear networks, reduced-rank regression | **Analyticity holds** (normalizations are analytic away from zero input) | Other conditions still must be checked |
| ReLU, ReLU² (RWKV channel-mix FFN), max-pooling, top-k MoE routing, hard quantization | **Theorem not established** | Not analytic, which Watanabe's resolution-of-singularities proof needs. Volume-scaling exponents may still exist, but the free-energy theorem is not proved here. |
| Separable logistic regression; unregularized nets whose loss infimum sits at ‖θ‖ → ∞ | **Does not apply** | No w₀ in a compact W (§3.5) |
| SGD or Adam endpoints, read as "posterior samples" | **Heuristic at best** | §5. E3 ≠ E2 in the simplest case. LLC at an SGD point needs w* to be a local min of K; report ‖∇L(w*)‖, as the user's `chl-slt-experiments` already does. |
| SGLD-estimated LLC | **Estimator with known bias** | Minibatch bias ∝ εN/B·λ_max(A), invisible to R-hat (user's `chl-acquisition-gate/RESULTS_SGLD_BIAS.md`; regular model only) |
| Finite n, especially d ≫ n | **Heuristic** | §3.2: λ_eff interpolates between rank(H)/2 and λ. The O_p(1) terms can dominate. |
| Restricted (LoRA- or layer-only) LLC | **Applies to a different model**: the restricted sub-model with the others frozen | It is a valid object, but its λ is not "the network's λ" |
| Non-realizable distillation (attention teacher → GDN student) | **Conditional** | Relatively finite variance must hold (§7.4) |
| RL, online or non-iid data streams | **Does not apply as stated** | The iid assumption fails, and the data distribution depends on θ |
| Transformers or SSMs at scale | **Conditional heuristic** | Analyticity often holds; compactness is a prior choice; posterior ≠ SGD; d ≫ n; estimator bias. Use as a logged covariate only, matching the repo policy. |

### 4.5 Bridge B6: parameter unobservability *is* singularity

**DERIVATION.** Take a Gaussian-output model y = f(x;θ) + ε, ε ~ N(0,σ²I). The Fisher information is

    I(θ) = σ⁻² E_x[J_θ(x)ᵀJ_θ(x)],    J_θ = ∂f/∂θ.

That is the noise-weighted **observability Gramian of the parameter** from input–output data: the parameter-sensitivity analogue of the source document's initial-state derivation. So:

- ker I(θ*) = parameter directions that are first-order **unobservable** from behavior;
- at a realizable point, I = H (§3.1), so these are exactly the directions counted by d − rank(H).

The SLT refinement is that *higher-order* observability (λ_g) decides how unobservable they are. "Non-identifiability" (statistics), "unobservability" (control) and "degeneracy" (SLT) are three names for one thing, the fibers of θ ↦ p(·|θ), measured at different orders.

**Caveat (from the source document, kept):** this concerns *parameter* identifiability from outputs. It is not *state* observability of a recurrent model; that is §7.

### 4.6 Worked example and correction: the RLCT of a diagonal LTI SSM

**Setup.** A real diagonal SISO LTI layer with n modes. Under white-noise input and Gaussian output noise, the population loss is proportional to

    K(w) = Σ_{k≥0} (g_k(w) − g*_k)²,    g_k(w) = Σ_{i=1}^{n} p_i λ_i^k,  with |λ_i| ≤ ρ < 1,

where p_i = c_i b_i, and the truth has r < n distinct active modes.

**(i) Generic true point.** The unused modes have p_i = 0 and λ_i distinct from every active pole. The point is Morse–Bott with rank(H) = 2r + (n−r), so λ_local = r + (n−r)/2 by (3.1). This matches review 02's local measurement (1.48–1.51 for n=2, r=1), and NUMERICAL this session gives slope 1.486.

**(ii) Coincident stratum (n = 2, r = 1).** Put both poles near λ* with total amplitude near p*. Let a = λ₂ − λ*, b = λ₁ − λ*, q = p₂, and p₁ = p* + u − q. Expanding λ^k in powers of the pole deviation, the coefficients on the linearly independent sequences C(k,j)·λ*^{k−j} are:

    j=0: u
    j=1: p₁ b + q a
    j=2: p₁ b² + q a²  =  q a²(p₁ + q)/p₁ + O(j=1 coefficient)

Change variables from (p₁, b) to (u, v = p₁b + qa). This is a local diffeomorphism because p₁ ≈ p* ≠ 0. Then K ≍ u² + v² + q²a⁴, and the higher j are dominated. The RLCT of q²a⁴ is min(1/2, 1/4) = 1/4 with multiplicity 1, so

    λ_local(coincident) = 1/2 + 1/2 + 1/4 = 5/4  <  3/2,     m = 1.

The same value holds on the split stratum (q ≠ 0, a = 0), where K ≍ u² + v² + a⁴.

**(iii) Global RLCT.** For n = 2, r = 1, the global RLCT is λ = 5/4: the minimum over the strata, with the prior positive on the coincident stratum.

**NUMERICAL (this session; script in the companion file, numpy, CPU).** Monte Carlo volume scaling log V(ε) vs log ε over local boxes, 1.2 × 10⁸ samples per case:

| True point | Predicted λ_local | Measured slope |
|---|---|---|
| G: generic (p₁,λ₁,p₂,λ₂) = (1, .5, 0, −.3) | 3/2 | 1.486 (ε ∈ [10⁻⁶, 10⁻⁴]) |
| S: split (.5, .5, .5, .5) | 5/4 | 1.248 / 1.264 / 1.293 on three ε windows |
| C: coincident (1, .5, 0, .5), box 0.1 | 5/4 (asymptotic) | 1.00 (pre-asymptotic, equal to rank(H)/2, §3.2) |
| C: coincident, box widened to 0.3 in (p₂, λ₂) | 5/4 | 1.13–1.18, rising as the window moves to smaller ε |

**Verdict.**

- Review 02's sentence "Additional loci (λ_i = λ_j coincidence…) do not lower λ; they raise m" is **wrong** for the global RLCT and for the local RLCT at coincident points.
- Its value r + (n−r)/2 is correct only as the **local coefficient at a generic point**.
- The Codex counterexample (5/4) is confirmed analytically and numerically.
- The repo PLAN's phrase "corrected formula λ = r + (n−r)/2" should read **"generic-point local LLC"**.

For general n and r, and for complex conjugate pairs (where the review's generic value n + r should likewise be a generic-point value), the global RLCT is open; see **CONJECTURE C1** in the companion file.

**Why this matters beyond the toy.**

1. *Which* λ an experiment measures depends on where SGD lands. A trained model's unused modes sit at generic poles unless training drives poles together. So a "staircase" of LLC against Hankel rank is a statement about generic local values, and it must be labelled that way.
2. Mode coalescence, where an unused pole migrates onto an active one, is a **lower-free-energy configuration**. Bayesian theory therefore predicts the posterior *prefers* it at large n. Whether SGD goes there is an open dynamical question, and it gives a clean, CPU-checkable E2-vs-E1 discrepancy test (T-C1b).

---

## 5. Bridge B7: SGD is not a posterior sampler (E3 ≠ E2), and exactly how it differs

### 5.1 Setting

Take a regular local quadratic regime: L(θ) ≈ L* + ½δᵀAδ with δ = θ − θ*, a constant step η, batch size B, and per-example gradient covariance C at θ*. The preconditioner is P; plain SGD has P = I. Linearizing (1.1) gives

    δ_{k+1} = (I − ηPA)δ_k − ηPζ_k,    Cov ζ_k = C/B.

**DERIVATION.** The stationary covariance Σ solves the discrete Lyapunov equation Σ = (I−ηPA)Σ(I−ηPA)ᵀ + η²PCPᵀ/B. To first order in η, and when P, A and C commute:

    PAΣ + ΣAP = (η/B)·PCP.                                                (5.1)

This is the same Lyapunov model the user's lab validated on real SGLD chains (cosine 0.9988 to prediction; `RESULTS_SGLD_BIAS.md`), and the OU picture of [Mandt, Hoffman & Blei 2017](https://arxiv.org/abs/1704.04289) (checked title).

### 5.2 Well-specified case C = A, with preconditioner P = A^{−α}

At a well-specified NLL optimum, the gradient covariance equals the Fisher, which equals the Hessian. Solving (5.1):

    Σ_SGD = (η/2B) · A^{−α}.                                              (5.2)

| Optimizer (idealized) | α | Σ_SGD | Matches the posterior (nβA)^{−1}? |
|---|---|---|---|
| plain SGD | 0 | (η/2B)·I, **isotropic and independent of curvature** | No |
| full-matrix RMSProp/Adam (√-preconditioning) | 1/2 | (η/2B)·A^{−1/2} | No |
| natural gradient (P = A^{−1}) | 1 | (η/2B)·A^{−1} | **Yes**, with β_eff = 2B/(ηn) |

**Consequences (all DERIVATION, in this regime):**

1. **The "SGD temperature" is not a posterior temperature, except under natural gradient.** The analogue of the LLC identity is nβ·E[K]. Under the posterior it equals d/2 = λ. Under plain-SGD fluctuations it equals nβ·(η/4B)·tr A. That **measures total curvature, not the learning coefficient.** So *any complexity proxy built from SGD fluctuations is a curvature trace, not an LLC*.
2. **Flat directions.** Along an eigenvector with small curvature a_i, the posterior variance 1/(nβ a_i) is large, while plain-SGD variance stays η/2B. As a_i → 0, C = A gives **zero first-order noise** there, so SGD does not diffuse along W₀ at all to first order. The posterior *does* spread along W₀, weighted by higher-order geometry. That spreading is precisely where λ < d/2 comes from.
   - SGD's position along W₀ is set by initialization, higher-order noise and drift: the "stochastic collapse" phenomenon ([Chen, Kunin et al. 2023, arXiv 2306.04251](https://arxiv.org/abs/2306.04251), identified by the citation check, not fetched by me).
3. **The "Almost Bayesian" theorem's diffusion assumption.** That theorem assumes isotropic diffusion along degenerate directions. In this first-order picture there is *no* diffusion along them (C = A = 0 there). So any Gibbs-like behavior along W₀ must come from higher-order noise, and the theorem's hypothesis has to be checked, not assumed.
4. **Diagonal Adam ≠ α = 1/2.** When A is not axis-aligned, diagonal preconditioning does not commute with A. That is the same reason the lab's diagonal SGLD correction failed (error 2.78).

**NUMERICAL (this session, CPU).** Setup: a well-specified 2-D Gaussian linear model, with A having eigenvalues 10 and 1 in rotated axes; η = 0.01, B = 8, 3.8 × 10⁵ stationary steps on fresh data.

- Measured Σ·(2B/η) matches A^{−α} with relative error **0.054** (α = 0), **0.013** (α = 1/2) and **0.037** (α = 1). The α = 0 residual is the O(ηλ_max) = O(0.1) correction.
- The "SGD-LLC" quantity tr(AΣ)·B/η equals **5.88** for plain SGD, against tr(A)/2 = 5.5 predicted and d/2 = λ = 1 for the posterior.
- It equals **1.02** under natural gradient.

Plain-SGD fluctuations measure curvature, not λ, as derived.

**Validity of the derivation.** It needs a quadratic regime, ηλ_max ≪ 1 (not edge of stability), C ≈ A (well-specified, large n), and commuting matrices for the closed forms. Outside these, use the general discrete Lyapunov solve; the conclusion "≠ posterior" generically persists.

### 5.3 The correct framing of "LLC at an SGD checkpoint"

λ(w*) is a **geometric invariant of the population loss at w*** (§4.2). It is well-defined however w* was found, *provided* w* is a local minimum of K. What is heuristic is not the number. It is two downstream readings of it:

- (a) treating the SGD-trained predictor as if the posterior theorems (free energy, generalization λ/n, transition law (4.1)) applied to it;
- (b) treating SGD's *choice* of w* on W₀ as posterior-typical. §5.2 says it is not: SGD under-explores W₀.

**Falsifiable test T-B7** (CPU): in the 2-mode LTI toy and in K = a²b², compare the distribution of SGD endpoints along W₀ with the posterior marginal along W₀ (MCMC with exact likelihood), using a KS test. The prediction is that they differ, and that the posterior puts excess mass at the lower-λ stratum (coalescence or the origin) relative to SGD.

---

## 6. Layer (d): training as a controlled system

### 6.1 The plant, actuators, sensors

From (1.1):

| Element | Meaning |
|---|---|
| **State** | s_k = (θ_k, m_k) |
| **Actuators** | u_k = (η_k, B_k, π_k, regularizer weights ω_k, constraint multipliers μ_k) |
| **Disturbance** | minibatch noise, with covariance C/B (§5) |
| **Sensors** | y_k: per-domain validation losses, behavioral metrics b(θ), and *logged* LLC/susceptibilities |

Controllability and observability mean the following here (DEFINITIONS, with the strongest statement I can defend):

- **Local behavioral controllability over horizon K:** the rank of the response map ∂b(θ_K)/∂u_{0:K−1}. From the source document's linearization, this is Σ_k (∂b/∂θ_K)·A_{K−1}⋯A_{k+1}·B_k with A_k = I − η_kH_k. **Parameter** controllability is irrelevant, and generically fails (§3.6). What matters is controllability of the *behavioral readout*.
- **Observability of training:** whether y_{0:K} determines the behavior-relevant part of θ_K. By B6 (§4.5), parameter directions in ker I(θ) are unobservable *and behaviorally irrelevant to first order*. That is a feature: only the quotient θ/~ needs to be observed.

### 6.2 Bridge B8: Bayesian susceptibility equals classical influence in the regular limit

**DERIVATION.** The source document's linear-response identity is d/dv E_v[O] = −β Cov_v(O, ℓ). Under a Gaussian (Laplace) posterior with covariance (nβA)^{−1} and smooth O, ℓ, first order gives Cov(O,ℓ) ≈ G_O (nβA)^{−1} G_ℓᵀ, so

    χ = d/dv E_v[O] ≈ −(1/n)·G_O A^{−1} G_ℓᵀ,

which is the classical (Koh–Liang) influence function. This is exactly the "point influence" the lab used as ground truth.

**Hence:** Bayesian influence functions and susceptibilities can only add information beyond classical influence through **non-Gaussian, singular structure** (λ_g, multimodality along W₀). This sharpens the source documents' baseline requirement. Any claimed BIF advantage must be located in a model and region where the Laplace approximation fails. The reported BIF result on Pythia-14M (LDS 0.110 ± 0.112 vs EK-FAC 0.221 ± 0.125 vs GradSim 0.198 ± 0.109, [Kreer et al. v2, App. C.3](https://arxiv.org/abs/2509.26544), checked) does not show such an advantage, and the differences are within one standard deviation.

### 6.3 Bridge B9: curriculum order is a Lie bracket

**DERIVATION.** Take one SGD step on domain i, then one on domain j, with gradients g_i = ∇L_i and Hessians H_i:

    θ_{ij} = θ − η g_i − η g_j(θ − ηg_i) = θ − η(g_i + g_j) + η² H_j g_i + O(η³).

Reversing the order and subtracting:

    θ_{ij} − θ_{ji} = η² (H_j g_i − H_i g_j) + O(η³) = η² [g_i, g_j] + O(η³),          (6.1)

where [g_i, g_j] = (Dg_j)g_i − (Dg_i)g_j is the Lie bracket of the gradient vector fields. For any behavioral readout b, the **order effect** is

    Δb_order ≈ η² ∇b(θ)ᵀ (H_j g_i − H_i g_j).                              (6.2)

This costs two Hessian-vector products.

**Reading.**

- The data-mixture actuator is a **control-affine, driftless system** θ̇ = Σ_j π_j(−∇L_j(θ)) with π_j ≥ 0.
- One step reaches only the cone spanned by {−∇L_j}: rank ≤ J.
- Further directions are reachable only through brackets, at O(η²) per bracket. So the *local* reachable directions over K steps are the span of the gradient fields and their iterated brackets. That is the Lie-algebra rank condition of geometric control. The nonnegativity of mixture weights makes this a cone-constrained version, so full accessibility needs the bracket span plus positive-span conditions.
- This is the precise, limited sense in which "curriculum design is a control problem": the order of curriculum presentation matters at **second order in η**, and the effect is computable in advance.

**Validity:** ηλ_max ≪ 1, and smooth L_j. It fails at edge of stability (§3.4), where O(η³) terms are not small.

**Test T-B9:** small two-domain runs; predict Δb_order from (6.2) at the start of each block, then compare with measured i→j vs j→i differences across η. The prediction is agreement scaling as η², degrading as ηλ_max → 2. Baselines: zero (no order effect) and a first-order gradient-alignment heuristic.

### 6.4 Constraints during training

- **KKT / primal–dual.** For min L₀ s.t. L_c ≤ b, gradient descent–ascent on the Lagrangian is the standard actuator. Constrained learning is PAC-learnable under conditions, with a near-zero duality gap for sufficiently rich parameterizations ([Chamon & Ribeiro 2020](https://arxiv.org/abs/2006.05487), checked title). THEOREM under their assumptions. Not re-derived here.
- **Safe gradient flow (CBF on parameters).** DERIVATION. For a constraint h(θ) ≥ 0 (for example h = τ − KL_offdomain), the minimal modification of the flow θ̇ = −g is the QP min‖v + g‖² s.t. ∇hᵀv ≥ −αh. It has the closed form

      v* = −g + max(0, (∇hᵀg − αh)/‖∇h‖²)·∇h.

  In continuous time, {h ≥ 0} is forward-invariant under the CBF conditions ([Ames et al.](http://ames.caltech.edu/ames2019control.pdf)). **Discrete steps** need a margin. With ‖∇²h‖ ≤ M_h,

      h(θ + ηv) ≥ h(θ) + η∇hᵀv − (η²M_h/2)‖v‖²,

  so the step constraint must be ∇hᵀv ≥ −αh + (ηM_h/2)‖v‖². With minibatch estimates of h and ∇h, the guarantee becomes probabilistic at best. This is the CBF prototype already in the user's `chl-slt-experiments` (`objective.mode=cbf`), which now has a stated discrete-time margin.
- **What a training-time constraint certifies.** It certifies only h along the training trajectory, measured on the data used to estimate h. It says nothing about h on held-out inputs unless combined with a generalization bound (§2) on h itself.

### 6.5 Optimal control of hyperparameters = backprop through training

**DERIVATION.** For a terminal cost J = b(θ_K) with dynamics θ_{k+1} = T(θ_k, u_k), Pontryagin's discrete costate is p_k = (∂T/∂θ)ᵀ p_{k+1} with p_K = ∇b, and ∂J/∂u_k = (∂T/∂u)ᵀ p_{k+1}. These are exactly hypergradients. The costate propagates through Π(I − η_kH_k)ᵀ.

- Under contraction (ηλ_max < 2 and H ≻ 0), it decays.
- At edge of stability, the eigenvalues near −1 keep it from decaying.
- Along ker H (§3.6) it is preserved exactly.

So long-horizon hypergradients are informative only about the behavior-relevant (observable) quotient, and they blow up or oscillate at edge of stability. This explains, rather than merely notes, why long-horizon training control is ill-conditioned.

---

## 7. Layers (e)+(f): latent state, interpretability as observability, and delta-rule SSMs

This is the user's focus: attention-free models, GDN/RWKV-7 conversion, and the State-Space-Decision-Model.

### 7.1 Bridge B10: a gated delta-rule layer is an online learner (two-time-scale structure)

**Gated DeltaNet** ([Yang, Kautz & Hatamizadeh 2024](https://arxiv.org/abs/2412.06464), checked): for each head,

    S_t = S_{t−1} · α_t (I − β_t k_t k_tᵀ) + β_t v_t k_tᵀ,     o_t = S_t q_t,

with α_t, β_t ∈ (0,1). The q and k paths are linear projection → short convolution → SiLU → **L2 normalization**, so ‖k_t‖ = ‖q_t‖ = 1.

**DERIVATION.** Define ℓ_t(S) = ½‖S k_t − v_t‖², so ∇_S ℓ_t = (S k_t − v_t) k_tᵀ.

- With α_t = 1, the update is exactly S_t = S_{t−1} − β_t ∇ℓ_t(S_{t−1}): **one SGD (LMS/Widrow–Hoff, Kaczmarz-type) step with learning rate β_t** on an in-context regression of values on keys.
- With α_t < 1: S_t = α_t(S_{t−1} − β_t∇ℓ_t) + (1−α_t)β_t v_t k_tᵀ, a decayed step plus an extra write.

So the fast state is an *optimizer state*, and §§3–6 apply on the token time scale j as well as the training time scale k:

- persistent excitation of the keys (the source documents' system-identification condition) is the convergence condition for this inner learner;
- β_t is its learning rate;
- α_t is its forgetting factor.

### 7.2 Bridge B11: delta-rule transitions are non-expansive (an exact structural property)

**DERIVATION.** Write A_t = α_t(I − β_t k_t k_tᵀ). With ‖k_t‖ = 1, its eigenvalues are α_t(1 − β_t) (along k_t) and α_t (multiplicity d_k − 1, orthogonal to k_t). A_t is symmetric, so

    ‖A_t‖₂ = α_t · max(1, |1 − β_t|) = α_t     for β_t ∈ [0, 2].

This range includes the negative-eigenvalue extension β ∈ (1,2) of [Grazzi et al. 2024](https://arxiv.org/abs/2411.12537) (checked title).

**Consequences.**

1. **Incremental stability, conditional on the layer's inputs.** α_t, β_t, k_t, v_t depend only on the layer *input* x_t, not on S. So for identical inputs and two initial states, ΔS_t = ΔS_0 Φ(0,t) with Φ(s,t) = A_{s+1}⋯A_t, and

       ‖ΔS_t‖_F ≤ (Π_{u=1}^{t} α_u) ‖ΔS_0‖_F.

   - This is **non-expansive always**, and contracting only when gates close (α < 1).
   - With α ≡ 1, decay happens only in directions the keys excite: the delta rule erases along k_t alone.
2. **Bounded state needs ᾱ < 1.** If α_t ≤ ᾱ < 1 and ‖v_t‖ ≤ v̄, then ‖S_t‖_F ≤ ᾱ‖S_{t−1}‖_F + β̄v̄, so sup_t ‖S_t‖_F ≤ ‖S_0‖ + β̄v̄/(1 − ᾱ). With α allowed to reach 1, I could not establish a uniform bound (the write term can accumulate), so I do not claim one.
3. **Scope.**
   - The result covers GDN, DeltaNet, and Mamba-2 (where A_t = α_t I).
   - **RWKV-7's** transition diag(w_t) − κ̂_tᵀ(a_t ⊙ κ̂_t) is *not symmetric*, so ‖·‖₂ ≤ 1 does not follow from its eigenvalues. **Open:** check ‖G_t‖₂ empirically and analytically ([RWKV-7](https://arxiv.org/abs/2503.14456), checked title).

### 7.3 Bridge B12: input-conditioned Gramians, the observability of interventions, and an exact persistence functional

Fix a layer's input sequence x_{1:T}. Then S evolves as a **linear time-varying (LTV)** system in S. The non-linearity sits entirely in how the inputs set (A_t, B_t).

**DERIVATION (observability of the initial state / an edit).** The output is o_t = S_0 Φ(0,t) q_t + (input-driven terms). Define

    W_o(T) = Σ_{t=1}^{T} Φ(0,t) q_t q_tᵀ Φ(0,t)ᵀ  ∈ ℝ^{d_k×d_k}.

The output energy caused by a state edit ΔS is Σ_t ‖ΔS Φ(0,t) q_t‖² = tr(ΔS W_o ΔSᵀ). Therefore:

- **Exact null-edit property.** Any edit with ΔS W_o = 0 (rows in ker W_o) changes *this layer's* outputs over the horizon by **exactly zero**.
- Across layers it also changes nothing, because the downstream layers' inputs are unchanged.

This is a perfect, free *negative control* for interpretability interventions. An internal "concept edit" whose effect is unobservable cannot be behaviorally meaningful. This is mechanistic interpretability recast as observability (layer (e)): **a concept is behaviorally real only to the extent its direction is observable through W_o, and controllable (writable) through the reachability Gramian**

    W_c(T) = Σ_s Φ(s,T)ᵀ k_s k_sᵀ Φ(s,T) · β_s²‖v_s‖².

The Hankel-style quantity λ_i(W_c W_o) is invariant to state-coordinate changes S → S M with M invertible, provided the readout and write maps are transformed consistently. Probe coordinates are not invariant in this way (source document, similarity-transform argument).

**DERIVATION (exact persistence functional).** The contribution of the write at token s to the output at token t > s is

    β_s v_s · (k_sᵀ Φ(s,t) q_t),       |k_sᵀ Φ(s,t) q_t| ≤ Π_{u=s+1}^{t} α_u.

This is **exact per sequence**, cheap to compute, and comes with a decomposition:

- **decay** (α);
- **overwrite** (later keys aligned with k_s, through the (I − βkkᵀ) factors);
- **readout alignment** (q_t).

**What it is NOT (respecting reviews 02/03).** It is not a certificate. The gates are input-dependent, and an adversary can keep α ≈ 1 and avoid overwriting keys, so no sequence-independent bound is useful. The user's plan already replaced the half-life certificate with *measured* knockout-at-distance flip rates. The functional is the natural *predictor* for those flip rates, and that is a testable claim (T-B12), not a guarantee.

### 7.4 The whole stack: what composes and what does not

- **Prefill / teacher forcing.** The stack is a causal *cascade*: layer ℓ at token t depends on layer ℓ−1 at t and on its own state at t−1, with no feedback from higher layers. Per-layer incremental bounds therefore compose by the cascade argument.
  - A perturbation to a layer's input changes A_t and B_t, so the per-layer incremental gain needs the state bound of §7.2(2): ‖ΔS_t‖ ≤ ᾱ‖ΔS_{t−1}‖ + (S̄·L_A + L_B)‖Δx_t‖, where L_A and L_B are Lipschitz constants of the gate and write maps.
  - This gives an ISS-type bound of the source document's form for each layer. The composition is a DERIVATION only when ᾱ < 1 and the Lipschitz constants are finite. Normalization layers make the L's input-scale dependent.
- **Autoregressive generation closes the loop.** Output token → embedding → input is a *feedback* interconnection, through a discontinuous map (argmax or sampling). The cascade argument no longer applies. Small-gain conditions would be needed and generically fail. **No stability statement above transfers to free-running generation.** This is a precise version of the source documents' "composition needs its own theorem".

### 7.5 SLT for attention→GDN conversion and fine-tuning: what can and cannot be said

1. **Distillation is generally non-realizable.** A fixed-size recurrent state cannot represent exact long-range retrieval that softmax attention can. Watanabe's theory then needs the relatively finite variance condition (§4.1(3)). It holds if the student's KL-optimal set represents a single distribution. If different student optima give different output distributions with equal KL, the condition can fail, and I make no claim about the asymptotics.
   - **Test:** train two or more seeds of the same conversion and compare their output distributions on held-out text. Distinct distributions at equal teacher-KL flag a possible failure.
2. **Symmetry counting predicts LLC upper bounds and changes them under conversion.** DERIVATION. Continuous parameter symmetries with s-dimensional orbits make K constant along orbits, so they contribute s null directions to H, and λ ≤ (d − s)/2 locally.
   - *Softmax attention head* (scores x W_q W_kᵀ x′ᵀ): the GL(d_k) symmetry W_q → W_q M, W_k → W_k M^{−T} gives s = d_k². With RoPE on d_rot of the d_k dimensions, M must commute with the rotary rotations on the rotary block. Commutants of SO(2) on each 2-D pair are complex scalars, so s = d_rot + (d_k − d_rot)² (distinct rotary frequencies assumed).
   - *GDN head:* the SiLU then L2-normalization breaks GL(d_k). What remains continuous is per-channel rescaling between the projection rows and the depthwise short-convolution kernels: s = d_k on each of the q and k paths, if the conv is linear, bias-free and depthwise. The v/output side has its own symmetries, depending on the output norm and gate.
   - **Prediction (exact parameterization must be checked against the code):** the q/k-block Hessian nullity per head changes from d_rot + (d_k − d_rot)² (teacher attention) to about 2·d_k (student GDN). It is checkable by Lanczos on a converted checkpoint (T-C5).
   - This is a statement about *flat directions*. It does not by itself say the converted model is "simpler".
3. **Fine-tuning phase transitions.** For a LoRA-restricted sub-model (a valid model in its own right), the transition law (4.1) predicts when the posterior switches between solutions, for example a narrow one versus a general one. In the user's `chl-slt-experiments` EM study, it predicts the dataset size n_c from the restricted losses and λ's. For SGD-trained adapters this is a *heuristic* (§5). It is logged, never gated.

---

## 8. What does NOT unify, and why

| Pair | Why there is no bridge (or only an analogy) |
|---|---|
| Hankel singular values ↔ LLC | They are different objects. Counterexamples are in the source document (*Learning Geometry*) and §4.6: the same transfer function can carry different λ, and λ can change at fixed Hankel rank (coalescence). **ANALOGY-ONLY**, apart from the generic-point local count. |
| "Singular" (HSV) ↔ "singular" (SLT) | Shared word only. |
| Contraction-based control of training ↔ SLT degeneracy | Directly in tension (§3.6). The degenerate directions are where contraction has no authority. |
| SLT ↔ the implicit bias of GD on separable data | There is no minimizer in a compact W (§3.5). |
| SLT ↔ mean-field / infinite-width limits | SLT has fixed finite d. No theorem connects them (§3.3). |
| SLT ↔ ReLU, ReLU², MoE top-k, quantized networks | The analyticity assumption fails. The theorem is not established. |
| SLT ↔ RL / online non-iid training | The iid assumption fails, and the data depend on θ. |
| Posterior (E2) ↔ SGD endpoint (E1/E3) | Not equal in the simplest regime (§5.2). Equal only under natural-gradient preconditioning with C = A, or under the conditions of the "Almost Bayesian" theorem. |
| LLC ↔ any alignment, safety or semantic property | The LLC is a volume exponent of the loss. It carries no semantics. The repo policy holds: log, never gate. |
| Stability of a prefill cascade ↔ stability of autoregressive generation | Generation closes a discontinuous feedback loop (§7.4). |
| Mechanistic interpretability "features" ↔ identified latent variables | A probe direction is defined only up to the realization equivalence class: the source's similarity-transform argument, and nonlinear non-identifiability (Locatello et al.). Only observable and controllable directions (§7.3) are behaviorally meaningful. Even those are not semantically identified without interventions. |
| PAC-Bayes certificate ↔ the SGD-trained deterministic network | B1/B2 bound the Gibbs (randomized) predictor. Transferring to a point predictor needs a derandomization argument. |
| Fixed-n LLC estimates ↔ the asymptotic λ | They differ (§3.2). At practical n they often sit near rank(H)/2. |

---

## 9. The regime map (summary)

The unification is this table. Each row is a *regime*; within it, the listed theories describe the same object.

| Regime (checkable flag) | Valid descriptions | Invalid there |
|---|---|---|
| Lazy (parameters move ≪ 1 relative to scale; NTK stable) | NTK linear model; λ = rank(NTK covariance)/2 (Morse–Bott); classical influence = Bayesian susceptibility | higher-order SLT effects (λ_g) |
| Small step, quadratic basin (ηλ_max ≪ 1, C ≈ A) | SGD-as-SDE/OU; Lyapunov covariance (5.1); stationary law ≠ posterior unless natural gradient; curriculum Lie brackets (6.1) | edge-of-stability phenomena |
| Edge of stability (ηλ_max ≈ 2) | Empirical only (Cohen et al.) | descent-lemma Lyapunov, the OU picture, contraction, the bracket expansion |
| Bayesian posterior, analytic, compact, iid, n → ∞ | SLT: F_n, WBIC (to λ log n), λ/n generalization, transition law (4.1); B1/B2 complexity = λ log n | — |
| Finite n, d ≫ n | λ_eff between rank(H)/2 and λ (§3.2); estimates with SGLD bias ∝ εN/B·λ_max | the asymptotic SLT formulas, as quantitative predictions |
| Delta-rule recurrent layer, fixed inputs | exact LTV analysis: non-expansive transitions, Gramians, the null-edit property, the persistence functional, the inner learner = LMS | anything about free-running generation; sequence-independent certificates |

---

## 10. References (checked this session unless marked)

**Singular learning theory and its estimators**
- Watanabe, *A Widely Applicable Bayesian Information Criterion*, JMLR 14 (2013): https://www.jmlr.org/papers/v14/watanabe13a.html
- Watanabe, *Asymptotic Equivalence of Bayes Cross Validation and WAIC in Singular Learning Theory*, JMLR 11 (2010): https://jmlr.org/papers/v11/watanabe10a.html (title checked)
- Lau, Furman, Wang, Murfet, Wei, *The Local Learning Coefficient*, AISTATS 2025: https://arxiv.org/abs/2308.12108
- Furman & Lau, *Estimating the LLC at Scale*: https://arxiv.org/abs/2402.03698
- Chen & Murfet, *Modes of Sequence Models and Learning Coefficients*: https://arxiv.org/abs/2504.18048
- Aoyagi & Watanabe, *Stochastic complexities of reduced rank regression in Bayesian estimation*, Neural Networks 18(7):924–933 (2005)

**Developmental interpretability and data attribution**
- Hoogland et al., *Loss Landscape Degeneracy and Stagewise Development in Transformers*: https://arxiv.org/abs/2402.02364
- Chen, Lau, Mendel, Wei, Murfet, *Dynamical versus Bayesian Phase Transitions in a Toy Model of Superposition*: https://arxiv.org/abs/2310.06301
- Wang et al., *Refined LLC*: https://arxiv.org/abs/2410.02984
- Lee, Smith, Adam, Hoogland, *Influence Dynamics and Stagewise Data Attribution*: https://arxiv.org/abs/2510.12071
- Kreer et al., *Bayesian Influence Functions*: https://arxiv.org/abs/2509.26544
- Urdshals et al., *MDL meets SLT*: https://arxiv.org/abs/2510.12077

**SGD dynamics and its relation to the posterior**
- Hennick & De Baerdemacker, *Almost Bayesian: Dynamics of SGD Through Singular Learning Theory*, ICLR 2026: https://arxiv.org/abs/2503.22478
- Mandt, Hoffman, Blei, *SGD as Approximate Bayesian Inference*: https://arxiv.org/abs/1704.04289
- Li, Malladi, Arora, *On the Validity of Modeling SGD with SDEs*: https://arxiv.org/abs/2102.12470
- Cohen et al., *Edge of Stability*: https://arxiv.org/abs/2103.00065
- Chen, Kunin et al., *Stochastic Collapse*: https://arxiv.org/abs/2306.04251 (ID from the citation check; not opened by me)

**Generalization theory**
- Germain, Bach, Lacoste, Lacoste-Julien, *PAC-Bayesian Theory Meets Bayesian Inference*: https://arxiv.org/abs/1605.08636
- Alquier, *User-friendly introduction to PAC-Bayes bounds*: https://arxiv.org/abs/2110.11216 (the exact bound form is from my memory of §2; not checked against the text)
- Dziugaite & Roy: https://arxiv.org/abs/1703.11008
- Zhang et al.: https://arxiv.org/abs/1611.03530
- Nagarajan & Kolter: https://arxiv.org/abs/1902.04742
- Xu & Raginsky: https://arxiv.org/abs/1705.07809
- Hardt, Recht, Singer: https://arxiv.org/abs/1509.01240

**Infinite-width limits and implicit bias**
- Jacot et al. (NTK): https://arxiv.org/abs/1806.07572
- Chizat, Oyallon, Bach (lazy training): https://arxiv.org/abs/1812.07956
- Mei, Montanari, Nguyen (mean field): https://arxiv.org/abs/1804.06561
- Soudry et al. (implicit bias): https://arxiv.org/abs/1710.10345

**Control of training**
- Chamon & Ribeiro, *PAC Constrained Learning*: https://arxiv.org/abs/2006.05487
- Ames et al., *Control Barrier Functions*: http://ames.caltech.edu/ames2019control.pdf (from the source documents)
- Lessard, Recht, Packard (IQC analysis of optimizers): https://arxiv.org/abs/1408.3595

**Delta-rule and linear-recurrent architectures**
- Yang, Kautz, Hatamizadeh, *Gated Delta Networks*: https://arxiv.org/abs/2412.06464
- Yang et al., *Parallelizing Linear Transformers with the Delta Rule*: https://arxiv.org/abs/2406.06484
- Peng et al., *RWKV-7 "Goose"*: https://arxiv.org/abs/2503.14456
- Grazzi et al., *Negative eigenvalues for state tracking*: https://arxiv.org/abs/2411.12537
- Miller & Hardt, *Stable Recurrent Models*: https://arxiv.org/abs/1805.10369

**Not checked this session (standard results, used only as named tools):** the Gromoll–Meyer splitting lemma; the Donsker–Varadhan formula; Pontryagin's discrete costate.

