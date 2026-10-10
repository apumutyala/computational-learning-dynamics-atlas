# Unifying Theory — Open Problems and Tests

This is the companion to `Unifying Theory — Draft v0.md`. It contains:

- **A.** A claim ledger for the five source documents, with errors and overclaims;
- **B.** The conjectures, each with a falsifiable test;
- **C.** Cheap experiments (CPU, or ≤ $5 GPU);
- **D.** The code run this session.

Status labels are the same as in the draft.

---

## A. Claim ledger for the five source documents

### A.1 Load-bearing mathematical claims

"Checked" means I re-derived the claim, or the primary source was fetched this session.

| # | Claim (source doc) | Stated status | My verification |
|---|---|---|---|
| 1 | Similarity transform (TAT⁻¹, TB, CT⁻¹) leaves I/O behavior unchanged (*Math Foundations*) | derivation | **Correct** (re-derived) |
| 2 | Min input energy rᵀW_H⁻¹r; invariant under consistent T (*Math F.*) | derivation | **Correct** |
| 3 | ISS-type bound ρᵗ‖Δz₀‖ + L_u η(1−ρᵗ)/(1−ρ) (*Math F.*) | derivation | **Correct** |
| 4 | Error propagation e_H ≤ Lᴴe₀ + εΣLⁱ; closed-loop constant L_x + L_aL_π (*Math F.*) | derivation | **Correct.** Implicit requirement: both the true and predicted states stay in the verified region. |
| 5 | Robust CBF margin −‖∇h‖_*·D (*Math F.*) | derivation | **Correct** (dual norm) |
| 6 | Potential-shaping telescoping sum (*Math F.*) | derivation | **Correct** |
| 7 | KL-regularized optimum π* ∝ π_ref·e^{r/β} (*Math F.*) | cited | **Correct** |
| 8 | Linear response d/dv E_v[O] = −β Cov_v(O, ℓ) (*Math F.*) | derivation | **Correct** (assumes ρ is independent of v) |
| 9 | F_n = nL_n(w₀) + λ log n − (m−1) log log n + O_p(1) (*Math F.*) | cited, Watanabe 2013 | **Correct**: Thm 2, checked |
| 10 | Zero-failure bound 1 − α^{1/N} ≈ 3/N at α = 0.05 (*Math F.*) | derivation | **Correct** (−ln 0.05 = 3.0) |
| 11 | Port-Hamiltonian energy balance Ḣ = −∇HᵀR∇H + yᵀu (*Math F.*) | derivation | **Correct** |
| 12 | Gramian Lyapunov equations, x̂ᵀW_c⁻¹x and xᵀW_ox energies, HSV = √λ(W_cW_o) invariance (*Learning Geometry*) | cited | **Correct** |
| 13 | Balanced-truncation bound 2Σσᵢ (*L. Geometry*) | cited | **Correct.** With repeated HSVs, the sharp form counts each distinct σ once. Counting with multiplicity, as written, is still valid but looser. |
| 14 | Initial-state Fisher = W_o/σ² (*L. Geometry*) | derivation | **Correct** |
| 15 | Scalar HSV σ_H = \|bc\|/(2a); LLC = 1/2 with HSV free (*L. Geometry*) | derivation | **Correct** |
| 16 | c = w² gives V(ε) ∝ ε^{1/4} (*L. Geometry*) | derivation | **Correct** |
| 17 | V(ε) ~ cε^λ(−log ε)^{m−1}; regular gives ε^{d/2} (*L. Geometry*) | cited | **Correct** |
| 18 | LLC invariant under local diffeomorphisms, "proved by Lau et al." (*L. Geometry*) | cited | **Supported**: App. C of arXiv 2308.12108 (checked) |
| 19 | Hennick & De Baerdemacker, ICLR 2026, under local homogeneity + isotropic diffusion (*L. Geometry*) | cited | **Supported** (arXiv 2503.22478; the title differs between versions) |
| 20 | BIF Pythia-14M LDS 0.110 vs EK-FAC 0.221 vs GradSim 0.198 (*Math F.*, *Building*) | cited | **Numbers correct** (v2, App. C.3, Table 5). See error E3 for the omitted spreads. |
| 21 | Lee et al. acknowledge the Bayesian–SGD gap (*Building*, *Field map*) | cited | **Supported**: quote checked |
| 22 | MDL ↔ SLT connection (*Field map*) | cited | **Supported** |
| 23 | GenCtrl estimates "reachable and controllable sets" with guarantees (*Math F.*, *Field map*) | cited | **Partly.** See error E4. |
| 24 | Miller & Hardt: stable RNNs are approximable by feed-forward models; LSTM conditions (*Math F.*, *Field map*) | cited | **Supported** |
| 25 | Hoogland et al.: stagewise development and LLC, 2 settings (*L. Geometry*) | cited | **Supported** |
| 26 | Ledger: 171 distinct fetched URLs (*Evidence Ledger*) | count | **Consistent**: the table has 171 rows. The "148 cited" figure was not re-counted. |

### A.2 Errors, overclaims and imprecisions found

The five documents are careful. I found **no outright mathematical error** in their derivations. These are the issues:

| ID | Source | Issue | Severity |
|---|---|---|---|
| E1 | *Math Foundations* §"Where SLT fits"; *Field map* `slt` row | The F_n expansion is stated "under Watanabe's stated analytic/statistical conditions" without listing them. In particular, it does not say the conditions concern the **Bayesian posterior** and allow **non-realizable** models under relatively finite variance. | imprecision |
| E2 | *Math Foundations* | It is not noted that **WBIC matches F_n only through λ log n** (it has an extra U_n√(λ log n/2) term) (checked). | omission |
| E3 | *Math Foundations*, *Building* | "Local BIF scores below EK-FAC and GradSim" omits the **±0.11–0.13 standard deviations** in Table 5. All three are within about one SD, so the ordering is not resolved. The source authors themselves say "underperforms" and attribute it to SGLD difficulty. | mild overclaim, inherited |
| E4 | *Math Foundations*, *Field map* | GenCtrl is described as estimating "reachable *and* controllable sets". The abstract only supports **controllable sets** with PAC bounds. "Reachable" may appear in the body, but I could not verify it. | unverified wording |
| E5 | *Learning Geometry* | V(ε) is written with "L(θ) − L(θ*)". For the SLT statement this must be the **population** excess loss K, not the empirical L_n. The text is ambiguous. | imprecision |
| E6 | *Learning Geometry* §"Why HSV and LLC cannot be universally equal" | The counterexamples are right. However, the doc does not address the converse trap that the user's later reviews fell into: treating the generic-point local count r + (n−r)/2 as *the* RLCT. See the draft §4.6. | omission (downstream consequence) |
| E7 | *Math Foundations* "Training as a controlled dynamical process" | The linearization is correctly flagged as not a controllability result. It misses that it **fails structurally at edge of stability** (eigenvalues of I − ηH near −1) and along **ker H** (eigenvalue exactly 1). The draft §3.4, §3.6 and §6.5 cover this. | omission |
| E8 | *Evidence Ledger* | Hoogland thesis metadata: the "University of Amsterdam physics master's, supervisor Greg Stephens" could not be re-checked by me. | unverified |

**Outside the five files, in the user's own repo** (relevant because this draft must not repeat it):

- **R1.** `State-Space-Decision-Model/reviews/02-technical-pressure-test.md` says "Additional loci (λ_i = λ_j coincidence, λ_i → 0) do not lower λ; they raise m". **This is wrong for coincidence.** The global RLCT for n = 2, r = 1 is 5/4, not 3/2 (draft §4.6, derived and numerically supported).
  - `plans/claude/PLAN.md` line 78 calls r + (n−r)/2 a "corrected learning coefficient … Monte-Carlo-checked". It should say "**local LLC at a generic true parameter**".
  - The Codex counterexample in `analysis/CODEX_CRITICAL_COMPARISON.md` is right.

---

## B. Conjectures with falsifiable tests (ranked by importance × testability)

### C1 — Coalescence lowers the RLCT of diagonal SSMs; the global value is not r + (n−r)/2

**Statement.** For a real diagonal SISO LTI model with n modes and true order r (distinct nonzero true poles), with white-noise input:

- the local LLC at a generic true point is r + (n−r)/2 (DERIVATION, §3.1/§4.6);
- the **global** RLCT is strictly smaller whenever n > r. For n = 2, r = 1 it equals 5/4 (DERIVATION). For general (n, r), the conjecture is that it is attained when all unused poles coalesce onto active poles, with a value given by the RLCT of the associated "moment" ideal;
- the complex-pair analogue lies strictly below n + r.

**Test T-C1 (CPU, $0, about 1 h).**

1. Compute the global volume scaling by thermodynamic integration of Z(t) = ∫e^{−tK}φ over t ∈ [10², 10⁸], for n ≤ 4 and r ≤ 2.
2. Run SGLD-LLC at generic vs coincident points, to check whether the estimator sees 3/2 or 5/4 or rank(H)/2.

**Falsified if** the global slope converges to r + (n−r)/2.

**T-C1b (the E1-vs-E2 version).** Train with SGD from random initializations and record whether unused poles drift onto active ones. The Bayesian prediction is that posterior mass concentrates on coalescence as n → ∞ (mass ratio ~ n^{1/4} for n = 2, r = 1). The E3 ≠ E2 prediction (§5) is that SGD need not go there.

### C2 — Finite-n LLC estimates sit near rank(H)/2 (the Morse–Bott part)

**Statement.** At practical nβ and localization γ, λ̂ ≈ rank(H)/2 + (a small, n-increasing correction), with λ̂ → λ only for nβ ≫ 1/(scale of higher-order terms).

**Test T-C2 (CPU).**

1. On the LTI toy and on small MLPs (tanh, analytic), compute the Hessian rank by Lanczos (with a threshold reported) and the SGLD-LLC across n spanning 10²–10⁶.
2. **Falsified if** λ̂ − rank(H)/2 is large and n-independent at small n.
3. Also check the sandwich rank(H)/2 ≤ λ̂ ≤ d/2. A violation means estimator bias; compare with the lab's εN/B·λ_max(A) diagnostic.

### C3 — SGD fluctuations measure curvature, not complexity, and SGD under-explores W₀

**Statement.**

- In the small-step quadratic regime, nβ·E_SGD[K] = nβ(η/4B)·tr(A^{1−α}). It equals d/2 only under natural-gradient preconditioning. NUMERICAL confirmation in the regular case is in §5.2.
- **Conjecture (singular case):** SGD endpoint distributions along W₀ differ from the posterior marginal, and put less mass on the lowest-λ strata.

**Test T-B7 (CPU).**

1. K = a²b², and the 2-mode LTI model.
2. 10⁴ SGD runs vs exact-likelihood MCMC (HMC on a 2–4-D model).
3. KS test on the position along W₀ (for example the angle on the a·b = 0 cross, or the pole gap |λ₁ − λ₂|).

**Falsified if** they agree within the KS noise.

### C4 — Curriculum order effect is the Lie bracket η²∇bᵀ(H_j g_i − H_i g_j)

**Statement.** For ηλ_max ≲ 0.3, the predicted order effect explains most of the variance of the measured i→j vs j→i difference in a behavioral metric. The agreement degrades as ηλ_max → 2 (edge of stability).

**Test T-B9 (CPU, or under $1 GPU).**

1. A small MLP or tiny GDN with two data domains.
2. Compute the prediction with two Hessian-vector products at each block start.
3. Measure the actual order effect over 20 seeds and an η sweep.

**Pass:** slope ≈ 1 and R² > 0.5 at small η, beating a gradient-cosine heuristic. **Falsified if** R² stays below 0.2 at small η.

### C5 — Gramian diagnostics for gated delta-rule layers: exact null edits, and a persistence functional that predicts knockout flip rates

**Statements.**

- (a) Exact null edits are already a DERIVATION (§7.3). The test checks the implementation.
- (b) **Conjecture.** The per-sequence persistence functional β_s‖v_s‖·|k_sᵀΦ(s,t)q_t|, aggregated through layers by the cascade, rank-predicts the measured knockout-at-distance flip rate (the repo's "injection persistence curve"), with Spearman ρ > 0.5 across spans and layers.
- (c) **Conjecture.** Converting attention to GDN changes the q/k-block Hessian nullity per head from d_rot + (d_k − d_rot)² to about 2d_k (§7.5, conditional on the exact parameterization).

**Tests T-B12, T-C5** (CPU on a tiny model, or under $5 GPU on a converted checkpoint; logged only, never a gate).

1. Check the null-edit property numerically: the output change should be 0 to float tolerance.
2. Correlate the persistence functional with knockout flip rates.
3. Lanczos nullity per head before and after conversion.
4. **RWKV-7:** measure ‖G_t‖₂ over real sequences to test whether non-expansiveness holds there.

### C6 (secondary) — The PAC-Bayes/MI complexity of the Gibbs predictor is λ log n

**Test T-B2 (CPU).** Thermodynamic integration of KL(Q*‖φ) in reduced-rank regression (Aoyagi–Watanabe λ known exactly) and in the LTI toy. Check the slope vs log n.

---

## C. Cheap experiments (ordered; all logged, none gate anything)

| # | Experiment | Hardware / cost | Reuses | Decides |
|---|---|---|---|---|
| X1 | **Global RLCT and SGLD-LLC on diagonal LTI SSMs (T-C1, T-C1b)** | CPU, $0, ~1–3 h | the script below; devinterp wrapper in `chl-slt-experiments/chl/llc_tools.py`; the Lyapunov bias check in `chl-acquisition-gate/acq/sgld_bias.py` | C1, C2. Fixes the H-SLT-1 side paper's central formula before it is written. |
| X2 | **SGD vs posterior along W₀ (T-B7)** | CPU, $0 | sampler code in `chl-acquisition-gate` | C3 |
| X3 | **Lie-bracket curriculum order (T-B9)** | CPU, or under $1 | any tiny trainer (`chl/ssm/train.py`) | C4: whether second-order control of curriculum is predictive |
| X4 | **Delta-rule Gramian diagnostics + null-edit control + persistence functional vs knockout flip rate (T-B12)** | CPU on a 2-layer GDN; under $5 on one converted Stage-B checkpoint | the repo's injection-persistence harness (`techniques/03` §3.4) | C5(a, b). Adds a free negative control to interpretability claims. |
| X5 | **Hessian nullity per head across conversion (T-C5)** | under $5 (Lanczos, 50 iterations per head on one layer) | conversion checkpoints | C5(c) |

**Budget discipline.** X1–X3 cost $0. X4 and X5 together stay under $5 on one rented H100-minute block. No GPU was used in this session.

---

## D. Code run this session (CPU, numpy only)

Interpreter: `CHLabs/.venv-gpu/Scripts/python.exe`, CPU only.

### D.1 `rlct_lti.py`: toy RLCT volume scaling for a 2-mode real diagonal SSM

Runtimes: about 9 s at 4 × 10⁶ samples; about 90 s per case at 1.2 × 10⁸ samples.

Results:

| Case | Box | Measured slope | Prediction |
|---|---|---|---|
| G (generic) | half-width 0.1 | 1.486 | 3/2 |
| S (split) | half-width 0.1 | 1.248–1.293 | 5/4 |
| C (coincident) | 0.1 | 1.00 | pre-asymptotic, = rank(H)/2 |
| C (coincident) | (0.1, 0.1, 0.3, 0.3) | 1.13–1.18 | rising toward 5/4 |

```python
"""Toy RLCT check for a 2-mode real diagonal SISO LTI model fitting a 1-mode truth.

g_k(w) = p1*l1^k + p2*l2^k,  truth g*_k = 1 * 0.5^k,  K(w) = sum_{k<T} (g_k - g*_k)^2
(white-noise input + Gaussian output noise => population KL is proportional to K).

Estimate the local volume scaling V(eps) = P_{w~Unif(box)}(K(w) < eps) ~ c eps^lambda (-log eps)^(m-1)
in small boxes around three true parameters:
  G : generic       (p1,l1,p2,l2) = (1, .5, 0, -.3)   predicted local lambda = 3/2
  C : coincident    (1, .5, 0, .5)                    predicted 5/4 (unused pole sits on active pole)
  S : split         (.5, .5, .5, .5)                  predicted 5/4 (active mode split in two)
CPU only, numpy only.
"""
import numpy as np, sys, time

T = 40
P_STAR, L_STAR = 1.0, 0.5
k = np.arange(T)
g_star = P_STAR * L_STAR ** k

def K(p1, l1, p2, l2):
    # vectorised over samples; shape (N,)
    out = np.zeros_like(p1)
    a = np.ones_like(l1); b = np.ones_like(l2)
    for t in range(T):
        r = p1 * a + p2 * b - g_star[t]
        out += r * r
        a *= l1; b *= l2
    return out

def volume_curve(center, half, n_total, chunk, eps_grid, rng):
    hits = np.zeros(len(eps_grid), dtype=np.int64)
    c = np.asarray(center)
    done = 0
    while done < n_total:
        u = rng.uniform(-1, 1, size=(chunk, 4)) * half + c
        kv = K(u[:, 0], u[:, 1], u[:, 2], u[:, 3])
        hits += (kv[:, None] < eps_grid[None, :]).sum(0)
        done += chunk
    return hits, done

def slope(eps, hits, lo, hi):
    m = (eps >= lo) & (eps <= hi) & (hits > 50)
    x, y = np.log(eps[m]), np.log(hits[m])
    A = np.vstack([x, np.ones_like(x)]).T
    s = np.linalg.lstsq(A, y, rcond=None)[0][0]
    return s, int(m.sum())

if __name__ == "__main__":
    n_total = int(float(sys.argv[1])) if len(sys.argv) > 1 else int(2e8)
    rng = np.random.default_rng(0)
    eps = np.logspace(-11, -3, 33)
    cases = {
        "G generic    (pred 1.50)": (1.0, 0.5, 0.0, -0.3),
        "C coincident (pred 1.25)": (1.0, 0.5, 0.0, 0.5),
        "S split      (pred 1.25)": (0.5, 0.5, 0.5, 0.5),
    }
    halves = {"G": (0.1,)*4, "C": (0.1, 0.1, 0.3, 0.3), "S": (0.1,)*4}
    if len(sys.argv) > 2: cases = {k2: v for k2, v in cases.items() if k2[0] in sys.argv[2]}
    for name, c in cases.items():
        half = np.asarray(halves[name[0]])
        t0 = time.time()
        hits, n = volume_curve(c, half, n_total, 2_000_000, eps, rng)
        print(f"\n{name}: n={n:.1e}  ({time.time()-t0:.0f}s)")
        for e, h in zip(eps, hits):
            if h > 0:
                print(f"   eps={e:9.2e}  hits={h:10d}")
        for lo, hi in [(1e-10, 1e-8), (1e-9, 1e-7), (1e-8, 1e-6), (1e-7, 1e-5), (1e-6, 1e-4)]:
            s, npts = slope(eps, hits, lo, hi)
            print(f"   local slope on [{lo:.0e},{hi:.0e}] = {s:.3f}  ({npts} pts)")
```

Run: `python rlct_lti.py 1.2e8 CS` (cases C and S), or `python rlct_lti.py 4e6` (all three, quick).

### D.2 `sgd_vs_posterior.py`: stationary covariance of constant-LR SGD vs the posterior

Runtime: about 1 min.

Results:

- Σ·(2B/η) vs A^{−α}: relative error 0.054 / 0.013 / 0.037 for α = 0 / ½ / 1.
- "SGD-LLC" tr(AΣ)B/η: **5.88** for plain SGD (tr A/2 = 5.5) and **1.02** for natural gradient (d/2 = 1).

```python
"""Check Derivation 5.2: constant-LR SGD on a well-specified Gaussian linear model.
Per-example NLL l = (y - x.th)^2 / (2 s2); x ~ N(0, Sx); population Hessian A = Sx/s2 = per-example grad cov C at th*.
Prediction: Sigma_SGD ~= (eta/2B) * A^{-alpha} for P = A^{-alpha}; posterior cov = (n A)^{-1}.
Sampling from the population (fresh data each step) isolates the stationary law. numpy, CPU.
"""
import numpy as np
rng = np.random.default_rng(1)
d, s2, eta, B, steps, burn = 2, 1.0, 0.01, 8, 400_000, 20_000
Sx = np.diag([10.0, 1.0]); R = np.array([[np.cos(.6), -np.sin(.6)], [np.sin(.6), np.cos(.6)]]); Sx = R @ Sx @ R.T
A = Sx / s2; th_star = np.array([1.0, -2.0]); L = np.linalg.cholesky(Sx)
w, V = np.linalg.eigh(A)
for alpha in (0.0, 0.5, 1.0):
    P = V @ np.diag(w ** -alpha) @ V.T
    th = th_star.copy(); acc = np.zeros((d, d)); cnt = 0
    X = rng.standard_normal((steps, B, d)) @ L.T
    Y = X @ th_star + np.sqrt(s2) * rng.standard_normal((steps, B))
    for k in range(steps):
        x, y = X[k], Y[k]
        g = -(x.T @ (y - x @ th)) / (B * s2)
        th = th - eta * P @ g
        if k >= burn:
            dl = th - th_star; acc += np.outer(dl, dl); cnt += 1
    S = acc / cnt; pred = eta / (2 * B) * V @ np.diag(w ** -alpha) @ V.T
    print(f"alpha={alpha}: rel err ||S-pred||/||pred|| = {np.linalg.norm(S-pred)/np.linalg.norm(pred):.3f}")
    print("   measured Sigma*(2B/eta):\n", np.round(S * 2 * B / eta, 3), "\n   predicted A^-alpha:\n", np.round(V @ np.diag(w ** -alpha) @ V.T, 3))
    print(f"   'SGD-LLC' n*beta_eff*E[K]: with beta_eff s.t. posterior matches only if alpha=1;  tr(A Sigma)/2*(2B/eta) = {np.trace(A@S)*B/eta:.3f}  (d/2 = {d/2})")
```
