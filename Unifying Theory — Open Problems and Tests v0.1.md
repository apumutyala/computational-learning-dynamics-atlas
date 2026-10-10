# Unifying Theory — Open Problems and Tests v0.1

This is the companion to `Unifying Theory — Draft v0.1.md`. It revises the v0 companion using `Unifying Theory — Review v0.md`. The v0 files were not edited.

The reviewer's scripts (`rlct_mc_rerun.py`, `rlct_fiber.py`, `checks_b9_b11.py`) are in the session scratchpad. They are referenced here and were not re-run by me, except where stated.

---

## A. Claim ledger and errors

### A.1 Source-document ledger

The v0 ledger (`Unifying Theory — Open Problems and Tests.md`, §A.1, 26 claims) stands unchanged: no mathematical errors in the five source documents' derivations.

The source-document issues E1–E8 also stand. E1 and E2 (unlisted SLT conditions; WBIC matching F only through λ log n) are now addressed inside the draft (§4.1–4.2).

### A.2 Errors in *our own* v0, found by Review v0 and fixed in v0.1

| ID | v0 claim | Status | Fix (draft v0.1 section) |
|---|---|---|---|
| V1 | I(S;W) ≈ λ log n for the Gibbs algorithm | **Wrong.** I(S;W) = O(1) at a fixed truth. | §2.3: KL and MDL scale as λ log n; MI does not. |
| V2 | t·E[L_n − L_n(θ₀)] → λ at fixed β | **Wrong.** The limit is λ − βν (equations of state). | §2.3, eq. (2.3) |
| V3 | Toy slope "1.00 ± 0.01 = rank(H)/2" | **Not reproduced.** Reviewer: 1.05–1.13. My v0.1 smoke run: 1.04–1.08. Fibre method → 1.250. | §3.2: now a hypothesis |
| V4 | Inlined code reproduces the 0.1-box C row | **Wrong.** The box was hard-coded. | D.1 below |
| V5 | B1 "exact with NLL" | **Mixed claims.** The bound holds for bounded loss and fixed t. | §2.2 |
| V6 | "Certificates never point-valued" | **Overreach.** Łojasiewicz gives convergence to a point. | §3.6 |
| V7 | "Non-identifiability = unobservability = degeneracy" | **Overreach.** Counterexamples: x⁴; permutation copies. | §4.5 |
| V8 | B7 consequences stated generally | **Overreach.** They need C = A. | §5.2–5.3 |
| V9 | "Nothing transfers to free-running generation" | **Wrong** for exact per-trajectory identities. | §7.3–7.4 |
| V10 | B1, B7, B10 labelled DERIVATION | **Mislabelled.** These are known results. | Now labelled KNOWN |

### A.3 Repo correction (kept, now independently confirmed)

`State-Space-Decision-Model/reviews/02-technical-pressure-test.md` (line ~202) claims two things, both **wrong**:

- "it is an equality";
- coincidence loci "do not lower λ; they raise m".

**Correct statement:** for n = 2, r = 1 the global RLCT is λ = 5/4, with m = 1.

- **Evidence:** a derivation (draft §4.6) and the reviewer's fibre method, which gives a slope of 1.250 at ε = 10⁻¹⁶.
- "r + (n−r)/2" is the **local coefficient at a generic true parameter only**. As an inequality it is literally true.
- `plans/claude/PLAN.md` line 78 should be relabelled the same way.

---

## B. Conjectures with falsifiable tests (revised)

### C1 — RLCTs of diagonal SSMs beyond (2,1)

**Settled (no longer a conjecture).** (n, r) = (2, 1): λ = 5/4, m = 1. This is a DERIVATION plus two numerical methods.

**Open (downgraded).** v0's claim that "the global RLCT is attained when all unused poles coalesce onto active poles" is a **guess**. Coalescing two unused poles onto one active pole and splitting them across active poles give *different* moment ideals. The complex-pair claim "strictly below n + r" is plausible but unproven.

**Test T-C1 (exact algebra, CPU, $0).**

1. For each coalescence pattern of (n, r) ≤ (4, 2), generate the moment ideal (c₀, c₁, …) in local coordinates.
2. Compute its RLCT by a Newton polyhedron in non-degenerate coordinates, or by explicit blow-ups. A computer-algebra system can do this, or a small hand computation for n ≤ 3.
3. Cross-check each case with the reviewer's fibre-minimization method (`rlct_fiber.py`), which reaches ε = 10⁻¹⁶ cheaply.

Do **not** rely on thermodynamic integration over t ∈ [10², 10⁸]: the pre-asymptotic drift (§3.2) can keep the slope away from λ in that range.

**Falsified if** the algebraic value for some pattern is below the "all-coalesce" value. That would refute the guess, not the (2,1) result.

**C1b (E1 vs E2).** Train SGD from a stated initialization law and record whether unused poles migrate onto active ones. Two predictions compete:

- the posterior concentrates on coalescence as n → ∞ (mass ratio ~ n^{1/4} at (2,1));
- SGD's drift toward smaller tr H (draft §5.3) may or may not point toward coalescence.

**Test:** compute tr H on both strata, and use that to predict the direction of drift *before* running.

### C2 — Scale-dependent finite-n exponent (Hypothesis B4)

**Statement.** Define the scale-tied rank r_τ(H) = #{h_i > τ}, with τ ≈ 1/(nβR²), where R is the localization radius (R ≈ γ^{−1/2} for SGLD-LLC). The hypothesis is that SGLD-LLC estimates lie between r_τ/2 and r_τ/2 + (d − r_τ)/4 when localization is weaker than the quadratic scale, and approach λ as n grows.

**Not a violation-detector.** Values below r_τ/2 can occur legitimately under strong localization. They are *not* automatically estimator bias. Check against the lab's εN/B·λ_max diagnostic separately.

**Test T-C2.** On the LTI toy and on small tanh MLPs, compute λ̂ (SGLD), r_τ(H) (Lanczos) and the fibre-method volume slope across n = 10²–10⁶ and γ values.

**Falsified if** λ̂ − r_τ/2 is large and n-independent where the localization scale exceeds the quadratic scale.

### C3 — Where SGD lands on W₀, versus the posterior (re-specified)

**The v0 test was mis-specified.** On K = a²b², every point of W₀ has λ = 1/2. The origin differs only in **multiplicity** (m = 2), so the posterior's preference there is a log n factor, not n^{Δλ}. Also, a KS test against an initialization-dependent SGD law rejects almost surely.

**New statement (sign is the falsifiable content).** Use a model whose strata differ in **λ**: the 2-mode LTI toy, where the generic stratum has 3/2 and the coalescence stratum 5/4. Under a stated initialization law and small η, consider

    the fraction of SGD endpoints within distance δ of the coalescence stratum,

compared with the posterior's fraction at the matched temperature β_eff. Stochastic collapse (Chen, Kunin et al. 2023) and drift toward flat regions (Blanc 2019/20; Damian, Ma & Lee 2021; Li, Wang & Arora 2022) could make **SGD favour the lower-noise, flatter stratum *more* than the posterior does**. That is the opposite of v0's prediction.

**Pre-register the sign** from a tr H / noise calculation on each stratum, then run:

- 10³–10⁴ SGD runs;
- exact-likelihood MCMC;
- a bootstrap CI on the difference of fractions.

**Falsified if** the CI excludes the pre-registered sign.

### C4 — Curriculum order effects: block curricula, not single swaps

The single-swap prediction η²∇bᵀ[g_i, g_j] is a **Taylor identity**. It was checked numerically by the reviewer (relative error O(η)) and needs no further test, beyond an implementation check with identical minibatches.

**New statement.** For block curricula (K_i steps on i, then K_j steps on j), the BCH leading term K_iK_jη²∇bᵀ[g_i, g_j]:

- predicts the sign and magnitude of the order effect while K·η·λ_max ≲ 0.3;
- **loses predictive power** (R² < 0.2) once K·η·λ_max ≳ 1.

**Test T-B9.** A small MLP or tiny GDN with two domains; identical minibatch sequences in both orders; a sweep over K·η·λ_max ∈ [0.03, 3]; 20 seeds.

**Falsified if** the crossover is absent, i.e. predictive at Kηλ ≳ 1, or unpredictive at Kηλ ≲ 0.1. A full-solve comparison (integrating the two block flows) is the reference.

### C5 — Delta-rule diagnostics

- **(a) Dropped as a conjecture.** The null-edit property is a tautology. Keep it only as an *implementation check* for intervention code: edits in ker W_o(T) must give zero output change to float tolerance.
- **(b) Kept.** The per-sequence persistence functional, composed through layers, rank-predicts the measured knockout-at-distance flip rate with Spearman ρ > 0.5 across spans and layers. Logged, never a gate.
- **(c) Restricted to critical points.** At a checkpoint with small ‖∇K‖ (report it), the Hessian nullity **restricted to symmetry tangents** should match the symmetry count of the exact parameterization:
  - attention q/k with RoPE: d_rot + (d_k − d_rot)²;
  - GDN: ≈ d_k per q and k path.

  Report a stated eigenvalue threshold. Expect data-dependent degeneracies to add nullity *outside* the symmetry tangents.
- **(d) New.** For RWKV-7, measure ‖G_t‖₂ over real sequences. Also measure ‖k_t‖ in GDN implementations, to check the exact-unit-norm assumption of B11.

### C6 — KL(Q*‖φ) = λ log n + O(log log n)

**Test T-B2.** Thermodynamic integration in reduced-rank regression (exact λ, Aoyagi–Watanabe 2005) and in the LTI toy. MI is *not* tested; it is O(1), as settled by the reviewer's numerics.

### C7 (new) — PAC-Bayes regime t ≍ √(n log n)

**Statement.** F_n(t) − tL_n(θ₀) = λ log t + o(log t) for t → ∞ with t = o(n).

**Test.** Exact-likelihood thermodynamic integration on reduced-rank regression, with t = n^{1/2}, n^{2/3} and n^{3/4}.

---

## C. Cheap experiments (all logged, none gate anything)

| # | Experiment | Cost | Decides |
|---|---|---|---|
| X1 | Algebraic RLCTs of LTI coalescence patterns, plus the fibre-method cross-check (T-C1) | CPU, $0 | C1 (general n, r) |
| X2 | Stratum-sign test, SGD vs posterior on the 2-mode LTI (C3) | CPU, $0 | C3, C1b |
| X3 | Block-curriculum order effects vs Kηλ_max (T-B9) | CPU, or under $1 | C4 |
| X4 | Persistence functional vs knockout flip rate; null-edit implementation check; ‖k‖ and RWKV-7 ‖G_t‖₂ audit | CPU tiny GDN; under $5 on one converted checkpoint | C5(b, d) |
| X5 | Restricted Hessian nullity at near-critical checkpoints, before and after conversion | under $5 | C5(c) |

---

## D. Code

Interpreter: `CHLabs/.venv-gpu/Scripts/python.exe`, CPU only.

### D.1 `rlct_lti_v01.py`: toy RLCT volume scaling (box bug fixed)

Each case now carries its own box. v0 hard-coded the C box to (0.1, 0.1, 0.3, 0.3), so the 0.1-box row could not be reproduced.

**Smoke run this session** (4 × 10⁶ samples, cases C1 and C3):

| Case | Slope on [1e-8, 1e-6] | [1e-7, 1e-5] | [1e-6, 1e-4] |
|---|---|---|---|
| C1 (box 0.1) | 1.075 | 1.074 | 1.041 |
| C3 (box 0.1/0.3) | — | 1.165 | 1.178 |

These match the reviewer's re-run (C1: 1.05–1.12). The asymptote from the fibre method is 1.250.

Usage: `python rlct_lti_v01.py 1.2e8 C1,C3,S` (about 90 s per case), or `python rlct_lti_v01.py 4e6` (all cases).

```python
"""Toy RLCT check for a 2-mode real diagonal SISO LTI model fitting a 1-mode truth.

g_k(w) = p1*l1^k + p2*l2^k,  truth g*_k = 1 * 0.5^k,  K(w) = sum_{k<T} (g_k - g*_k)^2
(white-noise input + Gaussian output noise => population KL is proportional to K).

Estimate the local volume scaling V(eps) = P_{w~Unif(box)}(K(w) < eps) ~ c eps^lambda (-log eps)^(m-1)
in boxes around true parameters. Finite-eps slopes are PRE-ASYMPTOTIC (see draft v0.1 §3.2);
the asymptotic value at the coincident/split points is 5/4 (fibre method reaches 1.250 at eps=1e-16).
CPU only, numpy only.
"""
import numpy as np, sys, time

T = 40
P_STAR, L_STAR = 1.0, 0.5
k = np.arange(T)
g_star = P_STAR * L_STAR ** k

def K(p1, l1, p2, l2):
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
    if m.sum() < 2:
        return float("nan"), int(m.sum())
    x, y = np.log(eps[m]), np.log(hits[m])
    A = np.vstack([x, np.ones_like(x)]).T
    return np.linalg.lstsq(A, y, rcond=None)[0][0], int(m.sum())

if __name__ == "__main__":
    n_total = int(float(sys.argv[1])) if len(sys.argv) > 1 else int(2e8)
    rng = np.random.default_rng(0)
    eps = np.logspace(-11, -3, 33)
    # name -> (center, per-coordinate box half-widths); v0 hard-coded the C box to (.1,.1,.3,.3) (Review v0, B4)
    cases = {
        "G  generic, box .1        (pred 1.50)": ((1.0, 0.5, 0.0, -0.3), (0.1,) * 4),
        "C1 coincident, box .1     (asympt 1.25)": ((1.0, 0.5, 0.0, 0.5), (0.1,) * 4),
        "C3 coincident, box .1/.3  (asympt 1.25)": ((1.0, 0.5, 0.0, 0.5), (0.1, 0.1, 0.3, 0.3)),
        "S  split, box .1          (pred 1.25)": ((0.5, 0.5, 0.5, 0.5), (0.1,) * 4),
    }
    if len(sys.argv) > 2:  # e.g. "C1,C3"
        want = set(sys.argv[2].split(","))
        cases = {k2: v for k2, v in cases.items() if k2.split()[0] in want}
    for name, (c, hw) in cases.items():
        half = np.asarray(hw)
        t0 = time.time()
        hits, n = volume_curve(c, half, n_total, 2_000_000, eps, rng)
        print(f"\n{name}: n={n:.1e}  ({time.time()-t0:.0f}s)")
        for lo, hi in [(1e-10, 1e-8), (1e-9, 1e-7), (1e-8, 1e-6), (1e-7, 1e-5), (1e-6, 1e-4)]:
            s, npts = slope(eps, hits, lo, hi)
            print(f"   local slope on [{lo:.0e},{hi:.0e}] = {s:.3f}  ({npts} pts)")
```

### D.2 `sgd_vs_posterior.py`

The script is unchanged from the v0 companion §D.2. Its results are now read as the **C = A case only** (draft §5.2).

With isotropic gradient noise, the same script would give Σ ∝ A⁻¹. Adding a `C = σ²I` option is a one-line change: replace the per-example gradient noise with a Gaussian of covariance σ²I/B. This is suggested as a quick check of the "flip" statement.

### D.3 Reviewer scripts (not re-run by me)

| Script | What it does |
|---|---|
| `rlct_mc_rerun.py` | Monte Carlo re-run |
| `rlct_fiber.py` | Fibre minimization over (p₁, l₁) with Gauss–Newton, log-refined 440² grid, reaching ε = 10⁻¹⁶ |
| `checks_b9_b11.py` | Bracket convergence; delta-rule norms; I(S;W) vs E KL |
