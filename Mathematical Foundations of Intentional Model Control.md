# Mathematical Foundations of Intentional Model Control

This document develops the mathematical bridges needed to design useful, correctable learning systems rather than infer alignment from a model’s name or appearance. Established results are cited, elementary derivations are shown, and original design hypotheses are labeled. Its scope includes statistical predictors, recurrent and attention models, generative systems, learned physical dynamics, LLMs, VLMs, VLAs, and world models.

The central distinction is between a theorem about a mathematical object and a claim about what a deployed system means or does. The former becomes evidence for the latter only when the observation, action, state, objective, and implementation correspondences have been established.

## Put the intended behavior in the mathematics

### Four different time axes

Use \(t\) for environment time, \(j\) for token or sequence position, \(s\) for diffusion/flow/ODE-generation time, and \(k\) for training updates. These can interact, but they are not interchangeable: neural ODE depth, generative flow time, and a robot’s executed action horizon are different objects in their respective constructions ([Chen et al.](https://arxiv.org/pdf/1806.07366), [Song et al.](https://arxiv.org/pdf/2011.13456), [Black et al.](https://arxiv.org/html/2410.24164)).

For example, a VLA may use ten integration steps in generation time to produce an action chunk covering many physical control steps. π₀’s reported construction explicitly separates its flow-matching timestep from its future-action chunk ([Black et al.](https://arxiv.org/html/2410.24164)). Our proposed assurance protocol must therefore check both the sampler’s numerical behavior and the robot’s behavior during execution.

### A complete system boundary

As an organizing model, write:

\[
x_{t+1}=f_\star(x_t,a_t,d_t),\qquad
o_t=g_\star(x_t,v_t),
\]

\[
z_t=E_\theta(o_{\leq t},a_{<t},m_t),\qquad
a_t=\Pi_\theta(z_t,q_t,m_t).
\]

Here \(x_t\) is the environment state; \(o_t\) is the observation; \(z_t\) is the learned representation or belief; \(m_t\) is persistent memory; \(q_t\) is the authorized goal/specification; and \(d_t,v_t\) are disturbances and observation errors. This separates the physical state, its uncertain estimate, and the controller, following the state-space and feedback distinction ([Särkkä and Svensson](https://users.aalto.fi/~ssarkka/pub/bfs_book_2023_online.pdf), [Åström and Murray](https://people.duke.edu/~hpgavin/SystemID/References/Astrom-Feedback-2006.pdf)).

For a tool-using LLM, expand the state to include conversation history, retrieved data, tool state, permissions, external memory, and other agents. This is a proposed boundary definition, not a claim that these variables are fully observable. Holding \(\theta\) fixed would not hold that expanded state fixed.

For a bounded task family, define the intended behavior by a tuple:

\[
\mathcal S=(\mathcal X_0,\mathcal D,\mathcal A_{\rm allowed},
J,\{c_i\},H,\delta,\mathcal F).
\]

It names starting conditions, an environment/disturbance family, authorized actions, utility, constraints, horizon, acceptable risk, and fallback behavior. For this document, “aligned on \(\mathcal S\)” means doing well on the named utility while honoring those constraints and responding appropriately to authorized correction; it does not mean general moral or semantic alignment.

An expected-cost constraint and a trajectory constraint are different:

\[
\mathbb E_\pi\!\left[\sum_{t=0}^{H}c(x_t,a_t)\right]\leq b
\quad\not\Rightarrow\quad
c(x_t,a_t)=0\text{ for every rollout and time}.
\]

This distinction matters directly for constrained policy optimization: Achiam and collaborators’ bounds concern expected discounted costs, with approximation and sampling qualifications in the implementable algorithm ([CPO](https://proceedings.mlr.press/v70/achiam17a/achiam17a.pdf)).

## Identifying a model is not choosing its coordinates

### The latent-coordinate ambiguity

Consider a learned linear state model:

\[
z_{t+1}=Az_t+Bu_t,\qquad y_t=Cz_t.
\]

For any invertible \(T\), let \(\widetilde z=Tz\). Direct substitution gives:

\[
\widetilde A=TAT^{-1},\quad
\widetilde B=TB,\quad
\widetilde C=CT^{-1},
\]

and exactly the same input-output behavior. This is a change of state realization, not evidence for a different physical mechanism; state-space realization and observability are treated in the feedback-systems reference ([Åström and Murray](https://people.duke.edu/~hpgavin/SystemID/References/Astrom-Feedback-2006.pdf)).

**Consequence of the derivation:** “dimension 17 means object mass” does not follow from prediction alone. Even an excellent predictive model may be identifiable only up to an equivalence class of representations.

For nonlinear latent learning, the ambiguity can be much larger: observational data alone cannot generally select a unique disentangled representation without suitable assumptions or supervision ([Locatello et al.](https://www.arxiv.org/pdf/2007.14184)). Ahuja and collaborators show that interventions can shrink the ambiguity under specific decoder and support assumptions, sometimes to permutation, shift, and scaling; that is a conditional identification result, not arbitrary semantic recovery ([full ICML paper](https://proceedings.mlr.press/v202/ahuja23a/ahuja23a.pdf)).

### Prediction, control sufficiency, and causal sufficiency

These are proposed tests for different representation objectives:

| Requirement | What must be preserved |
|---|---|
| Observational prediction | Future observations under the data-collection process |
| Decision sufficiency | Information needed to choose good actions for the named reward |
| Constraint sufficiency | Information needed to distinguish acceptable from unacceptable outcomes |
| Intervention prediction | Consequences of changing actions or mechanisms, including changes not common in the training data |
| Scientific identification | The desired mechanism, modulo explicitly accepted ambiguities |

A state abstraction useful for one reward need not preserve information needed for a different reward or safety rule. Bisimulation-based representation learning explicitly defines behavioral similarity relative to reward and transition structure, rather than arbitrary visual similarity ([Zhang and collaborators](https://iclr.cc/virtual/2021/oral/3480)).

For a learned model to support control, propose checking the following relation on a declared action family:

\[
\phi(f_\star(x,a))\approx F_\theta(\phi(x),a).
\]

This says that encoding after a real transition should agree with transitioning the encoded state. It is a testable version of the representation-to-dynamics correspondence used in Koopman learning, where finite learned embeddings can have nonzero invariance error ([Fan et al.](https://arxiv.org/html/2401.08153v2)).

### Excitation is part of understanding

If data never varies an action independently of the state, several action-response models may explain the observations. System-identification conditions therefore concern what inputs were tried, not just how much passive data was collected; persistency of excitation is a rank/richness condition on those inputs ([KTH system-identification notes](https://www.kth.se/social/files/542d70caf276545075276b14/lec10.pdf)).

The behavioral/data-driven control route makes this concrete: extensions of Willems’ fundamental lemma characterize when measured LTI trajectories can span desired trajectories under specified excitation and subspace conditions ([Yu, Talebi, van Waarde, Topcu, Mesbahi, and Açıkmeşe](https://arxiv.org/html/2102.02953v2)). This is not a theorem that arbitrary neural-world-model pretraining supplies enough action coverage.

## Reachability, observability, and semantic intervention

### What the classical rank tests actually say

For an \(n\)-state LTI model, repeated substitution yields:

\[
x_H=A^Hx_0+\sum_{t=0}^{H-1}A^{H-1-t}Bu_t.
\]

For \(H\geq n\), the usual controllability matrix is:

\[
\mathcal C=[B,\ AB,\ldots,A^{n-1}B].
\]

Full rank means arbitrary finite state displacement is reachable with unconstrained inputs in that model; the observability matrix \(\mathcal O=[C^\top,(CA)^\top,\ldots,(CA^{n-1})^\top]^\top\) tests whether initial states can be distinguished from output sequences ([Åström and Murray](https://people.duke.edu/~hpgavin/SystemID/References/Astrom-Feedback-2006.pdf)).

Neither statement specifies whether the inputs are affordable, physically permitted, semantically valid, or safe along the path. To see the distinction directly, take \(A=0.5I,B=0\): the system is stable but no input can move its state. Conversely, \(A=2I,B=I\) is controllable while its uncontrolled dynamics is unstable.

With quadratic input energy and full-rank finite-horizon Gramian,

\[
W_H=\sum_{i=0}^{H-1}A^iBB^\top(A^i)^\top,
\]

the least energy to produce displacement \(r=x_H-A^Hx_0\) is:

\[
E_{\min}=r^\top W_H^{-1}r.
\]

This follows by constrained least squares on the stacked input map. Rank is therefore not the whole engineering story: almost unreachable directions can demand large input energy. Under a consistent similarity transform \(r\mapsto Tr\), \(W_H\mapsto TW_HT^\top\), this energy is unchanged, but raw Gramian eigenvalues under arbitrary coordinate scaling are not invariant.

### The relevant target for a learned model

For LLMs and multimodal models, define a behavioral readout:

\[
b=\psi(\text{output or trajectory}).
\]

Examples include tool effects, solved-problem correctness, object-selection identity, task completion, calibrated abstention, or physical clearance. A proposed intervention interface \(v\) might be a structured goal, a corrected state estimate, an editable program, a curriculum parameter, or a permitted memory update.

Locally:

\[
\Delta b\approx J_v\Delta v,\qquad J_v=\frac{\partial b}{\partial v}.
\]

This is a local sensitivity approximation, not a controllability theorem. It may be undefined for discrete behavior, inaccurate for large changes, or compatible with unacceptable side effects in unmeasured components.

For a proposed correction experiment, partition \(b=(b_{\rm target},b_{\rm preserve})\). Seek:

\[
J_{\rm target}\Delta v\approx\Delta b_{\rm desired},\qquad
\|J_{\rm preserve}\Delta v\|\leq\epsilon,
\quad \Delta v\in\mathcal V_{\rm allowed}.
\]

The claim becomes useful only after the actual finite intervention is performed on held-out cases. This gives a concrete way to ask whether a model has an effective correction interface without reverse engineering its hidden circuitry.

GenCtrl provides a complementary output-space approach: sample specified prompt/intervention distributions, quantize bounded behavioral measurements, and estimate finite-horizon reachable and controllable sets with statistical guarantees ([Cheng et al.](https://arxiv.org/html/2601.05637v1)). Its guarantee is about sampled set estimation under that setup, not construction of a successful controller for every unseen instruction or verification that the measurement captures intent ([GenCtrl](https://arxiv.org/html/2601.05637v1)).

## Stability can help control and hurt memory

### A useful bound

Suppose a recurrent update obeys, in a fixed norm,

\[
\|F(z,u)-F(z',u')\|
\leq \rho\|z-z'\|+L_u\|u-u'\|,\quad 0\leq\rho<1.
\]

Iteration gives:

\[
\|z_t-z'_t\|
\leq \rho^t\|z_0-z'_0\|
+L_u\sum_{k=0}^{t-1}\rho^{t-1-k}\|u_k-u'_k\|.
\]

For uniformly bounded input difference \(\|\Delta u_k\|\leq\eta\):

\[
\|z_t-z'_t\|
\leq\rho^t\|\Delta z_0\|
+L_u\eta\frac{1-\rho^t}{1-\rho}.
\]

This elementary derivation explains both robustness and forgetting. Contracting recurrent models and RENs provide concrete architectures and conditions supporting this type of analysis ([Miller and Hardt](https://arxiv.org/pdf/1805.10369), [Revay, Wang, and Manchester](https://arxiv.org/pdf/2104.05942)).

A one-time correction encoded only as a changed initial state also decays as \(\rho^t\) under identical future inputs. Thus strong uniform contraction is not automatically desirable for a system that must retain a permanent rule change; Miller and Hardt explicitly connect stability to finite-context approximability ([Stable Recurrent Models](https://arxiv.org/pdf/1805.10369)).

### A construction hypothesis, not a solution

Propose separating:

\[
m_{t+1}=\operatorname{AuthorizedUpdate}(m_t,e_t),
\qquad
z_{t+1}=F_\theta(z_t,o_t,m_t).
\]

Here \(m_t\) stores persistent instructions or learned rules, while \(z_t\) is an adaptive state whose disturbance sensitivity can be constrained. The whole system is not globally contracting if different persistent memories remain different; the claim would concern contraction conditional on the same memory and inputs.

The decisive test is whether this decomposition preserves learned capability and retains legitimate corrections better than ordinary recurrence, replayed instructions, and retrieval. The hardest open part is not persistence itself but deciding which update is warranted and predicting its consequences.

### Composition needs its own theorem

Two stable components do not automatically form a stable feedback interconnection. Small-gain and passivity conditions supply specific compositional routes; port-Hamiltonian neural controllers use such structure to obtain bounded-gain or closed-loop results under compatible plant assumptions ([Zakwan and Ferrari-Trecate](https://ar5iv.labs.arxiv.org/html/2403.17785)).

Consequently, “every layer is stable” is not enough evidence for a deployed loop. Our proposed audit must include sensor transformations, actuator saturation, feedback delay, memory, and any policy-switching logic.

## From a learned world model to an actual trajectory claim

### Multi-step prediction error

Let \(G_\star\) and \(\widehat G\) be true and learned closed-loop transition maps for a fixed policy. Assume, on a verified region:

\[
\|G_\star(x)-\widehat G(x)\|\leq\varepsilon,
\qquad
\|\widehat G(x)-\widehat G(x')\|\leq L\|x-x'\|.
\]

Then \(e_{t+1}\leq\varepsilon+Le_t\), so:

\[
e_H\leq L^He_0+\varepsilon\sum_{i=0}^{H-1}L^i.
\]

For \(L=1\), \(e_H\leq e_0+H\varepsilon\); for \(L<1\), the accumulated mismatch is bounded by \(\varepsilon/(1-L)\); for \(L>1\), the bound can grow rapidly. This is our elementary error-propagation derivation, under the displayed assumptions, not a universal bound for learned world models.

If instead the available bounds are for open-loop \(f(x,a)\), feedback sensitivity matters. With state and action Lipschitz constants \(L_x,L_a\), and policy constant \(L_\pi\), a sufficient closed-loop constant is \(L_x+L_aL_\pi\); model MSE under a behavior policy does not establish the uniform \(\varepsilon\) needed above.

### Turning prediction error into a constraint margin

Suppose the real constraint is \(c(x)\leq0\) and \(c\) is \(L_c\)-Lipschitz. Then:

\[
c(x_t)\leq c(\widehat x_t)+L_ce_t.
\]

Therefore enforcing \(c(\widehat x_t)\leq-L_ce_t\) is sufficient at that step if the state-error bound is valid. This direct derivation explains constraint tightening: uncertainty must be converted into margin, not merely displayed as a confidence score.

Predictive safety filters implement a more developed version of this logic with model-confidence sets, tracking assumptions, tightened constraints, terminal safe sets, and fallback plans; initial feasibility and the specified uncertainty conditions are essential ([Wabersich and Zeilinger](https://ar5iv.labs.arxiv.org/html/1812.05506)). High-quality simulated video does not establish those ingredients; WorldLens reports substantial differences between visual/geometric quality and closed-loop driving results in its tested systems ([Liang et al.](https://openaccess.thecvf.com/content/CVPR2026/papers/Liang_WorldLens_Full-Spectrum_Evaluations_of_Driving_World_Models_in_Real_World_CVPR_2026_paper.pdf)).

### Beliefs instead of unjustified point states

Under partial observation, use a belief \(b_t(x)=P(x_t=x\mid o_{\leq t},a_{<t})\) when the model permits it. Bayesian filtering distinguishes the distribution of possible states from a single estimate, while nonlinear filtering methods introduce their own approximation errors ([Särkkä and Svensson](https://users.aalto.fi/~ssarkka/pub/bfs_book_2023_online.pdf)).

For our design program, uncertainty should sometimes trigger another measurement, a question, or a reversible action. That is a proposed decision rule to test against always-act and always-refuse baselines, not a guarantee obtained by naming a hidden vector a “belief state.”

## Which properties can be built in?

### Equivariance

If every module respects specified input/output group actions,

\[
f(T_gx)=T'_g f(x),
\]

then compatible composition preserves that transformation law. This is a genuine construction-level guarantee for equivariant architectures, but not a guarantee of an accurate prediction ([Kondor](https://pmc.ncbi.nlm.nih.gov/articles/PMC12541325/)).

Our design rule is to encode a symmetry only where the task and environment share it. Rotating an entire physical problem including gravity is different from rotating an object while gravity remains fixed.

### Energy and dissipation

For a chosen port-Hamiltonian model,

\[
\dot x=(J-R)\nabla H(x)+Bu,\qquad
J^\top=-J,\quad R\succeq0,\quad y=B^\top\nabla H,
\]

direct differentiation gives:

\[
\dot H=-\nabla H^\top R\nabla H+y^\top u.
\]

The skew-symmetric term cancels, exposing stored energy, dissipation, and supplied power. Port-Hamiltonian neural-control work uses this type of structure to support passivity/gain reasoning under stated assumptions ([Zakwan and Ferrari-Trecate](https://ar5iv.labs.arxiv.org/html/2403.17785)).

This equation does not identify the correct physical energy from arbitrary data or establish collision avoidance. Nor does it prove that a chosen finite-step integrator preserves the same balance; the cited work itself distinguishes continuous-time guarantees from digital implementation questions ([Zakwan and Ferrari-Trecate](https://ar5iv.labs.arxiv.org/html/2403.17785)).

### Barrier constraints

For \(\dot x=f(x)+g(x)u\), define the safe set \(\mathcal C=\{x:h(x)\geq0\}\). Under the regularity, boundary, feasibility, and controller conditions of the CBF theorem, enforcing

\[
L_fh(x)+L_gh(x)u\geq-\alpha(h(x))
\]

renders that set forward invariant ([Ames, Coogan, Egerstedt, Notomista, Sreenath, and Tabuada](http://ames.caltech.edu/ames2019control.pdf)).

If an additional disturbance \(d\) satisfies \(\|d\|\leq D\), then \(\nabla h^\top d\geq-\|\nabla h\|_*D\). A conservative sufficient inequality is therefore:

\[
L_fh+L_gh\,u-\|\nabla h\|_*D\geq-\alpha(h).
\]

This is a direct robust-margin derivation for the displayed additive-disturbance model. Uncertain perception, unknown obstacle geometry, discretization, and missed deadlines require further bounds; they are not all covered by inserting a single arbitrary \(D\).

AEGIS provides an instructive VLA bridge: its stated theorem assumes accurate safety assessment and complete enclosing representations of the obstacle and end-effector, and does not cover every other robot link ([Hu et al.](https://arxiv.org/html/2512.11891v2)). Translating “do not harm the object” into a correct \(h\) is thus a separate perception-and-specification problem.

## Preference learning does not uniquely identify intent

### A useful reward ambiguity

For a discounted MDP, define a shaped reward:

\[
r'(s,a,s')=r(s,a,s')+\gamma\Phi(s')-\Phi(s).
\]

Summing the shaped part over a trajectory telescopes:

\[
\sum_{t=0}^{T}\gamma^t
[\gamma\Phi(s_{t+1})-\Phi(s_t)]
=-\Phi(s_0)+\gamma^{T+1}\Phi(s_{T+1}).
\]

For bounded \(\Phi\) and \(\gamma<1\), the final term vanishes as \(T\to\infty\), leaving a policy-independent shift for a fixed initial state. This explains the potential-based policy-invariance construction associated with Ng, Harada, and Russell ([publication record](https://mlanthology.org/icml/1999/ng1999icml-policy/)).

The deduction matters for interpretation: observing a policy does not generally determine a unique reward. Finite-horizon tasks require treatment of terminal potentials; the preceding infinite-horizon cancellation should not be copied without that condition.

### What DPO’s equivalence does and does not buy

For a KL-regularized reward objective,

\[
\max_\pi \ \mathbb E_{y\sim\pi(\cdot\mid x)}r(x,y)
-\beta D_{\rm KL}(\pi\|\pi_{\rm ref}),
\]

the optimum has the form:

\[
\pi^\star(y\mid x)
=\frac{\pi_{\rm ref}(y\mid x)e^{r(x,y)/\beta}}{Z(x)}.
\]

DPO uses the resulting reward-policy reparameterization in a preference model, avoiding a separate explicit reward-model/RL optimization loop in its formulation ([Rafailov, Sharma, Mitchell, Ermon, Manning, and Finn](https://papers.baulab.info/papers/also/Rafailov-2023.pdf)). The equivalence concerns that formal objective and preference assumptions, not proof that the preferences encode all legitimate user intentions ([DPO](https://papers.baulab.info/papers/also/Rafailov-2023.pdf)).

Reward optimization can also exploit a proxy’s errors: Gao, Schulman, and Hilton study this in a synthetic setup with separate proxy and “gold” reward models ([reward-model overoptimization](https://arxiv.org/abs/2210.10760)). A test program should therefore measure intended outcomes with independent evidence, not use the training reward as its only judge.

CIRL offers a different formalization in which the robot is uncertain about a human reward and informative interaction can be valuable ([Hadfield-Menell, Dragan, Abbeel, and Russell](https://arxiv.org/abs/1606.03137)). Our construction hypothesis is to combine uncertainty about intent with explicit clarification and correction training; solving that hypothesis at scale remains open.

## Training as a controlled dynamical process

### A direct sensitivity model

Let training be:

\[
\theta_{k+1}=U(\theta_k,v_k,\xi_k),
\]

where \(v_k\) contains permitted choices such as data mixture, loss weights, or curriculum and \(\xi_k\) contains batch/optimizer randomness. For a fixed stochastic realization and differentiable updates, linearization gives:

\[
\delta\theta_{k+1}=A_k\delta\theta_k+B_k\delta v_k,
\]

\[
\delta\theta_K
=\sum_{k=0}^{K-1}
A_{K-1}\cdots A_{k+1}B_k\,\delta v_k
\]

when the initial parameter perturbation is zero. For a final behavioral measurement \(b(\theta_K)\), multiply by \(\partial b/\partial\theta_K\) to obtain a local training-to-behavior response map.

This is our linearization, not a general controllability result for SGD. Nonconvex transitions, optimizer state, finite updates, noise, and long products of Jacobians can break a useful local approximation.

### The statistical-physics response bridge

Define a localized Gibbs distribution:

\[
p_v(\theta)=Z(v)^{-1}
\exp[-\beta(L_0(\theta)+v\ell(\theta))]\rho(\theta).
\]

If differentiation can pass through the integral and \(O,\rho\) have no explicit \(v\)-dependence, direct differentiation yields:

\[
\frac{d}{dv}\mathbb E_v[O]
=-\beta\,\operatorname{Cov}_v(O,\ell).
\]

This relates a controlled change in a loss term to a posterior observable’s first-order response. It does not say that the optimizer samples \(p_v\), that a finite curriculum change is linear, or that estimated covariance predicts a realized retraining run.

The connection is developed in local Bayesian influence functions and stagewise data attribution: the former uses localized loss covariances, while the latter studies how influence changes during learning ([Kreer, Wu, Adam, Furman, and Hoogland](https://arxiv.org/html/2509.26544v2), [Lee, Smith, Adam, and Hoogland](https://arxiv.org/html/2510.12071v1)). At inverse temperature one, the displayed response identity has the same covariance form; temperature and localization conventions must be aligned before comparing estimators.

The evidence does not establish uniform superiority: the February 2026 BIF version reports Pythia-14M Wikitext-finetuning LDS of 0.110 for local BIF, versus 0.221 for EK-FAC and 0.198 for GradSim, with substantial seed variation; its CIFAR-10 comparison also changes ordering with data regime ([Kreer et al.](https://arxiv.org/html/2509.26544v2)). LDS is the paper’s retraining-prediction evaluation score, not an alignment score; this result supports demanding matched baseline comparisons rather than assuming that the richer statistical framework gives better interventions.

### Where singular learning theory fits

Under Watanabe’s stated analytic/statistical conditions, singular Bayesian free energy has the form:

\[
F_n=nL_n(w_0)+\lambda\log n-(m-1)\log\log n+O_p(1).
\]

Here \(\lambda\) is a learning coefficient and \(m\) a multiplicity; the regular \(d/2\) complexity penalty is not generally the right singular-model replacement ([Watanabe](https://www.jmlr.org/papers/volume14/watanabe13a/watanabe13a.pdf)). Localized estimators introduce practical dependence on the sampling and localization procedure ([Lau et al.](https://arxiv.org/pdf/2308.12108)).

The defensible bridge is: statistical geometry may help predict when a training intervention changes later behavior. The unproven transfer is: a scalar complexity estimate identifies the desired semantics, gives an actuator to change them, or guarantees that a correction survives further training.

Our decisive test is prospective: before retraining, predict the sign, magnitude, and side effects of a data/loss intervention. Compare against loss trends, gradients, finite-difference pilots, matched random interventions, and ordinary validation-based selection; count estimator cost in the comparison.

## Uncertainty guarantees and their limits

Split conformal prediction can provide marginal coverage for a prediction set \(C(X)\) under exchangeable calibration and test examples:

\[
P\{Y_{\rm test}\in C(X_{\rm test})\}\geq1-\alpha.
\]

It is not a guarantee for every individual input or arbitrary deployment shift ([Angelopoulos and Bates](https://arxiv.org/pdf/2107.07511)). Conformal risk control similarly controls a specified expected loss under conditions including exchangeability, boundedness, and monotonicity, not all meanings of safety ([Angelopoulos, Bates, Fisch, Lei, and Schuster](https://proceedings.iclr.cc/paper_files/paper/2024/file/f3549ef9b5ff520a7e41ff3cc306ab2b-Paper-Conference.pdf)).

For an original evaluation calculation, suppose \(N\) independent episodes from a fixed distribution produce zero failures. A one-sided \(1-\alpha\) binomial upper bound on the failure probability is:

\[
p_{\rm upper}=1-\alpha^{1/N}.
\]

At \(\alpha=0.05\), this is approximately \(3/N\), not zero. This calculation does not apply unchanged after adaptive test selection, correlated episodes, or an unmeasured shift.

For trajectory events \(F_t\), if valid bounds \(P(F_t)\leq\delta_t\) exist, then:

\[
P\left(\bigcup_{t=0}^{H}F_t\right)\leq
\sum_{t=0}^{H}\delta_t.
\]

This union bound needs no independence, but it does require valid event probabilities. Inventing “1% monitor failure,” “1% specification error,” and “1% enforcement failure” without evidence does not create a 3% assurance result.

## A mathematical contract for the research program

For every proposed model, record the following before interpreting its results. These are reporting requirements proposed by this investigation.

| Object | Required statement |
|---|---|
| Representation | What is encoded, what equivalences are allowed, and which interventions establish meaning? |
| Capability | Which unseen tasks or compositions can the model learn and solve, with what resources? |
| Actuator | What can the user or engineer change, and which changes are prohibited? |
| Response | What behavioral changes are predicted, including collateral effects and persistence? |
| Guarantee | Exact object, assumptions, region/distribution, horizon, numerical implementation, and confidence |
| Boundary | Which tools, memories, policies, external processes, and execution privileges are included? |
| Failure | What outcome refutes the hypothesis, and what fallback remains available? |

The promising objective is not unrestricted control of every hidden coordinate. It is reliable control of the right behavioral quantities through legitimate, efficient interfaces, while the model retains the ability to learn and solve unfamiliar problems.
