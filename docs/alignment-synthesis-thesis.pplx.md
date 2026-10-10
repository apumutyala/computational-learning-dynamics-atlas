# An Alignment Thesis: Capable, Correctable Belief-State Systems

The proposed route is to construct capable models whose representations, forecasts, objectives and corrections participate in an explicit, testable feedback loop. The central hypothesis is that alignment becomes more tractable when the system preserves the information needed for human-relevant decisions, predicts consequences under actual actions, remains uncertain about underspecified objectives, and supports corrections whose behavioral effects can be measured prospectively.

This is a research thesis, not a demonstrated solution to alignment. Its foundations supply useful component constructions: probabilistic filtering, latent planning, uncertain-objective assistance, and constrained feedback each solve narrower problems under different assumptions ([Murphy](https://www.cs.ubc.ca/~murphyk/Thesis/thesis_double.pdf), [Hafner et al.](https://proceedings.mlr.press/v97/hafner19a/hafner19a.pdf), [Hadfield-Menell et al.](https://arxiv.org/pdf/1606.03137), [Wabersich and Zeilinger](https://ar5iv.labs.arxiv.org/html/1812.05506)). The missing result is a validated composition that retains useful capability and correctability under new tasks, long horizons, changing users and strategic interference.

## The claim and the system boundary

For this program, aligned behavior means achieving a specified user's legitimate task within authorized constraints, preserving uncertainty when the task is unclear, accepting valid corrections, and avoiding unauthorized external effects. This operational definition is a proposal for experiments, not a claim that all human values reduce to a scalar reward.

The deployed system includes the model, learned and external memory, sensors, retrieval, tools, action decoder, execution harness, user interface, operator and environment. Two time axes must remain distinct: the interaction steps that change the world and the training steps that change the model; a third, slower process governs revisions to the user's specification and system authority.

| Loop element | Proposed implementation | What must be checked |
|---|---|---|
| Plant and state | Physical environment, digital workflow and relevant durable state | Which state variables can affect task utility or prohibited outcomes? |
| Observations | Sensor signals, text, tool results and authenticated corrections | Which inputs are evidence, which are instructions, and who has authority to issue them? |
| Belief | Distribution or ensemble over uncertain task state and model uncertainty | Does the representation retain distinctions required for good decisions and corrections? |
| Predictor | Action-conditioned physical or latent world model | Are forecasts accurate under the policy actually being executed, not only historical behavior? |
| Controller | Planner/policy with clarification and information-gathering actions | Does competence persist when observations are incomplete or objectives change? |
| Correction interface | Explicit updates to task facts, goals, constraints or model state | Does the intended effect persist without silently changing unrelated capabilities? |
| Independent enforcer | Tool permissions, resource limits, action constraints and fallback | Can the learned policy bypass the execution boundary? |
| Learning loop | Data, objective and architecture updates with prospective tests | Do the updates improve held-out capability and correction at the full measured cost? |

The enforcer is not a substitute for learning the task correctly. It limits consequences when the model is wrong; it does not make the model's chosen action semantically correct.

## How the fields fit together

| Field family | Concrete contribution | Missing transfer to alignment |
|---|---|---|
| Signal processing | Sampling, filtering, spectral analysis, source separation and designed invariances determine what information reaches inference ([Rowell](https://ocw.mit.edu/courses/2-161-signal-processing-continuous-and-discrete-fall-2008/80ec6fb9bd027f30adcdc16577e8ab16_samplingdft.pdf), [Bruna and Mallat](https://arxiv.org/abs/1203.1513)). | Cleaner or invariant signals can still discard a rare distinction that matters to the task. The relevant distinction must be specified and tested. |
| Classical ML and statistics | Regression, GLMs, sparse fitting, clustering and factor models provide economical models and comparison baselines ([linear models](https://scikit-learn.org/stable/modules/linear_model.html), [decomposition](https://scikit-learn.org/stable/modules/decomposition.html), [mixtures](https://scikit-learn.org/stable/modules/mixture.html)). | Interpretable coefficients, clusters or low-rank coordinates do not by themselves identify causes or intended behavior. |
| Probabilistic graphical models | Factorization exposes dependencies; message passing connects HMMs, Gaussian filtering and other inference algorithms ([Kschischang, Frey and Loeliger](https://cba.mit.edu/events/03.11.ASE/docs/Loeliger.pdf)). | A directed probabilistic graph is not automatically a causal model, and approximate inference can be wrong ([Wainwright and Jordan](https://people.eecs.berkeley.edu/~jordan/papers/wainwright-jordan-fnt.pdf)). |
| Forecasting and world models | Classical temporal baselines, data assimilation and learned latent dynamics predict future observations or consequences of actions ([Hyndman](https://robjhyndman.com/nyc2018/1-5-StateSpace.pdf), [Evensen](https://www.ecmwf.int/sites/default/files/elibrary/2003/9321-ensemble-kalman-filter-theoretical-formulation-and-practical-implementation.pdf), [PlaNet](https://proceedings.mlr.press/v97/hafner19a/hafner19a.pdf)). | Passive prediction does not establish an intervention model; action-conditioned prediction also needs action coverage and validation. |
| Information theory | Entropy, sufficient statistics and prediction-preserving compression quantify which information a chosen representation retains ([Wainwright and Jordan](https://people.eecs.berkeley.edu/~jordan/papers/wainwright-jordan-fnt.pdf), [Shalizi and Crutchfield](https://arxiv.org/pdf/cs/0001027v1)). | “Relevant information” depends on a declared future, objective and operating regime; mutual information is not a measure of moral correctness. |
| Statistical mechanics | Gibbs distributions, response identities, mean-field approximations and stochastic sampling provide mathematical tools for inference and learning response ([Wainwright and Jordan](https://people.eecs.berkeley.edu/~jordan/papers/wainwright-jordan-fnt.pdf), [Welling and Teh](https://www.stats.ox.ac.uk/~teh/research/compstats/WelTeh2011a.pdf)). | Equilibrium response is not automatically finite-optimizer response; statistical free energy is not automatically physical energy. |
| Physical and latent dynamics | State reduction, explicit memory and structure-preserving models supply testable models of evolution ([Mori–Zwanzig study](https://arxiv.org/html/2507.16058v1), [structured mechanics comparison](https://proceedings.mlr.press/v144/zhong21a/zhong21a.pdf)). | A learned latent state need not be physically identifiable, Markovian or semantically editable. |
| Classical and modern control | Observability, reachability, robust control, MPC and safety filters specify estimation, actions and operating conditions ([Åström and Murray](https://people.duke.edu/~hpgavin/SystemID/References/Astrom-Feedback-2006.pdf), [predictive safety filtering](https://ar5iv.labs.arxiv.org/html/1812.05506)). | Controlling a named state or constraint does not prove that the named quantity captures the human's intention. |
| Assistance and preference learning | Treat a human objective as partly unknown, and make clarification, teaching and communication part of decision-making ([CIRL](https://arxiv.org/pdf/1606.03137)). | The representation of preferences and the model of human feedback may be misspecified; better proxy return need not mean better true utility ([Skalse et al.](https://arxiv.org/pdf/2209.13085)). |
| Developmental and training control | Measure responses to data/objective changes and deliberately influence learning ([Patterning](https://arxiv.org/html/2601.13548v1), [Goodfire](https://arxiv.org/html/2606.12360v1)). | A successful bounded intervention does not establish arbitrary capability control, net efficiency or persistent alignment. |

The proposed composition is not “replace deep learning with classical methods.” It is to use explicit structure where it helps, learned components where they are necessary, and common experiments that reveal when a simpler statistical model is sufficient.

## Mathematical connections that do real work

### Factorization, filtering and world-model planning

A controlled state-space model can factor its joint distribution into an initial belief, transitions and observations; HMMs, linear-Gaussian models and richer DBNs instantiate different choices of state and conditional distribution ([Murphy](https://www.cs.ubc.ca/~murphyk/Thesis/thesis_double.pdf)). For the following formula, the action sequence is fixed externally: conditioning retrospectively on an entire adaptively chosen action sequence is not generally the same experiment.

\[
p(z_{0:T},y_{1:T}\mid u_{0:T-1})
=p(z_0)\prod_{t=1}^{T}
p(z_t\mid z_{t-1},u_{t-1})p(y_t\mid z_t).
\]

With the corresponding conditional-independence assumptions, the belief update combines a predictive transition with a likelihood update ([Murphy](https://www.cs.ubc.ca/~murphyk/Thesis/thesis_double.pdf)).

\[
b_t(z)\propto p(y_t\mid z)
\int p(z\mid z',u_{t-1})b_{t-1}(z')\,dz'.
\]

This is a meaningful common interface: a Kalman filter chooses a linear-Gaussian representation; a particle method approximates the distribution with weighted samples; a learned latent model can supply transition and observation functions ([Särkkä and Svensson](https://users.aalto.fi/~ssarkka/pub/bfs_book_2023_online.pdf), [PlaNet](https://proceedings.mlr.press/v97/hafner19a/hafner19a.pdf)). The resulting belief is sufficient only relative to the specified model, not automatically relative to every consequence that a user might care about.

The proposed alignment extension adds uncertainty over an objective parameter \(\phi\) and treats authenticated feedback as evidence about that objective. CIRL supplies a formal precedent for this idea under a shared-reward game, but does not establish that arbitrary human intentions are represented by \(\phi\) or that a real human follows the assumed feedback model ([Hadfield-Menell et al.](https://arxiv.org/pdf/1606.03137)).

### Statistical free energy, entropy and response

For a normalized statistical model \(p(y,z)\), define a variational free-energy functional over a distribution \(q(z)\). The following identity follows by substituting Bayes' rule into the KL divergence, and is the statistical inference identity used here, not a claim about physical heat:

\[
\mathcal F[q]
=\mathbb E_q[\log q(z)-\log p(y,z)]
=D_{\mathrm{KL}}(q(z)\Vert p(z\mid y))-\log p(y).
\]

The unrestricted optimum is the posterior; restricting the family of \(q\) turns inference into an approximation problem, connecting variational inference with entropy and log-partition duality ([Wainwright and Jordan](https://people.eecs.berkeley.edu/~jordan/papers/wainwright-jordan-fnt.pdf)). A small variational objective does not establish causal validity, calibrated uncertainty, useful decisions or agreement with a human.

For the regular fixed-support exponential family

\[
p_\eta(z)=\exp\{\eta^\top T(z)-A(\eta)\},
\]

the gradient of \(A\) is the mean of \(T\), and its Hessian is the covariance of \(T\), equal to the Fisher information in these canonical coordinates ([Wainwright and Jordan](https://people.eecs.berkeley.edu/~jordan/papers/wainwright-jordan-fnt.pdf)).

\[
\nabla A=\mathbb E[T],\qquad
\nabla^2A=\operatorname{Cov}(T).
\]

This supplies a precise connection between information geometry and statistical response. It does not equate a covariance, a local learning coefficient, a Hessian of any chosen loss, and a Hankel spectrum.

For a fixed observable \(g\), a normalized distribution
\(q_h(\theta)\propto\pi(\theta)\exp[-\beta(L(\theta)+\sum_i h_i\ell_i(\theta))]\)
gives the following local response by differentiating its normalization, assuming the integrals and derivatives can be interchanged:

\[
\frac{\partial}{\partial h_i}\mathbb E_{q_h}[g]
=-\beta\,\operatorname{Cov}_{q_h}(g,\ell_i).
\]

This derivation is an atlas synthesis of the exponential-family response identity. If \(g\) also depends on \(h\), an additional \(\mathbb E[\partial g/\partial h_i]\) term is needed; using a changing diagnostic without that qualification is not the same calculation.

Patterning provides a concrete posterior-susceptibility training intervention, while the recorded typed-control experiment tests actual reweighting responses and finds gradient alignment useful in its tested setup ([Patterning](https://arxiv.org/html/2601.13548v1), [Item 4](https://github.com/apumutyala/chl-typed-control/issues/1#issuecomment-5925146688)). The essential experiment is to test whether such response measurements predict and improve real learning, rather than assume a posterior identity describes Adam.

### Why compressed latent state can need memory

Consider this unforced linear system, used here as an exact elementary example. Let \(z\) be retained variables and \(w\) variables that will be eliminated:

\[
\dot z=Az+Bw,\qquad \dot w=Cz+Dw.
\]

Solving the second equation and substituting into the first yields

\[
\dot z(t)=Az(t)+Be^{Dt}w(0)
+\int_0^t Be^{D(t-s)}Cz(s)\,ds.
\]

Thus even this simple exact reduction produces a memory kernel \(Be^{D(t-s)}C\) and an initial-state term. Approximating the reduced process as memoryless discards information unless additional conditions justify that step; adding external inputs would introduce corresponding forcing terms.

Mori–Zwanzig formalism generalizes this separation into resolved dynamics, memory and orthogonal dynamics, and the cited data-driven turbulence study learns a truncated memory model ([de Wit et al.](https://arxiv.org/html/2507.16058v1)). This motivates a constructive hypothesis for RNNs, SSMs and world models: choose and test memory because the retained state needs it, not because recurrence itself guarantees meaningful state.

### From model error to decision quality: a bounded positive result

The following is a standard contraction-style derivation spelled out for this thesis. Assume two discounted MDPs have the same states and available actions, discount \(0\le\gamma<1\), rewards in \([0,R_{\max}]\), uniform reward error at most \(\epsilon_r\), and uniform transition total-variation error at most \(\epsilon_p\).

For a fixed policy, Bellman recursion and the bound
\(|\mathbb E_P V-\mathbb E_Q V|\le\operatorname{range}(V)\operatorname{TV}(P,Q)\)
give

\[
\|V^\pi-\widehat V^\pi\|_\infty
\le \epsilon_r+\gamma\|V^\pi-\widehat V^\pi\|_\infty
+\frac{\gamma R_{\max}\epsilon_p}{1-\gamma}.
\]

Rearranging gives

\[
B=\frac{\epsilon_r}{1-\gamma}
+\frac{\gamma R_{\max}\epsilon_p}{(1-\gamma)^2}
\]

as an upper bound on the fixed-policy value difference. If planning is exact and the bound holds uniformly over the policy class, the policy optimal in the approximate model has regret at most \(2B\) against the optimal policy in that class, by adding and subtracting the two approximate-model values.

This is a genuine route from modeling accuracy to competent decisions, but only under the stated uniform conditions. Average held-out pixel error does not establish either error bound; partial observations, changing action sets, an inadequate abstraction, or the wrong reward can invalidate the transfer.

The proposed alignment program must therefore validate the representation and the objective as well as the dynamics. Better prediction of the wrong task is not alignment, while perfect objective identification without sufficient capability is not useful assistance.

## Information theory and physics: useful bridges, false shortcuts

- **Predictive state is not automatically physical state:** computational mechanics groups histories by their future distributions under a stationary symbolic-process model; its “causal states” do not themselves establish intervention semantics ([Shalizi and Crutchfield](https://arxiv.org/pdf/cs/0001027v1)). Controlled predictive state representations instead use tests involving specified action sequences, with their own representation theorem and learning limitations ([Littman, Sutton and Singh](https://web.eecs.umich.edu/~baveja/Papers/psr.pdf)).
- **Compression needs a relevance target:** information bottleneck preserves information about a specified relevance variable, so the choice of task or target carries substance that entropy alone cannot supply ([Tishby, Pereira and Bialek](https://www.princeton.edu/~wbialek/our_papers/tishby+al_99.pdf)). The thesis proposes preserving action consequences, constraints and correction-relevant facts jointly, and testing whether this works.
- **Physical dissipation is not a universal ML loss:** Still and collaborators connect nonpredictive memory to dissipation for a specified driven system in contact with a heat bath, assuming no feedback from system to driving signal ([Still et al.](https://arxiv.org/pdf/1203.3271)). An agent changes its environment, and a GPU training run is not automatically that thermodynamic setup.
- **Criticality is not an alignment principle:** mean-field analyses can explain signal propagation in specified network ensembles, but do not select the right human objective ([Schoenholz et al.](https://arxiv.org/pdf/1611.01232v2)). Stability, chaos, information retention and useful task behavior require separate measurements.
- **Coverage is not trajectory safety:** conformal prediction's exchangeability-based marginal coverage and online conformal regret results concern different objects from safe adaptive control ([Angelopoulos, Barber and Bates](https://arxiv.org/pdf/2411.11824v4), [Gibbs and Candès](https://www.jmlr.org/papers/v25/22-1218.html)). To use a sequence of valid failure probabilities in a trajectory bound, those probabilities must actually apply under the deployed action-selection process; a benchmark's marginal interval does not automatically meet that condition.
- **Increasing uncertainty is not calibration:** the 2026 horizon-aware world-model study imposes a horizon-dependent ensemble-disagreement objective and separately learns an action-conditioned stream ([Wan, Gan and Zhan](https://proceedings.iclr.cc/paper_files/paper/2026/file/ad222e6e38e26e3f00d85829b9e02f43-Paper-Conference.pdf)). The thesis would test its uncertainty against proper scores and empirical errors, rather than treat the word “calibrated” as a theorem.

## The constructive architecture proposal

The proposal is a modular research system, not a requirement that every model use every method. Its interfaces are specified below so that a simpler statistical implementation and a deep learned implementation can be compared on the same task.

- **Observation interface:** preserve timing, units, provenance and authority labels. Use task-appropriate filtering and representations; do not smooth away task-changing events or treat retrieved instructions as authorized corrections.
- **Belief interface:** maintain a state estimate and uncertainty with explicit update semantics. Compare HMM/DBN, Kalman/particle, predictive-state and learned recurrent alternatives where their assumptions are plausible.
- **Consequence interface:** predict multi-step outcomes conditional on candidate actions, including observation quality and uncertainty. Require held-out action tests, not just plausible video or fluent narration.
- **Intent interface:** represent unresolved goals and constraints separately from uncertain world facts. Support clarification, multiple plausible interpretations, revocation and stakeholder-specific authority.
- **Correction interface:** identify whether a correction changes a belief, task rule, constraint, preference, durable memory or learned parameter. Test persistence and unrelated-task retention for each intervention class.
- **Planning interface:** choose useful actions or queries under belief and objective uncertainty. Make uncertainty-driven abstention compete with a competent baseline rather than win by refusing every difficult task.
- **Execution interface:** check authorized actions at the point of use, maintain bounded resources, and provide a tested fallback. Put enforcement outside the learned model's unilateral write access.
- **Learning interface:** use measurement-guided curriculum, replay or objective changes only when they improve held-out results after including measurement and tuning costs. Keep changes versioned and reversible.

For an LLM, the tested state may include task facts, working memory, goals and tool history without reconstructing internal circuits. For a VLM it additionally includes uncertain visual observations; for a VLA it includes the action decoder, robot dynamics and physical execution constraints.

## Gap register and failure tests

| Unresolved gap | Why it can break the thesis | Test that could reject the proposed bridge |
|---|---|---|
| Semantic sufficiency | A compact state fits ordinary examples but merges cases needing different actions | Construct matched observations with different relevant histories, constraints or intentions; test decision separation |
| Causal action validity | A predictor exploits correlations created by the behavior policy | Intervene on actions and mechanisms; evaluate beyond the training policy's support and report nonidentifiability |
| Objective identification | Several incompatible user objectives fit the same feedback | Require clarification or explicit uncertainty; test preference changes, mistakes and disagreement |
| Persistent correction | An edit changes one output but not subsequent behavior, or damages unrelated skills | Measure immediate effect, long-horizon retention, recovery and cross-task side effects |
| Uncertainty under adaptation | The policy visits regions where its calibration no longer applies | Evaluate after adversarially selected actions, abrupt changes and rare events |
| Optimization pressure | Increased capability finds a new way to exploit a proxy | Optimize the proxy strongly while independently measuring task utility and forbidden outcomes |
| Representation reduction | Removed low-variance or low-Hankel directions encode rare but important events | Test task-weighted reduction against PCA, balanced and unreduced models under rare constraints |
| Compositional control | Individually good estimators, predictors and controllers interact badly | Introduce delays, actuator limits, belief errors and feedback coupling, then measure closed-loop outcomes |
| Authority and adaptation | Tool use, persistent memory or successor updates bypass the intervention boundary | Attempt permission bypass, memory poisoning, unchecked tool delegation and unreviewed parameter changes |
| Compute and usefulness | The control machinery costs more than it helps or refuses most tasks | Compare total compute and accepted-task success at a fixed error/correction threshold |

These are not gaps already filled by terminology. They are the experimental obligations that separate a cross-field thesis from a demonstrated alignment method.

## Experiment ladder and connections to existing work

The proposed decisive experiment is a partially observed, multi-step task family with substantive learned capability, hidden regime changes and explicit user corrections. Suitable initial domains include a simulated resource-management workflow or a physical-control simulator with uncertain sensors and changing goals; the environment must contain genuine planning and generalization demands, not only a tiny known-state diagnostic.

| Stage | Required comparison | Pass condition |
|---|---|---|
| Forecast and belief competence | Naive/ARIMA/ETS where appropriate; HMM/DBN or Kalman/particle; learned recurrent/world model | Useful multi-horizon prediction and decisions, with no future-information leakage |
| Learned representation | Raw-history baseline; PCA/factor state; structured recurrent state; learned latent state | Preserves held-out action, rare-event and correction-relevant distinctions |
| Correction | Prompt/history update; external-memory edit; retraining; proposed explicit interface; sham/random controls | Better persistent correction without comparable loss of unrelated capability |
| Intent uncertainty | Fixed reward; preference-trained policy; clarification-capable assistance model | Better independently evaluated utility under ambiguity and valid goal changes |
| Training control | Uniform sampling; loss-based scheduling; gradient alignment; interference-aware replay; optional susceptibility | Lower full cost to the capability-and-correction threshold |
| Composition stress | Same trained systems with delays, sensor loss, shifted policies and adversarial feedback | Bounded degradation, functioning fallback and no bypass of the declared execution boundary |

The existing typed-control result establishes a useful forward-response baseline, not successful inverse capability control; the recorded SGLD outcome remains inconclusive ([Item 4](https://github.com/apumutyala/chl-typed-control/issues/1#issuecomment-5925146688)). The reviewed State-Space-Decision-Model PR supplies qualification gates rather than a learned-capability result, while the structured-models brief makes construction from initialization a distinct research track ([PR 215](https://github.com/apumutyala/State-Space-Decision-Model/pull/215), [structured-models brief](https://github.com/apumutyala/chl-structured-models/issues/1)).

The next step is to connect those tracks through a shared task and evaluation contract. No experiment described in this section was run as part of building the atlas.

## Supported claim and current verdict

- **Supported claim:** the cited fields provide concrete model classes, inference algorithms, response identities and bounded control constructions that can be composed into testable systems. They are useful ingredients, not merely analogies ([factor graphs](https://cba.mit.edu/events/03.11.ASE/docs/Loeliger.pdf), [PlaNet](https://proceedings.mlr.press/v97/hafner19a/hafner19a.pdf), [CIRL](https://arxiv.org/pdf/1606.03137)).
- **Unproven transfer:** a learned representation and objective must preserve the distinctions needed for human-relevant action and persistent correction under shift. The atlas proposes this as the central empirical target rather than asserting it has been solved.
- **Control gap:** the system needs reliable measurements of intent and consequences, an effective correction actuator, and an execution boundary that remains intact while capability and adaptation increase.
- **Next decisive test:** demonstrate both competent novel-task performance and lasting correctability against the declared simple baselines at matched total cost, then stress the same system rather than changing tasks to make each component look successful.

The thesis succeeds only if the combination produces stronger evidence than its components separately. A capable and correctable system in a declared domain would be meaningful progress; claiming general alignment would require additional evidence about specification, transfer, strategic behavior, long horizons and deployment authority.
