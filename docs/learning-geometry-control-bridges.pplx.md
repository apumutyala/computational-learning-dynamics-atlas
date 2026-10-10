# Learning Geometry Meets System Control

This extension separates three objects: the physical or digital system being controlled, the model used to predict it, and the training process that changes that model. State-space control analyzes input, state, and output dynamics; singular learning theory analyzes statistical models and the geometry of their loss, while developmental research studies how measured learning geometry and behavior change over training ([Åström and Murray](https://people.duke.edu/~hpgavin/SystemID/References/Astrom-Feedback-2006.pdf), [Watanabe](https://www.jmlr.org/papers/volume14/watanabe13a/watanabe13a.pdf), [Hoogland et al.](https://arxiv.org/abs/2402.02364)). The constructive program below is an atlas synthesis, not a claimed theorem unifying these fields.

## The map in one page

| Question | Mathematical object | What it supports | What it does not establish |
|---|---|---|---|
| Can allowed inputs move the state where we want? | Reachability map and Gramian | State transfer and input-energy calculations for a specified dynamics model ([Åström and Murray](https://people.duke.edu/~hpgavin/SystemID/References/Astrom-Feedback-2006.pdf), [Scherpen](https://personal.math.vt.edu/jborggaa/nlromc/Scherpen_NLROMc_tutorial.pdf)) | Unconstrained linear reachability does not certify feasible transfer with bounded actuators or semantic compliance ([Ames et al.](http://ames.caltech.edu/ames2019control.pdf)). |
| Can measurements distinguish states? | Observability map and Gramian | Recoverability of state from specified input/output histories ([Åström and Murray](https://people.duke.edu/~hpgavin/SystemID/References/Astrom-Feedback-2006.pdf)) | Full rank alone does not guarantee accurate noisy estimation ([Rafieisakhaei, Chakravorty and Kumar](https://arxiv.org/pdf/1801.09877)). |
| Which state directions matter to input/output behavior? | Hankel operator and its singular values | Coordinate-invariant dynamical importance and bounded LTI model reduction ([Scherpen](https://personal.math.vt.edu/jborggaa/nlromc/Scherpen_NLROMc_tutorial.pdf)) | Not semantic importance, intelligence, or a general nonlinear reduction certificate ([Lall, Marsden and Glavaški](https://www.cmor-faculty.rice.edu/~cox/neuro/lall_2002a.pdf)). |
| How much nearby parameter volume has nearly the same loss? | Local learning coefficient, or LLC | A singularity-aware local statistical complexity quantity ([Lau et al.](https://raw.githubusercontent.com/mlresearch/v258/main/assets/lau25a/lau25a.pdf)) | Not a reachability test, state estimator, or control-energy measure; these are different objects in the cited definitions ([Lau et al.](https://raw.githubusercontent.com/mlresearch/v258/main/assets/lau25a/lau25a.pdf), [Scherpen](https://personal.math.vt.edu/jborggaa/nlromc/Scherpen_NLROMc_tutorial.pdf)). |
| When does learned behavior change? | Training trajectories and stagewise diagnostics | Empirical associations between changing loss geometry and behavior in tested models ([Hoogland et al.](https://arxiv.org/abs/2402.02364)) | An association does not demonstrate that changing the diagnostic causes the desired capability ([Hoogland et al.](https://arxiv.org/abs/2402.02364)). |

“Singular” has different meanings here. Hankel singular values arise from an operator's singular-value decomposition; a singular statistical model violates regularity conditions such as identifiability or nondegenerate Fisher information ([Scherpen](https://personal.math.vt.edu/jborggaa/nlromc/Scherpen_NLROMc_tutorial.pdf), [Lau et al.](https://raw.githubusercontent.com/mlresearch/v258/main/assets/lau25a/lau25a.pdf)). Sharing the word does not supply a mathematical equivalence.

## Start with the system boundary

Use separate notation for environment state \(x_t\), model state \(z_t\), parameters \(\theta_k\), and optimizer state \(v_k\). A physical-time index \(t\) and a training-step index \(k\) prevent accidental transfer of a stability statement from one process to another; this is the atlas's modeling convention, not a universal standard.

\[
x_{t+1}=f(x_t,u_t,d_t),\quad y_t=h(x_t)+\eta_t,
\qquad
z_{t+1}=F_{\theta_k}(z_t,y_t,u_t).
\]

\[
(\theta_{k+1},v_{k+1})=\mathcal U(\theta_k,v_k,a_k,\xi_k).
\]

Here \(u_t\) is an execution-time action, \(d_t,\eta_t\) are disturbances and measurement noise, and \(a_k\) is a proposed training intervention such as curriculum choice or data weighting. Treating the second equation as a control problem requires a measured response model, a target, allowed interventions, cost, and a validation horizon; the equation alone does not supply those.

A plant is the system to which a controller applies inputs, rather than necessarily a literal industrial machine; plant models may be derived from physics or identified from input/output data ([Åström and Murray](https://people.duke.edu/~hpgavin/SystemID/References/Astrom-Feedback-2006.pdf)). For a robot, include the robot and relevant environment; for an agent, our proposed boundary includes the execution harness, memory, tools, and permissions rather than only a neural network.

## Exact bridge: reachability, observability, and Hankel values

### The stable linear case

For a finite-dimensional, continuous-time, stable linear time-invariant system, use

\[
\dot x=Ax+Bu,\qquad y=Cx,
\]

where stable means \(A\) is Hurwitz, with every eigenvalue strictly in the left half-plane; the standard infinite-horizon Gramians then converge ([Scherpen](https://personal.math.vt.edu/jborggaa/nlromc/Scherpen_NLROMc_tutorial.pdf)).

\[
W_c=\int_0^\infty e^{At}BB^\top e^{A^\top t}\,dt,\qquad
W_o=\int_0^\infty e^{A^\top t}C^\top Ce^{At}\,dt.
\]

The corresponding Lyapunov equations are \(AW_c+W_cA^\top+BB^\top=0\) and \(A^\top W_o+W_oA+C^\top C=0\); positive definiteness characterizes reachability and observability in this setting ([Rowley](https://cwrowley.princeton.edu/papers/bt_ijbc3.pdf)). A Gramian is a matrix that collects squared responses over time, not a new kind of neural layer.

With input cost \(\int\|u(t)\|^2dt\), minimum energy to reach \(x\) from the asymptotic zero state is \(x^\top W_c^{-1}x\); with zero future input, output energy from \(x\) is \(x^\top W_ox\), provided the stated stable reachable setting holds ([Scherpen](https://personal.math.vt.edu/jborggaa/nlromc/Scherpen_NLROMc_tutorial.pdf)). The source uses a factor of one-half in its energy functions; this document omits it consistently in these two energy expressions.

The Hankel operator maps past inputs to future outputs across a time boundary. Factor it as \(\mathcal H=\mathcal O\mathcal R\), where \(\mathcal R\) maps past input to current state and \(\mathcal O\) maps current state to future output ([Scherpen](https://personal.math.vt.edu/jborggaa/nlromc/Scherpen_NLROMc_tutorial.pdf)).

\[
\sigma_i(\mathcal H)=\sqrt{\lambda_i(W_cW_o)}.
\]

These \(\sigma_i\) are the Hankel singular values, or HSVs; they are invariant under invertible state-coordinate changes, unlike the separate numerical eigenvalues of each Gramian under arbitrary coordinate scalings ([Rowley](https://cwrowley.princeton.edu/papers/bt_ijbc3.pdf)). Balanced coordinates make \(W_c=W_o=\operatorname{diag}(\sigma_i)\), exposing directions that jointly transmit input energy into measured output energy ([Scherpen](https://personal.math.vt.edu/jborggaa/nlromc/Scherpen_NLROMc_tutorial.pdf)).

### Reduction, not semantic deletion

For a stable minimal balanced realization, retain the first \(r\) states and truncate across a strict gap \(\sigma_r>\sigma_{r+1}\). The reduced system remains stable and obeys the conservative bound

\[
\|G-G_r\|_{H_\infty}\le 2\sum_{i=r+1}^{n}\sigma_i,
\]

where the norm measures worst-case induced input/output gain for the LTI error system, not an average task score ([Stanford model-reduction notes](https://lall.stanford.edu/engr210a/lectures/lecture11_2001_11_07_01.pdf)). The notes credit the error-bound result to Dale Enns and Keith Glover in 1984; balancing and its nonlinear extensions are reviewed by Jacquelien Scherpen ([Stanford notes](https://lall.stanford.edu/engr210a/lectures/lecture11_2001_11_07_01.pdf), [Scherpen](https://personal.math.vt.edu/jborggaa/nlromc/Scherpen_NLROMc_tutorial.pdf)).

Our design implication: declare outputs that include the task quantities and rare-event constraints you care about before computing importance. A mode that is small relative to one chosen output map is not thereby unimportant to another output or to an omitted constraint; input/output selection is part of the operator definition ([Scherpen](https://personal.math.vt.edu/jborggaa/nlromc/Scherpen_NLROMc_tutorial.pdf)).

## Conditional bridge: observability and Fisher information

Here is an atlas derivation under explicit assumptions. Take known deterministic dynamics, unknown initial state \(x_0\), known inputs already subtracted, and independent Gaussian measurement noise with fixed positive-definite covariance \(R_t\):

\[
y_t=C_t\Phi(t,0)x_0+\epsilon_t,\qquad \epsilon_t\sim\mathcal N(0,R_t).
\]

Writing \(J_t=C_t\Phi(t,0)\), differentiating the Gaussian log likelihood gives

\[
\nabla_{x_0}\log p(y\mid x_0)=\sum_t J_t^\top R_t^{-1}(y_t-J_tx_0),
\qquad
\mathcal I_{x_0}=\sum_t J_t^\top R_t^{-1}J_t.
\]

Thus initial-state Fisher information equals a noise-weighted finite-horizon observability Gramian; for \(R_t=\sigma^2I\), it is \(W_o/\sigma^2\), consistent with the relation examined by Rafieisakhaei, Chakravorty and Kumar ([paper](https://arxiv.org/pdf/1801.09877)). With process noise, correlated measurements, unknown dynamics, or informative priors, the simple unweighted equality is not the relevant full estimation calculation ([same paper](https://arxiv.org/pdf/1801.09877)).

For parameter identification instead, replace \(J_t\) with the sensitivity \(\partial y_t/\partial\theta\). This is a derivation of a Gaussian parameter-information matrix, not permission to identify parameter sensitivity with state observability; the unknowns have changed.

The further statistical bridge is limited: regular models have nonsingular Fisher information and learning coefficient \(d/2\), whereas singular models require loss geometry beyond this nondegenerate quadratic approximation ([Watanabe](https://www.jmlr.org/papers/volume14/watanabe13a/watanabe13a.pdf), [Lau et al.](https://raw.githubusercontent.com/mlresearch/v258/main/assets/lau25a/lau25a.pdf)). A rank-deficient Hessian or Fisher matrix alone does not specify all higher-order terms that determine an LLC ([Lau et al.](https://raw.githubusercontent.com/mlresearch/v258/main/assets/lau25a/lau25a.pdf)).

## Statistical bridge: SLT, LLC, and development

For a suitable local minimum \(\theta^*\), the local low-loss volume has the asymptotic form

\[
V(\epsilon)=\int_{B(\theta^*)}
\mathbf1\{L(\theta)-L(\theta^*)<\epsilon\}\pi(\theta)\,d\theta
\sim c\,\epsilon^\lambda(-\log\epsilon)^{m-1}.
\]

The LLC is the exponent \(\lambda\); \(m\) records the logarithmic multiplicity, under the analytic and local regularity conditions of singular learning theory ([Lau et al.](https://raw.githubusercontent.com/mlresearch/v258/main/assets/lau25a/lau25a.pdf)). These are asymptotics as the loss tolerance shrinks, not a matrix singular-value spectrum.

In the regular locally identifiable \(d\)-parameter case, the quadratic loss neighborhood is an ellipsoid whose volume scales as \(\epsilon^{d/2}\). In singular cases, higher-order geometry changes that exponent or its logarithmic factor; the corresponding Bayesian free-energy asymptotics replace the regular dimension penalty with a learning-coefficient penalty ([Watanabe](https://www.jmlr.org/papers/volume14/watanabe13a/watanabe13a.pdf)).

Lau, Furman, Wang, Murfet, and Wei develop a practical localized, tempered sampling estimator, but its behavior depends on localization and sampling settings; their equality between the localized quantity and theoretical LLC under proposed localization conditions is not a general proved estimator identity ([AISTATS 2025 paper](https://raw.githubusercontent.com/mlresearch/v258/main/assets/lau25a/lau25a.pdf)). Report sample size, loss, data distribution, temperature, localization strength, sampler settings, and uncertainty rather than only a scalar estimate.

Hoogland, Wang, Farrugia-Roberts, Carroll, Wei, and Murfet report associations between loss-landscape degeneracy changes and training stages in a language transformer and an in-context regression transformer ([paper, revised 2025](https://arxiv.org/abs/2402.02364)). This atlas includes their training-geometry and input/output-behavior findings, not circuit reconstruction or attention-head analysis.

The October 9 extension adds an intervention result: Wang and Murfet's Patterning uses estimated susceptibilities to choose training-data reweightings, including LLC-targeted solution selection in a parentheses task ([Patterning](https://arxiv.org/html/2601.13548v1)). This is evidence beyond passive developmental diagnostics, but not established compute-efficient capability control: its language-model susceptibility estimation costs approximately 500× model training by forward-pass accounting, and uniform training has the lowest loss ([paper](https://arxiv.org/html/2601.13548v1)). See the companion Structure-Informed Capability Learning investigation for the comparison with Goodfire, structured recurrent models, and our typed-control experiments.

Recent work also attempts to bridge SLT and optimizer dynamics: Hennick and De Baerdemacker's ICLR 2026 paper states results under local homogeneity and isotropic-diffusion assumptions and explicitly leaves aspects of adaptive optimizers unresolved ([Almost Bayesian](https://proceedings.iclr.cc/paper_files/paper/2026/file/a7b562dac391e9c7af691e8ef886ad10-Paper-Conference.pdf)). This is a relevant research path, not an established general controller for SGD or Adam, and its specialized theorems have not been independently reproduced here.

Hoogland's previously verified thesis title is The Ergodic Theory of Random Neural Networks, not an inferred title about statistical control; his biography connects a dynamical-systems background to the later developmental agenda ([author biography](https://www.jessehoogland.com/about)). The full thesis still was not located, so no thesis-specific derivations or results are attributed to it.

## Why HSV and LLC cannot be universally equal

The following are original elementary counterexamples derived from the definitions above, not published experimental findings. They show why a useful empirical association would still need a task-specific explanation.

### Change a dynamical gain without changing the learning exponent

Consider the stable scalar plant

\[
\dot x=-ax+bu,\qquad y=cx,\qquad a>0.
\]

Direct integration gives

\[
W_c=\frac{b^2}{2a},\quad W_o=\frac{c^2}{2a},\quad
\sigma_H=\frac{|bc|}{2a}.
\]

Assume \(a,b\ne0\) are known, \(c\) is the only unknown parameter, measured trajectories excite a nonzero signal, and the likelihood has fixed Gaussian noise. The loss around an identifiable true \(c^*\) is quadratic, so the theoretical LLC is \(1/2\), while the HSV can vary continuously with nonzero \(c^*\).

Consequently an HSV can be made small by changing output gain without changing this learning exponent. The comparison must declare units, sensor scaling, data excitation, and which parameters are learned.

### Keep the fitted transfer function but change the local statistical singularity

Fix the same \(a,b\), set the true output gain to zero, and first parameterize the unknown gain directly as \(c=w\). The local loss is proportional to \(w^2\), so \(V(\epsilon)\propto\epsilon^{1/2}\).

Now use \(c=w^2\) at \(w=0\), with a smooth positive parameter prior and the same Gaussian observation experiment. The excess loss is proportional to \(w^4\), so \(V(\epsilon)\propto\epsilon^{1/4}\), although the fitted plant has the same zero transfer function and zero Hankel values.

This second parameterization changes the local model family and is not a local diffeomorphism; it does not contradict LLC invariance under smooth invertible local reparameterizations proved by Lau et al. ([paper](https://raw.githubusercontent.com/mlresearch/v258/main/assets/lau25a/lau25a.pdf)). A fitted input/output operator alone cannot specify its surrounding statistical model and prior.

## Bridges into nonlinear and learned systems

| Route | Researchers and verified connection | Boundary |
|---|---|---|
| Physical systems to reduced models | Juang and Pappa's ERA identifies modal models and reduces systems from test data; Brunton, Dawson, and Rowley use input/output identification for aerodynamic control ([NASA record](https://ntrs.nasa.gov/citations/19850022899), [application paper](https://faculty.washington.edu/sbrunton/papers/BrDaRo2014.pdf)). | Finite sampled Hankel matrices, noise, excitation, and preprocessing matter; the aerodynamic example removes integrator contributions before applying ERA ([Brunton et al.](https://faculty.washington.edu/sbrunton/papers/BrDaRo2014.pdf)). |
| Large systems to balancing | Rowley's balanced POD combines direct and adjoint response snapshots; its small cross-snapshot SVD approximates the balancing calculation ([Rowley](https://cwrowley.princeton.edu/papers/bt_ijbc3.pdf)). | Ordinary variance-based POD is not automatically input/output balancing ([Rowley](https://cwrowley.princeton.edu/papers/bt_ijbc3.pdf)). |
| Linear balancing to nonlinear energy geometry | Scherpen replaces quadratic energy descriptions with nonlinear controllability and observability energy functions and local balanced coordinates ([tutorial](https://personal.math.vt.edu/jborggaa/nlromc/Scherpen_NLROMc_tutorial.pdf)). | Smoothness, stability, minimal linearization and local conditions matter; nonlinear singular-value functions are not a universal fixed HSV list ([Scherpen](https://personal.math.vt.edu/jborggaa/nlromc/Scherpen_NLROMc_tutorial.pdf)). |
| Simulation to nonlinear reduction | Lall, Marsden, and Glavaški estimate empirical Gramians and project nonlinear systems; the procedure recovers standard Gramians for stable linear systems ([paper](https://www.cmor-faculty.rice.edu/~cox/neuro/lall_2002a.pdf)). | In nonlinear settings, the result depends on sampled inputs, states, amplitudes, and coordinates; no automatic global LTI tail bound transfers ([same paper](https://www.cmor-faculty.rice.edu/~cox/neuro/lall_2002a.pdf)). |
| Learned dynamics to guaranteed properties | Revay, Wang, and Manchester parameterize recurrent equilibrium networks with contraction and incremental gain/passivity conditions ([REN paper](https://arxiv.org/pdf/2104.05942)). | These are selected dynamical properties, not universal guarantees of accurate prediction or intended semantics ([REN paper](https://arxiv.org/pdf/2104.05942)). |
| Action selection to learning | Klenske, Hennig, Schölkopf, and Zeilinger's approximate dual control explicitly values actions that improve knowledge of performance-relevant dynamics ([paper](https://ei.is.mpg.de/uploads_file/attachment/attachment/341/dual_control_final.pdf)). | Approximate beliefs and optimization, with soft state constraints in that study, do not give unconditional safe exploration ([same paper](https://ei.is.mpg.de/uploads_file/attachment/attachment/341/dual_control_final.pdf)). |

For an RNN, LSTM, selective SSM, neural ODE, or world model, the proposed workflow is to choose an operating region and evaluate response to allowed inputs before computing local or empirical Gramians. A linear SSM block may admit exact LTI analysis when its matrices are fixed, but input-dependent selection and nonlinear compositions require a different analysis; Mamba's selective construction is explicitly input-dependent ([Gu and Dao](https://arxiv.org/html/2312.00752v2)).

For LLMs, VLMs, and VLAs, our proposed object is a behavioral interaction process or learned action-conditioned model, not a claimed recovered internal circuit. A dialogue-control construction such as GenCtrl studies specified output attributes, while VLA safety filtering concerns an explicitly modeled physical constraint; neither transfers its result automatically to all user intentions ([GenCtrl](https://arxiv.org/html/2601.05637v1), [VLA safety study](https://arxiv.org/html/2512.11891v2)).

## Classical and modern control belong together

Frequency-domain tools analyze loop gain, delay, disturbance rejection, and robustness; state-space tools expose state, reachability, observability, and feedback design, with transfer functions connecting the representations in the LTI case ([Åström and Murray](https://people.duke.edu/~hpgavin/SystemID/References/Astrom-Feedback-2006.pdf)). “Classical” and “modern” are useful organizing labels, not a claim that one replaces the other.

- **LQR and LQG:** quadratic control and Gaussian state estimation provide a constructive baseline for specified linear systems; separation is a model-dependent design result, not a general principle for learned perception and arbitrary policies ([Åström and Murray](https://people.duke.edu/~hpgavin/SystemID/References/Astrom-Feedback-2006.pdf)).
- **Robust control:** sensitivity functions and small-gain reasoning bound effects of declared uncertainty; omitted uncertainty is not covered by the certificate ([Åström and Murray](https://people.duke.edu/~hpgavin/SystemID/References/Astrom-Feedback-2006.pdf)).
- **MPC:** receding-horizon optimization plans actions through a model and constraints; predictive safety filters require model confidence, feasibility, terminal safety, and fallback conditions ([Wabersich and Zeilinger](https://ar5iv.labs.arxiv.org/html/1812.05506)).
- **Lyapunov and barrier methods:** a Lyapunov function establishes a stability property, whereas a control barrier function supports forward invariance of a specified set under admissible control and regularity conditions ([Åström and Murray](https://people.duke.edu/~hpgavin/SystemID/References/Astrom-Feedback-2006.pdf), [Ames et al.](http://ames.caltech.edu/ames2019control.pdf)).
- **Adaptive and dual control:** learning a model while using it creates an exploration/performance trade-off; the action can affect both the system and the information available to its controller ([Klenske et al.](https://ei.is.mpg.de/uploads_file/attachment/attachment/341/dual_control_final.pdf)).

## A constructive, falsifiable research program

These are proposed experiments, not completed results. Start with unknown systems whose learned models must improve prediction, planning, and correction, rather than treating a known toy plant as sufficient evidence of learned intelligence.

| Proposal | Measurement | Strong baseline and decisive failure |
|---|---|---|
| Dual complexity tracking | Across training checkpoints, track task-conditioned Hankel spectrum, LLC estimates, prediction error, action success, correction retention, and constraint violations. | Match loss, parameter count, compute, data, and seeds. Reject useful added value if spectra and LLC fail to improve held-out outcome prediction over cheap diagnostics. |
| Intervene at predicted developmental stages | Pre-register intervention windows using earlier checkpoints only; vary curriculum or data weights with equal training budgets. | Compare uniform schedules, validation-loss triggers, gradient-based influence and random timing. Retrospective stage labels alone do not pass. |
| Balanced world-model compression | Compare retained dynamics and closed-loop utility after balanced, PCA, magnitude, and random reduction at equal memory and compute. | Test rare outputs and shifted action distributions. Reject the proposed benefit if only average prediction improves while planning or constraint performance worsens. |
| Information-aware exploration | Choose bounded actions using state/parameter information weighted by task relevance, with an independently defined feasible action set. | Compare random exploration, prediction-error heuristics, and certainty-equivalent MPC. Include exploration cost and unsafe attempts in the score. |
| Learn a correction interface | Train explicit state/action abstractions and test whether allowed corrections produce predicted multi-step behavioral changes. | Compare retraining, prompt/history changes, retrieval updates, and simple controllers. Pass requires correction retention without equivalent capability loss. |

Keep the Pareto comparison task-specific: prediction quality, action success, correction accuracy, risk, data, memory, latency, and training cost are separate axes. This extension adds no invented benchmark points or universal ranking.

## Graph reading contract and research scope

Every original atlas topic remains a node; this extension adds twenty control and learning-geometry topics. Edges record selected, cited relationships rather than an exhaustive literature graph: absent edges mean “not mapped here,” not “unrelated.”

The evidence labels are conditional exact identities, established constructions or connections, empirical associations, and untested hypotheses. Relationship descriptions additionally say whether a method uses, extends, estimates, constrains, or merely might predict another object; graph proximity and edge count are not evidence strength.

Discovery included academic and web indexes, university-hosted papers, proceedings, author pages, NASA records, and Google Scholar profile results. It was not an exhaustive Scholar crawl, no new model experiments were run, and no universal HSV–LLC equivalence was established by the sources reviewed; the counterexamples above also rule out a naive universal identity.

The thesis full-text gap remains open. Circuit-level mechanistic interpretability and LLM reverse engineering remain outside this atlas.
