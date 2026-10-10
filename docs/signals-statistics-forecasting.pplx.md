# Signals, Statistics, Inference and World-Model Forecasting

Signal processing, classical statistics, probabilistic inference and forecasting form a practical sequence from measurements to beliefs and decisions. They are not obsolete alternatives to deep learning: factor graphs expose shared algorithms, state-space models connect temporal inference with planning, and structured signal representations can be combined with statistical classifiers ([Kschischang, Frey and Loeliger](https://cba.mit.edu/events/03.11.ASE/docs/Loeliger.pdf), [Murphy](https://www.cs.ubc.ca/~murphyk/Thesis/thesis_double.pdf), [Bruna and Mallat](https://arxiv.org/abs/1203.1513)).

This extension adds explicit nodes for these connections and separates published findings from proposed alignment applications. Existing regression, SVM, k-means, tree, Gaussian-process, convolution, Fourier, wavelet, Wiener, Kalman, control, information-theory and deep-model nodes remain in the atlas.

## Reading routes through the map

| Route | What connects it | Boundary to preserve |
|---|---|---|
| Sampling → spectra → adaptive filters → system identification | Sampling controls distinguishability; adaptive filters update predictors from input/output errors ([Rowell](https://ocw.mit.edu/courses/2-161-signal-processing-continuous-and-discrete-fall-2008/80ec6fb9bd027f30adcdc16577e8ab16_samplingdft.pdf), [adaptive-filter tutorial](https://www2.spsc.tugraz.at/www-archive/downloads/adaptivefilter.pdf)). | A clean reference signal is not a complete specification of human intent. |
| Wavelets → scattering → SVM/PCA | Designed multiscale features can feed ordinary statistical classifiers ([Bruna and Mallat](https://arxiv.org/abs/1203.1513)). | Invariance is useful only when the removed variation is irrelevant to the task. |
| Regression → sparse recovery → reduced representations | Regularization and decomposition express different assumptions about coefficients and latent structure ([linear models](https://scikit-learn.org/stable/modules/linear_model.html), [decomposition](https://scikit-learn.org/stable/modules/decomposition.html), [Candès](https://candes.su.domains/publications/downloads/CompressiveSampling.pdf)). | Sparsity, variance, independence and control relevance are different criteria. |
| Graphical models → factor graphs → filtering/smoothing | Marginalization and temporal factorization connect HMMs, Gaussian filtering and message passing ([Kschischang et al.](https://cba.mit.edu/events/03.11.ASE/docs/Loeliger.pdf), [Murphy](https://www.cs.ubc.ca/~murphyk/Thesis/thesis_double.pdf)). | Exact tree inference does not imply exact loopy inference or causal validity. |
| DBN → POMDP → action-conditioned world model | A learned recurrent stochastic state model can combine filtering with latent planning ([PlaNet](https://proceedings.mlr.press/v97/hafner19a/hafner19a.pdf)). | A learned latent state need not be a physically identified or semantically editable state. |
| ARIMA/ETS → backtesting → proper scores | Temporal models need horizon-specific evaluation; distributions need more than point-error metrics ([ARIMA](https://otexts.com/fpp3/arima-r.html), [ETS](https://robjhyndman.com/nyc2018/1-5-StateSpace.pdf), [Gneiting and Raftery](https://sites.stat.washington.edu/raftery/Research/PDF/Gneiting2007jasa.pdf)). | Passive forecasting is not evidence about arbitrary interventions or policy changes. |
| Gibbs models → variational inference → response covariance | Partition functions, entropy and sufficient-statistic covariance supply precise statistical identities ([Wainwright and Jordan](https://people.eecs.berkeley.edu/~jordan/papers/wainwright-jordan-fnt.pdf)). | Algorithmic temperature, physical temperature and optimizer noise are not interchangeable. |
| Projection → memory closure → recurrent modeling | Eliminating variables can leave history dependence and unresolved forcing ([de Wit et al.](https://arxiv.org/html/2507.16058v1)). | Finite memory and omitted forcing require validation under the actual operating regime. |

## Signal processing and estimation

The details below list each new topic with its cited design handle, scope and an original proposed test. Author fields distinguish original papers from teaching or implementation references; a missing publication date is left explicit rather than inferred.

## Signal processing and designed representations

### Sampling, aliasing & anti-aliasing

Sampling can make distinct continuous signals indistinguishable. Band limits and anti-alias filters determine what a measurement stream can preserve. [Research or technical reference](https://ocw.mit.edu/courses/2-161-signal-processing-continuous-and-discrete-fall-2008/80ec6fb9bd027f30adcdc16577e8ab16_samplingdft.pdf).

- **Authors/reference:** D. Rowell / MIT teaching reference; 2008. [Research or technical reference](https://ocw.mit.edu/courses/2-161-signal-processing-continuous-and-discrete-fall-2008/80ec6fb9bd027f30adcdc16577e8ab16_samplingdft.pdf).
- **Usable contribution:** Choose acquisition rate, filtering and units before fitting the downstream model. [Research or technical reference](https://ocw.mit.edu/courses/2-161-signal-processing-continuous-and-discrete-fall-2008/80ec6fb9bd027f30adcdc16577e8ab16_samplingdft.pdf).
- **Scope:** Nyquist reconstruction assumes an appropriate band-limited signal. Better inference cannot uniquely recover arbitrary information lost through aliasing. [Research or technical reference](https://ocw.mit.edu/courses/2-161-signal-processing-continuous-and-discrete-fall-2008/80ec6fb9bd027f30adcdc16577e8ab16_samplingdft.pdf).
- **Proposed test:** Vary sample rate and sensor bandwidth; test whether action-relevant events remain distinguishable. This is an atlas proposal, not a reported experiment.

### Spectral estimation & Welch averaging

Estimate signal power across frequency from finite records, trading spectral resolution against estimator variance through windowing and averaging. [Research or technical reference](https://ocw.mit.edu/courses/2-161-signal-processing-continuous-and-discrete-fall-2008/c7d1951d44fe78ea97d0e14ea5158aee_lecture_23.pdf).

- **Authors/reference:** D. Rowell / Welch-method teaching reference; 2008. [Research or technical reference](https://ocw.mit.edu/courses/2-161-signal-processing-continuous-and-discrete-fall-2008/c7d1951d44fe78ea97d0e14ea5158aee_lecture_23.pdf).
- **Usable contribution:** Use autocorrelation and spectral estimates to measure timescales, residual structure and oscillation. [Research or technical reference](https://ocw.mit.edu/courses/2-161-signal-processing-continuous-and-discrete-fall-2008/c7d1951d44fe78ea97d0e14ea5158aee_lecture_23.pdf).
- **Scope:** Stationarity, finite-window bias and dependence between segments matter. A peak is not evidence of a causal mechanism or useful model state. [Research or technical reference](https://ocw.mit.edu/courses/2-161-signal-processing-continuous-and-discrete-fall-2008/c7d1951d44fe78ea97d0e14ea5158aee_lecture_23.pdf).
- **Proposed test:** Compare estimated spectra and residual correlations against simulated ground truth and changes in operating regime. This is an atlas proposal, not a reported experiment.

### Adaptive filtering: LMS & RLS

Update linear filter coefficients online from a reference error; RLS uses recursively updated weighted least-squares information. [Research or technical reference](https://www2.spsc.tugraz.at/www-archive/downloads/adaptivefilter.pdf).

- **Authors/reference:** Christian Feldbauer · Franz Pernkopf · Erhard Rank (tutorial); Tutorial; date not stated. [Research or technical reference](https://www2.spsc.tugraz.at/www-archive/downloads/adaptivefilter.pdf).
- **Usable contribution:** LMS connects sample-gradient learning with adaptive FIR filters. RLS supplies a cheap online identification baseline. [Research or technical reference](https://www2.spsc.tugraz.at/www-archive/downloads/adaptivefilter.pdf).
- **Scope:** The desired signal must be specified; learning-rate, excitation and tracking assumptions matter. Tracking a reference is not learning human intent. [Research or technical reference](https://www2.spsc.tugraz.at/www-archive/downloads/adaptivefilter.pdf).
- **Proposed test:** Compare LMS, normalized LMS, RLS and a neural adapter on drifting systems at equal latency and labeled-data cost. This is an atlas proposal, not a reported experiment.

### Wavelet scattering representations

Cascade wavelet convolutions, modulus and averaging to construct representations with specified invariance and deformation stability. [Research or technical reference](https://arxiv.org/abs/1203.1513).

- **Authors/reference:** Joan Bruna · Stéphane Mallat; 2012 preprint. [Research or technical reference](https://arxiv.org/abs/1203.1513).
- **Usable contribution:** Combine designed signal features with SVM or PCA classifiers; the paper reports both routes. [Research or technical reference](https://arxiv.org/abs/1203.1513).
- **Scope:** The chosen invariance must preserve task distinctions. Representation stability does not imply semantic correctness or control stability. [Research or technical reference](https://arxiv.org/abs/1203.1513).
- **Proposed test:** Compare scattering, learned CNN features and raw signals under physically relevant nuisance changes and task-changing perturbations. This is an atlas proposal, not a reported experiment.

### Compressed sensing & sparse recovery

Recover sparse or compressible signals from underdetermined measurements under conditions on the sensing operator. [Research or technical reference](https://candes.su.domains/publications/downloads/CompressiveSampling.pdf).

- **Authors/reference:** Emmanuel J. Candès; 2006. [Research or technical reference](https://candes.su.domains/publications/downloads/CompressiveSampling.pdf).
- **Usable contribution:** Use sparsity and restricted-isometry/incoherence conditions to obtain specified recovery guarantees. [Research or technical reference](https://candes.su.domains/publications/downloads/CompressiveSampling.pdf).
- **Scope:** A latent representation is not automatically sparse in the right basis; arbitrary measurement operators do not inherit recovery bounds. [Research or technical reference](https://candes.su.domains/publications/downloads/CompressiveSampling.pdf).
- **Proposed test:** Check sensing assumptions, noise and sparsity mismatch before comparing learned reconstruction to sparse recovery. This is an atlas proposal, not a reported experiment.

### ICA & blind source separation

Separate additive signal components using statistical independence assumptions rather than only uncorrelated directions. [Research or technical reference](https://scikit-learn.org/stable/modules/decomposition.html).

- **Authors/reference:** scikit-learn contributors / FastICA reference; Documentation; reviewed 2026. [Research or technical reference](https://scikit-learn.org/stable/modules/decomposition.html).
- **Usable contribution:** Model superimposed sensor sources and compare independent-component features with PCA features. [Research or technical reference](https://scikit-learn.org/stable/modules/decomposition.html).
- **Scope:** Independence assumptions and remaining representation ambiguities do not identify arbitrary physical causes or semantic factors. [Research or technical reference](https://scikit-learn.org/stable/modules/decomposition.html).
- **Proposed test:** Mix known sources, vary dependence and noise, and test whether recovered components support unseen interventions. This is an atlas proposal, not a reported experiment.

## State inference and assimilation

### Particle filtering

Approximate nonlinear filtering distributions with weighted samples rather than a single Gaussian approximation. [Research or technical reference](https://users.aalto.fi/~ssarkka/pub/bfs_book_2023_online.pdf).

- **Authors/reference:** Simo Särkkä · Lennart Svensson (reference); 2023. [Research or technical reference](https://users.aalto.fi/~ssarkka/pub/bfs_book_2023_online.pdf).
- **Usable contribution:** Track multimodal beliefs when linear-Gaussian filtering is inadequate. [Research or technical reference](https://users.aalto.fi/~ssarkka/pub/bfs_book_2023_online.pdf).
- **Scope:** Finite-particle degeneracy, model mismatch and high dimension can make uncertainty estimates misleading. [Research or technical reference](https://users.aalto.fi/~ssarkka/pub/bfs_book_2023_online.pdf).
- **Proposed test:** Compare Kalman variants and particle filters with known hidden state, multimodal ambiguity and equal computational budgets. This is an atlas proposal, not a reported experiment.

### Bayesian smoothing & retrospective state inference

Infer an earlier hidden state using observations that arrive later; filtering uses only observations available at the current time. [Research or technical reference](https://www.cs.ubc.ca/~murphyk/Thesis/thesis_double.pdf).

- **Authors/reference:** Kevin P. Murphy; Särkkä · Svensson; 2002 / 2023. [Research or technical reference](https://www.cs.ubc.ca/~murphyk/Thesis/thesis_double.pdf).
- **Usable contribution:** Use forward-backward or state-space smoothing for offline learning and state reconstruction. [Research or technical reference](https://www.cs.ubc.ca/~murphyk/Thesis/thesis_double.pdf).
- **Scope:** Using smoothed state as an online input leaks future information unless the deployment has the same delay. [Research or technical reference](https://www.cs.ubc.ca/~murphyk/Thesis/thesis_double.pdf).
- **Proposed test:** Separate causal filtering from offline smoothing and audit timestamps in every forecast/control comparison. This is an atlas proposal, not a reported experiment.

### Data assimilation & ensemble Kalman filters

Propagate an ensemble through a dynamical model, then assimilate observations using ensemble covariance estimates. [Research or technical reference](https://www.ecmwf.int/sites/default/files/elibrary/2003/9321-ensemble-kalman-filter-theoretical-formulation-and-practical-implementation.pdf).

- **Authors/reference:** Geir Evensen; 2003. [Research or technical reference](https://www.ecmwf.int/sites/default/files/elibrary/2003/9321-ensemble-kalman-filter-theoretical-formulation-and-practical-implementation.pdf).
- **Usable contribution:** Couple physics or learned forecasts with noisy measurements rather than running open-loop indefinitely. [Research or technical reference](https://www.ecmwf.int/sites/default/files/elibrary/2003/9321-ensemble-kalman-filter-theoretical-formulation-and-practical-implementation.pdf).
- **Scope:** Finite ensembles and Gaussian-style updates are approximations for nonlinear systems; ensemble spread is not automatically calibrated coverage. [Research or technical reference](https://www.ecmwf.int/sites/default/files/elibrary/2003/9321-ensemble-kalman-filter-theoretical-formulation-and-practical-implementation.pdf).
- **Proposed test:** Measure innovation consistency, forecast error, spread and closed-loop performance under sensor loss and model bias. This is an atlas proposal, not a reported experiment.

## Traditional statistical learning and evaluation

### Logistic regression & generalized linear models

Use a link function to connect linear features with conditional probabilities, counts or other exponential-family responses. [Research or technical reference](https://scikit-learn.org/stable/modules/linear_model.html).

- **Authors/reference:** scikit-learn developers (implementation reference); Documentation; reviewed 2026. [Research or technical reference](https://scikit-learn.org/stable/modules/linear_model.html).
- **Usable contribution:** Build cheap, inspectable task and behavioral-surrogate baselines before adding nonlinear complexity. [Research or technical reference](https://scikit-learn.org/stable/modules/linear_model.html).
- **Scope:** Coefficients depend on representation, regularization and data; a probability output is not automatically calibrated or causal. [Research or technical reference](https://scikit-learn.org/stable/modules/linear_model.html).
- **Proposed test:** Compare accuracy, log score, calibration and correction response with neural predictors on identical features. This is an atlas proposal, not a reported experiment.

### Lasso, elastic-net & sparse regularization

Penalize coefficient size with L1 or combined L1/L2 terms to encourage sparse or stabilized predictors. [Research or technical reference](https://scikit-learn.org/stable/modules/linear_model.html).

- **Authors/reference:** scikit-learn developers; Tibshirani lineage; Documentation; reviewed 2026. [Research or technical reference](https://scikit-learn.org/stable/modules/linear_model.html).
- **Usable contribution:** Connect sparse regression to compressed sensing and sparse dynamical-model identification. [Research or technical reference](https://scikit-learn.org/stable/modules/linear_model.html).
- **Scope:** Sparsity is coordinate dependent; selected features need not be true causes or semantically sufficient. [Research or technical reference](https://scikit-learn.org/stable/modules/linear_model.html).
- **Proposed test:** Compare sparse, ridge and unregularized models under correlated predictors and shifted interventions. This is an atlas proposal, not a reported experiment.

### PCA & probabilistic factor analysis

Represent covariance through low-dimensional components; probabilistic factor analysis specifies latent Gaussian factors and observation noise. [Research or technical reference](https://scikit-learn.org/stable/modules/decomposition.html).

- **Authors/reference:** scikit-learn contributors (reference); Documentation; reviewed 2026. [Research or technical reference](https://scikit-learn.org/stable/modules/decomposition.html).
- **Usable contribution:** Use low-rank statistical baselines for compression and observation models. [Research or technical reference](https://scikit-learn.org/stable/modules/decomposition.html).
- **Scope:** High variance is not the same as decision relevance, observability or controllability. Factor coordinates can remain ambiguous. [Research or technical reference](https://scikit-learn.org/stable/modules/decomposition.html).
- **Proposed test:** Compare PCA against task-aware and balanced reduction on rare constraints and action-conditioned prediction. This is an atlas proposal, not a reported experiment.

### Gaussian mixtures & soft clustering

Represent a distribution as weighted Gaussian components with latent assignments, means and covariance structure. [Research or technical reference](https://scikit-learn.org/stable/modules/mixture.html).

- **Authors/reference:** scikit-learn contributors (reference); Documentation; reviewed 2026. [Research or technical reference](https://scikit-learn.org/stable/modules/mixture.html).
- **Usable contribution:** Generalize hard clustering with soft assignments and explicit distributional components. [Research or technical reference](https://scikit-learn.org/stable/modules/mixture.html).
- **Scope:** Components need not be physical regimes; covariance collapse and local optima require attention. [Research or technical reference](https://scikit-learn.org/stable/modules/mixture.html).
- **Proposed test:** Compare k-means, GMM and supervised regimes for forecasting and correction under overlapping modes. This is an atlas proposal, not a reported experiment.

### Expectation maximization & latent-variable fitting

Alternate inference about latent assignments or sufficient statistics with parameter fitting; GMMs and HMMs provide concrete examples. [Research or technical reference](https://scikit-learn.org/stable/modules/mixture.html).

- **Authors/reference:** Wainwright · Jordan; scikit-learn contributors; 2008 / implementation reference. [Research or technical reference](https://scikit-learn.org/stable/modules/mixture.html).
- **Usable contribution:** Separate state/assignment inference from parameter estimation in a structured learning loop. [Research or technical reference](https://scikit-learn.org/stable/modules/mixture.html).
- **Scope:** The tested procedure is not guaranteed to find the global optimum; unbounded likelihoods can occur in unregularized mixtures. [Research or technical reference](https://scikit-learn.org/stable/modules/mixture.html).
- **Proposed test:** Compare restarts, held-out likelihood, recovered regimes and downstream action utility, not training likelihood alone. This is an atlas proposal, not a reported experiment.

### Bootstrap & sampling uncertainty

Resampling estimates variability of a statistic and can construct approximate confidence intervals. [Research or technical reference](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html).

- **Authors/reference:** Bradley Efron · Robert Tibshirani (1993 reference); SciPy community; 1993 / implementation reference. [Research or technical reference](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html).
- **Usable contribution:** Report uncertainty on performance differences instead of ranking methods by a single run. [Research or technical reference](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html).
- **Scope:** Resampling must respect the data-generating structure; ordinary row resampling is not automatically valid for dependent trajectories. [Research or technical reference](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html).
- **Proposed test:** Resample independent experimental units or trajectories and check coverage in controlled simulations. This is an atlas proposal, not a reported experiment.

### Conformal prediction & coverage

Calibrate prediction sets with finite-sample marginal coverage under exchangeability and the specified construction. [Research or technical reference](https://arxiv.org/pdf/2411.11824v4).

- **Authors/reference:** Anastasios Angelopoulos · Rina Foygel Barber · Stephen Bates; 2025 manuscript. [Research or technical reference](https://arxiv.org/pdf/2411.11824v4).
- **Usable contribution:** Wrap a predictor with a measured uncertainty set while separating set size from coverage. [Research or technical reference](https://arxiv.org/pdf/2411.11824v4).
- **Scope:** Marginal coverage is not per-input correctness, adaptive-trajectory safety or protection against arbitrary distribution shift. [Research or technical reference](https://arxiv.org/pdf/2411.11824v4).
- **Proposed test:** Test coverage by horizon, subgroup and shifted policy; report failures of exchangeability rather than assume the theorem still applies. This is an atlas proposal, not a reported experiment.

### Proper scoring rules & calibration

Score full predictive distributions with rules that reward truthful distributions in expectation under stated conditions. [Research or technical reference](https://sites.stat.washington.edu/raftery/Research/PDF/Gneiting2007jasa.pdf).

- **Authors/reference:** Tilmann Gneiting · Adrian Raftery; 2007. [Research or technical reference](https://sites.stat.washington.edu/raftery/Research/PDF/Gneiting2007jasa.pdf).
- **Usable contribution:** Use log score, CRPS or energy scores rather than point error alone to compare probabilistic forecasts. [Research or technical reference](https://sites.stat.washington.edu/raftery/Research/PDF/Gneiting2007jasa.pdf).
- **Scope:** Proper scoring is an evaluation property, not evidence that a trained model is calibrated, causal or aligned. [Research or technical reference](https://sites.stat.washington.edu/raftery/Research/PDF/Gneiting2007jasa.pdf).
- **Proposed test:** Compare calibration and sharpness by horizon and regime, then separately measure decisions made from those forecasts. This is an atlas proposal, not a reported experiment.

### Online uncertainty under distribution shift

Adapt conformal thresholds online; the cited work establishes local-interval regret results under its stated setup. [Research or technical reference](https://www.jmlr.org/papers/v25/22-1218.html).

- **Authors/reference:** Isaac Gibbs · Emmanuel Candès; 2024. [Research or technical reference](https://www.jmlr.org/papers/v25/22-1218.html).
- **Usable contribution:** Track changing forecast errors rather than assume a fixed calibration distribution. [Research or technical reference](https://www.jmlr.org/papers/v25/22-1218.html).
- **Scope:** Online regret or long-run error control does not guarantee every next action is safe; transient and rare-event failures remain relevant. [Research or technical reference](https://www.jmlr.org/papers/v25/22-1218.html).
- **Proposed test:** Introduce abrupt regime and policy changes; measure transient misses, set sizes and decision consequences. This is an atlas proposal, not a reported experiment.

## Probabilistic graphical models and beliefs

### Probabilistic graphical models

Represent a probability distribution through graph-structured factorization and conditional independence. [Research or technical reference](https://people.eecs.berkeley.edu/~jordan/papers/wainwright-jordan-fnt.pdf).

- **Authors/reference:** Martin Wainwright · Michael Jordan; 2008 reference. [Research or technical reference](https://people.eecs.berkeley.edu/~jordan/papers/wainwright-jordan-fnt.pdf).
- **Usable contribution:** Make uncertainty dependencies and inference assumptions explicit across perception, state, action and objective variables. [Research or technical reference](https://people.eecs.berkeley.edu/~jordan/papers/wainwright-jordan-fnt.pdf).
- **Scope:** A probabilistic graph does not acquire causal intervention semantics merely by drawing arrows. [Research or technical reference](https://people.eecs.berkeley.edu/~jordan/papers/wainwright-jordan-fnt.pdf).
- **Proposed test:** Test conditional-independence assumptions and intervention predictions separately from observational likelihood. This is an atlas proposal, not a reported experiment.

### Bayesian networks

A directed acyclic graph factorizes a joint distribution into local conditional distributions. [Research or technical reference](https://people.eecs.berkeley.edu/~jordan/papers/wainwright-jordan-fnt.pdf).

- **Authors/reference:** Wainwright · Jordan (reference); 2008. [Research or technical reference](https://people.eecs.berkeley.edu/~jordan/papers/wainwright-jordan-fnt.pdf).
- **Usable contribution:** Build explicit observation, state and objective dependencies with modular conditional models. [Research or technical reference](https://people.eecs.berkeley.edu/~jordan/papers/wainwright-jordan-fnt.pdf).
- **Scope:** Observationally equivalent structures can differ causally; causal use requires additional assumptions or interventions. [Research or technical reference](https://people.eecs.berkeley.edu/~jordan/papers/wainwright-jordan-fnt.pdf).
- **Proposed test:** Compare candidate factorizations using held-out interventions, not only fits to passive data. This is an atlas proposal, not a reported experiment.

### Markov random fields & Gibbs potentials

Undirected graphs represent distributions through local compatibility factors, including Ising-style pairwise interactions. [Research or technical reference](https://people.eecs.berkeley.edu/~jordan/papers/wainwright-jordan-fnt.pdf).

- **Authors/reference:** Wainwright · Jordan (reference); 2008. [Research or technical reference](https://people.eecs.berkeley.edu/~jordan/papers/wainwright-jordan-fnt.pdf).
- **Usable contribution:** Connect image models, spatial dependencies, energy-based learning and statistical mechanics. [Research or technical reference](https://people.eecs.berkeley.edu/~jordan/papers/wainwright-jordan-fnt.pdf).
- **Scope:** Normalizing constants and inference can be intractable; an energy function is not a physical conservation law. [Research or technical reference](https://people.eecs.berkeley.edu/~jordan/papers/wainwright-jordan-fnt.pdf).
- **Proposed test:** Compare approximate marginals against exact small-system calculations and measure downstream decisions. This is an atlas proposal, not a reported experiment.

### Factor graphs

A bipartite graph separates variables from the factors of a global function or probability distribution. [Research or technical reference](https://cba.mit.edu/events/03.11.ASE/docs/Loeliger.pdf).

- **Authors/reference:** Frank Kschischang · Brendan Frey · Hans-Andrea Loeliger; 2001. [Research or technical reference](https://cba.mit.edu/events/03.11.ASE/docs/Loeliger.pdf).
- **Usable contribution:** Express filtering, decoding and selected transform algorithms through a common factorization language. [Research or technical reference](https://cba.mit.edu/events/03.11.ASE/docs/Loeliger.pdf).
- **Scope:** The factorization specifies a mathematical model, not the correctness of its semantics or assumptions. [Research or technical reference](https://cba.mit.edu/events/03.11.ASE/docs/Loeliger.pdf).
- **Proposed test:** Verify messages against direct marginalization on a small model before using approximate large-scale inference. This is an atlas proposal, not a reported experiment.

### Belief propagation & sum-product

Pass local messages to compute marginals; finite cycle-free factor graphs admit exact sum-product inference. [Research or technical reference](https://cba.mit.edu/events/03.11.ASE/docs/Loeliger.pdf).

- **Authors/reference:** Kschischang · Frey · Loeliger; Wainwright · Jordan; 2001 / 2008. [Research or technical reference](https://cba.mit.edu/events/03.11.ASE/docs/Loeliger.pdf).
- **Usable contribution:** Unify forward-backward, Gaussian filtering and communication-code inference at the algorithmic level. [Research or technical reference](https://cba.mit.edu/events/03.11.ASE/docs/Loeliger.pdf).
- **Scope:** Loopy belief propagation need not converge or return exact marginals; message passing in a GNN is not automatically Bayesian inference. [Research or technical reference](https://cba.mit.edu/events/03.11.ASE/docs/Loeliger.pdf).
- **Proposed test:** Measure approximation error and convergence on the actual graph topology and compare with exact or sampling baselines. This is an atlas proposal, not a reported experiment.

### Dynamic Bayesian networks

Factor a sequential distribution across time and within each time slice, with hidden states and observations. [Research or technical reference](https://www.cs.ubc.ca/~murphyk/Thesis/thesis_double.pdf).

- **Authors/reference:** Kevin Patrick Murphy; 2002. [Research or technical reference](https://www.cs.ubc.ca/~murphyk/Thesis/thesis_double.pdf).
- **Usable contribution:** Express HMMs, linear-Gaussian state models and richer switching/factored dynamics within one representation. [Research or technical reference](https://www.cs.ubc.ca/~murphyk/Thesis/thesis_double.pdf).
- **Scope:** Compact representation does not ensure efficient exact inference, identifiable latents or causal action models. [Research or technical reference](https://www.cs.ubc.ca/~murphyk/Thesis/thesis_double.pdf).
- **Proposed test:** Compare inferred beliefs and multi-step predictions under missing sensors, switched regimes and unseen actions. This is an atlas proposal, not a reported experiment.

### Hidden Markov models & switching regimes

Discrete latent states evolve by a Markov transition model and generate observations; forward-backward supports inference. [Research or technical reference](https://www.cs.ubc.ca/~murphyk/Thesis/thesis_double.pdf).

- **Authors/reference:** Kevin Patrick Murphy (reference); 2002. [Research or technical reference](https://www.cs.ubc.ca/~murphyk/Thesis/thesis_double.pdf).
- **Usable contribution:** Provide an explicit regime and memory baseline against generic recurrent latent models. [Research or technical reference](https://www.cs.ubc.ca/~murphyk/Thesis/thesis_double.pdf).
- **Scope:** A fitted discrete regime need not correspond to a real cause; its state count and Markov assumptions need validation. [Research or technical reference](https://www.cs.ubc.ca/~murphyk/Thesis/thesis_double.pdf).
- **Proposed test:** Test regime recovery, forecast likelihood and action response against ARIMA and neural state models. This is an atlas proposal, not a reported experiment.

### Variational inference & statistical free energy

Optimize over a tractable distribution family to approximate inference, with entropy and log-partition duality. [Research or technical reference](https://people.eecs.berkeley.edu/~jordan/papers/wainwright-jordan-fnt.pdf).

- **Authors/reference:** Martin Wainwright · Michael Jordan; 2008. [Research or technical reference](https://people.eecs.berkeley.edu/~jordan/papers/wainwright-jordan-fnt.pdf).
- **Usable contribution:** Link mean-field approximations, probabilistic graphical models and latent-model objectives. [Research or technical reference](https://people.eecs.berkeley.edu/~jordan/papers/wainwright-jordan-fnt.pdf).
- **Scope:** An ELBO is not physical work or an alignment objective; approximation error depends on the variational family. [Research or technical reference](https://people.eecs.berkeley.edu/~jordan/papers/wainwright-jordan-fnt.pdf).
- **Proposed test:** Compare posterior moments and downstream decisions with exact or better-sampled reference distributions. This is an atlas proposal, not a reported experiment.

### POMDPs & belief-state planning

Plan with partial observations using a belief over hidden state rather than treating the latest observation as complete state. [Research or technical reference](https://proceedings.mlr.press/v97/hafner19a/hafner19a.pdf).

- **Authors/reference:** Hafner et al.; Hadfield-Menell · Dragan · Abbeel · Russell; 2016 / 2019 references. [Research or technical reference](https://proceedings.mlr.press/v97/hafner19a/hafner19a.pdf).
- **Usable contribution:** Combine an observation model, action-conditioned dynamics and reward model with sequential inference and planning. [Research or technical reference](https://proceedings.mlr.press/v97/hafner19a/hafner19a.pdf).
- **Scope:** The model, reward and belief update can all be wrong; belief-state sufficiency is conditional on the specified model. [Research or technical reference](https://proceedings.mlr.press/v97/hafner19a/hafner19a.pdf).
- **Proposed test:** Require competent long-horizon action selection under ambiguity, information-gathering actions and objective corrections. This is an atlas proposal, not a reported experiment.

### Predictive state representations

Represent state by probabilities of future observation tests conditional on specified action sequences. [Research or technical reference](https://web.eecs.umich.edu/~baveja/Papers/psr.pdf).

- **Authors/reference:** Michael Littman · Richard Sutton · Satinder Singh; Foundational paper; year not in fetched text. [Research or technical reference](https://web.eecs.umich.edu/~baveja/Papers/psr.pdf).
- **Usable contribution:** A finite POMDP admits a linear PSR with no more core tests than its minimal POMDP state count. [Research or technical reference](https://web.eecs.umich.edu/~baveja/Papers/psr.pdf).
- **Scope:** The representation theorem does not show easy learning, correct causal identification or human-readable semantics. [Research or technical reference](https://web.eecs.umich.edu/~baveja/Papers/psr.pdf).
- **Proposed test:** Learn predictive states and test untrained action sequences, rare events and correction retention. This is an atlas proposal, not a reported experiment.

## Forecasting and learned physical models

### ARIMA & autoregressive forecasting

Use autoregression, differencing and moving-average innovations to model time-series dependence. [Research or technical reference](https://otexts.com/fpp3/arima-r.html).

- **Authors/reference:** Rob Hyndman · George Athanasopoulos; Hyndman · Khandakar; Textbook / 2008 algorithm. [Research or technical reference](https://otexts.com/fpp3/arima-r.html).
- **Usable contribution:** Provide transparent temporal baselines with residual diagnostics, stationarity and invertibility checks. [Research or technical reference](https://otexts.com/fpp3/arima-r.html).
- **Scope:** Passive forecasts do not identify effects of a new policy; order selection and regime changes matter. [Research or technical reference](https://otexts.com/fpp3/arima-r.html).
- **Proposed test:** Compare multi-horizon forecasts against naive, ETS and neural baselines using time-ordered evaluation. This is an atlas proposal, not a reported experiment.

### ETS & innovations state-space forecasting

Represent level, trend and seasonality with stochastic state and observation equations sharing innovations. [Research or technical reference](https://robjhyndman.com/nyc2018/1-5-StateSpace.pdf).

- **Authors/reference:** Rob J. Hyndman (author teaching material); Slides; year not stated. [Research or technical reference](https://robjhyndman.com/nyc2018/1-5-StateSpace.pdf).
- **Usable contribution:** Connect exponential smoothing to model-based forecast distributions rather than point updates alone. [Research or technical reference](https://robjhyndman.com/nyc2018/1-5-StateSpace.pdf).
- **Scope:** Equal point forecasts can have different uncertainty; level and trend are not automatically physical latent variables. [Research or technical reference](https://robjhyndman.com/nyc2018/1-5-StateSpace.pdf).
- **Proposed test:** Compare point error and interval quality at several horizons, including abrupt changes in seasonality. This is an atlas proposal, not a reported experiment.

### Forecast backtesting & horizon evaluation

Evaluate forecasts on successive future windows using only the information available at each forecast origin. [Research or technical reference](https://robjhyndman.com/hyndsight/tscv-fable/).

- **Authors/reference:** Rob J. Hyndman; 2021. [Research or technical reference](https://robjhyndman.com/hyndsight/tscv-fable/).
- **Usable contribution:** Keep chronological train/test boundaries and compare error separately for each horizon. [Research or technical reference](https://robjhyndman.com/hyndsight/tscv-fable/).
- **Scope:** Backtesting under one data-collection policy does not establish validity under a new intervention policy. [Research or technical reference](https://robjhyndman.com/hyndsight/tscv-fable/).
- **Proposed test:** Audit leakage, observation delays, repeated tuning and future-covariate availability. This is an atlas proposal, not a reported experiment.

### Physical forecasting & learned simulators

Learn multi-step physical forecasts from historical/reanalysis states; GraphCast supplies a concrete weather example. [Research or technical reference](https://www.science.org/doi/10.1126/science.adi2336).

- **Authors/reference:** Remi Lam and collaborators / GraphCast; 2023. [Research or technical reference](https://www.science.org/doi/10.1126/science.adi2336).
- **Usable contribution:** Separate multi-step forecast accuracy from the cost and fidelity of a physical simulator. [Research or technical reference](https://www.science.org/doi/10.1126/science.adi2336).
- **Scope:** A strong passive forecast is not a validated controlled world model or a causal simulator for arbitrary interventions. [Research or technical reference](https://www.science.org/doi/10.1126/science.adi2336).
- **Proposed test:** Evaluate forecast horizons, rare events, sensor shifts and any claimed action effects as separate tasks. This is an atlas proposal, not a reported experiment.

### Horizon-aware world-model uncertainty

Pretrain a probabilistic ensemble over varying forecast horizons, then learn an action-conditioned stream for downstream control. [Research or technical reference](https://proceedings.iclr.cc/paper_files/paper/2026/file/ad222e6e38e26e3f00d85829b9e02f43-Paper-Conference.pdf).

- **Authors/reference:** Shenghua Wan · Le Gan · De-Chuan Zhan; ICLR 2026. [Research or technical reference](https://proceedings.iclr.cc/paper_files/paper/2026/file/ad222e6e38e26e3f00d85829b9e02f43-Paper-Conference.pdf).
- **Usable contribution:** The HAUWM paper tests a horizon-dependent disagreement objective and downstream control performance. [Research or technical reference](https://proceedings.iclr.cc/paper_files/paper/2026/file/ad222e6e38e26e3f00d85829b9e02f43-Paper-Conference.pdf).
- **Scope:** Ensemble disagreement is not a calibrated-coverage theorem. Action-free video does not itself identify action effects. [Research or technical reference](https://proceedings.iclr.cc/paper_files/paper/2026/file/ad222e6e38e26e3f00d85829b9e02f43-Paper-Conference.pdf).
- **Proposed test:** Compare uncertainty with empirical errors by horizon and action regime, then test whether it improves decisions. This is an atlas proposal, not a reported experiment.

## Information, statistical physics and model reduction

### Maximum entropy & Gibbs distributions

Moment constraints and entropy optimization connect exponential-family distributions, partition functions and statistical mechanics. [Research or technical reference](https://people.eecs.berkeley.edu/~jordan/papers/wainwright-jordan-fnt.pdf).

- **Authors/reference:** Wainwright · Jordan (exponential-family reference); 2008. [Research or technical reference](https://people.eecs.berkeley.edu/~jordan/papers/wainwright-jordan-fnt.pdf).
- **Usable contribution:** Specify which statistics a model preserves instead of treating maximum entropy as assumption-free ignorance. [Research or technical reference](https://people.eecs.berkeley.edu/~jordan/papers/wainwright-jordan-fnt.pdf).
- **Scope:** The chosen constraints encode modeling decisions. A mathematical inverse-temperature parameter need not be physical temperature. [Research or technical reference](https://people.eecs.berkeley.edu/~jordan/papers/wainwright-jordan-fnt.pdf).
- **Proposed test:** Change preserved statistics and test sensitivity of predictions, uncertainty and task decisions. This is an atlas proposal, not a reported experiment.

### Statistical susceptibility & response covariance

Derivatives of an exponential-family log partition function yield expected sufficient statistics and their covariance. [Research or technical reference](https://people.eecs.berkeley.edu/~jordan/papers/wainwright-jordan-fnt.pdf).

- **Authors/reference:** Wainwright · Jordan (partition-function identities); 2008. [Research or technical reference](https://people.eecs.berkeley.edu/~jordan/papers/wainwright-jordan-fnt.pdf).
- **Usable contribution:** For fixed support and a regular normalizable family, response derivatives provide exact local sensitivity identities. [Research or technical reference](https://people.eecs.berkeley.edu/~jordan/papers/wainwright-jordan-fnt.pdf).
- **Scope:** A Gibbs/posterior response is not the same object as a finite optimizer trajectory or arbitrary causal intervention. [Research or technical reference](https://people.eecs.berkeley.edu/~jordan/papers/wainwright-jordan-fnt.pdf).
- **Proposed test:** Compare predicted local response with actual finite perturbations and separately test retrained-model response. This is an atlas proposal, not a reported experiment.

### Langevin dynamics & posterior sampling

Combine gradients with controlled Gaussian noise to approach posterior sampling under appropriate limiting conditions. [Research or technical reference](https://www.stats.ox.ac.uk/~teh/research/compstats/WelTeh2011a.pdf).

- **Authors/reference:** Max Welling · Yee Whye Teh; 2011. [Research or technical reference](https://www.stats.ox.ac.uk/~teh/research/compstats/WelTeh2011a.pdf).
- **Usable contribution:** Connect stochastic dynamics and Bayesian learning; the noise schedule and step size are part of the method. [Research or technical reference](https://www.stats.ox.ac.uk/~teh/research/compstats/WelTeh2011a.pdf).
- **Scope:** Finite-step bias, mixing and gradient noise matter. Ordinary SGD or Adam is not automatically an equilibrium sampler. [Research or technical reference](https://www.stats.ox.ac.uk/~teh/research/compstats/WelTeh2011a.pdf).
- **Proposed test:** Check effective sample size and between-chain agreement before using posterior response estimates to choose interventions. This is an atlas proposal, not a reported experiment.

### Mori–Zwanzig memory & closure

Projecting dynamics onto resolved variables leaves instantaneous dynamics, history-dependent memory and unresolved forcing. [Research or technical reference](https://arxiv.org/html/2507.16058v1).

- **Authors/reference:** Xander de Wit · Alessandro Gabbana · Michael Woodward · Yen Ting Lin · Federico Toschi · Daniel Livescu; 2025 preprint. [Research or technical reference](https://arxiv.org/html/2507.16058v1).
- **Usable contribution:** The cited turbulence model learns finite-memory operators and tests long-time statistics. [Research or technical reference](https://arxiv.org/html/2507.16058v1).
- **Scope:** Truncating memory or discarding unresolved forcing is an approximation; passive turbulence results do not establish controlled semantic dynamics. [Research or technical reference](https://arxiv.org/html/2507.16058v1).
- **Proposed test:** Compare Markov, delay-state and learned-memory models on multi-step prediction, regime shifts and controlled interventions. This is an atlas proposal, not a reported experiment.

### Computational mechanics & predictive states

Group histories that induce the same distribution of futures; predictive states minimize stored statistical complexity under stated conditions. [Research or technical reference](https://arxiv.org/pdf/cs/0001027v1).

- **Authors/reference:** Cosma Rohilla Shalizi · James P. Crutchfield; 2000. [Research or technical reference](https://arxiv.org/pdf/cs/0001027v1).
- **Usable contribution:** Connect state construction, past-future information and sufficient prediction for stationary symbolic processes. [Research or technical reference](https://arxiv.org/pdf/cs/0001027v1).
- **Scope:** The term causal state here means predictive equivalence, not automatically an intervention variable or physical state. [Research or technical reference](https://arxiv.org/pdf/cs/0001027v1).
- **Proposed test:** Test approximate predictive-state sufficiency and then separately test action-conditioned and nonstationary generalization. This is an atlas proposal, not a reported experiment.

### Thermodynamics of prediction

Relate nonpredictive memory to physical dissipation in a specified driven system coupled to a heat bath. [Research or technical reference](https://arxiv.org/pdf/1203.3271).

- **Authors/reference:** Susanne Still · David Sivak · Anthony Bell · Gavin Crooks; 2012. [Research or technical reference](https://arxiv.org/pdf/1203.3271).
- **Usable contribution:** The work provides a real information/physics bridge with physical inverse temperature and a stated protocol. [Research or technical reference](https://arxiv.org/pdf/1203.3271).
- **Scope:** The result assumes no feedback from the system to its driving signal. It is not an energy law for arbitrary neural training or a moral objective. [Research or technical reference](https://arxiv.org/pdf/1203.3271).
- **Proposed test:** Verify the physical setup before testing an energy relation; otherwise treat neural-system transfer as an explicit hypothesis. This is an atlas proposal, not a reported experiment.

## Objective learning and alignment boundaries

### Assistance games & uncertain human objectives

Model cooperation with a human whose reward parameter is initially unknown to the machine; communication and teaching can matter. [Research or technical reference](https://arxiv.org/pdf/1606.03137).

- **Authors/reference:** Dylan Hadfield-Menell · Anca Dragan · Pieter Abbeel · Stuart Russell; 2016. [Research or technical reference](https://arxiv.org/pdf/1606.03137).
- **Usable contribution:** CIRL connects objective uncertainty with a coordination POMDP and active information gathering. [Research or technical reference](https://arxiv.org/pdf/1606.03137).
- **Scope:** A shared, representable reward and a specified human model are assumptions, not proof that real intentions are recovered. [Research or technical reference](https://arxiv.org/pdf/1606.03137).
- **Proposed test:** Test clarification, corrections and deference under ambiguous instructions, mistaken feedback and conflicting stakeholders. This is an atlas proposal, not a reported experiment.

### Proxy objectives & reward misspecification

Analyze when increasing proxy return can reverse the ordering induced by a true reward within a specified policy set. [Research or technical reference](https://arxiv.org/pdf/2209.13085).

- **Authors/reference:** Joar Skalse · Nikolaus Howe · Dmitrii Krasheninnikov · David Krueger; 2022 / revised 2025. [Research or technical reference](https://arxiv.org/pdf/2209.13085).
- **Usable contribution:** Check the policy set and evaluate true task outcomes rather than treating a learned reward as a complete specification. [Research or technical reference](https://arxiv.org/pdf/2209.13085).
- **Scope:** The finite-MDP ordering theorem does not prove hacking occurs on every actual training trajectory. [Research or technical reference](https://arxiv.org/pdf/2209.13085).
- **Proposed test:** Optimize against the proxy deliberately and test for rising proxy score with falling independently measured user utility. This is an atlas proposal, not a reported experiment.

## How to compare the methods fairly

The comparison program is a proposed evaluation contract, not a new benchmark result. No model was trained while adding this extension.

- **Same information:** match sensor history, timestamps, future-covariate availability and action support. Do not compare an offline smoother with an online filter as if they see the same data.
- **Same target:** separate signal reconstruction, state inference, passive forecast, intervention forecast and task success. Each can improve while another fails.
- **Same cost boundary:** record training, tuning, measurement, memory, inference latency and failure handling. A cheap prediction can require expensive preprocessing or calibration.
- **Useful uncertainty:** report proper scores, coverage where justified, set width, rare-event misses and the decisions made from the forecasts.
- **Correction and retention:** test whether user-approved changes affect the intended behavior for the declared horizon without damaging unrelated capabilities.
- **Honest topology:** relationships are selected, typed records. Missing graph edges mean unreviewed or unmapped connections, not scientific independence.

## Retrieval depth and unresolved evidence

This extension uses primary papers, author-hosted technical material, university course notes and maintained implementation documentation. Some large textbooks or PDFs exposed only partial text through retrieval; detailed claims were instead tied to accessible specialist papers or documentation, and the attempted Stanford ESL PDF URL returned an access error.

The GraphCast record was read for its published forecasting result, not as evidence of an action-conditioned controller. The horizon-aware world-model paper was read as an empirical method, not a guarantee of calibrated coverage; the broader alignment synthesis remains an original proposal ([GraphCast](https://www.science.org/doi/10.1126/science.adi2336), [Wan, Gan and Zhan](https://proceedings.iclr.cc/paper_files/paper/2026/file/ad222e6e38e26e3f00d85829b9e02f43-Paper-Conference.pdf)).
