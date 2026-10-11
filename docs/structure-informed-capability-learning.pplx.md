# Structure-Informed Capability Learning

Yes: researchers have improved learning by designing useful state dynamics, predicting the effects of training data, shaping rewards, and reducing interference between tasks. These are different interventions with different evidence: architectural throughput, sample efficiency, cheaper supervision, and deliberate behavioral change must not be collapsed into one claim of “more intelligence per FLOP” ([Gu et al., HiPPO](https://hazyresearch.stanford.edu/blog/2020-12-05-hippo), [Orvieto et al., LRU](https://proceedings.mlr.press/v202/orvieto23a/orvieto23a.pdf), [Bergen et al., Anatomy of Post-Training](https://arxiv.org/html/2606.12360v1), [Wang and Murfet, Patterning](https://arxiv.org/html/2601.13548v1)).

This atlas extension focuses on constructive training interventions and measured behavior. It does not add LLM circuit reconstruction or reverse-engineering studies; where Goodfire uses probes or sparse autoencoders, their role is described only as measurement or supervision infrastructure.

## Evidence at a glance

| Research route and authors | Intervention and demonstrated result | Efficiency conclusion |
|---|---|---|
| HiPPO: Albert Gu, Tri Dao, Stefano Ermon, Atri Rudra, Christopher Ré | Derive recurrent memory updates from online polynomial projections of signal history; the reported permuted-MNIST result is 98.3% ([author explanation](https://hazyresearch.stanford.edu/blog/2020-12-05-hippo)). | A mathematically designed memory mechanism with benchmark gains, not evidence of cheaper general LLM capability acquisition ([HiPPO](https://hazyresearch.stanford.edu/blog/2020-12-05-hippo)). |
| LRU: Antonio Orvieto, Samuel L. Smith, Albert Gu, Anushan Fernando, Caglar Gulcehre, Razvan Pascanu, Soham De | Use linear recurrence, complex diagonal state transitions, stable initialization, normalization, and parallel scans; reported training-step throughput is 1.9–29× a tested tanh-RNN across tasks, with 94.2% PathX accuracy ([ICML paper](https://proceedings.mlr.press/v202/orvieto23a/orvieto23a.pdf)). | Strong evidence for faster steps and effective long-sequence learning in the tested setup; not a universal time-to-equal-quality speedup over Transformers or SSMs ([LRU](https://proceedings.mlr.press/v202/orvieto23a/orvieto23a.pdf)). |
| Anatomy of Post-Training: Leon Bergen, Usha Bhalla, Sidharth Baskaran, Max Loeffler, Ekdeep Singh Lubana and collaborators | Predict concept changes from preference data and modify the learning signal; targeted poetic reward shaping improves Creative Writing Bench v3 by about 40 Elo ([paper](https://arxiv.org/html/2606.12360v1)). | Demonstrates targeted output-quality and behavior changes; no complete end-to-end compute-saving result is established ([paper](https://arxiv.org/html/2606.12360v1)). |
| Features as Rewards: Aaditya Prasad, Connor Watts, Jack Merullo, Dhruvil Gala, Owen Lewis, Thomas McGrath, Ekdeep Singh Lubana | Use learned feedback in RL; approximately $2,500 funds 360 optimizer steps versus only 3–5 with the compared Gemini-plus-web reward pipeline ([Goodfire](https://www.goodfire.com/research/rlfr)). | Cheaper feedback enables more optimization; the 58% hallucination-reduction result includes an intervention harness and best-of-32, not training alone ([Goodfire](https://www.goodfire.com/research/rlfr)). |
| Probe-Based Data Attribution: Frank Xiao and Santiago Aranguri | Rank preference examples using their alignment with a behavior probe, then edit data and retrain; filtering the top 30,000 examples reduces the measured harmful-distractor compliance by 63% ([Goodfire](https://www.goodfire.com/research/probe-based-data-attribution)). | Reported ranking cost is 12 H100-hours versus 128 for the compared gradient method, after probe training; this is not total training cost ([Goodfire](https://www.goodfire.com/research/probe-based-data-attribution)). |
| Patterning: George Wang and Daniel Murfet | Estimate response to training-distribution changes, then reweight data to favor a desired solution; mean out-of-distribution parentheses accuracy rises from .310 to .497 in the favorable condition ([paper](https://arxiv.org/html/2601.13548v1)). | Actual developmental intervention, but not demonstrated net efficiency: susceptibility estimation in the language-model experiment costs approximately 500× model training by the paper's forward-pass accounting ([paper](https://arxiv.org/html/2601.13548v1)). |
| Capacity, interference and rare-task retention: Jing Huang, Daniel Wurgaft, Rachit Bansal, Laura Ruis, Naomi Saphra, David Alvarez-Melis, Andrew Lampinen, Christopher Potts, Ekdeep Singh Lubana | Vary capacity, task frequency, and spacing; study how common-task updates erase rare-task learning, including language-model pretraining experiments ([paper](https://arxiv.org/pdf/2605.29548)). | Supports retention and interference as learning bottlenecks; does not establish a new universally compute-saving optimizer ([paper](https://arxiv.org/pdf/2605.29548)). |
| μTransfer: Greg Yang, Edward J. Hu and collaborators | Use a width-aware parameterization to transfer hyperparameters from small to large models; a 40M-to-6.7B transfer experiment reports tuning cost of 7% of pretraining cost ([Microsoft Research](https://www.microsoft.com/en-us/research/publication/tuning-large-neural-networks-via-zero-shot-hyperparameter-transfer/)). | A practical reduction in large-model tuning expense, not a claim that pretraining itself costs only 7% as much ([publication](https://www.microsoft.com/en-us/research/publication/tuning-large-neural-networks-via-zero-shot-hyperparameter-transfer/)). |
| Structured mechanics: Samuel Greydanus, Misko Dzamba, Jason Yosinski; comparative work by Yaofeng Desmond Zhong, Biswadip Dey, Amit Chakraborty | Build Hamiltonian/Lagrangian structure and physical constraints into learned dynamics; the comparative study tests formulations and data regimes on mechanical systems ([Zhong et al.](https://proceedings.mlr.press/v144/zhong21a/zhong21a.pdf)). | Useful physical priors can improve data efficiency, but an incorrect conservation assumption or a more expensive derivative calculation can remove the advantage ([comparison](https://proceedings.mlr.press/v144/zhong21a/zhong21a.pdf)). |

These rows are not points on one shared Pareto frontier. They have different targets, baselines, training histories, hardware, and cost boundaries; use them to select experiments, not to rank the fields universally.

## What Goodfire contributes

### Predicting and shaping the learning signal

Anatomy of Post-Training characterizes chosen and rejected preference examples, predicts which concepts training will amplify, and tests interventions through reward shaping, training-time representation changes, prompting, and filtering ([Bergen et al.](https://arxiv.org/html/2606.12360v1)). It is directly relevant to intentional construction because the intervention changes learning rather than merely describing a finished model ([paper](https://arxiv.org/html/2606.12360v1)).

The reported \(R^2=0.90\) concerns feature-conditioned prediction of feature changes; the prompt-conditioned pipeline reports \(R^2=0.58\), so neither number means “90% of arbitrary future behavior is predictable” ([paper](https://arxiv.org/html/2606.12360v1)). The creative-writing intervention is a positive quality result, while several other experiments suppress unwanted behavior; the smaller behavioral categories are not all reliably corrected, and concept correlations remain a limitation ([paper](https://arxiv.org/html/2606.12360v1)).

The paper's idealized reward-tilting view is

\[
\pi_*(y\mid x)\propto \pi_0(y\mid x)\exp(r(x,y)/\beta).
\]

A concept score can be subtracted from the reward to reduce an unwanted training incentive, or added to amplify a desired one; the paper's concept-factorization assumptions matter to this interpretation ([Bergen et al.](https://arxiv.org/html/2606.12360v1)). This is reward and distribution control, not a proof that arbitrary semantic objectives are reachable.

### Making supervision cheaper

Features as Rewards uses probes on a frozen model to supply feedback while another model is trained, avoiding reliance on the student's own changing activations as the reward signal ([Prasad et al.](https://www.goodfire.com/research/rlfr)). The reported experiments use Gemma 3 12B and show that a low-cost feedback mechanism permits substantially more RL steps at the stated budget ([Goodfire](https://www.goodfire.com/research/rlfr)).

This is promising for efficient correction, but the accounting must include probe construction, labels, inference, and the deployed intervention policy. In particular, the headline 58% reduction includes monitoring/intervention and best-of-32 selection, so it must not be entered into the atlas as a training-only capability gain ([Goodfire](https://www.goodfire.com/research/rlfr)).

### Protecting learning from interference

The Huang et al. paper, featured in Goodfire's research index, argues that larger models can fit common tasks more completely, reducing the residual gradients that otherwise overwrite rare-task learning ([paper](https://arxiv.org/pdf/2605.29548), [Goodfire research index](https://www.goodfire.com/research)). It tests task frequency and spacing as well as size; its evidence includes synthetic regression and injected tasks during language-model pretraining ([paper](https://arxiv.org/pdf/2605.29548)).

The useful design question is therefore not only “Can the network represent the capability?” but also “Can updates accumulate long enough for it to learn and retain that capability?” The proposed atlas experiment below tests replay and scheduling as interventions; the paper does not already establish our proposed controller.

## Developmental interpretability becomes an intervention

Patterning is an important addition to the earlier learning/control guide: developmental work is not limited to observational diagnostics. Wang and Murfet estimate a susceptibility matrix, meaning a local response map from a change in training-distribution weights to a change in selected posterior observables ([Patterning](https://arxiv.org/html/2601.13548v1)).

\[
d\mu^\infty=\chi\,dh,\qquad
dh_{\mathrm{target}}\approx\chi^\dagger d\mu^\infty_{\mathrm{target}}.
\]

Here \(h\) specifies data weighting, \(\mu^\infty\) is a posterior expectation, and the pseudoinverse \(\chi^\dagger\) proposes a data perturbation; this is not an exact model of an arbitrary finite SGD or Adam trajectory ([paper](https://arxiv.org/html/2601.13548v1)). The parentheses experiment targets competing solutions using local learning coefficients and evaluates the resulting trained behavior, rather than only checking whether a diagnostic moves ([paper](https://arxiv.org/html/2601.13548v1)).

The favorable dataset condition improves mean held-out accuracy from .310 to .497, while another condition produces .004; these are evidence of influence over solution selection, not perfect control ([paper](https://arxiv.org/html/2601.13548v1)). The modified datasets also change their effective example counts, so the result should not be described as a fully cost-matched efficiency comparison ([paper](https://arxiv.org/html/2601.13548v1)).

The language-model experiment is small, around 3M parameters, and the uniform-training baseline has the lowest loss; the paper reports approximately 500× training-equivalent forward-pass cost for susceptibility estimation in its patterning experiment ([paper](https://arxiv.org/html/2601.13548v1)). The supported conclusion is bounded developmental control; scalable capability acquisition at lower total cost remains unproven.

## Mathematical bridge: two dynamical systems, not one

For our experiment design, distinguish inference-time model dynamics from training-time parameter dynamics. The following notation and local derivation are an atlas synthesis, not a claim that the cited methods share one theorem:

\[
z_{t+1}=F_{\theta_k}(z_t,x_t).
\]

\[
\theta_{k+1}=\theta_k-\eta\sum_i w_{k,i}\nabla_\theta\ell_i(\theta_k).
\]

The first equation describes the model processing a sequence; HiPPO and LRU deliberately structure this memory process ([HiPPO](https://hazyresearch.stanford.edu/blog/2020-12-05-hippo), [LRU](https://proceedings.mlr.press/v202/orvieto23a/orvieto23a.pdf)). The second gives a simplified SGD process in which example weights are an available training actuator; Patterning and Goodfire's data interventions operate on this second level, although their actual objectives and estimators differ ([Patterning](https://arxiv.org/html/2601.13548v1), [Anatomy](https://arxiv.org/html/2606.12360v1)).

For a differentiable behavioral surrogate \(g(\theta)\), compare two one-step SGD updates from the same checkpoint that differ only by \(\Delta w_i\). A first-order Taylor expansion gives the local response:

\[
\Delta g\approx-\eta\sum_i\Delta w_i
\left\langle\nabla_\theta g,\nabla_\theta\ell_i\right\rangle.
\]

This elementary derivation explains why gradient alignment is a sensible cheap baseline. It is not an exact multi-step predictor: changing gradients, optimizer momentum, curvature, discrete metrics, and distribution shifts can invalidate the approximation.

Likewise, for losses \(L_a,L_b\), a gradient step on task \(b\) changes task \(a\)'s loss approximately by

\[
\Delta L_a\approx-\eta\langle\nabla L_a,\nabla L_b\rangle.
\]

Negative gradient alignment therefore predicts local interference under these assumptions. Our proposed controller would use this as a measurement, not assume it proves that a particular replay policy will improve final capability.

### Where Hankel values and LLC belong

Hankel singular values concern an input/output dynamical operator and jointly reachable/observable state directions, whereas the LLC concerns local loss-volume geometry in parameter space ([Scherpen](https://personal.math.vt.edu/jborggaa/nlromc/Scherpen_NLROMc_tutorial.pdf), [Lau et al.](https://arxiv.org/pdf/2308.12108)). Patterning supplies one empirical route from learning-geometry measurements to data interventions, but not an identity between LLC and a Hankel spectrum ([Patterning](https://arxiv.org/html/2601.13548v1)).

For the atlas, the proposed bridge is to identify a local training-response model from actual perturbations, validate it prospectively, and then ask whether its finite-horizon controllability or observability analysis adds predictive value. A fixed stationary linear plant cannot simply be assumed for a learning process whose parameters, optimizer state, and data distribution all change.

## Structural comparison points

I recommend using HiPPO/LRU as explicit structural comparison points, rather than treating any attention-free network as inherently interpretable or controllable.

## Decisive experiment: capability acquisition and retention at equal total cost

This is an original proposal, not an experiment run for this atlas. It tests constructive learning instead of repeating only a known-state toy diagnostic.

- **Task:** train on several learnable mechanisms or compositional skills, including infrequent ones, with held-out compositions and changed operating conditions. Require actual task success and retained knowledge, not only a probe score.
- **Architecture axis:** compare a conventional baseline against an explicitly structured recurrent/SSM model; add physical structure only when the task really has the corresponding invariants.
- **Training axis:** compare uniform sampling, validation-loss scheduling, gradient-alignment scheduling, and interference-aware replay. Add susceptibility/LLC-guided scheduling only when its measurement overhead fits within the same total budget.
- **Correction axis:** after learning, change a task rule or provide an allowed correction. Measure whether the model adapts, preserves unrelated capabilities, and retains the correction over the declared horizon.
- **Budget:** count training, tuning, failed runs, measurement, probe construction, labels, and inference-time selection separately and in total. Report wall-clock and hardware as well as estimated FLOPs.
- **Decision rule:** require a reproducible improvement in time or total cost to a prespecified capability-and-retention threshold. A faster step, better probe fit, or smaller training loss alone does not pass.

The most defensible initial comparison is therefore **structural architecture priors × cheap response-aware training**, with more expensive developmental measurements earning their place through incremental predictive and intervention value. This recommendation follows from published structural results, rather than presuming a universal theory of intelligence or alignment ([LRU](https://proceedings.mlr.press/v202/orvieto23a/orvieto23a.pdf)).

## Atlas relationship semantics

The extension adds topic and project nodes, not a social graph of the researchers. Authors remain inspectable attribution; the graph lines are undirected topic connections, and any directional scientific claim is stated in the relationship text.

- **Structural construction:** an explicit mathematical design creates the relevant model component.
- **Measured intervention:** training data, objective, or architecture was changed and an outcome measured in the cited experiment.
- **Predictive measurement:** an estimator forecasts responses without itself establishing successful inverse control.
- **Project connection:** a named existing experiment or implementation addresses part of the same question.
- **Hypothesis:** a proposed combination or transfer, not a demonstrated result.

Evidence status remains separate from relationship meaning. Missing edges mean “not mapped in this reviewed dataset”; geometric proximity and connection count do not indicate scientific importance, causal strength, or guaranteed compatibility.
