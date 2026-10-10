# Building Capable Models Intentionally: LLMs, World Models, VLMs, and VLAs

This document turns the cross-field literature into a construction-oriented research program. The goal is not merely to constrain a fixed model after training. It is to learn useful capabilities while deliberately designing the representations, training signals, correction interfaces, uncertainty treatment, and execution boundaries that make those capabilities understandable and controllable.

The proposed system and experiments below are a synthesis, not a demonstrated general-alignment solution. Published results are cited separately from design recommendations. No reverse engineering of LLM circuits or other circuit-level mechanistic-interpretability research is required by this program.

## What “build it properly” can mean

There are already examples of capability being shaped by deliberate design rather than only discovered afterward: compositional meta-learning targets systematic generalization, verifiable-outcome RL develops reasoning performance, learned world models support policy training in imagination, and neural-symbolic systems assign different jobs to learned proposal and explicit deduction ([Lake and Baroni](https://pubmed.ncbi.nlm.nih.gov/37880371/), [Guo et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC12443585/), [Hafner et al.](https://www.nature.com/articles/s41586-025-08744-2), [Trinh et al.](https://www.nature.com/articles/s41586-023-06747-5)). None of these results supplies a complete account of general intelligence or an assurance that every new goal will be understood and followed.

For this research program, building properly means making the intended competence and correction behavior explicit enough that training choices can be tested against them. A strong result would be a model that learns unfamiliar systems, uses what it learned to solve new tasks, accepts valid corrections predictably, preserves unrelated useful knowledge, and knows when its current model is inadequate.

That objective is stronger than a low prediction loss, but narrower and more testable than “the model understands everything.” It also avoids a false choice between an unrestricted opaque learner and a fully hand-written symbolic system.

## The capability stack is not one architecture

### Language models

An LLM construction has at least five separable choices: backbone, pretraining data, training objective, post-training feedback, and inference procedure. Transformer structure, compute/data allocation, instruction training, and reasoning-oriented reinforcement learning address different parts of that stack ([Vaswani et al.](https://arxiv.org/abs/1706.03762v5), [Hoffmann and collaborators](https://deepmind.google/blog/an-empirical-analysis-of-compute-optimal-large-language-model-training/), [Ouyang et al.](https://arxiv.org/abs/2203.02155), [Guo et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC12443585/)).

| Lever | Published capability route | Design implication and test |
|---|---|---|
| Pretraining | Learn broad predictive representations; data and parameter allocation affect empirical compute efficiency ([Hoffmann et al.](https://deepmind.google/blog/an-empirical-analysis-of-compute-optimal-large-language-model-training/)). | Choose data for required competencies and coverage, not token count alone. Audit held-out families and contamination. |
| Instruction supervision | Demonstrations and rankings improve instruction-following on evaluated distributions ([Ouyang et al.](https://arxiv.org/abs/2203.02155)); LIMA reports strong behavior from curated small-scale instruction tuning of an already large pretrained model ([Zhou et al.](https://arxiv.org/abs/2305.11206)). | Separate inherited capability from what post-training changes. Do not infer that a small instruction set creates the underlying intelligence from scratch. |
| Preference training | RLHF, DPO, and AI-generated preference feedback shape output distributions ([Christiano et al.](https://arxiv.org/abs/1706.03741), [Rafailov et al.](https://papers.baulab.info/papers/also/Rafailov-2023.pdf), [Bai et al.](https://arxiv.org/abs/2212.08073)). | Use preferences for genuinely preference-dependent choices. Check evaluator disagreement, distribution shift, and reward exploitation independently. |
| Verifiable reasoning | R1 uses correctness/format rewards and a multi-stage pipeline; process supervision has separate evidence on mathematical tasks ([Guo et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC12443585/), [Lightman et al.](http://arxiv.org/abs/2305.20050)). | Prefer executable outcomes where possible. A learned process judge is still a model, and rewardable correctness depends on verifier validity. |
| Compositional training | Meta-learning over compositional tasks can improve systematic generalization in tested settings ([Lake and Baroni](https://pubmed.ncbi.nlm.nih.gov/37880371/)). | Hold out combinations and rules, not only randomly selected examples. Compare against ordinary next-token training using matched data volume. |
| Instruction authority | Hierarchy training improves robustness to tested instruction conflicts and attacks ([Wallace et al.](https://openai.com/index/the-instruction-hierarchy/)). | Make authority levels explicit in training and interfaces. Test conflicts, untrusted retrieved material, and legitimate user correction separately. |

**Recommendation:** train correction as a capability, not just as a refusal style. Include examples where the model must revise a mistaken world assumption, preserve an unchanged goal, explain what evidence would distinguish alternatives, and verify the consequences of its revision.

Do not equate a fluent rationale with verified understanding. Turpin and collaborators demonstrate that input biases can influence answers without being faithfully disclosed in chain-of-thought explanations; their result is behavioral evidence, not circuit reverse engineering ([NeurIPS paper](https://proceedings.neurips.cc/paper_files/paper/2023/hash/ed3fea9033a80fea1376299fa7863f4a-Abstract-Conference.html)).

### Vision-language models

CLIP provides a language-addressable similarity interface learned from image-text pairs; LLaVA connects a vision encoder and an LLM through visual instruction tuning ([Radford et al.](https://cdn.openai.com/papers/Learning_Transferable_Visual_Models_From_Natural_Language.pdf), [Liu, Li, Wu, and Lee](https://arxiv.org/abs/2304.08485)). These demonstrate useful multimodal transfer, but neither construction establishes that every named object, relation, or hidden property has been correctly identified.

For intentional VLM construction, propose four independently supervised outputs:

- **Observation:** what is directly visible, with spatial grounding and uncertainty.
- **State hypothesis:** what is inferred but not directly visible, including alternative explanations.
- **Task interpretation:** which entities and relations matter to the authorized instruction.
- **Evidence request:** which next view or measurement would resolve the most consequential uncertainty.

These are proposed interfaces. Test them with counterbalanced images and instructions: keep text fixed while changing the image, keep the image fixed while changing the relevant relation, and include situations where the correct answer is “not observable.”

### World models

“World model” covers several constructions, and they should not be treated as equivalent. Dreamer learns action-conditioned latent dynamics used to train behavior in imagined trajectories, while V-JEPA 2 uses action-free visual pretraining followed by an action-conditioned stage for robotic planning ([Hafner et al.](https://www.nature.com/articles/s41586-025-08744-2), [Assran et al.](https://ai.meta.com/research/publications/v-jepa-2-self-supervised-video-models-enable-understanding-prediction-and-planning/)).

For this research program, require a control-relevant world model to answer: “If I take this action, what changes, what remains unchanged, what could go wrong, and how certain am I?” That is an evaluation definition, not a claim that any particular latent architecture already satisfies it.

| World-model form | Useful construction | Critical evaluation |
|---|---|---|
| Latent recurrent model | Compact state plus action-conditioned prediction and reward estimates, as in Dreamer ([Hafner et al.](https://www.nature.com/articles/s41586-025-08744-2)). | Counterfactual actions, planner exploitation of model errors, hidden-state persistence |
| Video-generative model | Rich predicted observations; visual and action-following metrics can be evaluated separately ([Liang et al.](https://openaccess.thecvf.com/content/CVPR2026/papers/Liang_WorldLens_Full-Spectrum_Evaluations_of_Driving_World_Models_in_Real_World_CVPR_2026_paper.pdf)). | Whether attractive video corresponds to the real consequences of actions |
| Object/graph model | Entity-oriented slots and relational computations ([Locatello et al.](https://arxiv.org/abs/2006.15055), [Battaglia et al.](https://arxiv.org/abs/1806.01261)). | Object identity through occlusion, changing interactions, new object counts |
| Scientific dynamics model | Sparse equations, structured energy, Koopman coordinates, or learned PDE operators ([Brunton et al.](https://arxiv.org/abs/1605.06682), [Greydanus et al.](https://arxiv.org/abs/1906.01563), [Fan et al.](https://arxiv.org/html/2401.08153v2), [Lanthaler et al.](https://academic.oup.com/imatrm/article/6/1/tnac001/6542709?searchresult=1)). | Whether the structural assumptions hold, and how measured error affects the downstream controller |
| Program or rule model | Reusable executable abstractions, as in DreamCoder ([Ellis and collaborators](https://royalsocietypublishing.org/doi/10.1098/rsta.2022.0050)). | New mechanisms, incomplete rules, uncertainty, and search cost |

**Recommendation:** do not require pixel-perfect reconstruction if it wastes capacity on irrelevant appearance. But do not discard information merely because it is irrelevant to the current reward; a later instruction or safety constraint may depend on it. Compare task-specific representations with representations trained across changing goals and constraints.

### Vision-language-action models

RT-2 turns VLM outputs into discretized action tokens, OpenVLA provides a trainable VLM-to-action system, and π₀ combines a VLM backbone with a flow-matching action expert ([Brohan et al.](https://arxiv.org/html/2307.15818v1), [Kim et al.](https://github.com/openvla/openvla), [Black et al.](https://arxiv.org/html/2410.24164)). These are different action-generation interfaces, not interchangeable labels for the same control mechanism.

The deliberate design question is how language semantics, perceptual uncertainty, physical state, and actuator limits meet. Propose separately exposing a goal representation, a state/belief estimate, an action proposal, and the reason an action was permitted or modified.

The difficult part is not merely adding another module. AEGIS illustrates both the promise and the gap: a VLM selects an obstacle, geometric processing produces an enclosing shape, and a CBF layer modifies actions; the collision theorem assumes correct assessment and complete representation and does not cover all robot links ([Hu et al.](https://arxiv.org/html/2512.11891v2)).

## A constructive architecture to investigate

The following is an original research design. It is not a claim that assembling these components produces alignment, nor that every component must be neural.

```text
Authorized instruction + evidence + observations
                      |
         Goal and uncertainty interface
                      |
     Learned belief / object / rule state
            |                    |
  Persistent authorized memory   |
            |                    |
   Action-conditioned world model
                      |
    Learned proposer / policy / planner
                      |
  Task checks + constraint-aware execution
                      |
             Environment effects
                      |
      Independent measurements and updates
```

### Learn the content; structure the interface

The model should learn unknown dynamics, rules, and reusable skills rather than being handed a complete world description. The architecture can still specify that state predictions, uncertainties, action proposals, and memory changes must pass through named interfaces.

Concept bottlenecks demonstrate that explicit intermediate predictions can be edited, while also showing that model-level concept edits are not automatically real-world causal interventions ([Koh et al.](https://proceedings.mlr.press/v119/koh20a/koh20a.pdf)). Later work makes the leakage issue explicit: predicting a concept well does not ensure that its representation contains only that concept ([Almudévar, Hernández-Lobato, and Ortega](https://arxiv.org/html/2506.04877v3)).

**Proposed design rule:** keep important interfaces narrow and testable, but allow residual learned representations where named concepts are incomplete. Any residual path must be included in correction tests, because it may carry information that defeats the intended edit.

### Co-design the learner and the controller

The learned model should optimize useful decision quality, not only an unrelated forecast metric. Task-based end-to-end learning provides a published route to training probabilistic models around the downstream optimization task, with demonstrations in inventory and energy applications ([Donti, Amos, and Kolter](https://arxiv.org/abs/1703.04529)).

Use such decision-aware losses alongside independent prediction and calibration evaluations. Otherwise a model may become useful for one controller while losing scientific fidelity or failing under a changed goal.

### Make correction a persistent, scoped operation

Define a correction object containing its claim, evidence, scope, authority, expiration or revision condition, and expected consequences. A one-off instruction, a new physical observation, and a permanent rule update should not all use the same memory semantics.

This is a proposed interface. It avoids assuming that modifying an arbitrary hidden-state coordinate has the desired meaning, and creates observable tests for correction retention, rollback, and collateral effects.

### Keep an independent execution boundary

Independent enforcement is not the entire alignment strategy. It is a boundary around consequences while the learned model is still uncertain or fallible.

Predictive safety filters and shields illustrate conditional enforcement of specified trajectory or temporal properties, while also depending on their modeled state/action space and assumptions ([Wabersich and Zeilinger](https://ar5iv.labs.arxiv.org/html/1812.05506), [Alshiekh et al.](https://arxiv.org/abs/1708.08611)). For a tool-using language system, our analogous design recommendation is to keep authorization, resource limits, reversible execution, and immutable audit records outside the model’s unilateral control.

## Where the classical fields enter the construction

The 77-entry companion map gives the detailed coverage. This table identifies practical roles without insisting that every method belongs in every model.

| Role | Candidate methods | Selection criterion |
|---|---|---|
| Predict simple relationships | Linear/ridge models, kernels, SVMs, trees, additive models | Prefer the simplest model that passes intervention and shift tests |
| Estimate physical state | Kalman, extended/unscented/cubature filters, particles, learned residual filters | Match noise, nonlinearity, multimodality, and available compute |
| Preserve signal structure | FIR/IIR, Fourier, wavelets, scattering, bilateral or PDE filtering | Quantify information loss and downstream delay, not just cleaner images |
| Learn memory and sequence processing | RNN, LSTM/GRU, S4, Mamba, attention, Hyena, hybrids | Jointly test memory, state tracking, throughput, correction retention |
| Learn dynamics | SINDy, GP residuals, Koopman, NODE/CDE, HNN/LNN, graph models | Match physical assumptions and validate action-conditioned behavior |
| Generate proposals | Autoregression, diffusion, score models, flow matching, distribution bridges | Compare reachable desired outcomes, calibration, diversity, and latency |
| Choose and improve actions | Dynamic programming, TD/Q, policy gradients, PPO/SAC, model-based RL, constrained RL | Separate utility, exploration risk, and actual trajectory constraints |
| Analyze learning | Deep-linear theory, NTK, DMFT, criticality, information theory, SLT/LLC/MDL | Require prospective predictive value beyond cheaper diagnostics |

These selection criteria are recommendations rather than published universal rankings. The purpose of the map is to match a tool’s assumptions to a question, not to elevate one fashionable architecture into an entire research philosophy.

## The experiments that would make this more than a diagram

### Start with unfamiliar systems, not a solved world

Use procedurally generated digital environments whose rules are hidden from the learner. For example, objects can have unknown transition rules, tools can have context-dependent effects, and tasks can require learning a new composition of rules from observations and safe experiments.

The evaluator knows the simulator and can measure true consequences, but the learner receives only the permitted observations and actions. That creates genuine mechanism-learning and generalization questions without pretending that a small simulator proves real-world alignment.

Increase difficulty along distinct axes:

- **Unknown mechanisms:** new transition rules rather than only new initial states.
- **Partial observation:** hidden variables, delayed measurements, and uncertainty.
- **Composition:** familiar primitives in unseen combinations.
- **Changing goals:** a new objective that depends on previously irrelevant information.
- **Corrections:** valid updates to a mistaken rule, plus invalid or unauthorized updates.
- **Long horizons:** persistent instructions and consequences across many steps.
- **Multimodality:** symbolic observations first, then pixels and language, then embodiment.

These are proposed benchmark dimensions. Keep some families completely held out during architecture and hyperparameter selection.

### Compare capability-bearing baselines

Do not compare a sophisticated learner only against no intervention. Include:

| Baseline | What it rules out |
|---|---|
| Linear/ridge/tree next-state model with a simple planner | Gains due only to adding a predictive model |
| Tabular or explicit symbolic learner where feasible | Gains that a small explicit state machine already achieves |
| Ordinary transformer or recurrent world model | Gains due only to extra model capacity or context |
| Retrieval/replayed correction with unchanged weights | Gains due only to reminding the model |
| Output correction or action projection | Gains due only to a final-stage correction |
| Same architecture with random or ordinary validation-selected intervention | Gains due only to intervention budget and tuning |
| Oracle state or oracle dynamics | Whether failures come from perception, representation, dynamics, or planning |

These baselines need not all scale to every task. Where a baseline becomes inapplicable, document why rather than silently removing a strong competitor.

### Pre-register six hypotheses

| Hypothesis | Distinctive prediction | Failure criterion |
|---|---|---|
| H1: Identified interfaces improve correction | The same named edit produces consistent intended consequences across held-out mechanisms and equivalent representations | Edits work only in familiar settings, or cheap replay matches them |
| H2: Structured dynamics improves robust utility | A constrained/structured model gains task success under shift at matched total cost | Stability improves but useful capability falls enough to lose the trade-off |
| H3: Persistent memory and adaptive state should be separated | Corrections survive long distractor sequences without suppressing new learning | Ordinary context/retrieval achieves equal retention and utility more cheaply |
| H4: Action-grounded training improves transferable understanding | Learned representations predict consequences of unseen actions and compositions, not only observations | Video/predictive quality rises without better intervention response |
| H5: Developmental measures improve training decisions | Early geometry/LLC or response measures predict which data intervention will help later | They add no predictive or decision value beyond loss, gradients, or pilot retraining |
| H6: Capability and authority training can improve together | More task competence accompanies better correction, clarification, and instruction-conflict handling | Gains are only refusal, reduced action rate, or distribution-specific compliance |

These are original hypotheses, not literature conclusions. H5 is specifically a forward training-control use of developmental ideas; it does not require circuit discovery.

The baseline requirement is substantive: the February 2026 Bayesian influence-function paper reports a Pythia-14M retraining-prediction comparison in which local BIF scores below EK-FAC and GradSim, and sensitivity to SGLD sampling choices remains a limitation ([Kreer et al.](https://arxiv.org/html/2509.26544v2)). Stagewise influence results also acknowledge the unresolved connection between Bayesian learning theory and actual non-equilibrium SGD ([Lee et al.](https://arxiv.org/html/2510.12071v1)).

### Measure success jointly

For each task family and model, report:

\[
\bigl(
\text{utility},
\text{generalization},
\text{correction success},
\text{correction persistence},
\text{collateral change},
\text{constraint violations},
\text{uncertainty quality},
\text{total cost}
\bigr).
\]

Do not collapse these into a single “alignment score” before inspecting their conflicts. A model that refuses all actions can have few violations while failing the intended task; a model that completes every task can still take unacceptable actions along the way.

SafeVLA-Bench operationalizes this distinction by reporting task success, safety, success-but-unsafe rate, and violation severity under specified simulator-side clauses ([Fan et al.](https://arxiv.org/html/2606.00773v3)). Its thresholds are not hardware-safety certification thresholds, so use the measurement idea without reusing its numbers as universal physical limits ([SafeVLA-Bench](https://arxiv.org/html/2606.00773v3)).

## Pareto frontiers for intentional construction

A useful frontier compares models on the same task distribution, observations, action permissions, correction budget, training budget, and execution environment. The underlying Pareto rule is simple: an option is dominated if another is at least as good on every selected axis and strictly better on at least one.

The axes should change with the decision:

| Decision | Main axes | Additional constraints |
|---|---|---|
| Choose a sequence backbone | Task competence, state tracking, inference latency/memory | Matched data and training compute |
| Choose an interpretable interface | Correction benefit, intervention fidelity, collateral damage | Comparable concept supervision and edit budget |
| Choose a world model | Control utility, intervention prediction, calibrated uncertainty | Same environment-interaction budget |
| Choose a physical controller | Success, violation rate/severity, response time | Hardware, sensing, actuator limits, fallback feasibility |
| Choose a training-control diagnostic | Improvement per intervention, predictive accuracy, overhead | Held-out prospective predictions |
| Choose an LLM post-training recipe | Verified capability, preference quality, authority handling, cost | Independent evaluators and unseen task families |

### Published points worth using carefully

| Comparison | What is actually reported | Frontier interpretation |
|---|---|---|
| Contracting RENs against recurrent baselines | System-identification studies report favorable nominal-error/robustness trade-offs; observed sensitivity is estimated, while structural bounds are separate ([Revay et al.](https://arxiv.org/pdf/2104.05942)). | Evidence for a task-specific robust-model route, not an LLM capability frontier |
| AEGIS versus its base VLA in the full-action setting | In the cited SafeLIBERO experiment, collision-free episode rate is 68.9% versus 17.1%, and task success is 67.5% versus 57.8% ([Hu et al., v2](https://arxiv.org/html/2512.11891v2)). | Favorable on those two reported axes, but not a matched-total-cost dominance claim; the pipeline adds assessment, sensing/geometry processing, and QP work |
| Safety-oriented VLA post-training | SafeVLA-Bench’s one-policy, one-suite case raises three-seed mean Safety from 85.0% to 90.8%, while mean success is 98.7% versus 99.5% for the base policy ([Fan et al.](https://arxiv.org/html/2606.00773v3)). | A reported safety/success trade-off, not proof of a general frontier or transfer to other violation types |
| Concept minimality | MCBM regularization reduces measured nuisance information, often with a task-accuracy trade-off; complete removal remains difficult ([Almudévar et al.](https://arxiv.org/html/2506.04877v3)). | Relevant to interpretability/capability trade-offs; do not rank models from concept accuracy alone |
| Sparse structured state tracking | PD-SSM provides automaton-emulation and computational-structure results, but has generation overhead and does not win every tested long-range task ([Terzić et al.](https://proceedings.neurips.cc/paper_files/paper/2025/file/77b830c18836a9b2e1395a4936dd687a-Paper-Conference.pdf)). | A concrete candidate for state-tracking experiments, not a universal Mamba replacement |

No new frontier was measured in this investigation. The recommendation is to build separate empirical frontiers and include uncertainty intervals, failed runs, tuning effort, and correction overhead.

## A staged construction program

### Establish learning and correction together

Build a learner for unknown digital systems with a real predictive or planning objective. Require it to improve with evidence and outperform cheap baselines before adding an elaborate interpretation layer.

The first gate is joint: useful held-out capability plus reproducible correction of a learned misconception. If an intervention merely forces one output while leaving the learned model and later behavior unchanged, label it an output fix rather than learned correction.

### Add structure where it earns its cost

Compare one structural choice at a time: explicit state, object slots, a sparse dynamics library, a contracting submodule, a concept interface, or executable rules. Retain it only if it improves a declared trade-off and the purported mechanism predicts interventions.

The second gate is transfer across new mechanisms and combinations. Structural interpretability that works only because the benchmark already hands the model its entire ontology should not be mistaken for learned scientific understanding.

### Add language and visual grounding

Introduce natural-language tasks and visual observations while keeping an independent environment evaluator. Test whether the same learned mechanism can be referred to in different language, observed from different viewpoints, and corrected without corrupting unrelated behavior.

The third gate is grounded generalization: successful new combinations, valid uncertainty, stable entity identity, and correction transfer. Separate failures in instruction interpretation, perception, dynamics, and policy choice using controlled substitutions.

### Add embodiment and bounded authority

Move first to high-fidelity simulation and only then to physical experiments with appropriate hardware safeguards. Use actual deadlines, actuator limits, sensing failures, interruption tests, and an independently implemented fallback.

The fourth gate is safe useful execution over the declared envelope. Neither a simulated success score nor a conditional theorem should be presented as proof that the full real system remains safe outside it.

### Scale only a demonstrated advantage

Scale data, model size, task diversity, and horizon separately enough to see which property breaks. Track whether correction fidelity and uncertainty degrade as capability grows, rather than assuming the favorable small-scale relationship continues.

The final gate is not “alignment solved.” It is a precisely scoped result such as: this learner acquires new transition mechanisms and retains authorized corrections better than matched baselines across these held-out families, with these measured costs and these remaining failure modes.

## The researcher paths to follow

These are literature paths rather than a ranking of people or a claim that each researcher endorses the proposed combined architecture.

- **Interpretable construction and editable concepts:** Cynthia Rudin; Pang Wei Koh, Percy Liang, Been Kim and collaborators; Antonio Almudévar, José Miguel Hernández-Lobato, and Alfonso Ortega ([Rudin](http://www.arxiv.org/abs/1811.10154), [CBMs](https://proceedings.mlr.press/v119/koh20a/koh20a.pdf), [MCBMs](https://arxiv.org/html/2506.04877v3)).
- **Identifiable representations and causal learning:** Bernhard Schölkopf, Francesco Locatello, Kartik Ahuja, Divyat Mahajan, Yixin Wang, Yoshua Bengio and collaborators ([causal representation overview](https://arxiv.org/abs/2102.11107), [interventional identification](https://proceedings.mlr.press/v202/ahuja23a/ahuja23a.pdf)).
- **Learned systems with structural guarantees:** Max Revay, Ruigang Wang, Ian Manchester; Aaron Ames and collaborators; Kim Wabersich, Melanie Zeilinger; Felix Berkenkamp, Angela Schoellig, Andreas Krause and collaborators ([RENs](https://arxiv.org/pdf/2104.05942), [CBFs](http://ames.caltech.edu/ames2019control.pdf), [predictive filters](https://ar5iv.labs.arxiv.org/html/1812.05506), [safe model learning](https://arxiv.org/pdf/1705.08551)).
- **Scientific dynamics and physical inductive biases:** Steven and Bingni Brunton, J. Nathan Kutz, Bethany Lusch, Urban Fasel; Siddhartha Mishra, George Karniadakis; Risi Kondor ([E-SINDy](https://arxiv.org/abs/2111.10992), [Koopman](https://www.lri.fr/~gcharpia/deeppractice/2022/chap_5_biblio/Koopman/KoopmanAE.pdf), [operator analysis](https://academic.oup.com/imatrm/article/6/1/tnac001/6542709?searchresult=1), [equivariance](https://pmc.ncbi.nlm.nih.gov/articles/PMC12541325/)).
- **Capability, composition, and explicit reasoning:** Brenden Lake, Marco Baroni; Kevin Ellis and collaborators; Trieu Trinh, Yuhuai Wu, Quoc Le, He He, and Thang Luong ([MLC](https://pubmed.ncbi.nlm.nih.gov/37880371/), [DreamCoder](https://royalsocietypublishing.org/doi/10.1098/rsta.2022.0050), [AlphaGeometry](https://research.google/pubs/solving-olympiad-geometry-without-human-demonstrations/)).
- **World models and embodied foundation models:** Danijar Hafner, Jurgis Pasukonis, Jimmy Ba, Timothy Lillicrap; Mido Assran and collaborators; Anthony Brohan and collaborators; Moo Jin Kim, Karl Pertsch, Chelsea Finn, Sergey Levine and collaborators; Kevin Black and collaborators ([Dreamer](https://github.com/danijar/dreamerv3), [V-JEPA 2](https://ai.meta.com/research/publications/v-jepa-2-self-supervised-video-models-enable-understanding-prediction-and-planning/), [RT-2](https://arxiv.org/html/2307.15818v1), [OpenVLA](https://github.com/openvla/openvla), [π₀](https://arxiv.org/html/2410.24164)).
- **Training geometry and developmental hypotheses:** Sumio Watanabe; Jesse Hoogland, Daniel Murfet, Susan Wei, Edmund Lau, George Wang and collaborators ([SLT](https://www.jmlr.org/papers/volume14/watanabe13a/watanabe13a.pdf), [LLC estimation](https://arxiv.org/pdf/2308.12108), [Hoogland catalogue](https://www.jessehoogland.com/research/)). Follow the training-data/learning-dynamics branch here, not the excluded circuit-analysis branch.
- **Behavioral control measurement:** Emily Cheng, Carmen Amo Alonso and GenCtrl collaborators; Jialiang Fan, Weizhe Xu, Zijun Wang, Oleg Sokolsky, Insup Lee, Fanxin Kong; Songqiao Hu and AEGIS collaborators ([GenCtrl](https://arxiv.org/html/2601.05637v1), [SafeVLA-Bench](https://arxiv.org/html/2606.00773v3), [VLSA](https://arxiv.org/html/2512.11891v2)).

## Narrow verdict

- **Supported claim:** selected aspects of learned models can be made explicit and sometimes guaranteed by construction, and capability can be shaped through structured objectives, task distributions, and feedback; the cited evidence covers specific systems and properties, not general semantic alignment ([RENs](https://arxiv.org/pdf/2104.05942), [equivariance](https://pmc.ncbi.nlm.nih.gov/articles/PMC12541325/), [MLC](https://pubmed.ncbi.nlm.nih.gov/37880371/), [R1](https://pmc.ncbi.nlm.nih.gov/articles/PMC12443585/)).
- **Unproven transfer:** that these properties compose into scalable, generally intelligent models whose goals, meanings, and correction behavior remain aligned across novel environments.
- **Control gap:** a validated connection from authorized intent to learned representation, from representation to predicted action consequences, and from those predictions to actual effects within an enforceable authority boundary.
- **Next decisive test:** learn unfamiliar mechanisms, pre-register a correction-interface advantage, and measure held-out capability plus correction persistence against cheap controls and matched unconstrained learners.

The most productive next step is a capability-bearing experiment with a falsifiable transfer claim. Another map of elegant correspondences will not resolve whether the proposed structure actually teaches a model to learn and behave better.
