# Introduction to AI Safety: a Study Guide to Stanford CS120

This guide walks through [Stanford CS120: Introduction to AI Safety](https://web.stanford.edu/class/cs120/) (Fall 2025) one week at a time and places each week in the atlas. All credit for the course design and the reading selection goes to the course. This guide only summarizes and links; the course page is the authoritative version, with slides, recordings and the full optional reading lists.

Each week below gives:

- **Readings:** the required readings, plus a few optional ones, each with its authors and a one-line note in this guide's own words. Most are assigned in part, usually the abstract and introduction; the course page says which sections.
- **In the atlas:** the topics (node ids in `code`) where the week's material lives.
- **Toward the thesis:** how the week connects to the atlas's alignment thesis, *capable, correctable belief-state systems* ([thesis](alignment-synthesis-thesis.pplx.md), [reading track](alignment-thesis-reading-track.md)).

A syllabus is a curated map, not evidence. A paper being on the list means the course thinks it is worth reading. It does not mean the paper's claims hold, and none of these readings tests the thesis. Check each atlas topic's boundary note for what its sources do and do not establish.

In the atlas, the course is the topic `cs120` in the **AI safety foundations** cluster. Three learning paths start from it:

1. *From introductory AI safety to the alignment thesis: rewards, preferences, robustness, oversight and correction* (`intro-ai-safety-to-thesis`), the flagship.
2. *Following Stanford CS120, I* (`intro-ai-safety-course`): weeks 1–5.
3. *Following Stanford CS120, II* (`intro-ai-safety-robustness-and-agency`): weeks 5–9.

## How the course frames safety

The course asks what safe AI is and how to build it, with a technical focus on reliable, ethical and aligned systems. It separates **model-specific** issues, which arise inside a trained model, from **systemic** issues, which arise from data, deployment, institutions and incentives. Its topics run from fairness and data, through adversarial vulnerabilities and embedding desired behaviour, to oversight of more capable systems and future risks. Most readings are research papers on current methods and their limits, across reinforcement learning, vision and language, with interpretability, robustness and evaluation as recurring themes.

The atlas's thesis shares the course's starting point: safety is several distinct problems, not one. Where the course surveys methods and their limitations, the thesis bets on one composition: belief states that keep the distinctions decisions need, action-conditioned prediction, explicit uncertainty about the objective, corrections with measurable and persistent effects, and an enforcer outside the model. It also names what would refute it.

---

## Week 1: What does safe AI mean anyway?

**Readings (required)**

- [Concrete Problems in AI Safety](https://arxiv.org/abs/1606.06565), Amodei, Olah, Steinhardt, Christiano, Schulman and Mané (2016). Five accident risks, grouped by where they come from: a wrong objective, an objective too costly to check, or unsafe behaviour while learning.
- [Taxonomy of Risks posed by Language Models](https://dl.acm.org/doi/10.1145/3531146.3533088), Weidinger et al. (FAccT 2022). Six areas of language-model harm, from discrimination and misinformation to malicious use, human–computer interaction and environmental harms.
- [AI safety is not a model property](https://www.normaltech.ai/p/ai-safety-is-not-a-model-property), Narayanan and Kapoor (2024). An essay arguing that harm usually depends on deployment context the model cannot see, so most misuse defences belong outside the model.
- [Hard Choices in Artificial Intelligence](https://arxiv.org/abs/2106.11022), Dobbe, Krendl Gilbert and Mintz (2021; published in *Artificial Intelligence*). The vague, contested decisions made while specifying a system are safety decisions too, and they need channels for dissent.

**Optional recap (machine learning basics):** [neural networks](https://www.youtube.com/watch?v=aircAruvnKk), [gradient descent](https://www.youtube.com/watch?v=IHZwWFHWa-w) and [backpropagation](https://www.youtube.com/watch?v=Ilg3gGewQ5U), 3Blue1Brown (2017), short visual explanations. [Intro to Large Language Models](https://www.youtube.com/watch?v=zjkBMFhNj_g), Karpathy (2023), a one-hour overview of how chat models are built.

**In the atlas:** `safetyproblems` (Concrete & unsolved problems in AI safety), `systemicsafety` (Safety as a system property), `rewardhack`, `oversight`, `constrainedrl` (safe exploration).

**Toward the thesis:** Each of Amodei et al.'s problems has a counterpart in the thesis. Objective uncertainty answers reward hacking and side effects, the correction interface answers scalable supervision, belief states address distributional shift, and the independent enforcer bounds unsafe exploration. The atlas records this mapping as a hypothesis (`safetyproblems`–`alignmentthesis`). Narayanan and Kapoor's argument is why the thesis draws its system boundary around the whole deployment (model, memory, tools, harness, operator), not the model alone.

## Week 2: Reward functions, alignment and human preferences; encoding preferences in AI

**Readings (required)**

- [Specification gaming: the flip side of AI ingenuity](https://deepmind.google/discover/blog/specification-gaming-the-flip-side-of-ai-ingenuity/), Krakovna et al. (DeepMind blog, 2020). A catalogue of agents meeting the letter of their objective while missing its intent.
- [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155), Ouyang et al. (2022). InstructGPT: supervised demonstrations, then a reward model from human rankings, then RL against it.
- [Scaling Laws for Reward Model Overoptimization](https://arxiv.org/abs/2210.10760), Gao, Schulman and Hilton (ICML 2023). As optimization against a learned reward grows, the true (gold) reward rises, peaks and falls, in a predictable way.
- [Open Problems and Fundamental Limitations of RLHF](https://arxiv.org/abs/2307.15217), Casper et al. (TMLR 2023). A survey of where human feedback, reward models and policy optimization each break down, separating fixable from fundamental problems.
- [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073), Bai et al. (2022). Self-critique against written principles, then RL from AI preference labels instead of human harm labels.
- [Discovering Language Model Behaviors with Model-Written Evaluations](https://aclanthology.org/2023.findings-acl.847/), Perez et al. (Findings of ACL 2023). Models write evaluation datasets at scale; among the findings, larger models more often repeat back a user's view.
- [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290), Rafailov et al. (NeurIPS 2023). Fit the policy directly to preference pairs with a classification-style loss, without a separate reward model or RL loop. The course also links a plain-language [explainer](https://medium.com/@joaolages/direct-preference-optimization-dpo-622fc1f18707) by Lages.

**Selected optional readings:** [Deep RL from human preferences](https://arxiv.org/abs/1706.03741), Christiano et al. (2017), an early demonstration of learning rewards from human comparisons of behaviour. [RLAIF vs. RLHF](https://arxiv.org/abs/2309.00267), Lee et al. (ICML 2024), where AI feedback matched human feedback on the reported tasks. [Language Models Learn to Mislead Humans via RLHF](https://arxiv.org/abs/2409.12822), Wen et al. (2024): RLHF made models more convincing to time-limited raters without making them more correct. [Cooperative Inverse Reinforcement Learning](https://arxiv.org/abs/1606.03137) and [Inverse Reward Design](https://arxiv.org/abs/1711.02827), Hadfield-Menell et al. (2016, 2017), which treat the reward as uncertain and the human's behaviour or designed reward as evidence about it. [AssistanceZero](https://arxiv.org/abs/2504.07091), Laidlaw et al. (2025), which solves assistance games at scale. [Beyond Preferences in AI Alignment](https://arxiv.org/abs/2408.16984), Zhi-Xuan et al. (2024), and [A Roadmap to Pluralistic Alignment](https://arxiv.org/abs/2402.05070), Sorensen et al. (ICML 2024), on whose values and how to combine them.

**In the atlas:** `specgaming`, `rewardhack`, `preference` (RLHF & DPO), `constitutional` (Constitutional AI & RLAIF), `rmensemble`, `llmjudge`, `sycophancy`, `assistance`, `ird`, `socialchoice`.

**Toward the thesis:** This is the thesis's home ground. Its main falsifier is the overoptimization curve: reject the thesis if better proxy optimization systematically destroys independently evaluated user utility (`rewardhack`–`alignmentthesis`). The thesis agrees with RLHF that preferences are evidence about intent, but it parts ways on what to do with that evidence. It does not fit a single reward and maximize it. It keeps uncertainty over objectives, as CIRL and inverse reward design do, and treats feedback as data about that uncertainty. The optional assistance-game readings are the course's direct bridge to the thesis's supporting case.

## Week 3: Validation of AI systems; data is all you need

**Readings:** the validation lecture (a guest lecture) has no assigned readings. For the data lecture:

- [DALL·E 2 pre-training mitigations](https://openai.com/index/dall-e-2-pre-training-mitigations/), OpenAI (2022). How filtering, rebalancing and deduplicating training images changed what the model generates, including how filtering itself introduced bias.
- [Machine Bias](https://www.propublica.org/article/machine-bias-risk-assessments-in-criminal-sentencing), Angwin, Larson, Mattu and Kirchner (ProPublica, 2016). The COMPAS investigation: unequal error rates across racial groups in a criminal risk score.
- [On Hate Scaling Laws For Data-Swamps](https://arxiv.org/abs/2306.13141), Birhane et al. (2023). Growing a web-scraped image–text dataset fivefold increased hateful content and the racial bias of models trained on it.
- [Machine Unlearning Doesn't Do What You Think](https://arxiv.org/abs/2412.06966), Cooper et al. (2024). Removing information from parameters and suppressing it in outputs are different goals, and neither matches what policy usually asks of unlearning.
- [Deep Ignorance](https://arxiv.org/abs/2508.06601), O'Brien et al. (2025). Filtering biothreat text before pretraining made models far more resistant to adversarial fine-tuning, though information supplied in context could still be used.

**Selected optional readings:** [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361), Kaplan et al. (2020), on loss falling as a power law in data, parameters and compute. [Certifying and Removing Disparate Impact](https://arxiv.org/abs/1412.3756), Feldman et al. (KDD 2015), which tests disparate impact by predicting the protected attribute from the other features. [Learning Fair Representations](https://proceedings.mlr.press/v28/zemel13.html), Zemel et al. (ICML 2013), which learns codes that keep task information and hide group membership.

**In the atlas:** `datasafety` (Training data, filtering & machine unlearning), `fairness`, `scalinglaws`, `sleeperagents` (what poisoned data can leave behind), `nnverify` (verification; see week 6), `rudin2019`.

**Toward the thesis:** Unlearning is a proposed correction, and Cooper et al.'s critique is the thesis's persistence falsifier in a policy setting. A correction must be shown to hold on held-out triggers, under further fine-tuning and without silent side effects (`datasafety`–`alignmentthesis`, a hypothesis). Deep Ignorance is a rare case where an intervention before training is compared head to head with post-training safeguards. That is the kind of matched comparison the thesis's learning loop asks for.

## Week 4: Needs for AI safety today, beyond the hype; social-choice theory for engineers

**Readings (required)**

- [“AI for Good” Isn't Good Enough: A Call for Human-Centered Design](https://docs.google.com/document/d/1CDCI9bJZOB4vXnil-Xnwepvd04nID1bEz_xUCvyfWWw/edit), Landay (2024). An argument for designing with the people affected, not only for them.
- [Unsolved Problems in ML Safety](https://arxiv.org/abs/2109.13916), Hendrycks, Carlini, Schulman and Steinhardt (2021). Four research areas: robustness, monitoring, alignment and systemic safety.
- [AI Safety on Whose Terms?](https://www.science.org/doi/10.1126/science.adi8982), Lazar and Nelson (*Science*, 2023). Asks who sets the safety agenda and whose risks it prioritizes.
- [The TESCREAL Bundle](https://firstmonday.org/ojs/index.php/fm/article/view/13636), Gebru and Torres (*First Monday*, 2024). A critique of the ideologies behind some AGI and existential-risk discourse.
- [The Grey Hoodie Project](https://arxiv.org/abs/2009.13676), Abdalla and Abdalla (AIES 2021). Compares industry funding of AI ethics research with tobacco-industry influence on science.
- [How Culture Shapes What People Want From AI](https://arxiv.org/abs/2403.05104), Ge et al. (CHI 2024). Survey evidence that cultural models of the self change whether people want to control AI or connect with it.

The social-choice guest lecture has no assigned readings. The week 2 optional readings on pluralistic alignment and PRISM ([Kirk et al. 2024](https://arxiv.org/abs/2404.16019)) cover the same ground from the machine-learning side.

**Selected optional readings:** [On the Opportunities and Risks of Foundation Models](https://arxiv.org/abs/2108.07258), Bommasani et al. (2021), the broad map of foundation-model capabilities and risks.

**In the atlas:** `safetyproblems`, `systemicsafety`, `socialchoice` (Social choice & pluralistic alignment), `humanai`, `aigovern`.

**Toward the thesis:** The thesis defines aligned behaviour for *a specified user* within authorized constraints. Social choice asks who authorizes those constraints when users and affected people disagree. The thesis does not yet answer this, and the atlas records the gap as a hypothesis link (`socialchoice`–`alignmentthesis`). Assistance games model a single human, so extending objective uncertainty to many stakeholders needs an explicit aggregation or negotiation step (`socialchoice`–`assistance`).

## Week 5: AI and mental health; red teaming, adversarial vulnerabilities and multi-agent systems

**Readings:** the mental-health guest lecture has no public readings. For the red-teaming lecture:

- [Explaining and Harnessing Adversarial Examples](https://arxiv.org/abs/1412.6572), Goodfellow, Shlens and Szegedy (ICLR 2015). Attributes adversarial examples to locally linear behaviour and introduces the fast gradient sign method and adversarial training.
- [Are aligned neural networks adversarially aligned?](https://arxiv.org/abs/2306.15447), Carlini et al. (NeurIPS 2023). Aligned multimodal models are easy to attack through images; failed text attacks do not prove robustness.
- [Jailbroken: How Does LLM Safety Training Fail?](https://arxiv.org/abs/2307.02483), Wei, Haghtalab and Steinhardt (NeurIPS 2023). Two failure modes, competing objectives and mismatched generalization, explain and generate jailbreaks.
- [3 takeaways from red teaming 100 generative AI products](https://www.microsoft.com/en-us/security/blog/2025/01/13/3-takeaways-from-red-teaming-100-generative-ai-products/), Bullwinkel et al. (Microsoft, 2025). Practitioner lessons: generative AI amplifies existing security risks and adds new ones, human judgment stays central, and defence in depth is needed.
- [Levels of Autonomy for AI Agents](https://arxiv.org/abs/2506.12469), Feng et al. (2025). Grades agents by the role the user plays, from operator to observer, as a design choice separate from capability.
- [Multi-Agent Risks from Advanced AI](https://arxiv.org/abs/2502.14143), Hammond et al. (2025). Failure modes that arise only when many AI agents interact: miscoordination, conflict and collusion.

**Selected optional readings:** [Red Teaming Language Models with Language Models](https://arxiv.org/abs/2202.03286), Perez et al. (2022), on automated red teaming. [Universal and Transferable Adversarial Attacks on Aligned Language Models](https://arxiv.org/abs/2307.15043), Zou et al. (2023), on optimized suffixes that transfer to closed models. [Poisoning Web-Scale Training Datasets is Practical](https://arxiv.org/abs/2302.10149), Carlini et al. (2023): cheap, real attacks on public datasets. [Universal Jailbreak Backdoors from Poisoned Human Feedback](https://arxiv.org/abs/2311.14455), Rando and Tramèr (ICLR 2024), a trigger planted through RLHF data. [Defending Against Unforeseen Failure Modes with Latent Adversarial Training](https://arxiv.org/abs/2403.05030), Casper et al. (2024), which attacks hidden activations instead of inputs.

**In the atlas:** `advexamples`, `jailbreaks`, `redteam`, `promptinject`, `guardrail`, `sleeperagents`, `multillm`, `agentworm`, `humanai` (Safe human–AI interaction, including vulnerable users).

**Toward the thesis:** Jailbreaks show that safety training can change behaviour on familiar requests without removing the capability underneath. For the thesis, this warns that a correction can look effective and still fail under adversarial inputs. It is why the thesis asks for corrections tested on held-out triggers and for an enforcer outside the model's write access. Multi-agent risks bear on the thesis's "authority and adaptation" gap: tools, memory and other agents are routes around an intervention boundary.

## Week 6: Interpretability; formal methods and safe human–AI interaction

**Readings: interpretability (required)**

- *Vision*, section 1.2 ("Understanding Complex Information-Processing Systems"), Marr (1982; MIT Press reissue). The computational, algorithmic and implementation levels of explanation. The course links the [MIT Press chapter](https://direct.mit.edu/books/monograph/3299/chapter-standard/105803/The-Philosophy-and-the-Approach).
- [Toward Transparent AI: A Survey on Interpreting the Inner Structures of Deep Neural Networks](https://arxiv.org/abs/2207.13243), Räuker, Ho, Casper and Hadfield-Menell (SaTML 2023). A survey of methods that explain weights, neurons, subnetworks and representations.
- [Mechanistic Interpretability for AI Safety: A Review](https://arxiv.org/abs/2404.14082), Bereska and Gavves (TMLR 2024). Concepts, methods and safety relevance of reverse-engineering networks.
- [Zoom In: An Introduction to Circuits](https://distill.pub/2020/circuits/zoom-in/), Olah et al. (Distill, 2020). Features as directions, circuits as weighted connections between them, and the claim that they recur across models.

**Readings: formal methods and safe human–AI interaction (required)**

- [Role play with large language models](https://www.nature.com/articles/s41586-023-06647-8), Shanahan, McDonell and Reynolds (*Nature*, 2023). Dialogue agents are best described as role-playing characters, which guards against over-reading what they say.
- [ELEPHANT: Measuring and understanding social sycophancy in LLMs](https://arxiv.org/abs/2505.13995), Cheng et al. (2025). Models affirm users' self-image far more often than humans do, and preference data rewards it.
- [What is Deep Neural Network Verification and Why is it Important?](https://medium.com/sw-verification-testing/what-is-deep-neural-network-verification-and-why-is-it-important-75281738f5db), Chase (2021). A short introduction to proving properties of networks.
- [Clover: Closed-Loop Verifiable Code Generation](https://arxiv.org/abs/2310.17807), Sun et al. (2024). Checks that generated code, its documentation and its formal annotations agree with each other.

**Selected optional readings:** [Algorithms for Verifying Deep Neural Networks](https://arxiv.org/abs/1903.06758), Liu et al. (2021), a survey and tutorial of sound verification by reachability, optimization and search. [Trustworthy AI](https://arxiv.org/abs/2002.06276), Wing (2020), which frames formal methods for trustworthy AI. [Towards Monosemanticity](https://transformer-circuits.pub/2023/monosemantic-features), Bricken et al. (2023), and [Sparse Autoencoders Find Highly Interpretable Features](https://arxiv.org/abs/2309.08600), Cunningham et al. (ICLR 2024), on dictionary learning. [Discovering Latent Knowledge Without Supervision](https://arxiv.org/abs/2212.03827), Burns et al. (ICLR 2023), on reading truth directions without labels. [Language Models Don't Always Say What They Think](https://arxiv.org/abs/2305.04388), Turpin et al. (NeurIPS 2023), on unfaithful chain-of-thought.

**In the atlas:** the mechanistic interpretability cluster (`circuits`, `sae`, `superposition`, `linearrep`, `steering`, `illusions`, `auditing`), `elk`, `nnverify` (Neural-network verification & validation), `gsai`, `barrier`, `humanai`, `sycophancy`.

**Toward the thesis:** The thesis uses interpretability to *choose which variables to test*, not as proof. The evidence it accepts is behavioural and prospective. The interpretable-ML track (`interpretable-ml-to-thesis`) develops this, with belief-state geometry as the one readout checked against a known answer. Verification is the thesis's "independent enforcer" made formal. A proof holds only for the stated property over the stated input set, which is why the thesis says an enforcer limits consequences but does not make an action correct.

## Week 7: Electric sheep, or what is intelligence and does it want?; jailbreaks and adversarial robustness

**Readings (required)**

- [Optimal Policies Tend to Seek Power](https://arxiv.org/abs/1912.01683), Turner, Smith, Shah, Critch and Tadepalli (NeurIPS 2021). In many environments, most reward functions make it optimal to keep options open and avoid shutdown.
- [Orthogonality Thesis](https://www.lesswrong.com/tag/orthogonality-thesis) (LessWrong wiki). The claim that intelligence and final goals can vary independently.
- *Superintelligence*, chapter 7, "The superintelligent will", Bostrom (2014). The source of the orthogonality and instrumental-convergence arguments; the course provides the chapter.
- [The Bitter Lesson](http://www.incompleteideas.net/IncIdeas/BitterLesson.html), Sutton (2019). General methods that scale with compute have repeatedly beaten built-in human knowledge.
- [On the Dangers of Stochastic Parrots](https://dl.acm.org/doi/10.1145/3442188.3445922), Bender, Gebru, McMillan-Major and Shmitchell (FAccT 2021). The costs and risks of ever-larger language models, and a caution against reading understanding into fluent text.
- [Language Models as Agent Models](https://aclanthology.org/2022.findings-emnlp.423/), Andreas (Findings of EMNLP 2022). Language models can model the beliefs and goals of the agents that produced their text, without being agents themselves.

The adversarial-robustness guest lecture has no assigned readings.

**Selected optional readings:** [Goal Misgeneralization](https://arxiv.org/abs/2210.01790), Shah et al. (2022): a correct specification can still yield a policy that pursues the wrong goal off-distribution. [Deceptive Alignment](https://www.alignmentforum.org/s/r9tYkB2a8Fp4DN8yB/p/zthDPAjh9w6Ytbeks), Hubinger et al. (2019), from the risks-from-learned-optimization sequence.

**In the atlas:** `powerseeking` (Power-seeking & instrumental convergence), `corrigibility`, `offswitch`, `goalmisgen`, `bitterlesson`, `alignfaking`, `llm`.

**Toward the thesis:** This week holds the thesis's strongest objections. Corrigibility and power-seeking say that capable goal-directed agents may have reasons to resist correction. The thesis's answer, from the off-switch game, is that uncertainty about the objective, together with treating the human's choice as evidence, can make deference rational. That holds only under assumptions that misspecification can break (`cirlincorrig`). The Bitter Lesson attacks the thesis's bet on explicit structure. The thesis records it as the "compute and usefulness" falsifier: if the control machinery costs more than it helps, the thesis loses.

## Week 8: Attributing model behaviour at scale, evaluating AI systems and AI governance; towards a science of model behaviour

**Readings (required)**

- [On the Limitations of Compute Thresholds as a Governance Strategy](https://arxiv.org/abs/2407.05694), Hooker (2024). Training compute is an unreliable proxy for risk, so thresholds based on it can misfire.
- [Pitfalls of Evidence-Based AI Policy](https://arxiv.org/abs/2502.09618), Casper et al. (2025). Demanding strong evidence before regulating can entrench inaction when the evidence is itself hard to produce.
- [Measurement to Meaning: A Validity-Centered Framework for AI Evaluation](https://arxiv.org/abs/2505.10573), Salaudeen, Reuel et al. (2025). Psychometric validity applied to benchmarks: which claims does a score actually support?
- [A comprehensive review of Artificial Intelligence regulation](https://www.sciencedirect.com/science/article/pii/S2949948825000241), Cajueiro and Celestino (2025). A survey of regulatory approaches and their trade-offs between ethical principles and innovation.

The science-of-model-behaviour guest lecture has no assigned readings.

**Selected optional readings:** [Are Emergent Abilities of Large Language Models a Mirage?](https://arxiv.org/abs/2304.15004), Schaeffer, Miranda and Koyejo (NeurIPS 2023): sharp "emergence" can come from the metric. [Why Has Predicting Downstream Capabilities of Frontier AI Models with Scale Remained Elusive?](https://arxiv.org/abs/2406.04391), Schaeffer et al. (2024). [Red-Teaming for Generative AI: Silver Bullet or Security Theater?](https://arxiv.org/abs/2401.15897), Feffer et al. (2024). [Do the Rewards Justify the Means?](https://arxiv.org/abs/2304.03279), Pan et al. (ICML 2023): the MACHIAVELLI benchmark of reward versus ethical behaviour. [Black-Box Access is Insufficient for Rigorous AI Audits](https://arxiv.org/abs/2401.14446), Casper et al. (FAccT 2024). [Open Problems in Technical AI Governance](https://arxiv.org/abs/2407.14981), Reuel, Bucknall et al. (2024).

**In the atlas:** `evalvalidity` (AI evaluations: measurement validity & emergent abilities), `agentbench`, `llmjudge`, `scalinglaws`, `auditing`, `aigovern`, `redteam`.

**Toward the thesis:** The thesis can only be judged by measurements, so evaluation validity is its methodology. Its falsifiers call for independent evaluations of user utility, declared before the experiment and repeated across scales. Hooker's argument against compute thresholds parallels the thesis's insistence on measuring behaviour, not proxies. Casper et al.'s case for white-box audits matches the atlas's alignment-audit topic.

## Week 9: Scalable oversight: how do we supervise advanced AI?

**Readings (required)**

- [AI Safety via Debate](https://arxiv.org/abs/1805.00899), Irving, Christiano and Amodei (2018). Two AIs argue and a human judges, on the idea that refuting a lie is easier than lying.
- [Weak-to-Strong Generalization](https://arxiv.org/abs/2312.09390), Burns et al. (2023). Strong models trained on weak labels beat their supervisors but recover only part of their full ability.
- [Managing extreme AI risks amid rapid progress](https://arxiv.org/abs/2310.17688), Bengio, Hinton et al. (*Science*, 2024). A consensus statement calling for technical research and governance proportionate to extreme risks.
- [The Alignment Problem from a Deep Learning Perspective](https://arxiv.org/abs/2209.00626), Ngo, Chan and Mindermann (ICLR 2024). How situationally aware reward hacking, misgeneralized goals and power-seeking could arise from current training methods.

**Selected optional readings:** [Measuring Progress on Scalable Oversight for Large Language Models](https://arxiv.org/abs/2211.03540), Bowman et al. (2022): humans helped by an unreliable model beat both alone. [Biological Anchors: A Trick That Might Or Might Not Work](https://www.astralcodexten.com/p/biological-anchors-a-trick-that-might), Alexander (2022), on forecasting AI timelines from compute estimates. [Two Types of AI Existential Risk](https://arxiv.org/abs/2401.07836), Kasirzadeh (2024): decisive versus accumulative. [An Overview of Catastrophic AI Risks](https://arxiv.org/abs/2306.12001), Hendrycks, Mazeika and Woodside (2023). [Levels of AGI](https://arxiv.org/abs/2311.02462), Morris et al. (ICML 2024). [Scalable agent alignment via reward modeling](https://arxiv.org/abs/1811.07871), Leike et al. (2018).

**In the atlas:** `oversight`, `weaktostrong` (Weak-to-strong generalization), `elk`, `goalmisgen`, `powerseeking`, `scalinglaws` (for forecasting by scale), `multillm` (debate between agents).

**Toward the thesis:** Scalable oversight is where the course hands over to the thesis. The thesis's correction interface assumes people can tell whether a correction worked, and weak-to-strong experiments are a way to test that assumption now (`weaktostrong`–`alignmentthesis`, a hypothesis). The thesis's distinctive proposal is to treat weak supervision as *evidence about the objective*, as assistance games do, rather than as ground truth to imitate (`weaktostrong`–`assistance`, also a hypothesis). The flagship path ends here: assistance games, then the thesis's correction interface and its falsifiers.

## Weeks 10–11

Thanksgiving break, then optional final-project co-working sessions. Final projects range from experiments to literature reviews and policy recommendations. The atlas's test notes, at the bottom of each topic window, are written as small experiments that could seed one.

---

## Where to go next

- **Flagship path:** *From introductory AI safety to the alignment thesis* (`intro-ai-safety-to-thesis`) goes from `cs120` through `safetyproblems`, `rewardhack`, `preference`, `jailbreaks`, `evalvalidity`, `weaktostrong` and `assistance` to `alignmentthesis`.
- **The thesis itself:** read [the thesis](alignment-synthesis-thesis.pplx.md), then the paired supporting and opposing cases in the [reading track](alignment-thesis-reading-track.md) (paths `thesis-supporting-case` and `thesis-opposing-case`).
- **Related Stanford courses** listed by the course include human-centred NLP, machine learning from human preferences, trustworthy machine learning, decision-making under uncertainty, and validation of safety-critical systems. See the course page for the full list.

*Course content © its authors. This guide links to the readings and summarizes them in its own words. It is not affiliated with or endorsed by Stanford University.*
