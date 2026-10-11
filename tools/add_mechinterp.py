#!/usr/bin/env python3
"""Add (or refresh) the "Mechanistic interpretability" cluster and its learning data.

Usage (from the repo root):
    python3 tools/add_mechinterp.py [--root DIR]

All content lives in this file as data. The script is idempotent: it upserts by id,
so running it twice, or after another script has added nodes with the same ids
(`beliefgeometry`, `rudin2019`), leaves exactly one copy of each record.

What it writes:
  atlas.json           cluster, nodes and edges (minified, no trailing newline kept as found);
                       every cluster `count` is recomputed from the nodes
  learning/nodes.json  per-topic metadata (existing indentation / line endings preserved)
  learning/paths.json  the interpretable-ML path plus three in-cluster paths
  learning/background.json  one goal option
  learning/questions-7.json + learning/index.json registration
  README.md, index.html     topic / edge / cluster / path / item counts and the per-type table,
                            recomputed from the data (nothing hardcoded)
  app.js               the atlas.json cache-busting query, set to a hash of atlas.json

Edge ids are deterministic: "rel-" + sha1("a|b|relation")[:8].
Python 3 standard library only.
"""
import argparse
import hashlib
import json
import os
import re
import sys

CLUSTER = {
    "name": "Mechanistic interpretability",
    "color": "#9aa5ad",
    "center": [-398.0, -314.0, -55.9],
}
CLUSTER_NOTE = (
    "The mechanistic interpretability cluster is placed apart from the others on purpose: "
    "it is an adjacent methodology the atlas draws on, not the atlas's promoted route."
)
SL = "Statistical learning"
MI = CLUSTER["name"]


def L(label, url):
    return "[%s](%s)" % (label, url)


# ---- sources -----------------------------------------------------------------
U = {
    "zoomin": "https://distill.pub/2020/circuits/zoom-in/",
    "framework": "https://transformer-circuits.pub/2021/framework/index.html",
    "toy": "https://transformer-circuits.pub/2022/toy_model/index.html",
    "toyarx": "https://arxiv.org/abs/2209.10652",
    "induction": "https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html",
    "inductionarx": "https://arxiv.org/abs/2209.11895",
    "ioi": "https://arxiv.org/abs/2211.00593",
    "rome": "https://arxiv.org/abs/2202.05262",
    "bestpractice": "https://arxiv.org/abs/2309.16042",
    "howto": "https://arxiv.org/abs/2404.15255",
    "hase": "https://arxiv.org/abs/2301.04213",
    "atpblog": "https://www.neelnanda.io/mechanistic-interpretability/attribution-patching",
    "eap": "https://arxiv.org/abs/2310.10348",
    "atpstar": "https://arxiv.org/abs/2403.00745",
    "scrub": "https://www.alignmentforum.org/posts/JvZhhzycHu2Yd57RN/causal-scrubbing-a-method-for-rigorously-testing",
    "ca2021": "https://arxiv.org/abs/2106.02997",
    "ca2023": "https://arxiv.org/abs/2301.04709",
    "das": "https://arxiv.org/abs/2303.02536",
    "acdc": "https://arxiv.org/abs/2304.14997",
    "mono": "https://transformer-circuits.pub/2023/monosemantic-features/index.html",
    "cunningham": "https://arxiv.org/abs/2309.08600",
    "seeds": "https://arxiv.org/abs/2501.16615",
    "scaling": "https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html",
    "transcoders": "https://arxiv.org/abs/2406.11944",
    "crosscoders": "https://transformer-circuits.pub/2024/crosscoders/index.html",
    "crossart": "https://arxiv.org/abs/2504.02922",
    "methods": "https://transformer-circuits.pub/2025/attribution-graphs/methods.html",
    "biology": "https://transformer-circuits.pub/2025/attribution-graphs/biology.html",
    "sfc": "https://arxiv.org/abs/2403.19647",
    "lrh": "https://arxiv.org/abs/2311.03658",
    "notlinear": "https://arxiv.org/abs/2405.14860",
    "control": "https://arxiv.org/abs/1909.03368",
    "belinkov": "https://arxiv.org/abs/2102.12452",
    "logitlens": "https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens",
    "tuned": "https://arxiv.org/abs/2303.08112",
    "actadd": "https://arxiv.org/abs/2308.10248",
    "repe": "https://arxiv.org/abs/2310.01405",
    "refusal": "https://arxiv.org/abs/2406.11717",
    "axbench": "https://arxiv.org/abs/2501.17148",
    "grok": "https://arxiv.org/abs/2301.05217",
    "othello": "https://arxiv.org/abs/2210.13382",
    "othellolin": "https://arxiv.org/abs/2309.00941",
    "belief": "https://arxiv.org/abs/2405.15943",
    "constrained": "https://arxiv.org/abs/2502.01954",
    "bolukbasi": "https://arxiv.org/abs/2104.07143",
    "subspace": "https://arxiv.org/abs/2311.17030",
    "gdm": "https://www.alignmentforum.org/posts/4uXCAJNuPKtKBsi28/sae-progress-update-2-draft",
    "sparseprobe": "https://arxiv.org/abs/2502.16681",
    "saebench": "https://arxiv.org/abs/2503.09532",
    "audit": "https://arxiv.org/abs/2503.10965",
    "osdt": "https://arxiv.org/abs/1904.12847",
    "gosdt": "https://arxiv.org/abs/2006.08690",
    "guess": "https://arxiv.org/abs/2112.00798",
    "rashomon": "https://arxiv.org/abs/1908.01755",
    "treefarms": "https://arxiv.org/abs/2209.08040",
    "rudin": "https://arxiv.org/abs/1811.10154",
    "llc": "https://arxiv.org/abs/2308.12108",
    "stagewise": "https://arxiv.org/abs/2402.02364",
    "rllc": "https://arxiv.org/abs/2410.02984",
    "suscept": "https://arxiv.org/abs/2504.18274",
    "grokslt": "https://arxiv.org/abs/2603.01192",
    "bifstage": "https://arxiv.org/abs/2510.12071",
    "probeattr": "https://www.goodfire.com/research/probe-based-data-attribution",
    "cmech": "https://arxiv.org/pdf/cs/0001027v1",
}


def node(id, name, group, authors, year, text, url, handle, boundary, test, p, sourceLabel=None):
    return {"id": id, "name": name, "group": group, "authors": authors, "year": year, "text": text,
            "url": url, "sourceLabel": sourceLabel, "kind": "topic", "guide": None,
            "handle": handle, "boundary": boundary, "test": test, "p": p}


NODES = [
    node("circuits", "Circuits framework", MI,
         "Olah · Cammarata · Schubert · Goh · Petrov · Carter", "2020",
         "Treat features as directions and circuits as weighted connections between them, then read algorithms off the weights. The evidence is mainly case studies in one vision model.",
         U["zoomin"],
         "Three claims organize the programme: features are the fundamental unit, features connect through weights into circuits, and analogous features and circuits recur across models. Curve detectors in InceptionV1 are a worked example (%s)." % L("Olah et al.", U["zoomin"]),
         "The authors call the three claims speculative. Case studies in one vision model do not show that every behaviour decomposes into readable circuits, and polysemantic neurons already break the neuron-as-feature reading (%s)." % L("Olah et al.", U["zoomin"]),
         "Predeclare a circuit hypothesis from the weights alone, then check it on held-out inputs and by ablation; report the fraction of the behaviour's loss the circuit accounts for.",
         [-443.3, -337.2, -20.5]),
    node("superposition", "Superposition", MI,
         "Elhage · Hume · Olsson · Schiefer · Henighan et al.", "2022",
         "Small ReLU models store more sparse features than they have dimensions, as nearly orthogonal directions, trading interference for capacity.",
         U["toy"],
         "Whether a toy model stores features in superposition depends on feature sparsity and importance, with phase changes between regimes and features arranged in regular geometric configurations; the authors relate it to compressed sensing (%s)." % L("Elhage et al.", U["toy"]),
         "These are toy models with synthetic sparse features and a known ground truth. They motivate, but do not show, that large models store features this way, or that every feature is a one-dimensional direction (%s, %s)." % (L("Elhage et al.", U["toyarx"]), L("Engels et al.", U["notlinear"])),
         "Plant known sparse features in a small transformer's training data and measure how well dictionary methods recover them as sparsity and dictionary size vary.",
         [-355.3, -354.6, -75.1]),
    node("inductionheads", "Induction heads", MI,
         "Olsson · Elhage · Nanda · Joseph · DasSarma et al.", "2022",
         "A previous-token head composes with an induction head to complete [A][B] … [A] → [B]. Their formation coincides with a phase change in in-context learning.",
         U["induction"],
         "The two-head circuit is derived for attention-only transformers (%s). Six lines of evidence tie induction heads to most in-context learning in small models, including a bump in the loss curve when they form (%s)." % (L("Elhage et al.", U["framework"]), L("Olsson et al.", U["induction"])),
         "The authors grade their own evidence: interventional for small attention-only models, mostly correlational for large models with MLPs. That induction heads drive most in-context learning at scale remains an argued hypothesis (%s)." % L("Olsson et al.", U["inductionarx"]),
         "Predict the step at which induction heads form from early checkpoints, and compare it with the LLC-detected stage boundary in the same run.",
         [-450.6, -304.8, -72.7]),
    node("ioi", "Indirect-object identification circuit", MI,
         "Wang · Variengien · Conmy · Shlegeris · Steinhardt", "2022",
         "A circuit of 26 attention heads in 7 classes that lets GPT-2 small complete “When Mary and John went to the store, John gave a drink to” with “Mary”.",
         U["ioi"],
         "Path patching isolates name-mover, S-inhibition, induction and duplicate-token heads, among others; the explanation is scored for faithfulness, completeness and minimality (%s)." % L("Wang et al.", U["ioi"]),
         "One task in one small model. The authors report remaining gaps against their own criteria, and backup name-mover heads take over when the main ones are ablated, so ablation effects understate a component's role (%s)." % L("Wang et al.", U["ioi"]),
         "Run the same circuit-finding protocol blind on a second GPT-2-small seed and report how many heads and classes replicate.",
         [-415.3, -354.5, -55.5]),
    node("actpatching", "Activation & causal patching", MI,
         "Meng · Bau · Andonian · Belinkov; Zhang · Nanda; Heimersheim · Nanda", "2022–2024",
         "Copy an activation from one run into another and measure how the output changes. It shows where information that matters for one contrast flows.",
         U["howto"],
         "Causal tracing localizes factual recall to mid-layer MLPs in GPT models and motivated rank-one editing (%s). Results depend on the corruption method and the metric, so best-practice guides fix both in advance (%s, %s)." % (L("Meng et al.", U["rome"]), L("Zhang and Nanda", U["bestpractice"]), L("Heimersheim and Nanda", U["howto"])),
         "A patching effect shows that a component matters for one contrast on one input distribution, not what it computes. Causal-tracing localization did not predict which layer was best to edit (%s)." % L("Hase et al.", U["hase"]),
         "Report effects under at least two corruption schemes and two metrics across seeds; treat a component as implicated only if its sign and rank hold across them.",
         [-441.6, -295.2, -16.1]),
    node("attrpatching", "Attribution patching", MI,
         "Nanda; Syed · Rager · Conmy; Kramár · Lieberum · Shah · Nanda", "2023 / 2024",
         "Estimate every activation-patching effect at once with a first-order Taylor expansion: two forward passes and one backward pass.",
         U["atpblog"],
         "The estimate is (a_clean − a_corrupt) · ∂m/∂a (%s). Edge attribution patching recovered reference circuits better than ACDC on average (%s); AtP* repairs two false-negative failure modes (%s)." % (L("Nanda", U["atpblog"]), L("Syed, Rager and Conmy", U["eap"]), L("Kramár et al.", U["atpstar"])),
         "It is a linearization. Saturated attention and cancelling paths give false negatives, so a small estimate is not evidence that a component is unimportant (%s)." % L("Kramár et al.", U["atpstar"]),
         "On a model with a known circuit, compare attribution-patching rankings with exhaustive activation patching and report the false-negative rate among the true top-k components.",
         [-429.5, -326.3, -100.5]),
    node("causalscrubbing", "Causal scrubbing", MI,
         "Chan · Garriga-Alonso · Goldowsky-Dill · Greenblatt et al. (Redwood Research)", "2022",
         "Write a hypothesis as a map from an interpretable computation onto the network, then resample every activation the hypothesis says does not matter and measure how much behaviour survives.",
         U["scrub"],
         "Tests are derived mechanically from the hypothesis as behaviour-preserving resampling ablations, which lets hypothesis checking run inside the research loop; worked on induction and a parentheses-balancing model (%s)." % L("Chan et al.", U["scrub"]),
         "The score is relative to one dataset, metric and hypothesis format. Two incompatible hypotheses can both score well, so passing does not identify the mechanism, and the test says nothing about inputs outside the dataset (%s)." % L("Chan et al.", U["scrub"]),
         "Score two incompatible hypotheses for the same behaviour; if both recover most of the loss, record the test as non-identifying for that task.",
         [-439.4, -267.9, -58.9]),
    node("causalabstraction", "Causal abstraction & interchange interventions", MI,
         "Geiger · Lu · Icard · Potts; Geiger · Wu · Potts · Icard · Goodman; Geiger et al.", "2021–2023",
         "Ask whether a high-level causal model abstracts the network: interventions on aligned variables must have matching effects. Distributed alignment search learns the alignment.",
         U["ca2023"],
         "Interchange interventions set an internal representation to its value on another input and check the high-level model's counterfactual prediction (%s). The framework is offered as a common foundation for patching, circuit analysis and causal scrubbing (%s); DAS learns rotated subspaces to align (%s)." % (L("Geiger et al., 2021", U["ca2021"]), L("Geiger et al., 2023", U["ca2023"]), L("Geiger et al.", U["das"])),
         "Learned subspace alignments can succeed through a dormant pathway the model does not normally use, so interchange accuracy alone overstates faithfulness (%s). Abstraction is relative to the chosen high-level model and input distribution." % L("Makelov, Lange and Nanda", U["subspace"]),
         "Fit DAS on one input distribution, then report interchange accuracy on shifted inputs and for a sham high-level model with permuted variables.",
         [-399.1, -360.5, -87.1]),
    node("acdc", "Automated circuit discovery (ACDC)", MI,
         "Conmy · Mavor-Parker · Lynch · Heimersheim · Garriga-Alonso", "2023",
         "Automate one step of circuit work: patch edges of the computational graph and prune those whose removal changes a task metric by less than a threshold.",
         U["acdc"],
         "ACDC rediscovered all five component types of a hand-found greater-than circuit in GPT-2 small, selecting 68 of 32,000 edges, all previously found by hand (%s)." % L("Conmy et al.", U["acdc"]),
         "The output depends on the threshold, metric and corrupted distribution; it finds edges that matter for one contrast, not what they compute. A gradient-based alternative matched or beat it at far lower cost (%s)." % L("Syed, Rager and Conmy", U["eap"]),
         "Run several thresholds and corruption distributions, report circuit overlap, and compare with random subgraphs of equal size.",
         [-403.4, -344.7, -24.8]),
    node("sae", "Sparse autoencoders & dictionary learning", MI,
         "Bricken · Templeton · Batson · Chen · Jermyn et al.; Cunningham · Ewart · Riggs · Huben · Sharkey", "2023",
         "Learn an overcomplete, sparsely active dictionary of activation directions, so that each latent fires on a narrower and often nameable set of inputs than a neuron does.",
         U["mono"],
         "Dictionaries trained on a one-layer transformer's MLP activations give latents judged more monosemantic than neurons (%s); on Pythia models, SAE latents score as more interpretable than PCA or ICA directions and localize counterfactual behaviour more precisely (%s)." % (L("Bricken et al.", U["mono"]), L("Cunningham et al.", U["cunningham"])),
         "The objective trades reconstruction error against sparsity; it does not identify the model's “true” features. SAEs trained on the same data with different seeds share only part of their latents (%s), and the reconstruction error is part of what is left unexplained." % L("Paulo and Belrose", U["seeds"]),
         "Report seed stability, the loss increase when activations are replaced by reconstructions, and a dense linear probe's accuracy on the same labels.",
         [-398.9, -274.3, -74.9]),
    node("scalingmono", "Scaling monosemanticity", MI,
         "Templeton · Conerly · Marcus · Lindsey · Bricken et al.", "2024",
         "SAEs with up to 34 million latents on a production model's residual stream give abstract, multilingual and multimodal features, including safety-relevant ones.",
         U["scaling"],
         "A Golden Gate Bridge feature fires across languages and on images, and clamping it to ten times its maximum activation makes the model talk about, and identify as, the bridge (%s)." % L("Templeton et al.", U["scaling"]),
         "The authors caution that a safety-relevant feature's existence does not show the model uses it to act: knowing about lies differs from lying. The features found cover only part of what the model represents (%s)." % L("Templeton et al.", U["scaling"]),
         "For a safety-relevant feature, compare feature-based detection with a dense linear probe and a prompted classifier on held-out, distribution-shifted prompts.",
         [-372.4, -326.1, -7.6]),
    node("transcoders", "Transcoders & crosscoders", MI,
         "Dunefsky · Chlenski · Nanda; Lindsey · Templeton · Marcus · Conerly · Batson · Olah", "2024",
         "Sparse dictionaries trained to map a sublayer's input to its output (transcoders), or to read and write several layers or models at once (crosscoders).",
         U["transcoders"],
         "A transcoder replaces an MLP with a wider, sparse MLP, so feature-to-feature attributions split into input-dependent and input-invariant terms (%s). Crosscoders share features across layers and support model diffing between base and fine-tuned models (%s)." % (L("Dunefsky, Chlenski and Nanda", U["transcoders"]), L("Lindsey et al.", U["crosscoders"])),
         "A replacement layer is a different model, and its error is part of the explanation. Crosscoder diffs can report spurious model-specific latents caused by the sparsity penalty (%s)." % L("Minder et al.", U["crossart"]),
         "Report the loss when every MLP is replaced at once, and check model-diff latents against a control diff between two seeds of the same base model.",
         [-385.8, -331.9, -114.0]),
    node("attributiongraphs", "Attribution graphs & circuit tracing", MI,
         "Ameisen · Lindsey · Pearce · Gurnee · Turner · Chen · Citro et al.", "2025",
         "Replace the MLPs with a cross-layer transcoder, freeze attention, and compute linear feature-to-feature attributions for one prompt; check the hypotheses by perturbation.",
         U["methods"],
         "Error nodes carry what the replacement misses, and frozen attention makes every edge linear for that prompt (%s). Applied to Claude 3.5 Haiku, including multi-step reasoning, planning in poems and a model fine-tuned with a hidden goal (%s)." % (L("Ameisen et al.", U["methods"]), L("Lindsey et al.", U["biology"])),
         "Each graph explains one prompt through a replacement model and does not explain why attention attends where it does. The authors report satisfying insight for about a quarter of the prompts they tried (%s, %s)." % (L("Ameisen et al.", U["methods"]), L("Lindsey et al.", U["biology"])),
         "Before perturbing, predict the effect of suppressing the top three graph features on held-out paraphrases, and score the predictions against a random-feature control.",
         [-406.6, -275.6, -36.3]),
    node("linearrep", "Linear representation hypothesis", MI,
         "Park · Choe · Veitch", "2023",
         "Concepts correspond to directions in representation space. A causal inner product makes the probing and steering senses of “direction” agree.",
         U["lrh"],
         "Using counterfactual word pairs, the authors formalize linear representation in the embedding and unembedding spaces and give a causal inner product under which the two coincide, tying probes to steering vectors in LLaMA-2 (%s)." % L("Park, Choe and Veitch", U["lrh"]),
         "Some features are irreducibly multi-dimensional, such as circular day-of-week and month representations (%s). The hypothesis concerns how a concept is encoded, not whether downstream computation uses that direction." % L("Engels et al.", U["notlinear"]),
         "For each concept, compare probe accuracy along the estimated direction, the steering effect along the same direction and a nonlinear probe; flag concepts where they disagree.",
         [-369.4, -265.9, -55.9]),
    node("controlprobes", "Probing with control tasks", MI,
         "Hewitt · Liang; Belinkov", "2019 / 2021",
         "A probe's accuracy mixes what the representation encodes with what the probe can learn on its own. Control tasks with random labels measure the latter.",
         U["control"],
         "Selectivity is task accuracy minus control-task accuracy. Popular probes on ELMo were not selective, and a later layer with slightly lower accuracy was far more selective (%s). A review separates correlational probing from causal tests (%s)." % (L("Hewitt and Liang", U["control"]), L("Belinkov", U["belinkov"])),
         "Decodability does not show use: a probe can read out a property the network never relies on (%s)." % L("Belinkov", U["belinkov"]),
         "Report selectivity alongside accuracy, then test use by intervening along the probe direction and measuring the change in behaviour.",
         [-342.4, -331.5, -34.7]),
    node("tunedlens", "Logit lens & tuned lens", MI,
         "Belrose · Ostrovsky · McKinney · Furman · Smith et al.; logit lens (2020 post)", "2020 / 2023",
         "Decode intermediate residual-stream states into next-token distributions, with the final unembedding (logit lens) or a learned affine map per layer (tuned lens).",
         U["tuned"],
         "The logit lens applies the unembedding directly to intermediate layers (%s). The tuned lens trains one affine translator per layer and is reported to be more predictive, reliable and less biased than the logit lens (%s)." % (L("logit lens", U["logitlens"]), L("Belrose et al.", U["tuned"])),
         "A lens shows what a decoder can read from a layer, which is a probing result, not the computation the model performs; a trained translator can add structure of its own (%s)." % L("Belinkov", U["belinkov"]),
         "Compare lens predictions with activation-patching effects at the same layer; disagreement marks information that is decodable but unused.",
         [-379.0, -298.7, -92.5]),
    node("steering", "Activation steering & representation engineering", MI,
         "Turner · Thiergart · Leech · Udell · Vazquez et al.; Zou · Phan · Chen et al.; Arditi et al.", "2023 / 2024",
         "Add or remove a direction in the residual stream at inference to shift behaviour, with the direction found from contrasting prompts.",
         U["repe"],
         "Activation addition builds a steering vector from a pair of prompts (%s); representation engineering reads and controls concepts such as honesty with population-level directions (%s); ablating one direction removes refusal in open chat models (%s)." % (L("Turner et al.", U["actadd"]), L("Zou et al.", U["repe"]), L("Arditi et al.", U["refusal"])),
         "On AxBench, prompting beat every representation-based steering method, followed by fine-tuning, and SAE-based steering was not competitive (%s). A steering effect shows that a direction influences output, not that it is the concept's natural encoding." % L("Wu et al.", U["axbench"]),
         "Measure steering effect and capability loss together on held-out tasks, against prompting and LoRA baselines at matched compute.",
         [-375.7, -279.6, -17.8]),
    node("grokkingmech", "Mechanistic account of grokking", MI,
         "Nanda · Chan · Lieberum · Smith · Steinhardt", "2023",
         "A one-layer transformer that groks modular addition computes with discrete Fourier components and trigonometric identities; progress measures show the circuit forming before test accuracy jumps.",
         U["grok"],
         "Reverse-engineering the weights gives the Fourier multiplication algorithm; progress measures built on it split training into memorization, circuit formation and cleanup (%s)." % L("Nanda et al.", U["grok"]),
         "One algorithmic task in a small model, where the algorithm could be guessed and checked. The progress measures are derived from the reverse-engineered mechanism, so they are not a general method for predicting emergence in advance (%s)." % L("Nanda et al.", U["grok"]),
         "Compare the lead time of the progress measures over test accuracy with that of LLC and weight-norm trajectories in the same runs.",
         [-435.6, -336.0, -62.4]),
    node("othello", "Emergent world representations (Othello-GPT)", MI,
         "Li · Hopkins · Bau · Viégas · Pfister · Wattenberg; Nanda · Lee · Wattenberg", "2022 / 2023",
         "A GPT trained only on legal Othello move sequences encodes the board state, and editing the decoded state changes its move predictions.",
         U["othello"],
         "Nonlinear probes recover the board, and interventions on the probed state change predictions as expected (%s). In a mine/theirs basis instead of black/white, the board is linearly decodable and linearly steerable (%s)." % (L("Li et al.", U["othello"]), L("Nanda, Lee and Wattenberg", U["othellolin"])),
         "A synthetic game with a fully specified, fully observed state. Recovering a known state in a toy domain does not show that models trained on natural text keep coherent world states, or which variables they track (%s)." % L("Li et al.", U["othello"]),
         "Train on a game with hidden information and test whether probes recover the belief over hidden pieces, not only the visible board.",
         [-359.0, -314.0, -76.2]),
    node("beliefgeometry", "Belief-state geometry in the residual stream", MI,
         "Shai · Marzen · Teixeira · Gietelink Oldenziel · Riechers", "2024",
         "Transformers trained on data from hidden Markov processes linearly represent the Bayesian belief over hidden states, including fractal belief geometries, as computational mechanics predicts.",
         U["belief"],
         "The target geometry is computed in advance from the generator's mixed-state presentation; a linear map from residual activations recovers it, sometimes spread across layers, and the beliefs carry information about the whole future, not only the next token (%s). Follow-up work derives how attention implements a constrained version of the update (%s)." % (L("Shai et al.", U["belief"]), L("Piotrowski et al.", U["constrained"])),
         "Small transformers on stationary processes with known, finite generators and no actions. The result is a linear readout of belief, not proof the model uses it downstream, and it does not transfer automatically to natural language, whose generator is unknown (%s)." % L("Shai et al.", U["belief"]),
         "Add actions and regime switches to the generator; test whether the readout tracks the action-conditioned belief and stays linear after the switch.",
         [-341.0, -292.7, -43.9]),
    node("illusions", "Interpretability illusions", MI,
         "Bolukbasi · Pearce · Yuan · Coenen · Reif et al.; Makelov · Lange · Nanda", "2021 / 2023",
         "Explanations that look convincing on one dataset or under one intervention can be artefacts of the dataset, the probe or a dormant pathway.",
         U["subspace"],
         "In BERT, the same neuron or direction gets different, individually convincing interpretations depending on the dataset used to find its top activations (%s). Subspace patching can change output by activating a dormant parallel pathway, so the patched subspace is not where the feature lives (%s)." % (L("Bolukbasi et al.", U["bolukbasi"]), L("Makelov, Lange and Nanda", U["subspace"])),
         "These show that specific methods can mislead, not that they always do; the same paper shows a success case on IOI, where prior circuit analysis locates the feature (%s)." % L("Makelov, Lange and Nanda", U["subspace"]),
         "Pair every interpretation with a held-out dataset check and an intervention whose effect is predicted before it is run.",
         [-403.9, -308.1, -12.7]),
    node("saebaselines", "SAE negative results & simple baselines", MI,
         "Smith · Rajamanoharan · Conmy et al. (GDM); Kantamneni · Engels · Rajamanoharan · Tegmark · Nanda; Karvonen et al.", "2025",
         "On downstream tasks with strong baselines, SAE latents have so far not beaten dense linear probes, and SAE proxy metrics track practical usefulness poorly.",
         U["gdm"],
         "For out-of-distribution harmful-intent detection, dense linear probes performed nearly perfectly while SAE probes underperformed, and the team deprioritized fundamental SAE research (%s). Across sparse-probing regimes SAEs did not consistently beat baselines (%s), and gains on proxy metrics did not reliably translate into practical performance (%s)." % (L("GDM interpretability team", U["gdm"]), L("Kantamneni et al.", U["sparseprobe"]), L("Karvonen et al.", U["saebench"])),
         "Negative results on the tasks tested, not proof that SAEs are useless; the same work found SAEs helpful for debugging datasets and generating hypotheses (%s)." % L("GDM interpretability team", U["gdm"]),
         "For any proposed SAE use, run dense-probe and prompting baselines first, with matched labels and compute, and report the margin.",
         [-419.8, -287.0, -107.3]),
    node("auditing", "Alignment audits for hidden objectives", MI,
         "Marks · Treutlein · Bricken · Lindsey · Marcus et al.", "2025",
         "A model trained to exploit known reward-model errors is the target of a blind auditing game; teams use interpretability, behavioural attacks and training-data analysis.",
         U["audit"],
         "Three of four blind teams uncovered the hidden objective, using techniques including SAEs, behavioural attacks and training-data analysis; an unblinded follow-up compares eight auditing techniques (%s)." % L("Marks et al.", U["audit"]),
         "One planted objective, taught partly through documents the auditors could search. Success in a game with a known answer does not show that audits find objectives nobody planted (%s)." % L("Marks et al.", U["audit"]),
         "Repeat with an objective instilled without descriptive training documents, and compare interpretability-based audits with behaviour-only audits given equal time.",
         [-377.1, -363.9, -41.3]),
    # ---- inherently interpretable models (Statistical learning, near `trees`) ----
    node("sparsetrees", "Optimal sparse decision trees", SL,
         "Hu · Rudin · Seltzer; Lin · Zhong · Hu · Rudin · Seltzer; McTavish · Zhong · Achermann · Karimalis · Chen et al.", "2019–2022",
         "Find the tree that minimizes loss plus a per-leaf penalty, with a certificate of optimality, by branch-and-bound and dynamic programming instead of greedy splitting.",
         U["gosdt"],
         "OSDT minimizes misclassification plus λ × leaves, with analytical bounds that prune the search (%s); GOSDT extends this to other objectives and continuous features with dynamic programming (%s); guessing thresholds from a reference ensemble cuts runtime further (%s)." % (L("Hu, Rudin and Seltzer", U["osdt"]), L("Lin et al.", U["gosdt"]), L("McTavish et al.", U["guess"])),
         "Optimality is relative to the objective, the penalty and, with threshold guessing, the guessed thresholds (%s). A small optimal tree is inspectable; that does not make it causal or correct under shift." % L("McTavish et al.", U["guess"]),
         "Compare optimal sparse trees with boosted ensembles under covariate shift, and count the leaves needed to reach the ensemble's held-out accuracy.",
         [133.2, 192.9, 66.5]),
    node("rashomon", "Rashomon sets", SL,
         "Semenova · Rudin · Parr; Xin · Zhong · Chen · Takagi · Seltzer · Rudin", "2019 / 2022",
         "The set of models in a class whose loss is within ε of the best. When it is large, accurate simple models are likely to exist, and many different models fit equally well.",
         U["rashomon"],
         "The Rashomon ratio, the volume of the ε-near-optimal set over the volume of the hypothesis space, is proposed as a gauge of simplicity (%s). TreeFARMS enumerates the whole Rashomon set of sparse trees and computes variable importance over it (%s)." % (L("Semenova, Rudin and Parr", U["rashomon"]), L("Xin et al.", U["treefarms"])),
         "That many real data sets have large Rashomon sets is the authors' hypothesis, supported by experiments rather than proved. A large set also means one model's explanation is not unique (%s)." % L("Semenova, Rudin and Parr", U["rashomon"]),
         "Report variable importance across the Rashomon set rather than for one model, and check whether conclusions that hold across the set survive a shift.",
         [150.8, 164.6, 53.9]),
    node("rudin2019", "Interpretable models instead of post-hoc explanations", SL,
         "Cynthia Rudin", "2019",
         "For high-stakes decisions, build inherently interpretable models instead of explaining black boxes, because a post-hoc explanation cannot be fully faithful to the model it explains.",
         U["rudin"],
         "The paper argues that an explanation faithful everywhere would be the model itself, that on structured data with meaningful features simple models often match complex ones, and that the accuracy–interpretability trade-off is often assumed rather than measured (%s)." % L("Rudin", U["rudin"]),
         "A position paper. Its accuracy argument concerns structured data with meaningful features; it does not show that interpretable models match networks on raw text, images or long-horizon agent behaviour (%s)." % L("Rudin", U["rudin"]),
         "For each high-stakes task, search for an interpretable model first and record the accuracy gap before accepting a black box with explanations.",
         [158.3, 182.0, 92.3]),
]

# Nodes this script owns outright. `beliefgeometry` and `rudin2019` may also be added by
# tools/add_thesis_track.py; they are upserted, but edges others attach to them are kept.
SHARED = {"beliefgeometry", "rudin2019"}

# ---- edges: (a, b, type, relation, text, url) ---------------------------------
EDGES = [
    ("circuits", "inductionheads", "established", "circuit analysis applied to attention",
     "Induction heads are a two-head circuit derived by reading attention weights, the circuits programme carried from vision models to transformers.", U["framework"]),
    ("circuits", "superposition", "established", "polysemantic neurons motivate superposition",
     "Polysemantic neurons break the neuron-as-feature reading of circuits; superposition is the proposed explanation, studied in toy models.", U["toy"]),
    ("superposition", "sae", "established", "dictionary learning to undo superposition",
     "If features are stored in superposition, an overcomplete sparse dictionary may recover them; SAEs are trained for that purpose.", U["mono"]),
    ("sae", "scalingmono", "established", "scaled to a production model",
     "The same sparse-dictionary method, scaled to tens of millions of latents on a frontier model's residual stream.", U["scaling"]),
    ("sae", "transcoders", "established", "sparse dictionary variant",
     "Transcoders keep the sparse-dictionary objective but predict a sublayer's output from its input, making feature-to-feature attribution tractable.", U["transcoders"]),
    ("transcoders", "attributiongraphs", "established", "replacement model for circuit tracing",
     "Attribution graphs are computed through a cross-layer transcoder that replaces the MLPs; its error nodes record what it misses.", U["methods"]),
    ("inductionheads", "ioi", "established", "circuit contains induction heads",
     "The IOI circuit includes induction and duplicate-token heads that detect the repeated name.", U["ioi"]),
    ("ioi", "actpatching", "established", "found by path patching",
     "The IOI circuit was isolated with path patching, a variant of activation patching that restricts the effect to chosen paths.", U["ioi"]),
    ("actpatching", "attrpatching", "exact", "first-order estimate; equal when the metric is affine in the patched activation",
     "Attribution patching estimates the patching effect as (a_clean − a_corrupt) · ∂m/∂a. The two agree exactly only when the metric is affine in that activation along the patch; otherwise the estimate can miss effects, e.g. through saturated attention.", U["atpstar"]),
    ("attrpatching", "acdc", "empirical", "edge attribution patching beats ACDC at circuit recovery",
     "Averaged over tasks, pruning edges by attribution-patching scores recovered reference circuits with higher AUC than ACDC, at the cost of two forward passes and one backward pass.", U["eap"]),
    ("actpatching", "acdc", "established", "automates iterative patching",
     "ACDC automates the patch-and-prune step that manual circuit work did with activation patching.", U["acdc"]),
    ("actpatching", "causalscrubbing", "established", "resampling ablation driven by a hypothesis",
     "Causal scrubbing generalizes patching: the hypothesis determines which activations are resampled from other inputs.", U["scrub"]),
    ("causalscrubbing", "causalabstraction", "established", "both test a hypothesized high-level computation",
     "Causal abstraction is proposed as a common foundation under which causal scrubbing and patching are instances of interchange-intervention tests.", U["ca2023"]),
    ("actpatching", "causalabstraction", "established", "interchange intervention",
     "An interchange intervention sets an internal variable to its value on another input, which is activation patching read against a high-level causal model.", U["ca2023"]),
    ("causalabstraction", "illusions", "empirical", "subspace-patching illusion",
     "Learned subspace interventions can reproduce a feature's effect through a dormant pathway, so interchange accuracy can overstate faithfulness.", U["subspace"]),
    ("controlprobes", "othello", "empirical", "probes plus interventions",
     "Othello-GPT goes beyond probe accuracy: editing the probed board state changes move predictions, the causal check that probing alone lacks.", U["othello"]),
    ("othello", "linearrep", "empirical", "board state is linear in the right basis",
     "Relative to the player to move (mine/theirs), the board is linearly decodable and steerable, though black/white linear probes fail.", U["othellolin"]),
    ("linearrep", "steering", "established", "causal inner product unifies probe and steering directions",
     "Under the causal inner product, the probing direction and the steering direction for a concept coincide.", U["lrh"]),
    ("linearrep", "superposition", "established", "features as directions",
     "Superposition assumes features are linear directions and asks how more of them than dimensions can be stored.", U["toy"]),
    ("scalingmono", "steering", "empirical", "feature clamping steers behaviour",
     "Clamping a dictionary feature to a high value changes behaviour in the direction its label predicts, e.g. the Golden Gate Bridge feature.", U["scaling"]),
    ("steering", "saebaselines", "empirical", "prompting beats representation steering",
     "On AxBench, prompting outperformed every representation-based steering method and SAEs were not competitive for steering or concept detection.", U["axbench"]),
    ("sae", "saebaselines", "empirical", "SAE probes vs. dense baselines",
     "Under data scarcity, class imbalance, label noise and covariate shift, SAE-based probes did not consistently beat simple baselines.", U["sparseprobe"]),
    ("tunedlens", "controlprobes", "established", "a lens is a per-layer probe",
     "The tuned lens trains an affine probe per layer into the vocabulary, so the probe-capacity caveats of control tasks apply to it.", U["tuned"]),
    ("grokkingmech", "circuits", "established", "full reverse-engineering of a small model",
     "The modular-addition network is reverse-engineered completely from its weights, a rare case where the circuits programme succeeds end to end.", U["grok"]),
    ("grokkingmech", "inductionheads", "established", "abrupt change, gradual mechanism",
     "Both pair an abrupt change in behaviour with a mechanism that forms over training, and read the change through that mechanism.", U["grok"]),
    ("attributiongraphs", "auditing", "empirical", "applied to the hidden-objective model",
     "Attribution graphs on the model fine-tuned to exploit reward-model biases show reward-model-bias features in the circuits for its behaviour.", U["biology"]),
    ("beliefgeometry", "othello", "established", "world state vs. belief state",
     "Othello-GPT recovers a fully observed board; belief-state geometry recovers a distribution over hidden states, the object an agent under partial observation needs.", U["belief"]),
    ("attrpatching", "sae", "established", "attribution over dictionary features",
     "Sparse feature circuits use attribution patching over SAE features and error terms to find and edit circuits.", U["sfc"]),
    ("beliefgeometry", "sae", "established", "a target fixed in advance vs. features found without one",
     "SAE features are discovered and then labelled; the belief geometry is computed from the generator before the network is inspected, so the regression is checked against a known answer.", U["belief"]),
    # ---- bridges ----
    ("beliefgeometry", "predictivestate", "established", "mixed-state presentation predicts the geometry",
     "The predicted geometry is the set of belief states over the generator's hidden states, from computational mechanics; the network is found to represent it linearly.", U["belief"]),
    ("beliefgeometry", "hmm", "established", "belief updating over an HMM",
     "The training data come from hidden Markov processes, and the represented belief is the HMM forward-filter posterior over hidden states.", U["belief"]),
    ("beliefgeometry", "psr", "hypothesis", "passive belief vs. action-conditioned predictive state",
     "Extend the readout to action-conditioned processes and test whether it matches a predictive state representation; the published setting has no actions.", None),
    ("beliefgeometry", "alignmentthesis", "hypothesis", "a readout check for the belief interface",
     "A known-target belief readout could check the thesis's belief interface in a synthetic task. It is evidence about representation, not about semantic correctness or correction.", None),
    ("beliefgeometry", "attention", "empirical", "attention implements constrained belief updates",
     "In single-layer transformers on a family of HMMs, attention carries out a constrained, parallel version of the Bayesian update, with the attention pattern and OV vectors predicted in advance.", U["constrained"]),
    ("grokkingmech", "grokkingslt", "established", "two accounts of grokking",
     "One account reads grokking as circuit formation and cleanup in a small transformer; the other as a move to a lower-LLC basin in shallow quadratic networks. The settings differ.", U["grokslt"]),
    ("inductionheads", "stagewise", "empirical", "an LLC stage coincides with induction-head formation",
     "In a two-layer language transformer, previous-token heads form in LM3 and induction heads in LM4, stages found from the LLC curve.", U["stagewise"]),
    ("inductionheads", "rllc", "empirical", "refined LLCs separate induction heads",
     "Weight-refined LLC curves cluster heads into previous-token, induction and multigram types, confirmed with ablations.", U["rllc"]),
    ("inductionheads", "susceptibility", "empirical", "susceptibilities separate the induction circuit",
     "A principal component of the data-perturbation response matrix loads on the induction-circuit heads, found without circuit-level patching.", U["suscept"]),
    ("superposition", "compressed", "established", "relation to compressed sensing",
     "Recovering more sparse features than dimensions is the compressed-sensing setting; the recovery guarantees need conditions that learned representations are not known to meet.", U["toy"]),
    ("sae", "lasso", "established", "L1-penalized sparse coding",
     "The classic SAE loss is reconstruction error plus an L1 penalty on codes; for a fixed decoder the optimal code solves a lasso, which the encoder only approximates.", U["cunningham"]),
    ("linearrep", "probeattr", "established", "assumes a linear behaviour probe",
     "Probe-based attribution aligns representation differences with a linear behaviour probe, so it relies on that behaviour being linearly readable.", U["probeattr"]),
    ("controlprobes", "featurerewards", "hypothesis", "probe selectivity under optimization",
     "A probe used as a reward is optimized against. Test whether the trained policy raises the probe score without the target property, using held-out graders.", None),
    ("steering", "preference", "empirical", "post-training behaviour mediated by a direction",
     "Refusal learned in safety post-training is mediated by a single direction in open chat models; ablating it removes refusal.", U["refusal"]),
    ("auditing", "rewardhack", "empirical", "planted reward-model exploitation",
     "The audited model was trained to exploit errors in RLHF reward models and generalized to behaviours it believed reward models rate highly.", U["audit"]),
    ("auditing", "specgaming", "established", "the same failure class, examined internally",
     "Specification gaming is observed in behaviour; the auditing game asks whether a planted tendency to exploit the reward signal can be found from the model's internals and data.", U["audit"]),
    ("auditing", "redteam", "empirical", "behavioural attacks vs. interpretability in audits",
     "Blind teams combined behavioural attacks with interpretability and data analysis; the unblinded study compares the techniques' strengths and limits.", U["audit"]),
    ("auditing", "oversight", "hypothesis", "internals as an oversight signal",
     "Test whether interpretability-based audits add detection power beyond output-level oversight when the overseen model is stronger than the overseer on the task.", None),
    ("saebaselines", "guardrail", "empirical", "activation probes as runtime monitors",
     "Dense linear probes on activations detected harmful user intent nearly perfectly, including on new jailbreaks; SAE-based probes were worse.", U["gdm"]),
    ("saebaselines", "logistic", "established", "logistic-regression probes as the baseline",
     "The baselines that SAE probes failed to beat are regularized logistic regressions on raw activations.", U["sparseprobe"]),
    ("scalingmono", "sycophancy", "empirical", "a sycophancy feature",
     "The authors report features for sycophantic praise that causally influence outputs when clamped; a feature's existence does not show the behaviour occurs unprompted.", U["scaling"]),
    ("rudin2019", "gam", "established", "interpretable by design",
     "Additive and monotonic predictors are one of the inherently interpretable model classes the paper recommends over post-hoc explanation.", U["rudin"]),
    ("rudin2019", "trees", "established", "interpretable model classes",
     "Sparse trees and rule lists are central examples of the inherently interpretable models the paper advocates.", U["rudin"]),
    ("rudin2019", "sae", "established", "opposing positions on post-hoc explanation",
     "SAEs explain a trained network after the fact with a reconstruction that has residual error; the paper argues that, for high-stakes use, such explanations should not replace an interpretable model.", U["rudin"]),
    ("rudin2019", "attributiongraphs", "established", "the fidelity gap made explicit",
     "Attribution graphs carry error nodes for what the replacement model misses, a measured version of the fidelity gap the paper says every post-hoc explanation has.", U["methods"]),
    ("sparsetrees", "trees", "established", "globally optimal instead of greedy",
     "Optimal sparse trees replace greedy splitting and pruning with a certified optimum of loss plus a leaf penalty.", U["gosdt"]),
    ("sparsetrees", "rashomon", "established", "enumerating all near-optimal sparse trees",
     "TreeFARMS extends the dynamic-programming search for one optimal sparse tree to enumerate every tree within ε of the optimum.", U["treefarms"]),
    ("rashomon", "rudin2019", "established", "why simple accurate models often exist",
     "A large Rashomon set makes it likely that it contains a simple model, the paper's argument for searching before accepting a black box.", U["rashomon"]),
    ("rashomon", "sae", "hypothesis", "many near-equivalent decompositions",
     "SAEs trained with different seeds learn partly different features at similar loss, which resembles a Rashomon set of dictionaries. Whether Rashomon-set tools (enumeration, importance across the set) carry over is untested.", U["seeds"]),
    ("rashomon", "llc", "hypothesis", "volume of the near-optimal set",
     "The Rashomon ratio measures the volume of the ε-sublevel set at fixed ε; the LLC is the exponent of that volume as ε → 0 in parameter space. Whether large Rashomon ratios and low LLCs co-occur in a shared model class is untested.", U["llc"]),
    ("predictivestate", "development", "hypothesis", "belief geometry as a developmental target",
     "In a transformer trained on an HMM process, test whether LLC-detected stages coincide with the residual stream acquiring the predicted belief geometry.", None),
    ("development", "probeattr", "hypothesis", "stage-aware data attribution",
     "Checkpoint-wise influence can change sign across stages, while probe-based attribution ranks data at one checkpoint. Compare their rankings across checkpoints at matched cost.", U["bifstage"]),
    ("probeattr", "alignmentthesis", "hypothesis", "a candidate correction actuator",
     "Probe-guided data editing is a candidate for the thesis's correction and learning interfaces. It counts only if the correction persists and unrelated capability holds on independent evaluations.", None),
    ("causalabstraction", "causal", "established", "structural causal models as the formal basis",
     "Both use structural causal models and interventions; causal abstraction asks whether a network implements a given high-level model, causal representation learning whether latents are identifiable.", U["ca2023"]),
    ("othello", "world", "established", "a different sense of “world model”",
     "Othello-GPT's world model is an internal state decoded from a sequence model; model-based RL's world model is an action-conditioned predictor used for planning.", U["othello"]),
    ("transcoders", "mlp", "established", "replaces the MLP sublayer",
     "A transcoder is trained to imitate an MLP sublayer's input–output map with a wider, sparsely active hidden layer.", U["transcoders"]),
]

# ---- learning metadata ----------------------------------------------------------
META = {
    "circuits": (2, ["deep-learning", "linear-algebra"], ["cnn"]),
    "superposition": (3, ["linear-algebra", "deep-learning", "theory"], ["circuits", "compressed"]),
    "inductionheads": (3, ["deep-learning"], ["attention", "circuits"]),
    "ioi": (3, ["deep-learning"], ["inductionheads"]),
    "actpatching": (2, ["deep-learning", "stats-ml"], ["attention"]),
    "attrpatching": (3, ["deep-learning", "optimization"], ["actpatching"]),
    "causalscrubbing": (3, ["deep-learning", "theory"], ["actpatching"]),
    "causalabstraction": (3, ["theory", "probability", "deep-learning"], ["actpatching", "causal"]),
    "acdc": (3, ["deep-learning", "optimization"], ["actpatching"]),
    "sae": (3, ["deep-learning", "linear-algebra", "optimization"], ["superposition", "lasso"]),
    "scalingmono": (3, ["deep-learning"], ["sae", "llm"]),
    "transcoders": (4, ["deep-learning", "linear-algebra"], ["sae", "mlp"]),
    "attributiongraphs": (4, ["deep-learning", "linear-algebra"], ["transcoders", "attrpatching"]),
    "linearrep": (3, ["linear-algebra", "deep-learning", "theory"], ["controlprobes"]),
    "controlprobes": (2, ["stats-ml", "deep-learning"], ["logistic"]),
    "tunedlens": (2, ["deep-learning", "linear-algebra"], ["mlp"]),
    "steering": (3, ["deep-learning", "linear-algebra"], ["linearrep"]),
    "grokkingmech": (3, ["deep-learning", "theory", "signals-estimation"], ["circuits", "fourier"]),
    "othello": (2, ["deep-learning", "stats-ml"], ["controlprobes", "attention"]),
    "beliefgeometry": (4, ["probability", "deep-learning", "theory"], ["hmm", "linearrep"]),
    "illusions": (3, ["deep-learning", "stats-ml"], ["actpatching", "controlprobes"]),
    "saebaselines": (3, ["stats-ml", "deep-learning"], ["sae", "logistic"]),
    "auditing": (4, ["deep-learning", "rl"], ["sae", "rewardhack"]),
    "sparsetrees": (2, ["optimization", "stats-ml"], ["trees"]),
    "rashomon": (3, ["stats-ml", "theory", "optimization"], ["sparsetrees"]),
    "rudin2019": (1, ["stats-ml"], ["trees"]),
}

# ---- paths -----------------------------------------------------------------------
# `via` is given as (a, b, relation) of an edge above, or an existing atlas edge id.

def S(node, objective, why, via):
    return {"node": node, "objective": objective, "why": why, "via": via}


PATHS = [
    {
        "id": "interpretable-ml-to-thesis",
        "title": "Interpretable-ML researcher: from sparse trees and Rashomon sets through mechanistic interpretability to the thesis",
        "summary": "Start from inherently interpretable models and Rashomon sets, test post-hoc network explanation (SAEs, circuits, their critiques) against that standard, then follow belief-state geometry into predictive states, developmental SLT and measured correction, ending at the atlas's alignment thesis.",
        "audience": "Interpretable-ML researchers: people who build sparse optimal decision trees and Rashomon sets, or who work in mechanistic interpretability, and who work on agent safety, with strong math (real analysis, probability, high-dimensional statistics) and hands-on RL post-training and agent-infrastructure experience.",
        "level": "advanced",
        "skills": ["stats-ml", "theory", "deep-learning", "probability"],
        "estHours": 20,
        "steps": [
            S("sparsetrees",
              "State the OSDT/GOSDT objective (loss + λ × leaves), explain how bounds and dynamic programming certify optimality, and say what the certificate is relative to.",
              "Home territory, and the evidence standard the rest of the path is measured against: an explicit objective, a certified optimum, and a model that is its own explanation.",
              None),
            S("rashomon",
              "Define the Rashomon set and ratio, explain how TreeFARMS enumerates it, and compute a variable-importance range across the set rather than for one tree.",
              "The atlas edge: TreeFARMS enumerates every tree within ε of the optimum. The habit of asking “which of the many near-optimal models?” is the one to carry into network interpretability.",
              ("sparsetrees", "rashomon", "enumerating all near-optimal sparse trees")),
            S("rudin2019",
              "Summarize the argument against post-hoc explanation for high-stakes decisions, and say which part concerns structured data and which part is general.",
              "The atlas edge: a large Rashomon set makes a simple, accurate model likely to exist. This is the position from which mechanistic interpretability looks like explaining a black box after the fact.",
              ("rashomon", "rudin2019", "why simple accurate models often exist")),
            S("sae",
              "Write the SAE objective, explain why it does not identify a unique dictionary, and list the three checks a claim should report: seed stability, reconstruction loss, and a dense-probe baseline.",
              "The atlas edge: SAEs are exactly the post-hoc explanation the previous step argues against, and their reconstruction error is the fidelity gap. Your Rashomon intuition applies directly: different seeds give partly different dictionaries at similar loss (the hypothesis edge to Rashomon sets). Mechanistic interpretability's evidence is weaker than a certified optimum: on downstream tasks SAEs have not beaten linear probes, and illusions show that convincing labels can be dataset artefacts. It still informs the thesis by naming candidate internal variables to test.",
              ("rudin2019", "sae", "opposing positions on post-hoc explanation")),
            S("beliefgeometry",
              "Explain how the target belief simplex is computed from a known HMM before the network is examined, and what a linear readout of it does and does not show.",
              "The atlas edge: a target fixed in advance, unlike features found and then labelled. This is the piece of interpretability the thesis can use, because it is checked against a known answer, much as a certified tree is. It remains a readout in toy processes with no actions, not evidence that the model uses the belief.",
              ("beliefgeometry", "sae", "a target fixed in advance vs. features found without one")),
            S("predictivestate",
              "State the predictive-equivalence definition of causal states and why they are the minimal sufficient statistic for prediction under stationarity.",
              "The atlas edge: computational mechanics supplies the predicted geometry. The thesis needs belief as an explicit object with update semantics, not as a direction found inside a network, and this is where that object is defined.",
              ("beliefgeometry", "predictivestate", "mixed-state presentation predicts the geometry")),
            S("development",
              "Describe how LLC-detected stages are found, and design the test that would show whether belief geometry forms at a stage boundary.",
              "A hypothesis edge: whether stages coincide with the belief geometry forming is untested. SLT is the natural home for Rashomon intuitions: the LLC is the exponent of the volume of near-optimal parameters (see the hypothesis edge from Rashomon sets to the LLC), and singular models have degenerate, non-unique minima. Developmental claims here are associations in small transformers, a weaker standard than an optimality certificate.",
              ("predictivestate", "development", "belief geometry as a developmental target")),
            S("probeattr",
              "Explain how a behaviour probe ranks preference data for filtering, and how stage-aware attribution would change the ranking.",
              "A hypothesis edge: compare checkpoint-wise influence with one-checkpoint probe attribution. This is where interpretability becomes an actuator: a probe changes the training data and the retrained outcome is measured. Features used as RL rewards are the RL-side analogue, and their probes face the same selectivity and optimization-pressure caveats as any probe.",
              ("development", "probeattr", "stage-aware data attribution")),
            S("alignmentthesis",
              "State the thesis's correction and learning interfaces and its falsifier, and say what evidence would make probe-guided correction count.",
              "A hypothesis edge: probe-guided editing is a candidate correction actuator only if the correction persists and unrelated capability holds on independent evaluations. The thesis asks for behavioural, prospective evidence; internal explanations inform which variables to test but do not substitute for that test.",
              ("probeattr", "alignmentthesis", "a candidate correction actuator")),
        ],
    },
    {
        "id": "mechinterp-circuits-causal-tests",
        "title": "Mechanistic interpretability: circuits and the causal tests behind them",
        "summary": "From the circuits programme and two fully worked case studies to the patching, scrubbing and abstraction tests used to check circuit claims, ending with the illusions those tests can produce.",
        "audience": "Learners who know transformers and want to judge circuit-level claims, including what each test can and cannot establish.",
        "level": "advanced",
        "skills": ["deep-learning", "theory"],
        "estHours": 14,
        "steps": [
            S("circuits", "State the three claims of the circuits programme and give one worked example.",
              "The programme every later step either uses or tests.", None),
            S("grokkingmech", "Explain the Fourier multiplication algorithm in the modular-addition network and the three phases the progress measures reveal.",
              "The atlas edge: a rare complete reverse-engineering, possible because the task's algorithm could be guessed and checked.",
              ("grokkingmech", "circuits", "full reverse-engineering of a small model")),
            S("inductionheads", "Describe the previous-token/induction-head composition and the grade of evidence for its role at scale.",
              "The atlas edge: both read an abrupt behaviour change through a mechanism that forms gradually.",
              ("grokkingmech", "inductionheads", "abrupt change, gradual mechanism")),
            S("ioi", "Name the IOI head classes and explain why backup heads complicate ablation evidence.",
              "The atlas edge: the IOI circuit contains induction and duplicate-token heads.",
              ("inductionheads", "ioi", "circuit contains induction heads")),
            S("actpatching", "Run activation patching with two corruption schemes and two metrics, and explain why results can differ.",
              "The atlas edge: IOI was found with path patching, so its evidence is only as good as patching's.",
              ("ioi", "actpatching", "found by path patching")),
            S("causalscrubbing", "Turn a circuit hypothesis into resampling ablations and compute the fraction of loss recovered.",
              "The atlas edge: scrubbing derives which activations to resample from the hypothesis itself.",
              ("actpatching", "causalscrubbing", "resampling ablation driven by a hypothesis")),
            S("causalabstraction", "Define an interchange intervention and explain what distributed alignment search learns.",
              "The atlas edge: causal abstraction is proposed as the common foundation for scrubbing and patching.",
              ("causalscrubbing", "causalabstraction", "both test a hypothesized high-level computation")),
            S("illusions", "Explain the dormant-pathway illusion and the dataset-dependence illusion, and the extra evidence each calls for.",
              "The atlas edge: learned subspace interventions can pass interchange tests through pathways the model does not normally use.",
              ("causalabstraction", "illusions", "subspace-patching illusion")),
        ],
    },
    {
        "id": "mechinterp-features-representations",
        "title": "Mechanistic interpretability: probes, linear features, dictionaries and steering",
        "summary": "From lenses and probes to the linear representation and superposition hypotheses, sparse dictionaries at scale and steering, ending with the negative results against simple baselines.",
        "audience": "Learners with linear algebra and transformer experience who want to judge feature-level claims against simple baselines.",
        "level": "advanced",
        "skills": ["deep-learning", "linear-algebra", "stats-ml"],
        "estHours": 15,
        "steps": [
            S("tunedlens", "Decode intermediate layers with the logit lens and the tuned lens, and say why a lens is a probe.",
              "A low-cost first look at what a layer encodes, and the first probing caveat.", None),
            S("controlprobes", "Compute probe selectivity with a control task and explain why decodability does not show use.",
              "The atlas edge: the tuned lens is a per-layer affine probe, so control-task caveats apply.",
              ("tunedlens", "controlprobes", "a lens is a per-layer probe")),
            S("othello", "Describe the Othello-GPT probes and the intervention that changes its moves.",
              "The atlas edge: Othello-GPT adds the intervention that probing alone lacks.",
              ("controlprobes", "othello", "probes plus interventions")),
            S("linearrep", "State the linear representation hypothesis, the causal inner product, and a known counterexample.",
              "The atlas edge: the board becomes linear in a mine/theirs basis.",
              ("othello", "linearrep", "board state is linear in the right basis")),
            S("superposition", "Explain when toy models store features in superposition and the link to compressed sensing.",
              "The atlas edge: superposition assumes linear features and asks how more than d fit in d dimensions.",
              ("linearrep", "superposition", "features as directions")),
            S("sae", "Train or describe an SAE and report seed stability and reconstruction loss.",
              "The atlas edge: an overcomplete sparse dictionary is the proposed way to undo superposition.",
              ("superposition", "sae", "dictionary learning to undo superposition")),
            S("scalingmono", "Summarize what scaling SAEs to a production model found, and the authors' caution on safety-relevant features.",
              "The atlas edge: the same method at tens of millions of latents.",
              ("sae", "scalingmono", "scaled to a production model")),
            S("steering", "Build a steering vector from contrasting prompts and measure its side effects.",
              "The atlas edge: clamping a feature steers behaviour as its label predicts.",
              ("scalingmono", "steering", "feature clamping steers behaviour")),
            S("saebaselines", "Compare an SAE-based method with a dense linear probe and with prompting at matched data.",
              "The atlas edge: on AxBench prompting beat representation steering. The baseline comparison is the evidence standard to apply to every feature claim on this path.",
              ("steering", "saebaselines", "prompting beats representation steering")),
        ],
    },
    {
        "id": "mechinterp-circuit-discovery-scale",
        "title": "Mechanistic interpretability: automating circuit discovery, up to alignment audits",
        "summary": "From manual patching to automated and gradient-based circuit discovery, feature-level circuits through transcoders, attribution graphs, and their use in an alignment audit.",
        "audience": "Learners who know activation patching and SAEs and want the scaled-up circuit-discovery toolchain and its limits.",
        "level": "advanced",
        "skills": ["deep-learning", "linear-algebra", "optimization"],
        "estHours": 12,
        "steps": [
            S("actpatching", "Explain what an activation-patching effect measures and how corruption choices change it.",
              "Every method on this path approximates or automates this measurement.", None),
            S("acdc", "Run or describe ACDC and how its threshold changes the recovered circuit.",
              "The atlas edge: ACDC automates the patch-and-prune loop.",
              ("actpatching", "acdc", "automates iterative patching")),
            S("attrpatching", "Derive the first-order attribution-patching estimate and name two ways it produces false negatives.",
              "The atlas edge: edge attribution patching recovered circuits better than ACDC on average, far more cheaply.",
              ("attrpatching", "acdc", "edge attribution patching beats ACDC at circuit recovery")),
            S("sae", "Explain why attributions over SAE features plus error terms give feature-level circuits.",
              "The atlas edge: sparse feature circuits run attribution patching over dictionary features.",
              ("attrpatching", "sae", "attribution over dictionary features")),
            S("transcoders", "Contrast an SAE with a transcoder, and say what model diffing with a crosscoder shows and what can mislead it.",
              "The atlas edge: transcoders predict a sublayer's output, making feature-to-feature attribution tractable.",
              ("sae", "transcoders", "sparse dictionary variant")),
            S("attributiongraphs", "Read an attribution graph, including error nodes, and design the perturbation that tests it.",
              "The atlas edge: attribution graphs are computed through a cross-layer transcoder replacement.",
              ("transcoders", "attributiongraphs", "replacement model for circuit tracing")),
            S("auditing", "Describe the hidden-objective auditing game and what its success does and does not show.",
              "The atlas edge: attribution graphs were applied to the audited model. One planted objective with a known answer is a test bed, not evidence that audits find unplanted ones.",
              ("attributiongraphs", "auditing", "applied to the hidden-objective model")),
        ],
    },
]

BG_QUESTION = "bg-goal"
BG_OPTION = {"id": "goal-interp",
             "label": "I come from interpretability research (inherently interpretable models such as sparse trees and Rashomon sets, or mechanistic interpretability) and want to see how it connects to the alignment thesis",
             "sets": {}, "goal": "interpretable-ml-to-thesis"}

# ---- question bank ---------------------------------------------------------------
QFILE = "questions-7.json"


def Q(id, node, skill, difficulty, prompt, choices, answer, explain, source):
    return {"id": "q-mi-" + id, "node": node, "skill": skill, "difficulty": difficulty, "prompt": prompt,
            "choices": choices, "answer": answer, "explain": explain, "source": source}


QUESTIONS = [
    Q("circuits-1", "circuits", "deep-learning", 1,
      "In the circuits framework, what is a circuit?",
      ["A set of training examples that activate one neuron", "Features connected by weights, forming a computational subgraph", "A path in the loss landscape between two minima", "A sequence of layers with residual connections"], 1,
      "Zoom In defines circuits as features connected by weights; features are the nodes and weights the edges, read directly from the trained network.", U["zoomin"]),
    Q("circuits-2", "circuits", "deep-learning", 2,
      "Why do polysemantic neurons undermine reading circuits neuron by neuron?",
      ["They only appear in untrained networks", "They make gradients vanish", "One neuron responds to several unrelated features, so it is not a single feature", "They cannot be ablated"], 2,
      "If one neuron mixes unrelated features, a neuron-level circuit diagram mislabels its nodes; this motivates superposition and dictionary methods.", U["zoomin"]),
    Q("superposition-1", "superposition", "linear-algebra", 2,
      "In toy models of superposition, which change makes a model more likely to store features in superposition?",
      ["Making features sparser", "Making features denser", "Adding more hidden dimensions than features", "Removing the ReLU nonlinearity entirely"], 0,
      "Sparse features rarely co-occur, so interference between nearly orthogonal directions is cheap; the toy models show superposition growing with sparsity.", U["toy"]),
    Q("superposition-2", "superposition", "theory", 3,
      "Which established field studies recovering more sparse signals than measured dimensions, and what does it require that learned networks are not known to satisfy?",
      ["Kalman filtering; Gaussian noise", "Compressed sensing; conditions on the measurement operator such as restricted isometry", "Fourier analysis; band-limited signals", "Control theory; observability"], 1,
      "Superposition is related to compressed sensing, whose recovery guarantees need conditions on the sensing operator; a trained network's weights are not known to meet them.", U["toy"]),
    Q("inductionheads-1", "inductionheads", "deep-learning", 1,
      "Given the token pattern [A][B] … [A], what does an induction head promote as the next token?",
      ["[A]", "The most frequent token in the context", "[B]", "An end-of-sequence token"], 2,
      "A previous-token head marks what followed each token; the induction head then attends to the token after the earlier [A] and copies [B].", U["framework"]),
    Q("inductionheads-2", "inductionheads", "deep-learning", 3,
      "How strong is the evidence that induction heads drive most in-context learning in large models with MLPs, according to the authors themselves?",
      ["Proven by ablation in every model size", "Mainly correlational; the interventional evidence is for small attention-only models", "Refuted by later work", "Established by a theorem"], 1,
      "Olsson et al. grade their evidence by model type: causal for small attention-only models, mostly correlational for large models.", U["inductionarx"]),
    Q("ioi-1", "ioi", "deep-learning", 2,
      "In the IOI circuit, what complicates reading a head's importance from an ablation?",
      ["Backup name-mover heads take over when the main ones are ablated", "GPT-2 small has no attention heads", "Ablation always increases accuracy", "The task has no correct answer"], 0,
      "Wang et al. found backup heads that compensate, so a single ablation can understate a component's role.", U["ioi"]),
    Q("ioi-2", "ioi", "deep-learning", 1,
      "Which technique was used to isolate the IOI circuit?",
      ["Gradient descent on the input", "Path patching", "Principal component analysis of weights", "Dropout"], 1,
      "Path patching, a variant of activation patching that restricts effects to selected paths, located the 26 heads.", U["ioi"]),
    Q("actpatching-1", "actpatching", "deep-learning", 1,
      "What does activation patching measure?",
      ["The gradient of the loss with respect to the weights", "The change in output when an activation is replaced by its value from another input", "The number of parameters in a layer", "The entropy of the attention pattern"], 1,
      "Patching copies an activation from a source run into a target run and measures the effect on a chosen metric.", U["howto"]),
    Q("actpatching-2", "actpatching", "stats-ml", 3,
      "Hase et al. compared causal-tracing localization with model editing. What did they find?",
      ["Editing works only at the localized layer", "Localization perfectly predicts edit success", "Localization did not predict which layer was best to edit", "Causal tracing cannot be computed for GPT models"], 2,
      "Knowing where a fact is localized by causal tracing gave little information about where editing it works best.", U["hase"]),
    Q("attrpatching-1", "attrpatching", "optimization", 2,
      "Attribution patching approximates the patching effect of activation a on metric m as:",
      ["(a_clean − a_corrupt) · ∂m/∂a", "∂²m/∂a²", "m(a_clean) · m(a_corrupt)", "‖a_clean‖ − ‖a_corrupt‖"], 0,
      "It is the first-order Taylor term of m around the corrupted activation, computed for every component from one backward pass.", U["atpblog"]),
    Q("attrpatching-2", "attrpatching", "deep-learning", 3,
      "When is a small attribution-patching score weak evidence that a component is unimportant?",
      ["Never; the estimate is exact", "When the metric is affine in the activation", "When the model is a linear regression", "When attention is saturated or paths cancel, so the linear estimate misses the true effect"], 3,
      "Kramár et al. identify saturated attention softmaxes and cancellation as sources of false negatives.", U["atpstar"]),
    Q("causalscrubbing-1", "causalscrubbing", "deep-learning", 2,
      "In causal scrubbing, which activations are resampled from other inputs?",
      ["All activations in the model", "Those the hypothesis claims do not matter for the behaviour", "Only the input embeddings", "None; the method uses gradients"], 1,
      "The hypothesis determines which activations can be swapped between inputs without changing the behaviour; performance after resampling scores it.", U["scrub"]),
    Q("causalscrubbing-2", "causalscrubbing", "theory", 3,
      "Two incompatible hypotheses both recover most of the loss under causal scrubbing. What follows?",
      ["Both are true mechanisms", "The test does not identify the mechanism on this dataset", "The model is linear", "Scrubbing was run incorrectly"], 1,
      "A pass is consistency with the dataset and metric, not uniqueness; incompatible hypotheses passing shows the test is non-identifying there.", U["scrub"]),
    Q("causalabstraction-1", "causalabstraction", "theory", 2,
      "What does an interchange intervention do?",
      ["Swaps two layers of the network", "Sets an internal representation to its value on a different input and checks the high-level model's prediction", "Retrains the model on swapped labels", "Exchanges the encoder and decoder of an SAE"], 1,
      "It tests whether intervening on an aligned internal variable has the effect the high-level causal model predicts.", U["ca2021"]),
    Q("causalabstraction-2", "causalabstraction", "deep-learning", 3,
      "Why can a learned subspace pass interchange tests yet misrepresent where a feature lives?",
      ["Subspaces cannot be learned", "The intervention can work by activating a dormant parallel pathway", "Interchange tests ignore the output", "Rotations change the model's weights"], 1,
      "Makelov et al. show subspace patching can produce the right output through a pathway the model does not normally use.", U["subspace"]),
    Q("acdc-1", "acdc", "deep-learning", 1,
      "What does ACDC automate?",
      ["Tokenization", "Pruning edges of the computational graph whose removal barely changes a task metric", "Hyperparameter search for training", "Data labelling"], 1,
      "ACDC iteratively patches edges and removes those below a threshold of effect on the metric.", U["acdc"]),
    Q("acdc-2", "acdc", "optimization", 2,
      "Which finding about ACDC concerns cost and accuracy?",
      ["It needs no forward passes", "Edge attribution patching recovered circuits better on average with two forward passes and one backward pass", "It always finds a unique circuit regardless of threshold", "It replaces the need for a metric"], 1,
      "Syed, Rager and Conmy report higher average circuit-recovery AUC for the gradient-based method at much lower cost.", U["eap"]),
    Q("sae-1", "sae", "linear-algebra", 1,
      "What does a sparse autoencoder trained on activations learn?",
      ["A low-rank projection with fewer directions than dimensions", "An overcomplete dictionary of directions, few of which are active per input", "The model's training data", "A new attention pattern"], 1,
      "SAEs use more latents than dimensions, with a sparsity penalty so each input activates only a few.", U["mono"]),
    Q("sae-2", "sae", "stats-ml", 3,
      "SAEs trained on the same model and data but different seeds share only part of their features. Which conclusion fits best?",
      ["The model has no features", "An SAE's features are one useful decomposition, not a unique list of the model's true features", "Seeds should be averaged to get the true features", "Only TopK SAEs are valid"], 1,
      "Paulo and Belrose conclude that SAE features are a pragmatically useful decomposition rather than a universal list.", U["seeds"]),
    Q("scalingmono-1", "scalingmono", "deep-learning", 1,
      "What happened when the Golden Gate Bridge feature was clamped to a high value?",
      ["The model refused all requests", "The model's outputs became about the bridge, even identifying as it", "Nothing changed", "The model's loss went to zero"], 1,
      "Clamping the feature to ten times its maximum activation induced bridge-related behaviour, evidence that the feature influences outputs.", U["scaling"]),
    Q("scalingmono-2", "scalingmono", "deep-learning", 2,
      "The authors find a feature related to deception. What do they caution?",
      ["The feature proves the model lies", "The feature should be deleted", "Knowing about lies differs from lying, so the feature's existence does not show the behaviour", "Deception features appear only in images"], 2,
      "The paper warns against reading too much into a safety-relevant feature's existence.", U["scaling"]),
    Q("transcoders-1", "transcoders", "deep-learning", 2,
      "How does a transcoder differ from an SAE?",
      ["It reconstructs the same activation it reads", "It predicts a sublayer's output from that sublayer's input through a sparse hidden layer", "It has no sparsity penalty", "It operates on tokens, not activations"], 1,
      "A transcoder imitates the MLP sublayer's input–output map, which makes feature-to-feature attributions tractable.", U["transcoders"]),
    Q("transcoders-2", "transcoders", "deep-learning", 3,
      "A crosscoder diff between a base and a chat model shows chat-only latents. What control would help check them?",
      ["Train on more chat data", "A diff between two seeds of the same base model, which should show few model-specific latents", "Remove the sparsity penalty", "Use a larger learning rate"], 1,
      "Sparsity artefacts can create spurious model-specific latents, so a null diff gives a baseline rate.", U["crossart"]),
    Q("attributiongraphs-1", "attributiongraphs", "linear-algebra", 2,
      "Why are edges in an attribution graph linear for a given prompt?",
      ["The model is linear", "Attention patterns and normalization are frozen, and transcoders bridge the MLP nonlinearities", "Only the last layer is used", "Edges are fitted by regression"], 1,
      "Freezing the remaining nonlinearities makes direct feature-to-feature interactions linear in the local replacement model.", U["methods"]),
    Q("attributiongraphs-2", "attributiongraphs", "deep-learning", 2,
      "For roughly what share of the prompts they tried did the authors report satisfying insight from attribution graphs?",
      ["About a quarter", "Nearly all", "About half", "None"], 0,
      "The biology paper states the graphs gave satisfying insight for about a quarter of prompts tried.", U["biology"]),
    Q("linearrep-1", "linearrep", "linear-algebra", 2,
      "What does the causal inner product in Park, Choe and Veitch accomplish?",
      ["It orthogonalizes all weights", "It makes the probing direction and the steering direction for a concept coincide", "It removes superposition", "It speeds up training"], 1,
      "Under that inner product, the embedding-space and unembedding-space notions of a concept direction agree.", U["lrh"]),
    Q("linearrep-2", "linearrep", "theory", 3,
      "Which finding is a counterexample to the claim that every feature is one-dimensionally linear?",
      ["Induction heads", "Circular representations of days of the week and months", "The logit lens", "Dropout"], 1,
      "Engels et al. found irreducibly multi-dimensional, circular features.", U["notlinear"]),
    Q("controlprobes-1", "controlprobes", "stats-ml", 1,
      "What is a probe's selectivity?",
      ["Accuracy on the linguistic task minus accuracy on a control task with random labels", "The number of parameters in the probe", "Accuracy on the training set", "The rank of the probe's weight matrix"], 0,
      "Control tasks can only be learned by the probe itself, so the difference isolates what the representation contributes.", U["control"]),
    Q("controlprobes-2", "controlprobes", "stats-ml", 2,
      "A probe decodes a property with 95% accuracy. What additional evidence shows the model uses that property?",
      ["A higher-capacity probe", "An intervention along the probe direction that changes behaviour as predicted", "Training the probe longer", "Reporting the accuracy on the training set"], 1,
      "Decodability is correlational; use needs an intervention.", U["belinkov"]),
    Q("tunedlens-1", "tunedlens", "deep-learning", 1,
      "How does the tuned lens differ from the logit lens?",
      ["It uses the input embedding", "It learns an affine translator per layer before the unembedding", "It decodes only the final layer", "It requires retraining the model"], 1,
      "The logit lens applies the unembedding directly; the tuned lens first applies a trained affine map per layer.", U["tuned"]),
    Q("tunedlens-2", "tunedlens", "deep-learning", 2,
      "Why should a lens reading be treated as a probing result?",
      ["Because lenses modify the model", "Because it shows what a decoder can read from a layer, not what computation the model performs with it", "Because lenses only work on vision models", "Because lenses are exact"], 1,
      "A lens is a decoder trained or applied post hoc; information it reads may be unused downstream.", U["belinkov"]),
    Q("steering-1", "steering", "deep-learning", 1,
      "How is an activation-addition steering vector typically built?",
      ["From the difference in activations between two contrasting prompts", "By retraining the model", "From the gradient of the loss", "By random initialization"], 0,
      "Activation addition takes the activation difference for a prompt pair and adds it during the forward pass.", U["actadd"]),
    Q("steering-2", "steering", "deep-learning", 2,
      "On AxBench, which method steered best?",
      ["SAE features", "Prompting", "Random directions", "Difference-in-means steering"], 1,
      "Prompting outperformed all representation-based steering methods, followed by fine-tuning; SAEs were not competitive.", U["axbench"]),
    Q("grokkingmech-1", "grokkingmech", "signals-estimation", 2,
      "What algorithm did the modular-addition network learn?",
      ["A lookup table of all pairs", "Discrete Fourier components combined with trigonometric identities", "Gradient boosting", "A recurrent counter"], 1,
      "Nanda et al. reverse-engineered a Fourier multiplication algorithm from the weights.", U["grok"]),
    Q("grokkingmech-2", "grokkingmech", "theory", 3,
      "Why are the progress measures not a general method for predicting emergence in advance?",
      ["They need test labels", "They were derived from the reverse-engineered mechanism, which must be known first", "They only work on images", "They are random"], 1,
      "The measures rely on knowing the algorithm, which was found after training in a small, checkable task.", U["grok"]),
    Q("othello-1", "othello", "deep-learning", 2,
      "Why did linear probes for Othello-GPT's board fail with black/white labels but succeed later?",
      ["The model was retrained", "The board is linear relative to the player to move (mine/theirs)", "More data was used", "Nonlinear probes were mislabelled as linear"], 1,
      "Nanda, Lee and Wattenberg found a linear representation in a mine/theirs basis.", U["othellolin"]),
    Q("othello-2", "othello", "stats-ml", 3,
      "What makes Othello-GPT stronger evidence than probe accuracy alone?",
      ["Larger model size", "Editing the decoded board state changes move predictions as expected", "Training on natural language", "Using a control task"], 1,
      "The intervention shows the model uses the representation, not only that it is decodable.", U["othello"]),
    Q("beliefgeometry-1", "beliefgeometry", "probability", 2,
      "In belief-state geometry, what is represented linearly in the residual stream?",
      ["The next token only", "The posterior belief over the generating HMM's hidden states", "The training loss", "The attention weights"], 1,
      "Shai et al. find the Bayesian belief state over hidden states, as computed from the mixed-state presentation.", U["belief"]),
    Q("beliefgeometry-2", "beliefgeometry", "theory", 3,
      "What makes the belief-geometry result checkable in a way typical SAE feature labels are not?",
      ["It uses larger models", "The target geometry is computed from the known generator before the network is examined", "It uses human raters", "It avoids linear regression"], 1,
      "The prediction is fixed in advance from computational mechanics, so the readout is compared with a known answer.", U["belief"]),
    Q("illusions-1", "illusions", "deep-learning", 2,
      "What did Bolukbasi et al. show about interpreting a BERT neuron from its top-activating examples?",
      ["Interpretations are always unique", "Different datasets give different, each convincing, interpretations", "Neurons never activate", "Top examples are random"], 1,
      "The interpretation depended on the dataset used to find the examples.", U["bolukbasi"]),
    Q("illusions-2", "illusions", "stats-ml", 3,
      "Which practice best guards against interpretability illusions?",
      ["Using more top-activating examples from the same dataset", "Held-out dataset checks plus interventions whose effects are predicted beforehand", "Reporting only successful cases", "Increasing model size"], 1,
      "Illusions arise from dataset dependence and unintended pathways; out-of-sample checks and predeclared interventions address both.", U["subspace"]),
    Q("saebaselines-1", "saebaselines", "stats-ml", 1,
      "On out-of-distribution harmful-intent detection, how did SAE probes compare with dense linear probes in the GDM study?",
      ["SAE probes were much better", "SAE probes underperformed dense linear probes", "Both failed completely", "They were identical"], 1,
      "Dense linear probes performed nearly perfectly, including out of distribution; SAE probes were worse.", U["gdm"]),
    Q("saebaselines-2", "saebaselines", "stats-ml", 2,
      "SAEBench found what about SAE proxy metrics?",
      ["They perfectly predict downstream use", "Gains on them did not reliably translate into better practical performance", "They are not computable", "They only apply to vision"], 1,
      "Karvonen et al. report that proxy-metric gains do not reliably carry over to practical tasks.", U["saebench"]),
    Q("auditing-1", "auditing", "rl", 2,
      "What hidden objective was the audited model trained to have?",
      ["Maximizing output length", "Exploiting known errors in RLHF reward models", "Refusing all questions", "Minimizing perplexity"], 1,
      "The model learned about reward-model errors and was trained to exploit some of them.", U["audit"]),
    Q("auditing-2", "auditing", "deep-learning", 3,
      "Why does success in this auditing game not show that audits find unplanted objectives?",
      ["The teams were not blind", "One objective was planted, partly via searchable documents, and the answer was known to organizers", "SAEs were not used", "The model was too small"], 1,
      "A planted objective with traces in the training data gives auditors routes that an unplanted objective may not.", U["audit"]),
    Q("sparsetrees-1", "sparsetrees", "optimization", 2,
      "What objective does OSDT minimize?",
      ["Misclassification error plus λ times the number of leaves", "Squared error plus an L2 weight penalty", "Information gain at the root only", "The number of features used, ignoring accuracy"], 0,
      "OSDT trades training error against a per-leaf penalty and finds the certified optimum.", U["osdt"]),
    Q("sparsetrees-2", "sparsetrees", "stats-ml", 3,
      "With threshold guessing from a reference ensemble, optimality is guaranteed relative to what?",
      ["All possible thresholds", "The guessed thresholds", "The test set", "A random forest's predictions"], 1,
      "Guessing restricts the candidate splits; the tree is optimal over that restricted set.", U["guess"]),
    Q("rashomon-1", "rashomon", "stats-ml", 1,
      "What is the Rashomon set?",
      ["The single best model", "The set of models in a class whose loss is within ε of the best", "The test set", "All models with zero training error"], 1,
      "It collects almost-equally-accurate models from a function class.", U["rashomon"]),
    Q("rashomon-2", "rashomon", "theory", 3,
      "The Rashomon ratio and the local learning coefficient both relate to which quantity?",
      ["Number of training examples", "The volume of a near-optimal (sublevel) set of the loss", "Learning rate", "The Hessian's largest eigenvalue"], 1,
      "The Rashomon ratio is a volume ratio at fixed ε; the LLC is the exponent of the near-optimal parameter volume as ε → 0. Their relationship is an atlas hypothesis.", U["llc"]),
    Q("rudin2019-1", "rudin2019", "stats-ml", 1,
      "What does Rudin (2019) recommend for high-stakes decisions?",
      ["Explaining black-box models with saliency maps", "Using inherently interpretable models instead of explaining black boxes", "Using the largest available model", "Avoiding machine learning"], 1,
      "The paper argues for interpretable models because post-hoc explanations are not fully faithful.", U["rudin"]),
    Q("rudin2019-2", "rudin2019", "stats-ml", 2,
      "Which scope limits the paper's claim that simple models often match complex ones?",
      ["Raw images", "Structured data with meaningful features", "Reinforcement learning", "Language generation"], 1,
      "The accuracy argument concerns structured data with meaningful features, not raw perceptual data.", U["rudin"]),
]

# ===================================================================================


def edge_id(a, b, rel):
    return "rel-" + hashlib.sha1(("%s|%s|%s" % (a, b, rel)).encode("utf-8")).hexdigest()[:8]


def read(path):
    with open(path, "r", encoding="utf-8", newline="") as fh:
        return fh.read()


def write(path, text):
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write(text)


def style_of(text):
    nl = "\r\n" if "\r\n" in text else "\n"
    return nl, text.endswith(nl) or text.endswith("\n")


def dump_like(original, obj, indent):
    nl, trail = style_of(original)
    out = json.dumps(obj, indent=indent, ensure_ascii=False)
    if nl != "\n":
        out = out.replace("\n", nl)
    return out + (nl if trail else "")


def detect_indent(text):
    m = re.search(r"\n( +)\"", text)
    return len(m.group(1)) if m else 1


def apply_atlas(root):
    path = os.path.join(root, "atlas.json")
    raw = read(path)
    atlas = json.loads(raw)
    nodes, edges = atlas["nodes"], atlas["edges"]
    ids = {n["id"] for n in nodes} | {n["id"] for n in NODES}
    for e in EDGES:
        for x in e[:2]:
            if x not in ids:
                sys.exit("edge endpoint %r is not a node" % x)
    if any(c["name"] == CLUSTER["name"] for c in atlas["clusters"]):
        for c in atlas["clusters"]:
            if c["name"] == CLUSTER["name"]:
                c["color"], c["center"] = CLUSTER["color"], CLUSTER["center"]
    else:
        atlas["clusters"].append({"name": CLUSTER["name"], "color": CLUSTER["color"], "count": 0, "center": CLUSTER["center"]})
    # upsert nodes in place
    pos = {n["id"]: i for i, n in enumerate(nodes)}
    for n in NODES:
        if n["id"] in pos:
            nodes[pos[n["id"]]] = dict(n)
        else:
            nodes.append(dict(n))
    # upsert edges: drop old copies by id or by unordered pair; drop stale edges on owned nodes
    owned = {n["id"] for n in NODES} - SHARED
    new_edges = []
    for a, b, t, rel, text, url in EDGES:
        new_edges.append({"id": edge_id(a, b, rel), "a": a, "b": b, "type": t, "relation": rel, "text": text, "url": url})
    new_ids = {e["id"] for e in new_edges}
    pairs = {frozenset((e["a"], e["b"])) for e in new_edges}
    keep = []
    for e in edges:
        if e["id"] in new_ids or frozenset((e["a"], e["b"])) in pairs:
            continue
        if (e["a"] in owned or e["b"] in owned):
            continue
        keep.append(e)
    # keep this cluster's edges in a stable position: replace where the first old one was, else append
    atlas["edges"] = keep + new_edges
    present = {n["id"] for n in nodes}
    atlas["edges"] = [e for e in atlas["edges"] if e["a"] in present and e["b"] in present]
    counts = {}
    for n in nodes:
        counts[n["group"]] = counts.get(n["group"], 0) + 1
    for c in atlas["clusters"]:
        c["count"] = counts.get(c["name"], 0)
    unknown = set(counts) - {c["name"] for c in atlas["clusters"]}
    if unknown:
        sys.exit("nodes reference unknown clusters: %s" % ", ".join(sorted(unknown)))
    out = json.dumps(atlas, ensure_ascii=False, separators=(",", ":")) if "\": " not in raw[:200] else json.dumps(atlas, ensure_ascii=False)
    out += raw[len(raw.rstrip("\r\n")):]
    write(path, out)
    return atlas


def apply_nodes_meta(root):
    path = os.path.join(root, "learning", "nodes.json")
    raw = read(path)
    doc = json.loads(raw)
    for nid, (lvl, skills, pre) in META.items():
        doc["nodes"][nid] = {"level": lvl, "skills": list(skills), "prereqs": list(pre)}
    write(path, dump_like(raw, doc, detect_indent(raw)))


def resolve_via(via, atlas_edges):
    if via is None or isinstance(via, str):
        return via
    a, b, rel = via
    eid = edge_id(a, b, rel)
    if eid not in atlas_edges:
        sys.exit("path via edge %s|%s|%s not found" % via)
    return eid


def apply_paths(root, atlas):
    path = os.path.join(root, "learning", "paths.json")
    raw = read(path)
    doc = json.loads(raw)
    ae = {e["id"] for e in atlas["edges"]}
    pos = {p["id"]: i for i, p in enumerate(doc["paths"])}
    for p in PATHS:
        q = dict(p)
        q["steps"] = [dict(s, via=resolve_via(s["via"], ae)) for s in p["steps"]]
        if q["id"] in pos:
            doc["paths"][pos[q["id"]]] = q
        else:
            doc["paths"].append(q)
    write(path, dump_like(raw, doc, detect_indent(raw)))
    return doc


def apply_background(root):
    """background.json keeps one option per line; edit it textually to preserve that layout."""
    path = os.path.join(root, "learning", "background.json")
    raw = read(path)
    nl, _ = style_of(raw)
    json.loads(raw)
    line = json.dumps(BG_OPTION, ensure_ascii=False)
    lines = raw.split(nl)
    # replace an existing copy
    for i, ln in enumerate(lines):
        if ('"id": "%s"' % BG_OPTION["id"]) in ln:
            indent = ln[:len(ln) - len(ln.lstrip())]
            lines[i] = indent + line + ("," if ln.rstrip().endswith(",") else "")
            break
    else:
        # find the question block, then its last option line
        qi = next(i for i, ln in enumerate(lines) if ('"id": "%s"' % BG_QUESTION) in ln)
        oi = next(i for i in range(qi, len(lines)) if lines[i].strip().startswith('"options"'))
        end = next(i for i in range(oi + 1, len(lines)) if lines[i].strip().startswith("]"))
        last = end - 1
        indent = lines[last][:len(lines[last]) - len(lines[last].lstrip())]
        if not lines[last].rstrip().endswith(","):
            lines[last] = lines[last].rstrip() + ","
        lines.insert(end, indent + line)
    text = nl.join(lines)
    json.loads(text)
    write(path, text)


def balanced(items):
    """Move each item's correct choice to position k mod n so answer indices are spread evenly."""
    out = []
    for k, it in enumerate(items):
        ch = list(it["choices"])
        right = ch.pop(it["answer"])
        tgt = k % (len(ch) + 1)
        ch.insert(tgt, right)
        out.append(dict(it, choices=ch, answer=tgt))
    return out


def apply_questions(root):
    ldir = os.path.join(root, "learning")
    qpath = os.path.join(ldir, QFILE)
    write(qpath, json.dumps({"version": 1, "items": balanced(QUESTIONS)}, indent=2, ensure_ascii=False) + "\n")
    ipath = os.path.join(ldir, "index.json")
    raw = read(ipath)
    doc = json.loads(raw)
    if QFILE not in doc["questions"]:
        doc["questions"].append(QFILE)
    write(ipath, dump_like(raw, doc, detect_indent(raw)))
    return doc


def cluster_phrase(names):
    parts = [n.replace(" & ", " and ") for n in names]
    parts = [p[0].lower() + p[1:] for p in parts]
    return "; ".join(parts[:-1]) + "; and " + parts[-1] if len(parts) > 1 else parts[0]


def sub1(pattern, repl, text, what):
    new, n = re.subn(pattern, repl, text, count=1)
    if n == 0:
        print("  warning: could not find %s" % what)
    return new


def apply_counts(root, atlas, paths_doc, index_doc):
    nn, ne, nc = len(atlas["nodes"]), len(atlas["edges"]), len(atlas["clusters"])
    by = {t: 0 for t in ("exact", "established", "empirical", "hypothesis")}
    for e in atlas["edges"]:
        by[e["type"]] = by.get(e["type"], 0) + 1
    banks = []
    for f in index_doc["questions"]:
        try:
            banks.append(len(json.loads(read(os.path.join(root, "learning", f)))["items"]))
        except (OSError, ValueError, KeyError):
            pass
    rpath = os.path.join(root, "README.md")
    r = read(rpath)
    r = sub1(r"The atlas has \d+ topics in \d+ clusters, joined by \d+ recorded connections\. The clusters are [^.]*\.",
             "The atlas has %d topics in %d clusters, joined by %d recorded connections. The clusters are %s." % (
                 nn, nc, ne, cluster_phrase([c["name"] for c in atlas["clusters"]])), r, "README topic sentence")
    labels = {"exact": "Exact, conditional", "established": "Established", "empirical": "Empirical", "hypothesis": "Hypothesis"}
    for t, lab in labels.items():
        r = sub1(r"\| %s \| \d+ \|" % re.escape(lab), "| %s | %d |" % (lab, by[t]), r, "README row " + lab)
    r = sub1(r"There are \d+ curated paths", "There are %d curated paths" % len(paths_doc["paths"]), r, "README path count")
    if banks:
        r = sub1(r"The current banks hold \d+ items \([^)]*\)\.",
                 "The current banks hold %d items (%s)." % (sum(banks), " + ".join(str(b) for b in banks)), r, "README bank count")
    if CLUSTER_NOTE not in r:
        r = sub1(r"(Only a drawn, labelled connection records a relationship\.)", lambda m: m.group(1) + " " + CLUSTER_NOTE, r, "README layout note")
    write(rpath, r)
    hpath = os.path.join(root, "index.html")
    h = read(hpath)
    h = re.sub(r"\d+ research topics and \d+ (typed|evidence-labelled) connections",
               lambda m: "%d research topics and %d %s connections" % (nn, ne, m.group(1)), h)
    h = sub1(r"\d+ topics · \d+ typed connections", "%d topics · %d typed connections" % (nn, ne), h, "index.html eyebrow")
    write(hpath, h)
    return nn, ne, nc, by


def apply_cachebust(root):
    apath = os.path.join(root, "app.js")
    a = read(apath)
    v = hashlib.sha1(read(os.path.join(root, "atlas.json")).encode("utf-8")).hexdigest()[:8]
    a2 = re.sub(r"fetch\('atlas\.json(\?v=[^']*)?'\)", "fetch('atlas.json?v=%s')" % v, a, count=1)
    write(apath, a2)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=".", help="repo root (default: cwd)")
    root = os.path.abspath(ap.parse_args(argv).root)
    atlas = apply_atlas(root)
    apply_nodes_meta(root)
    paths_doc = apply_paths(root, atlas)
    apply_background(root)
    index_doc = apply_questions(root)
    nn, ne, nc, by = apply_counts(root, atlas, paths_doc, index_doc)
    apply_cachebust(root)
    mine = [n["id"] for n in NODES if n["group"] == MI]
    print("add_mechinterp: %d topics, %d edges, %d clusters; cluster %r has %d nodes; %d edges written by this script"
          % (nn, ne, nc, MI, len(mine), len(EDGES)))
    print("  edge types (all): " + ", ".join("%s %d" % kv for kv in by.items()))


if __name__ == "__main__":
    main()
