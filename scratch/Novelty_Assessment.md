# Novelty Assessment — Probe-Heads as PPO Rewards for VLM Debiasing

**Project:** Debias_VLMs (Qwen2.5-VL-3B + LoRA + custom PPO, SB-Bench training, BBQ transfer eval)
**Reward variants under review:**
- **Variant A — SVM heads:** linear SVM probes trained on penultimate hidden states of (chosen, rejected) SB-Bench pairs, used as the dense PPO reward (Phase 0.6 F).
- **Variant B — PCA-DRM heads:** 100-component PCA basis fit on the (chosen − rejected) embedding-difference matrix, 90/200 kept via `kept_heads_pca.json`, used as a decomposed PPO reward (Phase 0.7 PCA branch).

**Empirical status at time of writing:** SB-Bench acc = 1.0000 across all 18 checkpoints for the SVM variant; PCA-DRM training+eval pipeline mirrored. Gates G1 (polarity), G2 (cyclic-1 perturbation), G3 (POPE) PASS; **G4 (BBQ transfer) PENDING.** The 1.0 in-distribution number is interpreted as a ceiling of reward-model distillation — the probes themselves score ~1.0 on the same (chosen, rejected) split during D3.

**Target venues considered:** SoLaR @ NeurIPS, Trustworthy ML, SafeGenAI workshops.

---

## Section A — SVM Probe-as-Reward: Novelty Analysis

### A.1 What it is, precisely

For each bias category, a linear SVM is fit on the penultimate-layer hidden state at the EOS position of `(prompt + full_answer_text)` for chosen vs. rejected SB-Bench completions. At PPO training time, the signed margin `w·h + b` from these probes is normalized and used as the per-token reward. Composition:
`r_t = r_task_normalized + max(0, Δ)*0.1 + causal_per_token − 0.1*(exp(Δ) − 1 − Δ)`.

### A.2 Direct precedents (this is **not** novel as a mechanism)

| Work | What they did | Why it dominates the novelty claim |
|---|---|---|
| **Papadatos & Freedman, "Linear probe penalties reduce LLM sycophancy" (arXiv 2412.00967, Dec 2024)** | Train linear probes on hidden activations for an unwanted behaviour, use the probe score as an RL penalty during fine-tuning. | This is essentially the same architectural template ("classifier-probe-as-RL-reward-signal"), one bias-axis upstream. Any "novelty of using a linear classifier head as a reward" claim is foreclosed by this paper. |
| **SWIFT (Guo et al., KDD 2026, arXiv 2505.12225)** | Probe-guided RL fine-tuning using internal-representation classifiers as reward terms. | Concurrent confirmation that the probe-as-reward template is now a recognised pattern, not a novel one. |
| **Yang et al., Rewards-in-Context (NeurIPS 2024, arXiv 2406.10216)** | Treat a lightweight learned scoring head as a weak-supervision reward inside the RLHF loop. | Same family. |

### A.3 Conceptual lineage (older work the reviewer will cite)

- **TCAV** (Kim et al., ICML 2018) — concept activation vectors from linear probes.
- **Representation Engineering / RepE** (Zou et al., 2023) — linear-probe direction as a control signal.
- **Inference-Time Intervention (ITI)** (Li et al., NeurIPS 2023) — probe directions added at inference for truthfulness.
- **Contrastive Activation Addition (CAA)** (Xia et al., NAACL 2024) — direction = mean(chosen) − mean(rejected) hidden states; steering at inference.

The user's SVM is closer to CAA-style "mean-difference direction" than to a learned RM, but **interpreted as a reward**, it collapses back to the Papadatos & Freedman template.

### A.4 What is genuinely new about Variant A in this project

1. **Modality.** All cited precedents are LLM-only. Applying probe-as-reward to a **vision-language model** (Qwen2.5-VL-3B) is mildly new (~1 increment); MJ-Bench (arXiv 2407.04842) already establishes multimodal RM evaluation, so this alone is thin.
2. **Per-category probe bank.** A separate SVM per bias dimension (rather than one global probe) is an engineering wrinkle, not a research contribution.
3. **Composite reward formulation** with the asymmetric `max(0, Δ)*0.1` bonus and the `−0.1·(exp(Δ) − 1 − Δ)` regulariser is original phrasing, but probably re-derivable from "shaped logistic" baselines.
4. **The oracle-distillation finding itself** — that probe-as-reward + on-distribution PPO + on-distribution eval = vacuous 1.0 — is the most defensible contribution. It is a **diagnostic / negative result** about the methodology, not a positive performance claim.

### A.5 The decisive risk: oracle distillation

The probe achieves ~1.0 accuracy on the (chosen, rejected) split during D3. Using that probe as the PPO reward on prompts drawn from the same distribution drives the policy to output the chosen-class direction; evaluating that policy on SB-Bench then re-measures the same property the probe was trained for. The policy converges to "what the probe rewards," not to "what an unbiased VLM should do." This is the same failure mode catalogued by **reward hacking / Goodhart in RLHF** (Gao et al., Skalse et al.); the novelty of *naming and isolating it for probe-rewards in the bias setting* is what would survive review.

### A.6 Required baselines (to make Variant A defensible)

1. **Random-projection probe.** Same shape, random `w`. If PPO with this also hits 1.0 on SB-Bench, the SVM signal is decorative.
2. **Frozen-embedding logistic regression as RM.** Same data, simpler probe — does it produce the same policy?
3. **DPO on the same (chosen, rejected) pairs.** Without any probe-as-reward, on the same data.
4. **CAA steering with no PPO.** Add `mean(chosen) − mean(rejected)` direction at inference; cheapest baseline.
5. **Out-of-distribution evaluation only** (BBQ, StereoSet, CrowS-Pairs). Drop SB-Bench from the eval table for the headline number.

### A.7 Drafted related-work paragraph (SVM)

> Probe-as-reward methods extend a long line of representation-engineering work — TCAV (Kim et al., 2018), RepE (Zou et al., 2023), ITI (Li et al., 2023), CAA (Xia et al., 2024) — by promoting a linear classifier on internal activations to the role of an RL reward signal. The most direct precedent is Papadatos & Freedman (2024), who use probe scores as RLHF penalties to suppress sycophancy in LLMs; SWIFT (Guo et al., 2026) and Rewards-in-Context (Yang et al., 2024) instantiate the same pattern in adjacent settings. Our contribution is not the mechanism but its application to multimodal social-bias debiasing under a controlled in-distribution-vs-transfer evaluation, where we isolate an oracle-distillation failure that the existing literature does not flag explicitly.

### A.8 Recommended framing for Variant A

> *"We adapt the probe-as-reward paradigm (Papadatos & Freedman, 2024) to multimodal social-bias debiasing and characterise a previously-unreported oracle-distillation failure: when the probe reward and the evaluation set are drawn from the same distribution, PPO converges to a vacuous 1.0 accuracy without any genuine bias reduction transfer (BBQ)."*

---

## Section B — PCA-DRM Heads as PPO Reward: Novelty Analysis

### B.1 What it is, precisely

For each bias category, compute embedding differences `d_i = h_chosen_i − h_rejected_i` over SB-Bench pairs; PCA-fit a 100-component basis on `D = [d_1, …, d_N]`; keep the top 90/200 components per `kept_heads_pca.json` (selected by per-head separability on a held-out split). At PPO time, the per-head signed projection `pc_k · h_t` is used as a *decomposed* reward signal, mirroring Phase 0.6 F's SVM pipeline but with PCA components in place of SVM normals. All other hyperparameters identical (lr=1e-4, kl_beta=0.1, target_kl=0.02, max_grad_norm=0.1, cosine schedule, ckpt every 10 steps, `lambda_causal=0.0`).

### B.2 Direct precedent (this is where the novelty ceiling sits)

**Luo et al., "Rethinking Diverse Human Preference Learning through Principal Component Analysis" (arXiv 2502.13131, Feb 2025; v2 Jun 2025).** The paper introduces *Decomposed Reward Models (DRMs)*:

> "We represent human preferences as vectors and analyze them using Principal Component Analysis (PCA). By constructing a dataset of embedding differences between preferred and rejected responses, DRMs identify orthogonal basis vectors that capture distinct aspects of preference. These decomposed rewards can be flexibly combined to align with different user needs … DRMs effectively extract meaningful preference dimensions (e.g., helpfulness, safety, humor) and adapt to new users without additional training."

Code: <https://github.com/amandaluof/DRMs>.

**This is the exact mechanism of Variant B**: PCA on (chosen − rejected) embedding differences → orthogonal basis vectors → each component used as a reward. The "PCA on preference-embedding differences → decomposed rewards" contribution belongs to Luo et al. and must be cited as the source of the method, not as related work.

### B.3 Adjacent work (DRM-flavoured)

| Work | Connection |
|---|---|
| **MJ-Bench** (Chen et al., arXiv 2407.04842, 2024) | Multimodal reward-model benchmark; the natural evaluation harness once Variant B is ported to a VLM. |
| **Proximalized Preference Optimization for Diverse Feedback Types: A Decomposed Perspective on DPO** (arXiv 2505.23316, 2025) | Decomposed objective view of DPO, complementary algorithmic decomposition. |
| **TGDPO: Token-Level Reward Guidance for DPO** (arXiv 2506.14574, ICML 2025) | Token-level decomposition of preference reward; relevant to how PCA-DRM heads aggregate per-token. |

### B.4 What is **not** novel about Variant B

1. **The PCA-on-embedding-differences construction.** This is the headline contribution of Luo et al. 2025 verbatim.
2. **Treating each component as an independent reward dimension.** Also Luo et al.; their pitch is exactly "interpretable, combinable basis vectors."
3. **Using preference-embedding orthogonal bases for alignment.** Foundational claim of DRM.

Any paper-level novelty claim for Variant B has to start with "we apply DRM (Luo et al., 2025) to …" — the framework itself is borrowed.

### B.5 What **is** plausibly novel about Variant B in this project

1. **Multimodal extension (VLM, not LLM).** Luo et al. evaluated DRM only on text LLMs. Lifting it to Qwen2.5-VL with image conditioning on the embedding extractor is a genuine first-of-kind operationalisation. This is the strongest novelty axis.
2. **DRM as a PPO training reward, not an inference-time combination.** Luo et al. compose head weights at inference for personalised alignment; they do **not** roll the heads into a PPO loop. Using DRM components as the dense reward inside on-policy RL is, to current best knowledge, unreported.
3. **Bias-axis decomposition vs. personalisation-axis decomposition.** Luo et al. interpret the PCs as helpfulness/safety/humor; using PCA components as a basis for *social-bias categories* is a different semantic claim and would need a per-PC interpretability section (which would also be a publishable mini-contribution).
4. **Kept-head subset selection (`kept_heads_pca.json`, 90/200).** Luo et al. weight all heads. A filtered subset selected by held-out probe separability is an engineering addition that, if shown to matter empirically (ablation: kept vs. all), is a small contribution.
5. **Cross-method comparison.** Running SVM-decomposed (Variant A) and PCA-decomposed (Variant B) rewards under identical PPO settings on the same VLM and the same evaluation suite is itself useful — most published work isolates one decomposition.

### B.6 Same decisive risk — oracle distillation

Variant B inherits Variant A's failure mode unchanged. The PCA basis is fit on SB-Bench embeddings; if SB-Bench is also the evaluation set, the PPO policy is being scored against the directions in which the reward was learned. The 1.0 accuracy is again a ceiling artefact, not evidence of debiasing. The same diagnostic (BBQ T1 transfer) is the only thing that can separate "decomposed reward genuinely reduces bias" from "policy learned to push along the PCA basis trained on its own eval distribution."

### B.7 Required baselines (to make Variant B defensible)

1. **Luo et al.'s DRM recipe ported verbatim.** Inference-time head combination, no PPO. Establishes whether PPO adds anything beyond the published DRM.
2. **Random-projection heads of the same dimensionality (90/200).** Sanity baseline — does the PCA structure matter, or would any 90-dim basis on these activations work?
3. **All 200 PCA heads with no filtering.** Tests whether `kept_heads_pca.json` is doing real work.
4. **SVM heads (Variant A) under identical PPO settings.** Apples-to-apples decomposition comparison.
5. **Activation-steering with the top-k PCA directions, no PPO.** Direct CAA-style sanity check using the same directions.
6. **Out-of-distribution eval primary (BBQ, StereoSet, CrowS-Pairs).** SB-Bench reported only as an in-distribution sanity number.

### B.8 Drafted related-work paragraph (DRM)

> Our PCA-decomposed reward instantiation directly adapts Decomposed Reward Models (Luo et al., 2025), which extract orthogonal preference dimensions by running PCA on the matrix of (chosen − rejected) embedding differences and use the resulting basis as a combinable set of interpretable reward heads. Where Luo et al. apply DRM at inference time for personalised LLM alignment along open-ended dimensions (helpfulness, safety, humor), we (i) lift the construction to a multimodal backbone (Qwen2.5-VL-3B), (ii) repurpose the orthogonal basis to span pre-specified social-bias categories, and (iii) integrate the per-head signals as the dense reward of an on-policy PPO loop rather than an inference-time mixture. Concurrent work on decomposed DPO objectives (Proximalized PO, arXiv 2505.23316; TGDPO, arXiv 2506.14574) and multimodal reward-model evaluation (MJ-Bench, arXiv 2407.04842) addresses adjacent slices of this design space but does not combine PCA-decomposed preference bases with multimodal on-policy fine-tuning for bias mitigation.

### B.9 Recommended framing for Variant B

> *"We present, to our knowledge, the first integration of Decomposed Reward Models (Luo et al., 2025) into an on-policy PPO loop for multimodal social-bias debiasing, and the first head-to-head comparison of SVM-decomposed and PCA-decomposed reward signals under identical fine-tuning conditions. We further characterise an oracle-distillation failure mode shared by both decompositions: when the head basis and the evaluation distribution coincide, in-distribution accuracy saturates at 1.0 without delivering measurable cross-benchmark bias reduction."*

---

## Section C — Combined Verdict and Recommended Path

### C.1 Where the novelty actually sits

Neither **probe-as-reward** (Variant A) nor **PCA-decomposed reward** (Variant B) is a novel mechanism. The mechanisms belong to Papadatos & Freedman (2024) and Luo et al. (2025) respectively. What is genuinely new in this project, in decreasing order of defensibility:

1. **Cross-decomposition comparison.** A controlled SVM-vs-PCA reward ablation on the same VLM, same PPO recipe, same data, same evaluation. — *Defensible regardless of G4 outcome.*
2. **Multimodal lift of DRM into a PPO loop.** First operationalisation of DRM on a VLM and inside on-policy RL. — *Defensible regardless of G4 outcome, but weaker if BBQ transfer is null.*
3. **Oracle-distillation diagnostic.** A clean demonstration that probe-as-reward + on-distribution eval = vacuous 1.0, with per-head causal-ablation evidence. — *This is the strongest contribution if both variants fail G4.*
4. **Per-category head-bank engineering and `kept_heads_pca.json` filtering.** Engineering contributions; small alone, useful for ablations.

### C.2 Decision matrix conditional on G4 (BBQ transfer)

| BBQ outcome | Recommended paper framing |
|---|---|
| **G4 PASS (BBQ acc > base, ≥1pp, both variants)** | Positive result: "DRM- and SVM-decomposed rewards transfer to held-out bias benchmarks in VLMs." Lead with multimodal-DRM-in-PPO as the new methodological piece; in-distribution 1.0 becomes upper-bound evidence rather than the headline. |
| **G4 PASS for one variant only** | Comparative result: "PCA decomposition transfers, SVM does not (or vice versa); we characterise why." This is the most publishable outcome. |
| **G4 FAIL for both** | Diagnostic/negative-result paper: "Probe-as-reward debiasing collapses to oracle distillation under on-distribution evaluation." Lead with the failure mode, support with ablations 1–6 from §A.6/§B.7. Workshop-fit (SoLaR / SafeGenAI / Trustworthy ML) is *better* for this framing than for a thin positive result. |

### C.3 Shared required ablations (minimum viable contribution set)

Regardless of which framing wins:

- **A1.** Random-projection-head baseline (both variants).
- **A2.** OOD-only evaluation table (BBQ, StereoSet, CrowS-Pairs) as the headline; SB-Bench in supplementary.
- **A3.** Kept-vs-all PCA heads ablation.
- **A4.** Inference-time DRM (Luo et al. recipe) as a no-PPO baseline.
- **A5.** DPO on the same (chosen, rejected) pairs.
- **A6.** Per-head causal ablation (zero-out a probe direction at inference; measure SB-Bench drop) — supports the oracle-distillation argument.

### C.4 Target venues (unchanged from Section A)

- **SoLaR @ NeurIPS** — best fit for the diagnostic framing.
- **Trustworthy ML workshop** — accepts negative results and methodology critiques.
- **SafeGenAI** — accepts multimodal alignment work.

A main-conference NeurIPS/ICML submission is **not** advisable on the current evidence: the headline 1.0 is a ceiling, and the mechanism novelty is foreclosed by Papadatos & Freedman 2024 (for SVM) and Luo et al. 2025 (for PCA).

---

## Appendix — Source list (verified via arXiv / Semantic Scholar)

| ID | Authors / Year | Title |
|---|---|---|
| arXiv:2412.00967 | Papadatos & Freedman, 2024 | Linear probe penalties reduce LLM sycophancy |
| arXiv:2505.12225 | Guo et al., KDD 2026 | SWIFT (probe-guided RL fine-tuning) |
| arXiv:2406.10216 | Yang et al., NeurIPS 2024 | Rewards-in-Context |
| arXiv:2502.13131 | Luo, Yang, Sun, Deng, Yao, Shen, Zhang, Chen, 2025 | Rethinking Diverse Human Preference Learning through Principal Component Analysis (DRM) |
| arXiv:2407.04842 | Chen et al., 2024 | MJ-Bench: Multimodal RM evaluation |
| arXiv:2505.23316 | 2025 | Proximalized Preference Optimization for Diverse Feedback Types: Decomposed Perspective on DPO |
| arXiv:2506.14574 | 2025, ICML | TGDPO: Token-Level Reward Guidance for DPO |
| — | Kim et al., ICML 2018 | TCAV |
| — | Zou et al., 2023 | Representation Engineering |
| — | Li et al., NeurIPS 2023 | Inference-Time Intervention (ITI) |
| — | Xia et al., NAACL 2024 | Contrastive Activation Addition (CAA) |
| — | Gao et al. / Skalse et al. | Scaling-laws of reward model overoptimization / Defining reward hacking |
