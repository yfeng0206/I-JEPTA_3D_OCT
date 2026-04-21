# I-JEPA for FairVision OCT Glaucoma Classification

Self-supervised pretraining using [I-JEPA](https://github.com/facebookresearch/ijepa) (Assran et al., CVPR 2023) on [Harvard FairVision](https://github.com/Harvard-Ophthalmology-AI-Lab/FairVision) OCT data, evaluated via frozen probe + fine-tune on binary glaucoma classification.

## Headline results

All on FairVision glaucoma held-out test split (3000 volumes). Encoder: random-init I-JEPA ViT-B/16, 100 epochs SSL on 600K OCT slices.

| Method | Probe | Params (trainable) | **Test AUC** |
|---|---|---|---|
| **Fine-tune + LLRD γ=0.5** | AttentiveProbe d=1 + Linear | 7.17M + 86M encoder | **0.8878** |
| **Fine-tune + LLRD γ=0.5** | CrossAttnPool + Linear | 277K + 86M encoder | **0.8872** |
| **Fine-tune + LLRD γ=0.5** | **MeanPool + Linear (0 probe params)** | **2.3K + 86M encoder** | **0.8868** |
| Frozen probe | CrossAttnPool + Linear | 277K | 0.8791 |
| Frozen probe | MeanPool + Linear (no attention, no pos) | 2.3K | 0.8746 |
| Frozen probe | AttentiveProbe d=1 + Linear | 7.17M | 0.8706 |

**Headline finding — fine-tune collapses the probe ablation.** All three fine-tune runs land within 0.001 AUC of each other (p > 0.6 all pairwise). Under fine-tuning, the probe architecture is irrelevant: MeanPool (0 probe params, just a 2.3K linear head) matches the 7.17M AttentiveProbe and the 277K CrossAttnPool. Whatever slice-weighting the attention probes learn in the frozen regime, the encoder's top-block + encoder.norm adaptation absorbs under fine-tune.

**Ablation findings** (paired bootstrap, B=2000, 95% CI):
- **Frozen: probe architecture matters.** CrossAttnPool (277K) beats d=1 (7.17M) by +0.009 (p=0.002). CrossAttnPool beats MeanPool by +0.005 (p=0.04). d=1 fails to improve over MeanPool (p=0.08, ns) despite 3000× more params — d=1 is over-parameterized.
- **Fine-tune: probe architecture is noise.** d=1 vs MeanPool: Δ=+0.0009 (p=0.69, ns). CrossAttnPool vs MeanPool: Δ=+0.0004 (p=0.63, ns). d=1 vs CrossAttnPool: Δ=+0.0005 (p=0.81, ns).
- **Fine-tune uplift is real on every probe**: +0.0172 on d=1 (p<0.001), +0.0122 on MeanPool (p<0.001), +0.0080 on CrossAttnPool (p=0.009). Uplift scales inversely with probe capacity — d=1 had the most room to recover, CrossAttnPool the least.
- **Practical takeaway**: for fine-tune protocols, MeanPool is Pareto-optimal (zero probe params, matches best). For frozen-probe protocols, CrossAttnPool (277K) is Pareto-optimal.

Full statistical analysis: [`docs/experiments/frozen/ablation_analysis.md`](docs/experiments/frozen/ablation_analysis.md).

![Probe-architecture ranking on ep100](results/summary/probe_ranking_ep100.png)

## Interpretability — why the three probes tie

Architecture-agnostic occlusion attribution on all three fine-tune probes. **At the slice level, all three converge on the same bimodal structure along the disc-region axis** — MeanPool and CrossAttnPool curves correlate at r = 0.94, confirming the tied Test AUCs aren't coincidental.

Window occlusion (W=7 consecutive slices) gives ~7× cleaner signal than single-slice zero-masking and makes the shared structure obvious across all three probes:

![Window occlusion (W=7) across 3 fine-tune probes](results/summary/04_window_occlusion_W7.png)

At the patch level, per-volume correlation between probes drops to ~0.10 — the probes agree on *which slices* matter but each picks a different patch subset within those slices. Full writeup + 10 findings (including an OD/OS retraction on the earlier "bilateral disc rim" reading): [`docs/experiments/interpretability.md`](docs/experiments/interpretability.md).

Pretraining-epoch sweep (ep25/50/75/100) lives at [`docs/experiments/frozen/d1_sweep.md`](docs/experiments/frozen/d1_sweep.md).

## Method

- **Pretraining**: I-JEPA on 256×256 OCT slices (FairVision Training split, 600K slices). ViT-B/16, 100 epochs, peak LR 0.00025, EMA 0.996→1.0, effective batch 512.
- **Downstream input**: Frozen ViT encodes each slice, patches mean-pooled within slice → per-slice 768-dim token. 100 slices per volume.
- **Slice-aggregation probe**: CrossAttnPool (learnable query, single-head cross-attention head_dim=64, slice-axis pos_embed) OR AttentiveProbe d=1 (I-JEPA paper style).
- **Fine-tune**: LLRD γ=0.5 with base LR 2e-4, 50 epochs planned / early-stopped by patience=15.

See [`docs/architecture.md`](docs/architecture.md) for the full spec.

## Dataset

Harvard FairVision Glaucoma subset: 10,000 subjects (6K Train / 1K Val / 3K Test), each with a 200×200×200 OCT B-scan volume. Binary label glaucoma/not. ~48.5% positive prevalence — balanced. Available on [HuggingFace](https://huggingface.co/datasets/ming0100/Harvard_FairVision).

## Roadmap

- Phase 1 (done): Random-init I-JEPA SSL → frozen probe + fine-tune evaluation
- Phase 2 (done): Probe architecture ablations — full 2×3 matrix (3 probes × frozen/fine-tune)
- Phase 3 (done): Interpretability — occlusion attribution, patch aggregate, bootstrap CI
- Phase 4 (in progress): Foundation-model baselines on same Test split (DINOv3, OCTCube)
- Phase 5 (planned): 3D-aware SSL extension (multi-view / axial)

Details and backlog: [`docs/research_log.md`](docs/research_log.md).

## Links

| | |
|---|---|
| Pretraining | [`docs/experiments/pretraining`](docs/experiments/pretraining) |
| Frozen probe (d=1, CrossAttnPool, MeanPool) | [`docs/experiments/frozen`](docs/experiments/frozen) |
| Fine-tune (LLRD on d=1) | [`docs/experiments/finetune`](docs/experiments/finetune) |
| Model architecture | [`docs/architecture.md`](docs/architecture.md) |
| Lessons learned | [`docs/lessons_learned.md`](docs/lessons_learned.md) |
| Research log + paper bibliography | [`docs/research_log.md`](docs/research_log.md) |

## References

- Assran et al., *Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture* (I-JEPA), CVPR 2023. [arxiv 2301.08243](https://arxiv.org/abs/2301.08243)
- Bardes et al., *V-JEPA: Revisiting Feature Prediction for Learning Visual Representations from Video*, 2024. [arxiv 2404.08471](https://arxiv.org/html/2404.08471v1)
- Zhou et al., *Generalist vs Specialist Vision Foundation Models for Ocular Disease and Oculomics*, 2025. [arxiv 2509.03421](https://arxiv.org/abs/2509.03421v1)
- Zhou et al., *A Foundation Model for Generalizable Disease Detection from Retinal Images* (RETFound), Nature 2023. [paper](https://www.nature.com/articles/s41586-023-06555-x)
- Kakogeorgiou et al., *Attention, Please! Revisiting Attentive Probing for Masked Image Modeling*, ICLR 2026. [arxiv 2506.10178](https://arxiv.org/abs/2506.10178)
- Luo et al., *FairVision: Equitable Deep Learning for Eye Disease Screening*, 2024. [arxiv 2310.02492](https://arxiv.org/abs/2310.02492)

Full bibliography with context: [`docs/research_log.md`](docs/research_log.md#paper-bibliography).
