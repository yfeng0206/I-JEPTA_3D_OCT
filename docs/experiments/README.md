# Experiments

Three sections: pretraining (the SSL runs), frozen (probe-only evaluation with encoder frozen), finetune (encoder unfrozen).

All results use FairVision glaucoma held-out Test split (3000 volumes). Encoder: ViT-B/16.

## Headline

| Stage | Probe | Params (trainable) | Test AUC | Detail |
|---|---|---|---|---|
| **Finetune** | AttentiveProbe d=1 + Linear, LLRD γ=0.5 | 7.17M + 86M encoder | **0.8878** | [finetune/llrd.md](finetune/llrd.md) |
| **Finetune** | CrossAttnPool + Linear, LLRD γ=0.5 | 277K + 86M encoder | **0.8872** | [finetune/llrd.md](finetune/llrd.md) |
| **Finetune** | MeanPool + Linear, LLRD γ=0.5 | 2.3K + 86M encoder | **0.8868** | [finetune/llrd.md](finetune/llrd.md) |
| Frozen | CrossAttnPool + Linear | 277K | 0.8791 | [frozen/cross_attn_pool.md](frozen/cross_attn_pool.md) |
| Frozen | MeanPool + Linear | 2.3K | 0.8746 | [frozen/mean_pool.md](frozen/mean_pool.md) |
| Frozen | AttentiveProbe d=1 + Linear | 7.17M | 0.8706 | [frozen/d1_sweep.md](frozen/d1_sweep.md) |

Best overall: **fine-tune with MAE-style LLRD at Test AUC 0.8878**. +0.017 over the frozen d=1 baseline.

Primary ablation finding: **under fine-tune, the probe architecture is irrelevant.** All three probes land within 0.001 AUC of each other (pairwise p > 0.6). MeanPool (0 probe params) matches AttentiveProbe d=1 (7.17M probe params) when the encoder is unfrozen.

Secondary finding (frozen regime only): **CrossAttnPool beats d=1 at 26× fewer params** (+0.009 AUC, p=0.002) — the self-attn + FFN in the I-JEPA-style attentive probe is over-parameterized for frozen-probe protocols.

## Structure

```
docs/experiments/
  pretraining/
    README.md
    random_100ep.md      random-init ViT-B/16, 100 ep SSL
  frozen/
    README.md
    d1_sweep.md          AttentiveProbe d=1 sweep across ep25/50/75/100
    cross_attn_pool.md   minimal cross-attention (277K params) on ep100
    mean_pool.md         mean-pool + linear (ablation floor) on ep100
  finetune/
    README.md
    llrd.md              unfrozen encoder + LLRD γ=0.5 on ep100
```

## Reference

- [research_log.md](../research_log.md) — chronological problem/solution log + paper bibliography + backlog
- [lessons_learned.md](../lessons_learned.md) — mistakes, fixes, invariants
- [architecture.md](../architecture.md) — model architecture spec
