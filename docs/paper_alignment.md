# Paper Alignment Notes (MiniOneRec-JAX)

This document records the current metric alignment state against the MiniOneRec paper (Table 1).

## Paper Targets (Table 1, MiniOneRec row)

- Industrial: `HR@3=0.1143`, `NDCG@3=0.1011`, `HR@5=0.1321`, `NDCG@5=0.1084`, `HR@10=0.1586`, `NDCG@10=0.1167`
- Office: `HR@3=0.1217`, `NDCG@3=0.1088`, `HR@5=0.1420`, `NDCG@5=0.1172`, `HR@10=0.1634`, `NDCG@10=0.1242`

## Current Reproduction Artifacts

### Industrial (full set)

- Predictions: `artifacts/eval/industrial_full_ropefix.json`
- Runtime log: `artifacts/logs/industrial_full_beam50_batch16_rerun.log` (source run log)
- `n_beam=50`, `sample_count=4533`, `missing_item_predictions=0`

| Metric | Run | Paper | Delta |
|---|---:|---:|---:|
| HR@3 | 0.113611294948 | 0.1143 | -0.000688705052 |
| NDCG@3 | 0.101758723470 | 0.1011 | +0.000658723470 |
| HR@5 | 0.132142069270 | 0.1321 | +0.000042069270 |
| NDCG@5 | 0.109439794417 | 0.1084 | +0.001039794417 |
| HR@10 | 0.157732186190 | 0.1586 | -0.000867813810 |
| NDCG@10 | 0.117574795587 | 0.1167 | +0.000874795587 |

### Office (full set, paper profile)

- Predictions: `artifacts/eval/office_full_beam16_batch16.json`
- Runtime log: `artifacts/logs/office_full_beam16_batch16.log`
- `n_beam=16`, `sample_count=4866`, `missing_item_predictions=0`

| Metric | Run | Paper | Delta |
|---|---:|---:|---:|
| HR@3 | 0.128853267571 | 0.1217 | +0.007153267571 |
| NDCG@3 | 0.114330274527 | 0.1088 | +0.005530274527 |
| HR@5 | 0.142416769420 | 0.1420 | +0.000416769420 |
| NDCG@5 | 0.120000640653 | 0.1172 | +0.002800640653 |
| HR@10 | 0.164611590629 | 0.1634 | +0.001211590629 |
| NDCG@10 | 0.127101142105 | 0.1242 | +0.002901142105 |

## Important Note on Beam Width

- The paper training/evaluation description explicitly mentions rollout with beam width `16`.
- `official-minionerec/evaluate.py` defaults to `num_beams=50`.
- For Office, this repo reproduces paper-level `HR@10` when using `--num-beams 16`.

## Recompute Metrics Locally

```bash
cd /home/john/workdir3/minionerec-jax
PYTHONPATH=src python -m minionerec_jax.cli eval-metrics \
  --predictions-json artifacts/eval/office_full_beam16_batch16.json \
  --item-info ../official-minionerec/data/Amazon/info/Office_Products_5_2016-10-2018-11

PYTHONPATH=src python -m minionerec_jax.cli eval-metrics \
  --predictions-json artifacts/eval/industrial_full_ropefix.json \
  --item-info ../official-minionerec/data/Amazon/info/Industrial_and_Scientific_5_2016-10-2018-11
```
