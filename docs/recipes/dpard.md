# D-PARD and DPALA for DFlash1

D-PARD is selected with `training.strategy: dflash` and
`training.loss_type: dpard`. It trains the DFlash1 draft with full-vocabulary
Rényi-half divergence and detached position weights based on exact
rejection-sampling acceptance. It adds no confidence head or candidate selector.
DFlash2, DSpark, and LK-loss combinations are not supported for this objective.

For teacher distribution `p_t` and draft distribution `q_t` at temperature 1:

```text
R_t = -2 log sum_v sqrt(p_t(v) q_t(v))
a_t = sum_v min(p_t(v), q_t(v))
s_t = alpha + (1 - alpha) a_t
W_t = stop_gradient(sum_{k=t}^D product_{i=1}^k s_i)
```

The objective sums `W_t R_t` over supervised predicted positions. It uses
D-PACE's native sequence-anchor reduction: average over valid anchors within
each sequence, then average over valid sequences. The position weights are
not normalized by their sum. The clean anchor token is excluded from the
prediction loss; masked positions contribute no loss or continuation weight.
`training.dpard_alpha` is in `[0, 1]` and defaults to `0.5`.

DPALA selects `training.loss_type: dpala` and uses the same detached overlap
weights and sequence-anchor reduction, with actor `-log(a_t)` instead of
Rényi-half divergence. Its smoothing parameter is `training.dpala_alpha`
(default `0.5`, in `[0, 1]`); the training example uses `0.3`.
The actor is computed in log space as `-logsumexp(min(log_p, log_q))`.
Both objectives reduce to hard-label CE with a point-mass teacher.

For both training objectives, the frozen target head projects each sequence
position once per forward, outside activation-checkpoint recomputation. Each objective
chunk gathers its teacher logits by predecessor position. This avoids repeated
target-head matrix products for overlapping anchors without materializing an
anchor-by-vocabulary cache. The sequence logits remain live until backward
finishes; memory scales with batch size, sequence length, and vocabulary size.
Optional detailed metrics retain their separate teacher-projection path.

## Prepare offline features

From the repository root, with the offline SGLang capture dependencies installed:

```bash
torchrun --nproc_per_node=1 scripts/prepare_hidden_states.py \
  --target-model-path Qwen/Qwen3-4B \
  --strategy dflash \
  --loss-type dpard \
  --draft-model-config configs/qwen3-4b-dflash-3l-b16.json \
  --data-path ./cache/dataset/sharegpt_train.jsonl \
  --output-path ./cache/hidden_states/qwen3-4b-dflash-dpard \
  --chat-template qwen \
  --max-length 3072 \
  --tp-size 1 \
  --batch-size 1
```

Use the same draft JSON for capture and training. This configuration captures
target layers `[1, 17, 33]` and saves four tensors per sample: `input_ids`,
`loss_mask`, `hidden_states`, and `target_last_hidden_states`.
`--loss-type dpard` is required to retain the final teacher states. Ordinary
three-field DFlash/D-PACE caches remain valid for those objectives but cannot
train D-PARD; regenerate them with the command above. Capture still uses the
DFlash backend, not DSpark.

## Train

```bash
specforge train --config examples/configs/offline/colocated/qwen3-4b-dflash-dpard-offline.yaml --plan
specforge train --config examples/configs/offline/colocated/qwen3-4b-dflash-dpard-offline.yaml

# DPALA uses the same captured features.
specforge train --config examples/configs/offline/colocated/qwen3-4b-dflash-dpala-offline.yaml --plan
specforge train --config examples/configs/offline/colocated/qwen3-4b-dflash-dpala-offline.yaml
```

The example uses Qwen3-4B, three full-attention draft layers, B16, up to 512
anchors per sequence, alpha 0.5, and seed 42. It trains for six epochs on two
GPUs with per-rank batch size 1 and accumulation 2. Change the model, data,
output, and process-count settings for your environment before launching.

D-PARD and DPALA automatically enable sequence-anchor reduction. To use the same
reduction with a static DFlash baseline, set `training.loss_type: dflash` and
`training.dflash_normalize_by_anchors: true`; its default `false` preserves
legacy static DFlash normalization. D-PACE already uses sequence-anchor
reduction without that flag.

`dpard_loss` and `dpala_loss` report their objectives through the standard trainer metrics.
The selected objective, effective alpha, and anchor-normalization setting are
recorded in checkpoint resume contracts.
