# Prefix Mode Constraints and Data Loss in SAP VallE Training

## Background

VallE's training pipeline supports several "prefix modes" that control how a NAR (non-autoregressive) acoustic prompt is constructed during training. For standard TTS training (e.g. LibriTTS), `prefix_mode=1` is the default: a random 3-second window is sampled from the source audio and used as a voice-style prompt, with the model trained to predict the remainder. For SAP voice conversion (VC) training, this approach **breaks down entirely** — both semantically and mechanically.

---

## Plain English: Why This Matters

Three things combine to make prefix_mode=1 wrong for SAP VC:

1. **The forward pass concatenates source and target into one tensor** (`valle.py:208`). prefix_mode=1 takes a random prefix from the *start* of this combined tensor — which is always source audio, since source comes first.

2. **The NAR targets are sliced to `targets[:, prefix_len:]`**, meaning the remaining source frames (from `prefix_len` up to `max_atypical_len`) are **included in the loss**. The masking that prevents this (`targets[:, :max_atypical_len] = NUM_AUDIO_TOKENS`) only fires for `prefix_mode == 0` (`valle.py:349-350`). With prefix_mode=1, there is no such masking — the model is penalised for predictions over source tokens it should never be predicting.

3. **Even if the masking were fixed**, a random 3-second window from the source has no frame-level correspondence to anything in the target. Source audio is longer than target in 87.4% of pairs (median 1.43× longer) — the two speakers produce the same utterance with different timing and speech rate. Position `t` in the source does not correspond to position `t` in the target.

prefix_mode=0 resolves all three issues: the full source is embedded as context with no prefix split, and the masking at `valle.py:349-350` ensures only target frames enter the loss.

---

## Why prefix_mode=1 Cannot Be Used for SAP VC

### The concatenation architecture

In VC training mode (`--voice-conversion 1`, `many_to_one=True`), the forward pass in `valle/models/valle.py` concatenates source (atypical) and target (typical) audio into a single sequence before the model sees it:

```python
# valle/models/valle.py:208
y = torch.cat([atypical_audio, y], dim=1)  # [source_frames | target_frames]
```

The model then predicts over this combined sequence. Frames 0 to `max_atypical_len` are source; frames `max_atypical_len` onwards are target. For correct VC training, only target frames should contribute to the loss.

### What prefix_mode=1 actually does to this concatenation

In `_prepare_prompts` (`valle/models/core.py:245-261`), prefix_mode=1 takes a random prefix from the **start** of the combined `y` tensor:

```python
int_low = (0.25 * y_lens.min()).type(torch.int64).item()
prefix_len = torch.randint(int_low, int_low * 2, size=()).item()
prefix_len = min(prefix_len, 225)  # 24000/320 * 3s = 225 frames

y_prompts = self.nar_audio_embeddings[0](y[:, :prefix_len])   # first prefix_len frames
y_emb     = self.nar_audio_embeddings[0](y[:, prefix_len:])   # remainder
```

Since source audio comes first in `y`, the prefix is always drawn from **source audio**. The remainder includes the rest of the source plus all of the target.

The NAR targets are then sliced to match (`valle/models/valle.py:346-347`):

```python
elif self.prefix_mode in [1, 5]:
    targets = targets[:, prefix_len:]
```

This means the NAR loss is computed over frames `prefix_len` to the end — which **includes source audio frames** (frames `prefix_len` to `max_atypical_len`). The source-frame masking that prevents this is **only applied for prefix_mode=0**:

```python
# valle/models/valle.py:349-350
if many_to_one and self.prefix_mode == 0:
    targets[:, :max_atypical_len] = NUM_AUDIO_TOKENS  # mask source from loss
```

With prefix_mode=1 in VC mode, there is no such masking — the model is penalised for its predictions over source audio tokens that are neither meaningful targets nor aligned to anything in the target. This corrupts the training signal for the NAR decoder.

### The alignment problem

Even setting aside the masking bug, a random 3-second prefix from source audio does not correspond to any aligned window in the target audio. Atypical and typical renditions of the same utterance have different speech rates, pausing patterns, and overall durations. Across the SAP training set:

| | Mean | Median | Max |
|---|---|---|---|
| **Source (atypical)** | 8.16s | 5.33s | 180.15s |
| **Target (typical)** | 5.73s | 3.55s | 302.94s |

- Source is longer than target in **87.4%** of pairs
- Median source/target ratio: **1.43×** (mean: 1.57×)
- When source is longer, it exceeds target duration by **2.94s on average**

There is no frame-level correspondence between source position `t` and target position `t`. A 3-second window starting at, say, second 2 of the source captures a different portion of the utterance content than second 2 of the target. Using it as the NAR acoustic prompt for that target would give the model inconsistent conditioning.

### prefix_mode=0 is the correct choice

With `prefix_mode=0` (`core.py:238-244`):

```python
if self.prefix_mode == 0:
    prefix_len = 0
    y_emb = self.nar_audio_embeddings[0](y)  # embed full [source | target]
```

The entire source audio is embedded as context with no prefix split. Combined with the VC-specific masking at line 349-350, only target frames contribute to the NAR loss. This is the only prefix mode that correctly implements "condition on the full atypical speaker context, predict only the typical output" for parallel VC training.

---

## Data Loss Consequence

Using `prefix_mode=0` means the **full source sequence** is fed through the model on every step, not just a 3-second snippet. This creates two constraints that force us to discard a portion of the training data.

### 1. Memory underestimation by the sampler

The `DynamicBucketingSampler` estimates batch memory usage from the **source** cuts only (from `sap_recordings_train_source.jsonl`). But in VC mode the model actually receives `source_len + target_len` tokens per example. Given the typical ratio, actual GPU memory per item is roughly **1.57× higher** than the sampler assumes. This means the effective safe `--max-duration` is lower than the sampler's parameter suggests.

### 2. Quadratic duration penalty

The sampler is configured with `quadratic_duration=10` (`datamodule.py:333`). This accounts for transformer attention's quadratic memory cost by inflating the effective batch cost of long sequences:

```
effective_cost = duration × max(1, duration / quadratic_duration)
```

With `--max-duration 40` and `quadratic_duration=10`, the breakeven point — where a single source clip exhausts the entire batch budget — is:

```
D × (D / 10) = 40  →  D = sqrt(400) = 20s
```

Any source clip longer than 20 seconds has an effective cost exceeding 40 seconds and is placed in a single-item batch, producing the warning:

```
WARNING: Batch is taking 1 cut because the cuts are too long.
```

Source clips between 20–25 seconds (our current `--filter-max-duration 25`) consistently trigger this. Setting `--filter-max-duration 20` eliminates it.

### Clips removed at filter-max-duration=20

| Duration range | Count | % of total |
|---|---|---|
| 0–5s | 53,362 | 45.1% |
| 5–10s | 44,461 | 37.6% |
| 10–15s | 9,154 | 7.7% |
| 15–20s | 3,498 | 3.0% |
| **≤ 20s (kept)** | **110,475** | **93.5%** |
| 20–30s | 3,513 | 3.0% |
| 30–60s | 3,310 | 2.8% |
| 60s+ | 896 | 0.8% |
| **> 20s (removed)** | **7,719** | **6.5%** |

**6.5% of source clips are removed** (7,719 of 118,194). These are typically multi-sentence or run-on recordings that are less representative of the sentence-level VC task anyway.

### Recommended training flags

```bash
--max-duration 40 \
--filter-min-duration 0.5 \
--filter-max-duration 20 \
--prefix-mode 0 \
--voice-conversion 1
```

If GPU OOM is still observed on clips near the 20s boundary (which contribute ~20s source + ~14s target = ~34s total to the model), reduce `--filter-max-duration` to 15 or lower `--max-duration`. To guarantee batches of at least 2 clips for all retained audio, clips must satisfy `2 × D²/10 ≤ 40`, giving `D ≤ ~14s` — so `--filter-max-duration 14` is a conservative safe bound.

---

## Summary

| Issue | Root cause | Fix |
|---|---|---|
| prefix_mode=1 corrupts NAR loss in VC | No source-frame masking for prefix_mode≠0; source frames included in target loss | Use `--prefix-mode 0` |
| Random 3s prefix misaligns with target | Source/target have different durations (1.57× ratio); no frame-level correspondence | Use `--prefix-mode 0` (full source context) |
| 1-cut batch warnings | Source clips >20s exceed effective batch budget after quadratic penalty | Set `--filter-max-duration 20` |
| Memory underestimation | Sampler accounts for source duration only; model processes source+target | Account for ~1.57× memory overhead when setting `--max-duration` |
