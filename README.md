# sap-data-preparation
Series of scripts to extract, generate synthetic pairs, and create the Lhotse recipe for the SAP dataset.

---

## Pipeline Overview

```
Step 1: extract_sap_data.py
    Raw SAP tar archives
        └─> extracted/{TRAIN,DEV}/<speaker_id>/

Step 2: extract_speaker_data_and_ratings.py
    extracted/
        └─> speaker_ratings_{TRAIN,DEV}.csv

Step 3: generate_synthetic_speech.py          [StyleTTS2 container]
    extracted/ + speaker_ratings_TRAIN.csv  (run once per etiology)
        └─> synthetic/Parkinsons_Disease/<speaker_id>/*_synthetic.wav
        └─> synthetic/Parkinsons_Disease/speaker_pairs_{TRAIN,DEV}.csv
        └─> synthetic/ALS/<speaker_id>/*_synthetic.wav
        └─> synthetic/ALS/speaker_pairs_{TRAIN,DEV}.csv

Step 4: calculate_sap_wer.py                  [NeMo container, GPU]
    extracted/ + speaker_ratings_TRAIN.csv
        └─> pd_train_wer.csv

Step 5: select_validation_speakers.py         (run once per etiology)
    pd_train_wer.csv  --etiology "Parkinson's Disease"
        └─> pd_val_speakers.csv
        └─> pd_val_speakers_distribution.png
    als_train_wer.csv  --etiology "ALS"
        └─> als_val_speakers.csv
        └─> als_val_speakers_distribution.png

Step 6: sap.py                                [StyleTTS2 container]
    Parkinsons_Disease/speaker_pairs_{TRAIN,DEV}.csv [+ ALS/...]
    + pd_val_speakers.csv [+ als_val_speakers.csv ...]
        └─> sap_recordings_{train,val,test}_{source,target}.jsonl.gz
        └─> sap_supervisions_{train,val,test}_{source,target}.jsonl.gz
```

Steps 3 and 6 require the **StyleTTS2 container**. Step 4 requires the **NeMo container** with GPU access. Steps 1, 2, and 5 can be run in either container or a standard Python environment.

---

Build Container Image:

In dir of Dockerfile:

```podman build -t <container_name>:latest .```

```podman tag <container_name>:latest <docker.io username>/<container_name>:latest```

```podman push <docker.io username>/<container_name>:latest```

On Cluster:

```apptainer build <container_name>.sif docker://<docker.io username>/<container_name>:latest```


## Environment Setup

Two separate containers are used to avoid dependency conflicts between StyleTTS2 and NeMo.

### StyleTTS2 container (`docker/Dockerfile`)
Used for Steps 3 and 6 (synthesis and Lhotse manifest generation).

```bash
bash run_sap_container.sh
```

Update `run_sap_container.sh` before running:
- `apptainer_image` — path to your `sap_data_prep.sif` file
- `--bind` — your scratch/working directory (format: `/your/path:/your/path`)

### NeMo container (`docker_nemo/Dockerfile`)
Used for Step 4 (WER calculation). Kept separate to avoid conflicts with StyleTTS2 dependencies.

```bash
bash run_nemo_container.sh
```

Update `run_nemo_container.sh` before running:
- `apptainer_image` — path to your `sap_data_prep_nemo.sif` file
- `--bind` — your scratch/working directory

Build and push the NeMo container the same way as the StyleTTS2 container (see top of this README), pointing at `docker_nemo/Dockerfile`.

---

## Order of Operations

### Step 1 — Extract the SAP dataset (`extract_sap_data.py`)

Extracts the raw SAP tar archives into per-speaker directories.

```bash
python extract_sap_data.py \
    --raw-sap-dir /path/to/sap/download \
    --output-dir /path/to/sap
```

**Output:** `<output-dir>/extracted/{DEV,TRAIN}/<speaker_id>/`
Each speaker directory contains a `.json` metadata file and `.wav` audio files.

---

### Step 2 — Extract speaker ratings (`extract_speaker_data_and_ratings.py`)

Parses each speaker's JSON and computes average intelligibility ratings per speaker.

```bash
python extract_speaker_data_and_ratings.py \
    --extracted-dir /path/to/sap/extracted \
    --output-dir /path/to/ratings
```

**Output:** `<output-dir>/speaker_ratings_TRAIN.csv` and `speaker_ratings_DEV.csv`
Columns include `Speaker_ID` and `Etiology`, which are required by `generate_synthetic_speech.py`.

---

### Step 3 — Generate synthetic speech (`generate_synthetic_speech.py`)

Uses the extracted speaker audio and the ratings CSV to synthesize speech with StyleTTS2. Must be run inside the Apptainer container (see [Environment Setup](#environment-setup)).

```bash
# Start the container
bash run_sap_container.sh

# Then inside the container:
python generate_synthetic_speech.py \
    --sap-data-dir /path/to/sap \
    --split TRAIN \
    --mode all \
    --ratings-csv /path/to/ratings/speaker_ratings_TRAIN.csv
```

Run again for the DEV split (used as the test set):

```bash
python generate_synthetic_speech.py \
    --sap-data-dir /path/to/sap \
    --split DEV \
    --mode all \
    --ratings-csv /path/to/ratings/speaker_ratings_DEV.csv
```

**Transcript used for synthesis:** The `Transcript` field from each utterance's JSON is used as the TTS input (not `Prompt Text`). This ensures the source (atypical speaker audio) and target (synthetic audio) share the same linguistic content — critical for voice conversion training. Two transcript cleaning steps are applied before synthesis:

- **`[bracketed]` annotations are stripped** — these appear at the start of spontaneous prompts and represent the cue shown on screen, not something the speaker said.
- **Utterances with `(cs: ...)` annotations are skipped entirely** — these recordings contain a second speaker (a clinician reading phrases for the speaker to repeat) and cannot be used for voice conversion.

Disfluencies and false starts written in `(parentheses)` are retained — they are present in the audio.

**Output:** `<sap-data-dir>/synthetic/<etiology>/<speaker_id>/<utterance>_synthetic.wav`

A `speaker_pairs_{SPLIT}.csv` is written per etiology:
- `--mode etiology` → `<sap-data-dir>/synthetic/<etiology>/speaker_pairs_{SPLIT}.csv`
- `--mode all` → `<sap-data-dir>/synthetic/speaker_pairs_{SPLIT}.csv` (combined)
- Use `--output-csv` to override the path in either mode.

CSV columns: `speaker_id`, `etiology`, `original_audio`, `synthetic_audio`, `transcript`, `prompt_text`, `category_description`, `status`.

#### `--mode` options

| Mode | Description | Additional required args |
|---|---|---|
| `all` | Synthesize all speakers in the split | `--ratings-csv` |
| `etiology` | Synthesize all speakers with a given etiology | `--ratings-csv`, `--etiology` |
| `speaker` | Synthesize a single speaker | `--speaker-id` |

Example — synthesize a single etiology:

```bash
python generate_synthetic_speech.py \
    --sap-data-dir /path/to/sap \
    --split TRAIN \
    --mode etiology \
    --etiology "Parkinson's Disease" \
    --ratings-csv /path/to/ratings/speaker_ratings_TRAIN.csv
```

---

### Step 4 — Calculate speaker WER (`calculate_sap_wer.py`)

Runs NeMo ASR on the extracted reference audio for each speaker and computes per-speaker Word Error Rate (WER). WER is used as an intelligibility proxy to stratify unrated speakers when selecting a validation set. Must be run inside the **NeMo container** with GPU access (see [Environment Setup](#environment-setup)).

Run on the TRAIN split, optionally filtered to a single etiology:

```bash
# Start the NeMo container first
bash run_nemo_container.sh

# Then inside the container:
python calculate_sap_wer.py \
    --extracted-dir /path/to/sap/extracted \
    --split TRAIN \
    --etiology "Parkinson's Disease" \
    --ratings-csv /path/to/ratings/speaker_ratings_TRAIN.csv \
    --output /path/to/pd_train_wer.csv
```

To process all etiologies, omit `--etiology` and `--ratings-csv`:

```bash
python calculate_sap_wer.py \
    --extracted-dir /path/to/sap/extracted \
    --split TRAIN \
    --output /path/to/train_wer.csv
```

The script saves results incrementally so it is safe to interrupt and resume — already-processed speakers are skipped automatically on restart.

**Ground truth source:** Defaults to `--gt-source transcript`, using the `Transcript` field (stripped of `[bracketed]` annotations) to match what is used for synthesis. Utterances with `(cs: ...)` annotations are skipped for the same reason as in Step 3. Use `--gt-source prompt-text` to score against the original `Prompt Text` instead.

**Output:** CSV with columns `Speaker_ID`, `Etiology`, `Average_Rating`, `Split`, `Num_Utterances`, `Num_Failed`, `Average_WER`.
A log file is written alongside the output CSV (e.g. `pd_train_wer.log`) unless overridden with `--log-file`.

---

### Step 5 — Select validation speakers (`select_validation_speakers.py`)

Uses the WER results from Step 4 to select a stratified validation set from TRAIN speakers. Each rating bin contributes a fixed fraction of its available speakers, ensuring coverage across the full severity range. This generalises across etiologies — run once per WER CSV.

Rated speakers are binned by `Average_Rating`. Unrated speakers are assigned a predicted bin using Gaussian likelihood over the WER distribution of each rated bin (same method as `plot_wer_ratings.py`).

**Sampling rule per bin:**
- 1 speaker available → 0 selected (cannot spare from training)
- 2+ speakers available → `max(1, round(n × fraction))`, capped at `n − 1` (always retains at least one speaker per bin in training)

Run once per etiology:

```bash
# Parkinson's Disease
python select_validation_speakers.py \
    --wer-csv /path/to/pd_train_wer.csv \
    --etiology "Parkinson's Disease" \
    --output /path/to/data/pd_val_speakers.csv \
    --val-fraction 0.15 \
    --seed 42

# ALS (when available)
python select_validation_speakers.py \
    --wer-csv /path/to/als_train_wer.csv \
    --etiology "ALS" \
    --output /path/to/data/als_val_speakers.csv \
    --val-fraction 0.15 \
    --seed 42
```

If your WER CSV contains only one etiology you can omit `--etiology`, but it is recommended to always pass it explicitly so the output is clearly labelled.

**Options:**

| Flag | Default | Description |
|---|---|---|
| `--wer-csv` | required | WER CSV from Step 4 |
| `--etiology` | `None` | Filter to this etiology (required for multi-etiology CSVs) |
| `--output` | required | Output CSV path (must contain `Speaker_ID` column) |
| `--val-fraction` | `0.15` | Fraction of each bin to hold out |
| `--min-utterances` | `50` | Exclude speakers with fewer utterances |
| `--seed` | `42` | Random seed for reproducibility |
| `--no-plot` | off | Skip saving the distribution plot |

**Output:** `<etiology>_val_speakers.csv` with columns `Speaker_ID`, `Etiology`, `Effective_Bin`, `Average_Rating`, `Average_WER`, `Num_Utterances`. The `Speaker_ID` column is required by `sap.py --val-speakers`.

A distribution plot (`<etiology>_val_speakers_distribution.png`) is also saved alongside the CSV, showing train vs val speaker counts per bin (rated and predicted separately).

---

### Step 6 — Prepare Lhotse manifests (`sap.py`)

Builds Lhotse `RecordingSet` and `SupervisionSet` manifests for TRAIN, VAL, and TEST.

Built from per-etiology `speaker_pairs_{SPLIT}.csv` files for all splits. Pass one CSV per etiology for both `--train-csv` and `--test-csv`; they are merged automatically before processing.

```bash
python sap.py \
    --train-csv /path/to/synthetic/Parkinsons_Disease/speaker_pairs_TRAIN.csv \
                /path/to/synthetic/ALS/speaker_pairs_TRAIN.csv \
    --test-csv  /path/to/synthetic/Parkinsons_Disease/speaker_pairs_DEV.csv \
                /path/to/synthetic/ALS/speaker_pairs_DEV.csv \
    --val-speakers /path/to/data/pd_val_speakers.csv \
                   /path/to/data/als_val_speakers.csv \
    --output-dir /path/to/manifests
```

For a single etiology:

```bash
python sap.py \
    --train-csv /path/to/synthetic/Parkinsons_Disease/speaker_pairs_TRAIN.csv \
    --test-csv  /path/to/synthetic/Parkinsons_Disease/speaker_pairs_DEV.csv \
    --val-speakers /path/to/data/pd_val_speakers.csv \
    --output-dir /path/to/manifests
```

If `--val-speakers` is omitted, only TRAIN and TEST manifests are produced:

```bash
python sap.py \
    --train-csv /path/to/synthetic/Parkinsons_Disease/speaker_pairs_TRAIN.csv \
    --test-csv  /path/to/synthetic/Parkinsons_Disease/speaker_pairs_DEV.csv \
    --output-dir /path/to/manifests
```

To write human-readable manifests for debugging, add `--json`:

```bash
python sap.py \
    --train-csv /path/to/synthetic/Parkinsons_Disease/speaker_pairs_TRAIN.csv \
    --test-csv  /path/to/synthetic/Parkinsons_Disease/speaker_pairs_DEV.csv \
    --val-speakers /path/to/data/pd_val_speakers.csv \
    --output-dir /path/to/manifests \
    --json
```

**Output:** One pair of files per split per side:
- `sap_recordings_{split}_{source|target}.jsonl.gz` — audio metadata
- `sap_supervisions_{split}_{source|target}.jsonl.gz` — transcript, speaker ID, custom fields: `prompt_text`, `etiology` (source only), `category_description`

Possible splits: `train`, `val`, `test`. With `--json`, files are written as uncompressed `.jsonl`.


```
python bin/tokenizer.py \
    --src-dir /scratch/lewis.jor/VallE/egs/sap/data/manifests \
    --output-dir /scratch/lewis.jor/VallE/egs/sap/data/tokenized \
    --prefix sap \
    --dataset-parts sap_vc \
    --suffix jsonl \
    --tts 0
```