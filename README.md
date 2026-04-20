# sap-data-preparation
Series of scripts to extract, generate synthetic pairs, and create the Lhotse recipe for the SAP dataset

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
A `speaker_pairs_{SPLIT}.csv` is also written to `<sap-data-dir>/synthetic/speaker_pairs_{SPLIT}.csv` mapping original audio to synthetic audio. Use `--output-csv` to override the path.

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

Uses the WER results from Step 4 alongside the speaker ratings CSV to select a stratified validation set from TRAIN speakers. Speakers are binned by intelligibility (using SAP ratings where available, WER otherwise) and sampled to cover the full severity range.

```bash
python select_validation_speakers.py \
    --wer-csv /path/to/pd_train_wer.csv \
    --ratings-csv /path/to/ratings/speaker_ratings_TRAIN.csv \
    --etiology "Parkinson's Disease" \
    --output /path/to/val_speakers_PD.csv
```

**Output:** CSV with a `Speaker_ID` column listing the selected validation speakers. This file is passed directly to `sap.py` via `--val-speakers`.

---

### Step 6 — Prepare Lhotse manifests (`sap.py`)

Reads `speaker_pairs_TRAIN.csv` and `speaker_pairs_DEV.csv` and builds Lhotse `RecordingSet` and `SupervisionSet` manifests. The DEV split is treated as the held-out **test set**. A validation set is carved out of TRAIN by passing a pre-selected speaker list via `--val-speakers` (produced in Step 5). Failed syntheses are excluded; skipped entries (already-generated audio) are included.

```bash
python sap.py \
    --train-csv /path/to/sap/synthetic/speaker_pairs_TRAIN.csv \
    --test-csv  /path/to/sap/synthetic/speaker_pairs_DEV.csv \
    --val-speakers /path/to/val_speakers_PD.csv \
    --output-dir /path/to/manifests
```

If `--val-speakers` is omitted, only TRAIN and TEST manifests are produced:

```bash
python sap.py \
    --train-csv /path/to/sap/synthetic/speaker_pairs_TRAIN.csv \
    --test-csv  /path/to/sap/synthetic/speaker_pairs_DEV.csv \
    --output-dir /path/to/manifests
```

To write human-readable manifests for debugging, add `--json`:

```bash
python sap.py \
    --train-csv /path/to/sap/synthetic/speaker_pairs_TRAIN.csv \
    --test-csv  /path/to/sap/synthetic/speaker_pairs_DEV.csv \
    --val-speakers /path/to/val_speakers_PD.csv \
    --output-dir /path/to/manifests \
    --json
```

**Output:** One pair of files per split per side:
- `sap_recordings_{split}_{source|target}.jsonl.gz` — audio metadata (path, duration, channels)
- `sap_supervisions_{split}_{source|target}.jsonl.gz` — transcript, speaker ID, and custom fields: `prompt_text`, `etiology` (source only), `category_description`

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