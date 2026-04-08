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

All scripts that use StyleTTS2 must be run inside an Apptainer container. Before running Step 3, start the container with:

```bash
bash run_sap_container.sh
```

Before running, update `run_sap_container.sh` for your environment:
- `apptainer_image` — set to the path of your `.sif` file
- `--bind` — set to your scratch/working directory (format: `/your/path:/your/path`)

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

**Output:** `<sap-data-dir>/synthetic/<etiology>/<speaker_id>/<utterance>_synthetic.wav`
A `speaker_pairs_{SPLIT}.csv` is also written to `<sap-data-dir>/synthetic/speaker_pairs_{SPLIT}.csv` (e.g. `speaker_pairs_TRAIN.csv`, `speaker_pairs_DEV.csv`) mapping original audio to synthetic audio with transcripts and synthesis status. Use `--output-csv` to override the path.

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

### Step 4 — Prepare Lhotse manifests (`sap.py`)

Reads `speaker_pairs_TRAIN.csv` and `speaker_pairs_DEV.csv` and builds Lhotse `RecordingSet` and `SupervisionSet` manifests for each split. Failed syntheses are excluded; skipped entries (already-generated audio) are included.

```bash
python sap.py \
    --train-csv /path/to/sap/synthetic/speaker_pairs_TRAIN.csv \
    --dev-csv /path/to/sap/synthetic/speaker_pairs_DEV.csv \
    --output-dir /path/to/manifests
```

To hold out a TEST set carved from TRAIN by speaker (no speaker bleed guaranteed), use `--test-size`:

```bash
python sap.py \
    --train-csv /path/to/sap/synthetic/speaker_pairs_TRAIN.csv \
    --dev-csv /path/to/sap/synthetic/speaker_pairs_DEV.csv \
    --output-dir /path/to/manifests \
    --test-size 30 \
    --seed 42
```

To write human-readable manifests for debugging, add `--json`:

```bash
python sap.py \
    --train-csv /path/to/sap/synthetic/speaker_pairs_TRAIN.csv \
    --dev-csv /path/to/sap/synthetic/speaker_pairs_DEV.csv \
    --output-dir /path/to/manifests \
    --json
```

**Output:** One pair of files per split per side:
- `sap_recordings_{split}_{source|target}.jsonl.gz` — audio metadata (path, duration, channels)
- `sap_supervisions_{split}_{source|target}.jsonl.gz` — transcript, prompt text, speaker ID, and etiology

With `--json`, files are written as uncompressed `.jsonl`. With `--test-size`, a `test` split is produced in addition to `train` and `dev`.
