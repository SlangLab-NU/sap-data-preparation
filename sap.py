"""
Lhotse recipe for the SAP dataset.

TRAIN and VAL manifests are built from per-etiology speaker_pairs_{SPLIT}.csv
files produced by generate_synthetic_speech.py. Each produces source (atypical
speaker) and target (StyleTTS2 synthetic) manifest pairs.

TEST manifests are built directly from the extracted DEV speaker JSON files —
no synthetic speech is generated for the test set.

Output files per split:
  sap_recordings_{split}_source.jsonl[.gz]
  sap_supervisions_{split}_source.jsonl[.gz]
  sap_recordings_{split}_target.jsonl[.gz]   (TRAIN and VAL only)
  sap_supervisions_{split}_target.jsonl[.gz]  (TRAIN and VAL only)

Source SupervisionSegment custom fields:
  - prompt_text: the Prompt Text shown to the speaker
  - etiology: speaker's etiology label
  - category_description: prompt category (e.g. "Novel Sentences", "Spontaneous Speech Prompts")

Target SupervisionSegment custom fields:
  - prompt_text: the Prompt Text shown to the speaker
  - category_description: prompt category
"""

import argparse
import json
import logging
import re
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

from lhotse.audio import Recording, RecordingSet
from lhotse.supervision import SupervisionSegment, SupervisionSet

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def get_args():
    parser = argparse.ArgumentParser(
        description="Prepare Lhotse manifests for the SAP dataset"
    )
    parser.add_argument(
        "--train-csv",
        type=Path,
        nargs="+",
        required=True,
        help="One or more speaker_pairs_TRAIN.csv files produced by generate_synthetic_speech.py. "
             "Pass one per etiology (e.g. synthetic/Parkinsons_Disease/speaker_pairs_TRAIN.csv "
             "synthetic/ALS/speaker_pairs_TRAIN.csv); they are merged before processing."
    )
    parser.add_argument(
        "--extracted-test-dir",
        type=Path,
        required=True,
        help="Path to the extracted DEV speaker directory (e.g. data/extracted/DEV). "
             "Test manifests are built directly from speaker JSON files — "
             "no synthetic speech is required for the test set."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write manifest files"
    )
    parser.add_argument(
        "--val-speakers",
        type=Path,
        nargs="+",
        default=None,
        help="One or more CSVs of speaker IDs to hold out from TRAIN as the VAL set. "
             "Each must contain a 'Speaker_ID' column. Produced by select_validation_speakers.py. "
             "Pass one file per etiology; they are merged before splitting."
    )
    parser.add_argument(
        "--json",
        action="store_true",
        default=False,
        help="Write manifests as readable .jsonl instead of compressed .jsonl.gz"
    )
    return parser.parse_args()


def clean_transcript(transcript):
    """Strip [bracketed] system annotations; keep parenthesised disfluencies."""
    return re.sub(r'\[.*?\]', '', transcript).strip()


def filter_valid(df: pd.DataFrame, label: str) -> pd.DataFrame:
    total = len(df)
    df = df[~df["status"].apply(lambda s: str(s).startswith("failed"))].copy()
    logger.info(f"[{label}] {len(df)}/{total} valid pairs (excluded failed)")
    return df


def split_by_val_speakers(train_df: pd.DataFrame, val_speakers_csvs: list):
    """
    Split train_df into TRAIN and VAL using one or more CSVs of pre-selected speaker IDs.
    Multiple CSVs (one per etiology) are merged before splitting.
    Returns (train_df, val_df) with no speaker overlap guaranteed.
    """
    val_ids = set()
    for csv_path in val_speakers_csvs:
        ids = pd.read_csv(csv_path)["Speaker_ID"].tolist()
        val_ids.update(ids)
        logger.info(f"Loaded {len(ids)} val speakers from {csv_path.name}")

    available = set(train_df["speaker_id"].unique())
    missing = val_ids - available
    if missing:
        logger.warning(
            f"{len(missing)} val speakers not found in TRAIN CSV and will be skipped: "
            + ", ".join(sorted(missing)[:5]) + ("..." if len(missing) > 5 else "")
        )
    val_ids = val_ids & available

    val_df = train_df[train_df["speaker_id"].isin(val_ids)].copy()
    train_df = train_df[~train_df["speaker_id"].isin(val_ids)].copy()

    assert len(set(train_df["speaker_id"]) & val_ids) == 0, \
        "Speaker bleed detected between TRAIN and VAL — this is a bug."

    logger.info(
        f"Train/val speaker split: "
        f"{len(train_df['speaker_id'].unique())} train speakers, "
        f"{len(val_df['speaker_id'].unique())} val speakers"
    )
    return train_df, val_df


def build_manifests(df: pd.DataFrame, split: str) -> dict:
    """
    Build source (atypical) and target (synthetic) Recording and SupervisionSegment
    lists for TRAIN or VAL. Returns dict with 'source' and 'target' sub-dicts.
    """
    result = {
        "source": {"recordings": [], "supervisions": []},
        "target": {"recordings": [], "supervisions": []},
    }

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Building {split} manifests"):
        recording_id = f"{row['speaker_id']}_{Path(row['original_audio']).stem}"

        # --- source: atypical speaker recording ---
        try:
            src_recording = Recording.from_file(row["original_audio"], recording_id)
        except Exception as e:
            logger.error(f"Failed to load source recording {row['original_audio']}: {e}")
            continue

        src_segment = SupervisionSegment(
            id=recording_id,
            recording_id=recording_id,
            start=0.0,
            duration=src_recording.duration,
            language="English",
            speaker=row["speaker_id"],
            text=row["transcript"],
            custom={
                "prompt_text": row["prompt_text"],
                "etiology": row["etiology"],
                "category_description": row.get("category_description", ""),
            }
        )

        result["source"]["recordings"].append(src_recording)
        result["source"]["supervisions"].append(src_segment)

        # --- target: synthetic recording ---
        if not row["synthetic_audio"] or pd.isna(row["synthetic_audio"]):
            logger.warning(f"No synthetic audio for {recording_id}, skipping target.")
            continue

        try:
            tgt_recording = Recording.from_file(row["synthetic_audio"], recording_id)
        except Exception as e:
            logger.error(f"Failed to load target recording {row['synthetic_audio']}: {e}")
            continue

        tgt_segment = SupervisionSegment(
            id=recording_id,
            recording_id=recording_id,
            start=0.0,
            duration=tgt_recording.duration,
            language="English",
            speaker=row["speaker_id"],
            text=row["transcript"],
            custom={
                "prompt_text": row["prompt_text"],
                "category_description": row.get("category_description", ""),
            }
        )

        result["target"]["recordings"].append(tgt_recording)
        result["target"]["supervisions"].append(tgt_segment)

    return result


def build_test_manifests(extracted_test_dir: Path) -> dict:
    """
    Build source-only TEST manifests by reading speaker JSON files directly
    from the extracted DEV directory. No synthetic speech is required.

    Applies the same filtering as generate_synthetic_speech.py:
      - Skips utterances with (cs: ...) cued-speech annotations
      - Strips [bracketed] annotations from transcripts
    """
    result = {"source": {"recordings": [], "supervisions": []}}

    speaker_dirs = sorted([d for d in extracted_test_dir.iterdir() if d.is_dir()])
    logger.info(f"Building TEST manifests from {len(speaker_dirs)} speakers in {extracted_test_dir}")

    counts = {
        "no_json":          0,
        "cued_speech":      0,
        "empty_transcript": 0,
        "audio_missing":    0,
        "load_error":       0,
        "ok":               0,
    }

    for speaker_dir in tqdm(speaker_dirs, desc="Building TEST manifests"):
        json_files = list(speaker_dir.glob("*.json"))
        if not json_files:
            counts["no_json"] += 1
            continue

        with open(json_files[0], 'r') as f:
            data = json.load(f)

        speaker_id = data.get('Contributor ID', speaker_dir.name)
        etiology = data.get('Etiology', 'Unknown')

        for file_entry in data.get('Files', []):
            filename = file_entry.get('Filename', '')
            if not filename:
                continue

            prompt = file_entry.get('Prompt', {})
            raw_transcript = prompt.get('Transcript', '')

            if '(cs:' in raw_transcript:
                counts["cued_speech"] += 1
                continue

            transcript = clean_transcript(raw_transcript)
            if not transcript:
                counts["empty_transcript"] += 1
                continue

            audio_path = speaker_dir / filename
            if not audio_path.exists():
                counts["audio_missing"] += 1
                continue

            recording_id = f"{speaker_id}_{audio_path.stem}"

            try:
                recording = Recording.from_file(str(audio_path), recording_id)
            except Exception as e:
                logger.error(f"Failed to load {audio_path}: {e}")
                counts["load_error"] += 1
                continue

            segment = SupervisionSegment(
                id=recording_id,
                recording_id=recording_id,
                start=0.0,
                duration=recording.duration,
                language="English",
                speaker=speaker_id,
                text=transcript,
                custom={
                    "prompt_text": prompt.get('Prompt Text', ''),
                    "etiology": etiology,
                    "category_description": prompt.get('Category Description', ''),
                }
            )

            result["source"]["recordings"].append(recording)
            result["source"]["supervisions"].append(segment)
            counts["ok"] += 1

    logger.info(
        f"[TEST] utterance counts — "
        f"ok: {counts['ok']}, "
        f"audio missing: {counts['audio_missing']}, "
        f"cued-speech: {counts['cued_speech']}, "
        f"empty transcript: {counts['empty_transcript']}, "
        f"load error: {counts['load_error']}, "
        f"no JSON: {counts['no_json']}"
    )
    return result


def save_manifests(split: str, data: dict, output_dir: Path, use_json: bool):
    suffix = "jsonl" if use_json else "jsonl.gz"
    split_lower = split.lower()

    for side, contents in data.items():
        recordings = contents["recordings"]
        supervisions = contents["supervisions"]

        if not recordings:
            logger.warning(f"No data for split {split} / {side}, skipping.")
            continue

        rec_set = RecordingSet.from_recordings(recordings)
        sup_set = SupervisionSet.from_segments(supervisions)

        rec_path = output_dir / f"sap_recordings_{split_lower}_{side}.{suffix}"
        sup_path = output_dir / f"sap_supervisions_{split_lower}_{side}.{suffix}"

        rec_set.to_file(rec_path)
        sup_set.to_file(sup_path)

        logger.info(f"[{split}/{side}] {len(recordings)} recordings -> {rec_path}")
        logger.info(f"[{split}/{side}] {len(supervisions)} supervisions -> {sup_path}")


def main():
    args = get_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_parts = []
    for csv_path in args.train_csv:
        logger.info(f"Reading TRAIN CSV: {csv_path}")
        train_parts.append(pd.read_csv(csv_path))
    train_df = filter_valid(pd.concat(train_parts, ignore_index=True), "TRAIN")

    splits = {"TRAIN": train_df}

    if args.val_speakers:
        train_df, val_df = split_by_val_speakers(train_df, args.val_speakers)
        splits["TRAIN"] = train_df
        splits["VAL"] = val_df

    for split, df in splits.items():
        manifests = build_manifests(df, split)
        save_manifests(split, manifests, args.output_dir, args.json)

    test_manifests = build_test_manifests(args.extracted_test_dir)
    save_manifests("TEST", test_manifests, args.output_dir, args.json)

    logger.info("Done.")


if __name__ == "__main__":
    main()
