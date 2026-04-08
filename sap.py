"""
Lhotse recipe for the SAP dataset.
Builds Recording and SupervisionSet manifests from the speaker_pairs_{SPLIT}.csv
files produced by generate_synthetic_speech.py.

Produces separate source (atypical speaker) and target (StyleTTS2 synthetic)
manifest pairs per split:
  sap_recordings_{split}_source.jsonl[.gz]
  sap_supervisions_{split}_source.jsonl[.gz]
  sap_recordings_{split}_target.jsonl[.gz]
  sap_supervisions_{split}_target.jsonl[.gz]

Source SupervisionSegment custom fields:
  - prompt_text: the Prompt Text shown to the speaker
  - etiology: speaker's etiology label

Target SupervisionSegment custom fields:
  - prompt_text: the Prompt Text shown to the speaker
"""

import argparse
import logging
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
        required=True,
        help="Path to speaker_pairs_TRAIN.csv produced by generate_synthetic_speech.py"
    )
    parser.add_argument(
        "--dev-csv",
        type=Path,
        required=True,
        help="Path to speaker_pairs_DEV.csv produced by generate_synthetic_speech.py"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write manifest files"
    )
    parser.add_argument(
        "--test-size",
        type=int,
        default=0,
        help="Number of speakers to hold out from TRAIN as a TEST set. "
             "Speakers are selected randomly and will not appear in TRAIN manifests."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible train/test speaker split (default: 42)"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        default=False,
        help="Write manifests as readable .jsonl instead of compressed .jsonl.gz"
    )
    return parser.parse_args()


def filter_valid(df: pd.DataFrame, label: str) -> pd.DataFrame:
    total = len(df)
    df = df[~df["status"].apply(lambda s: str(s).startswith("failed"))].copy()
    logger.info(f"[{label}] {len(df)}/{total} valid pairs (excluded failed)")
    return df


def split_by_speaker(train_df: pd.DataFrame, test_size: int, seed: int):
    """
    Randomly sample `test_size` speakers from train_df for the TEST set.
    Returns (train_df, test_df) with no speaker overlap guaranteed.
    """
    speakers = train_df["speaker_id"].unique()
    if test_size >= len(speakers):
        raise ValueError(
            f"--test-size ({test_size}) must be less than the number of "
            f"TRAIN speakers ({len(speakers)})"
        )
    test_speakers = set(
        pd.Series(speakers).sample(n=test_size, random_state=seed).values
    )
    test_df = train_df[train_df["speaker_id"].isin(test_speakers)].copy()
    train_df = train_df[~train_df["speaker_id"].isin(test_speakers)].copy()

    # Verify no bleed
    assert len(set(train_df["speaker_id"]) & test_speakers) == 0, \
        "Speaker bleed detected between TRAIN and TEST — this is a bug."

    logger.info(
        f"Train/test speaker split (seed={seed}): "
        f"{len(train_df['speaker_id'].unique())} train speakers, "
        f"{len(test_speakers)} test speakers"
    )
    return train_df, test_df


def build_manifests(df: pd.DataFrame, split: str) -> dict:
    """
    Build source (atypical) and target (synthetic) Recording and SupervisionSegment
    lists for a single split. Returns dict with 'source' and 'target' sub-dicts.
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
            }
        )

        result["target"]["recordings"].append(tgt_recording)
        result["target"]["supervisions"].append(tgt_segment)

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

    logger.info(f"Reading TRAIN CSV: {args.train_csv}")
    train_df = filter_valid(pd.read_csv(args.train_csv), "TRAIN")

    logger.info(f"Reading DEV CSV: {args.dev_csv}")
    dev_df = filter_valid(pd.read_csv(args.dev_csv), "DEV")

    splits = {"TRAIN": train_df, "DEV": dev_df}

    if args.test_size > 0:
        train_df, test_df = split_by_speaker(train_df, args.test_size, args.seed)
        splits["TRAIN"] = train_df
        splits["TEST"] = test_df

    for split, df in splits.items():
        manifests = build_manifests(df, split)
        save_manifests(split, manifests, args.output_dir, args.json)

    logger.info("Done.")


if __name__ == "__main__":
    main()
