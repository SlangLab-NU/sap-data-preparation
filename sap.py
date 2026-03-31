"""
Lhotse recipe for the SAP dataset.
Builds Recording and SupervisionSet manifests from the speaker_pairs.csv
produced by generate_synthetic_speech.py.

Each SupervisionSegment contains:
  - text: the Transcript (what was said)
  - prompt_text: the Prompt Text (the prompt shown to the speaker)
  - custom.synthetic_audio: path to the StyleTTS2-generated paired audio
  - custom.etiology: speaker's etiology label
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

SPLITS = ["TRAIN", "DEV"]


def get_args():
    parser = argparse.ArgumentParser(
        description="Prepare Lhotse manifests for the SAP dataset"
    )
    parser.add_argument(
        "--pairs-csv",
        type=Path,
        required=True,
        help="Path to speaker_pairs.csv produced by generate_synthetic_speech.py"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write manifest files"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        default=False,
        help="Write manifests as readable .jsonl instead of compressed .jsonl.gz"
    )
    return parser.parse_args()


def infer_split(audio_path: str) -> str:
    """
    Infer TRAIN or DEV from the original audio path.
    Expects the extracted dir structure: .../extracted/{TRAIN,DEV}/speaker_id/...
    """
    parts = Path(audio_path).parts
    for split in SPLITS:
        if split in parts:
            return split
    return "TRAIN"


def build_manifests(df: pd.DataFrame) -> dict:
    """
    Build Recording and SupervisionSegment lists per split.
    Returns dict keyed by split name.
    """
    manifests = {split: {"recordings": [], "supervisions": []} for split in SPLITS}

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Building manifests"):
        split = infer_split(row["original_audio"])

        recording_id = f"{row['speaker_id']}_{Path(row['original_audio']).stem}"

        try:
            recording = Recording.from_file(row["original_audio"], recording_id)
        except Exception as e:
            logger.error(f"Failed to load recording {row['original_audio']}: {e}")
            continue

        segment = SupervisionSegment(
            id=recording_id,
            recording_id=recording_id,
            start=0.0,
            duration=recording.duration,
            language="English",
            speaker=row["speaker_id"],
            text=row["transcript"],
            custom={
                "prompt_text": row["prompt_text"],
                "synthetic_audio": row["synthetic_audio"],
                "etiology": row["etiology"],
            }
        )

        manifests[split]["recordings"].append(recording)
        manifests[split]["supervisions"].append(segment)

    return manifests


def save_manifests(manifests: dict, output_dir: Path, use_json: bool):
    suffix = "jsonl" if use_json else "jsonl.gz"

    for split, data in manifests.items():
        recordings = data["recordings"]
        supervisions = data["supervisions"]

        if not recordings:
            logger.warning(f"No data for split {split}, skipping.")
            continue

        rec_set = RecordingSet.from_recordings(recordings)
        sup_set = SupervisionSet.from_segments(supervisions)

        split_lower = split.lower()
        rec_path = output_dir / f"sap_recordings_{split_lower}.{suffix}"
        sup_path = output_dir / f"sap_supervisions_{split_lower}.{suffix}"

        rec_set.to_file(rec_path)
        sup_set.to_file(sup_path)

        logger.info(f"[{split}] {len(recordings)} recordings -> {rec_path}")
        logger.info(f"[{split}] {len(supervisions)} supervisions -> {sup_path}")


def main():
    args = get_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Reading pairs CSV: {args.pairs_csv}")
    df = pd.read_csv(args.pairs_csv)

    total = len(df)
    df = df[df["status"] != "failed"].copy()
    df = df[df["status"].apply(lambda s: not str(s).startswith("failed"))].copy()
    logger.info(f"Loaded {len(df)}/{total} valid pairs (excluded failed)")

    manifests = build_manifests(df)
    save_manifests(manifests, args.output_dir, args.json)

    logger.info("Done.")


if __name__ == "__main__":
    main()
