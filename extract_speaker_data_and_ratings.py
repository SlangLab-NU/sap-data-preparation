"""
Generate speaker ratings CSVs from extracted SAP data.
Reads each speaker's JSON metadata and outputs speaker_ratings_TRAIN.csv
and speaker_ratings_DEV.csv, which are required by generate_synthetic_speech.py.
"""

import csv
import json
import logging
import argparse
from pathlib import Path
from collections import defaultdict
from tqdm.auto import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

SPLITS = ["DEV", "TRAIN"]


def get_args():
    parser = argparse.ArgumentParser(
        description="Extract speaker ratings CSVs from extracted SAP data"
    )
    parser.add_argument(
        "--extracted-dir",
        type=Path,
        required=True,
        help="Path to the extracted SAP directory (contains DEV/ and TRAIN/ subdirs)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write speaker_ratings_DEV.csv and speaker_ratings_TRAIN.csv"
    )
    return parser.parse_args()


def extract_ratings_for_split(extracted_dir, split, output_dir):
    split_dir = extracted_dir / split
    if not split_dir.exists():
        logger.warning(f"Split directory not found: {split_dir}")
        return

    speaker_dirs = [d for d in split_dir.iterdir() if d.is_dir()]
    csv_data = []

    for speaker_dir in tqdm(speaker_dirs, desc=f"Processing {split} speakers"):
        json_files = list(speaker_dir.glob("*.json"))
        if not json_files:
            continue

        try:
            with open(json_files[0], 'r') as f:
                data = json.load(f)

            speaker_id = data.get("Contributor ID", "Unknown")
            etiology = data.get("Etiology", "Unknown")

            all_ratings = []
            rated_utterances = 0
            unrated_utterances = 0

            for file_entry in data.get("Files", []):
                ratings = file_entry.get("Ratings", [])
                if not ratings:
                    unrated_utterances += 1
                else:
                    rated_utterances += 1
                    for rating in ratings:
                        level = rating.get("Level")
                        if level is not None:
                            try:
                                all_ratings.append(int(level))
                            except ValueError:
                                continue

            avg_rating = round(sum(all_ratings) / len(all_ratings), 2) if all_ratings else None

            csv_data.append({
                'Speaker_ID': speaker_id,
                'Etiology': etiology,
                'Average_Rating': avg_rating if avg_rating is not None else 'N/A',
                'Number_of_Ratings': rated_utterances,
                'Number_Not_Rated': unrated_utterances,
                'Total_Utterances': rated_utterances + unrated_utterances,
            })

        except Exception as e:
            logger.error(f"Failed to process {speaker_dir.name}: {e}")

    output_path = output_dir / f"speaker_ratings_{split}.csv"
    fieldnames = ['Speaker_ID', 'Etiology', 'Average_Rating', 'Number_of_Ratings',
                  'Number_Not_Rated', 'Total_Utterances']

    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_data)

    logger.info(f"Saved {len(csv_data)} speakers to {output_path}")


def main():
    args = get_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for split in SPLITS:
        extract_ratings_for_split(args.extracted_dir, split, args.output_dir)


if __name__ == "__main__":
    main()
