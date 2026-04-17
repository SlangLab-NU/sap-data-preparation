import json
import pandas as pd
from pathlib import Path
import argparse
from tqdm import tqdm
import logging
from jiwer import wer
import torch
import re
import soundfile as sf
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from num2words import num2words as _num2words
    _NUM2WORDS_AVAILABLE = True
except ImportError:
    _NUM2WORDS_AVAILABLE = False


def get_args():
    parser = argparse.ArgumentParser(description="Calculate WER for SAP speakers using NeMo ASR")

    parser.add_argument(
        "--extracted-dir",
        type=Path,
        required=True,
        help="Path to extracted SAP data directory (contains TRAIN/, DEV/ subdirs)"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="TRAIN",
        help="Dataset split to process (TRAIN or DEV)"
    )
    parser.add_argument(
        "--etiology",
        type=str,
        default=None,
        help="Only process speakers with this etiology (e.g. \"Parkinson's Disease\"). "
             "Processes all etiologies if not specified."
    )
    parser.add_argument(
        "--ratings-csv",
        type=Path,
        default=None,
        help="Path to speaker_ratings CSV. Required when --etiology is set to filter speakers."
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output CSV path for WER results"
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Detailed log file path. Defaults to <output>.log"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="stt_en_conformer_ctc_large",
        help="NeMo ASR model name"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for ASR inference"
    )
    parser.add_argument(
        "--max-speakers",
        type=int,
        default=None,
        help="Maximum speakers to process (for testing)"
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=10,
        help="Log example transcription every N utterances"
    )
    parser.add_argument(
        "--gt-source",
        type=str,
        choices=["transcript", "prompt-text"],
        default="prompt-text",
        help=(
            "Field to use as ground truth for WER. "
            "'prompt-text' uses Prompt.Prompt Text (the original script shown to the speaker). "
            "'transcript' uses Prompt.Transcript (a cleaned transcription of what was said). "
            "Default: prompt-text"
        )
    )

    return parser.parse_args()


def setup_file_logging(log_file):
    # Remove any existing file handlers (e.g. if called again after NeMo loads)
    logger.handlers = [h for h in logger.handlers if not isinstance(h, logging.FileHandler)]
    file_handler = logging.FileHandler(log_file, mode='a')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)


def load_nemo_model(model_name):
    try:
        import nemo.collections.asr as nemo_asr
    except ImportError:
        logger.error("NeMo not installed. Install with: pip install nemo_toolkit[asr]")
        return None

    logger.info(f"Loading NeMo model: {model_name}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name)
    asr_model.to(device)
    asr_model.eval()

    logger.info("Model loaded successfully")
    return asr_model


def load_and_convert_to_mono(audio_path):
    try:
        data, sr = sf.read(audio_path)
        if len(data.shape) > 1 and data.shape[1] > 1:
            data = np.mean(data, axis=1)
        if not np.isfinite(data).all():
            return None, None, "Audio contains NaN or Inf"
        return data, sr, None
    except Exception as e:
        return None, None, str(e)


_NUMBER_WORDS = [
    'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine',
    'ten', 'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen', 'sixteen',
    'seventeen', 'eighteen', 'nineteen', 'twenty', 'thirty', 'forty', 'fifty',
    'sixty', 'seventy', 'eighty', 'ninety', 'hundred', 'thousand', 'million'
]
_NUMBER_WORDS_SORTED = sorted(_NUMBER_WORDS, key=len, reverse=True)


def split_number_words(token):
    """Split concatenated number words, e.g. 'ninethirty' -> 'nine thirty'."""
    parts = []
    while token:
        matched = False
        for nw in _NUMBER_WORDS_SORTED:
            if token.startswith(nw):
                parts.append(nw)
                token = token[len(nw):]
                matched = True
                break
        if not matched:
            parts.append(token)
            break
    return ' '.join(parts)


def expand_numbers(text):
    if not _NUM2WORDS_AVAILABLE:
        return text
    try:
        return re.sub(r'\b\d+\b', lambda m: _num2words(int(m.group())), text)
    except Exception:
        return text


def normalize_text(text):
    text = text.lower()
    text = expand_numbers(text)
    # expand punctuation that corresponds to spoken words in email/URL context
    text = text.replace('@', ' at ')
    text = re.sub(r'(?<=\w)\.(?=\w)', ' dot ', text)  # dot between word chars (URLs, emails)
    text = re.sub(r'[^\w\s]', '', text)                # remove remaining punctuation
    text = ' '.join(split_number_words(t) for t in text.split())
    return text.strip()


def get_speaker_metadata(speaker_dir):
    json_files = list(speaker_dir.glob("*.json"))
    if not json_files:
        return None

    with open(json_files[0], 'r') as f:
        data = json.load(f)

    all_rating_levels = []
    for file_entry in data.get('Files', []):
        for rating in file_entry.get('Ratings', []):
            level = rating.get('Level')
            if level is not None:
                try:
                    all_rating_levels.append(int(level))
                except ValueError:
                    continue
    avg_rating = sum(all_rating_levels) / len(all_rating_levels) if all_rating_levels else None

    return {
        'speaker_id': data.get('Contributor ID', 'Unknown'),
        'etiology': data.get('Etiology', 'Unknown'),
        'files': data.get('Files', []),
        'avg_rating': avg_rating
    }


def get_already_processed(output_path):
    if not output_path.exists():
        return set()
    try:
        df = pd.read_csv(output_path)
        return set(df['Speaker_ID'].tolist())
    except Exception:
        return set()


def append_result(result, output_path):
    df_new = pd.DataFrame([result])
    if output_path.exists():
        df_new.to_csv(output_path, mode='a', header=False, index=False)
    else:
        df_new.to_csv(output_path, mode='w', header=True, index=False)


def calculate_speaker_wer(speaker_dir, asr_model, batch_size, log_every, gt_source="prompt-text"):
    metadata = get_speaker_metadata(speaker_dir)
    if not metadata:
        logger.warning(f"No metadata found for {speaker_dir.name}")
        return None

    speaker_id = metadata['speaker_id']
    etiology = metadata['etiology']
    files_data = metadata['files']

    gt_field = 'Prompt Text' if gt_source == 'prompt-text' else 'Transcript'

    audio_paths = []
    ground_truths = []
    skipped = 0

    temp_dir = speaker_dir / "temp_mono"
    temp_dir.mkdir(exist_ok=True)

    for file_entry in files_data:
        filename = file_entry.get('Filename', '')
        transcript = file_entry.get('Prompt', {}).get(gt_field, '')

        if not transcript:
            continue

        audio_path = speaker_dir / filename
        if not audio_path.exists():
            continue

        data, sr, error = load_and_convert_to_mono(audio_path)
        if error:
            logger.warning(f"Skipping {filename}: {error}")
            skipped += 1
            continue

        temp_path = temp_dir / filename
        sf.write(temp_path, data, sr)
        audio_paths.append(str(temp_path))
        ground_truths.append(transcript.strip())

    if skipped > 0:
        logger.warning(f"Skipped {skipped} problematic files for {speaker_id}")

    if not audio_paths:
        logger.warning(f"No valid audio files for {speaker_id}")
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
        return None

    tqdm.write(f"\nProcessing {speaker_id[:12]}... ({etiology}) — {len(audio_paths)} utterances")

    predictions = []
    for i in range(0, len(audio_paths), batch_size):
        batch = audio_paths[i:i + batch_size]
        try:
            preds = asr_model.transcribe(batch)
            predictions.extend(preds)
        except Exception as e:
            logger.error(f"Batch failed: {e}")
            for path in batch:
                try:
                    pred = asr_model.transcribe([path])
                    predictions.extend(pred)
                except Exception:
                    predictions.append("")

    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)

    wer_scores = []
    for idx, (gt, pred) in enumerate(zip(ground_truths, predictions)):
        if not pred:
            continue
        try:
            gt_norm = normalize_text(gt)
            pred_norm = normalize_text(pred)
            score = wer(gt_norm, pred_norm)
            wer_scores.append(score)

            if idx % log_every == 0:
                msg = (
                    f"\n  Example {idx + 1}/{len(ground_truths)}:\n"
                    f"    GT  : {gt_norm}\n"
                    f"    ASR : {pred_norm}\n"
                    f"    WER : {score:.2%}"
                )
                logger.info(msg)
                tqdm.write(msg)
        except Exception as e:
            logger.warning(f"WER calculation error: {e}")

    avg_wer = sum(wer_scores) / len(wer_scores) if wer_scores else None
    tqdm.write(
        f"  Done: {len(wer_scores)}/{len(audio_paths)} utterances, "
        f"Avg WER = {avg_wer:.2%}" if avg_wer is not None else "  Done: no valid utterances"
    )

    return {
        'Speaker_ID': speaker_id,
        'Etiology': etiology,
        'Average_Rating': round(metadata['avg_rating'], 2) if metadata['avg_rating'] is not None else None,
        'Num_Utterances': len(wer_scores),
        'Num_Failed': skipped,
        'Average_WER': avg_wer
    }


def get_speaker_dirs(split_dir, etiology_filter, ratings_csv):
    """
    Return speaker directories for the given split, optionally filtered by etiology.
    When etiology_filter is set, uses the ratings CSV to identify speaker IDs for that etiology.
    """
    if etiology_filter:
        if ratings_csv is None:
            raise ValueError("--ratings-csv is required when --etiology is specified")
        df = pd.read_csv(ratings_csv)
        speaker_ids = set(
            df[df['Etiology'] == etiology_filter]['Speaker_ID'].tolist()
        )
        dirs = [d for d in split_dir.iterdir() if d.is_dir() and d.name in speaker_ids]
        logger.info(
            f"Etiology filter '{etiology_filter}': {len(dirs)} speakers found "
            f"(from {len(speaker_ids)} in ratings CSV)"
        )
    else:
        dirs = [d for d in split_dir.iterdir() if d.is_dir()]
        logger.info(f"No etiology filter: {len(dirs)} speakers found")

    return dirs


def main():
    args = get_args()

    log_file = args.log_file or args.output.with_suffix('.log')
    setup_file_logging(log_file)

    if not _NUM2WORDS_AVAILABLE:
        logger.warning("num2words not installed — digit normalization disabled. "
                       "Install with: pip install --user num2words")

    logger.info("=" * 60)
    logger.info("SAP Speaker WER Calculation")
    logger.info(f"  Split     : {args.split}")
    logger.info(f"  Etiology  : {args.etiology or 'all'}")
    logger.info(f"  GT source : {args.gt_source}")
    logger.info(f"  Output    : {args.output}")
    logger.info(f"  Log       : {log_file}")
    logger.info("=" * 60)

    split_dir = args.extracted_dir / args.split
    if not split_dir.exists():
        logger.error(f"Split directory not found: {split_dir}")
        return

    processed = get_already_processed(args.output)
    if processed:
        logger.info(f"Resuming: {len(processed)} speakers already processed")

    speaker_dirs = get_speaker_dirs(split_dir, args.etiology, args.ratings_csv)

    # Filter already processed
    to_process = []
    for d in speaker_dirs:
        meta = get_speaker_metadata(d)
        if meta and meta['speaker_id'] not in processed:
            to_process.append(d)

    logger.info(f"Remaining to process: {len(to_process)}")

    if args.max_speakers:
        to_process = to_process[:args.max_speakers]
        logger.info(f"Limited to {args.max_speakers} speakers")

    asr_model = load_nemo_model(args.model_name)
    if not asr_model:
        return

    # NeMo modifies Python logging on import — re-attach our file handler after it loads
    setup_file_logging(log_file)

    for speaker_dir in tqdm(to_process, desc=f"WER [{args.split}]"):
        try:
            result = calculate_speaker_wer(
                speaker_dir, asr_model, args.batch_size, args.log_every, args.gt_source
            )
            if result:
                result['Split'] = args.split
                append_result(result, args.output)
        except Exception as e:
            logger.error(f"Error processing {speaker_dir.name}: {e}")

    # Print summary from full output CSV
    if args.output.exists():
        df = pd.read_csv(args.output)
        valid = df.dropna(subset=['Average_WER'])
        logger.info("\n" + "=" * 60)
        logger.info(f"Total speakers: {len(df)}, valid WER: {len(valid)}")
        if len(valid):
            logger.info(f"Mean WER  : {valid['Average_WER'].mean():.2%}")
            logger.info(f"Median WER: {valid['Average_WER'].median():.2%}")
            logger.info(f"Range     : {valid['Average_WER'].min():.2%} – {valid['Average_WER'].max():.2%}")
        logger.info("=" * 60)


if __name__ == "__main__":
    main()
