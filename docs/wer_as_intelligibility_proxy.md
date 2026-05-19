# WER as an Intelligibility Proxy for SAP Speakers

## Motivation

The SAP dataset includes a mix of rated and unrated speakers. Speakers are rated on a 1–7 scale
(1 = highest intelligibility, 7 = lowest). Approximately 71% of Parkinson's Disease TRAIN speakers
have a human rating; the remaining ~29% are unrated. To stratify the validation set by intelligibility
across all speakers, we use Word Error Rate (WER) from an automatic speech recognition (ASR) model
as a proxy for unrated speakers.

The core assumption is that speakers who are harder for a human to understand will also produce
higher ASR error rates. This has empirical support in the dysarthria literature, though the
correlation is imperfect — ASR models and human listeners fail in different ways.

---

## ASR Setup

**Model:** `stt_en_conformer_ctc_large` (NeMo 1.22.0)

This is a CTC-based conformer model trained on large-scale English speech. We deliberately use
a CTC model (rather than transducer or attention-encoder-decoder) because:

- CTC models predict tokens frame-by-frame with no language model component. This makes them
  more sensitive to acoustic quality and less likely to "correct" a dysarthric speaker's errors
  using context.
- A model with a strong LM could partially compensate for articulation errors, artificially
  lowering WER and making it a weaker intelligibility proxy.

**WER is computed per speaker** by averaging utterance-level WER scores across all valid
utterances for that speaker.

**Container:** NeMo 23.10 base image (`nvcr.io/nvidia/nemo:23.10`) with added dependencies:
`pandas`, `jiwer`, `soundfile`, `tqdm`, `num2words`.

---

## Ground Truth: Prompt Text vs. Transcript

The SAP JSON for each utterance contains two related but distinct text fields under `Prompt`:

- **`Prompt Text`** — the original script presented to the speaker on screen. Retains natural
  formatting including punctuation, capitalisation, and symbols (e.g. `maria.sari@wilderness.net.org`).
- **`Transcript`** — a human transcription of what the speaker actually said. For scripted
  prompts this usually matches `Prompt Text` closely; for spontaneous prompts and severe
  speakers it can diverge significantly. Also contains annotation markers (see below).

### Which to use as GT for WER?

We use **stripped `Transcript`** as the ground truth (the default in `calculate_sap_wer.py`).

The primary reason is **alignment with synthesis**: synthetic speech targets are generated from
the cleaned `Transcript` (see `generate_synthetic_speech.py`). WER must be scored against the
same text that was used for synthesis, otherwise the intelligibility metric and the training
target are measuring against different references.

A secondary reason is correctness for non-scripted prompts. For spontaneous speech prompts,
the speaker produces a free-form response — WER against the cue text (`Prompt Text`) is
meaningless and routinely exceeds 200% (see Example 101 below). The `Transcript` captures
what was actually said and is the only valid GT for these utterances.

The `--gt-source` flag allows switching if needed:
```bash
--gt-source transcript    # default — stripped Prompt.Transcript
--gt-source prompt-text   # Prompt.Prompt Text (original script)
```

### Note on email/URL prompts and `--gt-source prompt-text`

If scoring against `Prompt Text`, email addresses and URLs require special normalisation
before WER is computed (see section below). The `Transcript` field captures how the speaker
said the address as words, so this issue does not arise when using the default `transcript` GT.

---

## Transcript Annotation Types

The `Transcript` field uses several annotation conventions that must be handled before the
text is used for synthesis or WER scoring.

### `[Bracketed]` annotations — strip, do not synthesize

Square-bracket blocks appear at the start of spontaneous prompt transcripts. They contain
the cue text shown to the speaker and are **not spoken** — they are a system annotation.

```
Transcript: "[Tell us about one of your favorite bands or singers.] One of my
             favorite bands is Eagles. Period. They did their first concert..."
```

**Fix:** Strip `[...]` blocks with a regex before use. The `clean_transcript()` function in
both `generate_synthetic_speech.py` and `calculate_sap_wer.py` handles this.

---

### `(cs: ...)` annotations — exclude utterance entirely

Parenthesised blocks prefixed with `cs:` indicate **cued speech**: a clinician read a phrase
aloud and the speaker repeated it. These recordings contain two speakers mixed in the audio.

```
Transcript: "(cs: He then resorted) He then resorted (cs: to a trick) to a trick
             (cs: which is in itself) which is in itself (cs: objectionable)
             (object) objectionable. (cs: Good.)"
```

These utterances **cannot be used** for voice conversion training (source audio has two
speakers) or for meaningful WER scoring (ASR would transcribe both speakers). They are
excluded entirely in both `generate_synthetic_speech.py` and `calculate_sap_wer.py` — any
utterance whose `Transcript` contains `(cs:` is skipped and counted in the logs.

---

### `(parenthesised)` content without `cs:` — keep as-is

Round brackets without a `cs:` prefix denote disfluencies, false starts, or partial words
produced by the speaker. These **are present in the audio** and should be retained.

```
Transcript: "One of my favorite bands is Eagles. Period. They did their first
             concert in New York City (when) on my fifth (anna-) wedding anniversary."
```

Here `(when)` is an interjection and `(anna-)` is a false start on "anniversary". Both are
audible in the recording. They are left in the transcript for synthesis and WER.

**Note:** Partial words with trailing dashes (e.g. `(anna-)`) may cause StyleTTS2 to behave
unexpectedly since a dash is not a pronounceable character. This is a known minor issue.

---

### Truncated transcripts (severe speakers)

For severely impaired speakers, the transcript may be shorter than the prompt — the speaker
did not finish the sentence. This is not an annotation artefact; it reflects what was actually
said. The truncated transcript is used as-is for both synthesis and WER.

```
Prompt Text : "A vague fear crept over her that he might finally succeed in proving
               to her that it was her duty to resign."
Transcript  : "A vague fear crept over her that he might finally succeed in proving
               to her that."
Comment     : "audio cuts off at 2 minute mark"
```

Using the truncated `Transcript` for synthesis means the target audio covers only what the
speaker said — source and target remain content-aligned even for incomplete utterances.

---

## Known Transcript Artifacts and Normalisation

The SAP transcripts were not designed for ASR evaluation, and several systematic mismatches
between transcript conventions and ASR output inflate WER artificially. The following are
handled in `normalize_text()` in `calculate_sap_wer.py` before WER is computed.

### 1. Digit vs. Word Form

Transcripts use numerals; ASR outputs words.

```
GT  : what team won the world series in 1938
ASR : what team won the world series in nineteen thirty eight
WER : 37.50%  (without normalisation)
WER :  0.00%  (with normalisation)
```

**Fix:** Expand digits to word form in the GT using `num2words` before scoring.

---

### 2. Concatenated Number/Time Expressions

Transcripts write time and date expressions as single tokens without spaces;
ASR correctly splits them into individual words.

```
GT  : cancel my meeting with sib and gertrud at ninethirty am sunday
ASR : cancel my meeting with sib and gertrude at nine thirty a m sunday
```

```
GT  : show me photos from fourthirtytwo thousand and twentytwo
ASR : show me photos from four thirty twenty twenty two
```

**Fix:** Greedy prefix matching against a list of known number words splits
concatenated tokens before scoring. E.g. `"ninethirty"` → `"nine thirty"`,
`"twentytwo"` → `"twenty two"`.

---

### 3. Email Addresses and URLs

When using `Prompt Text` as GT, email addresses and URLs are present in their
written form (e.g. `Maria.Sari@wildernessnet.org`). Speakers say these as word
sequences ("maria dot sari at wildernessnet dot org"), which is also what the
ASR outputs. Simply stripping punctuation collapses the written form to a single
token (`mariasariwildernessnetorg`), producing WER > 100%.

```
GT  (raw)        : Send an email to Maria.Sari@wildernessnet.org.
GT  (normalised) : send an email to maria dot sari at wildernessnet dot org
ASR              : send an email to maria dot sari at wilderness net dot org
WER              : ~12%  (only "wildernessnet" vs "wilderness net" differs)
```

**Fix:** Before stripping punctuation, expand characters that correspond to
spoken words in email/URL context:
- `@` → `at` (always spoken as "at")
- `.` between two word characters → `dot` (mid-word dot in domain names, emails)
- All remaining punctuation removed as normal

The lookahead/lookbehind `(?<=\w)\.(?=\w)` ensures only mid-word dots are
expanded. Sentence-ending periods (no word character after) and abbreviation
trailing dots (space after) are removed rather than expanded.

**Edge case:** Abbreviations like `e.g.` — the first dot matches (word chars
on both sides) and becomes "dot", giving "eg dot use this". This is a minor
distortion but abbreviations are rare in the SAP command-style prompts.

**Residual error:** `wildernessnet` (one token) vs `wilderness net` (two tokens)
remains after normalisation. This reflects how the speaker articulated the domain
name and is a genuine intelligibility signal — not an artefact.

---

### 4. "AM/PM" Splitting

ASR sometimes splits abbreviations: `"am"` → `"a m"`. This is not currently
normalised. The effect is small (1 extra token per occurrence) and normalising
it risks false positives elsewhere.

---

---

## Example Output — Speaker `0c3278fa-e89` (Parkinson's Disease, 449 utterances, Avg WER 53.89%)

These examples illustrate the range of WER outcomes and the normalisation decisions in practice.

**Example 1/449 — kindhearted hyphen, contraction**
```
Prompt  : I'm quite sure you are too kind-hearted to enjoy giving pain to any living creature.
GT norm : im quite sure you are too kindhearted to enjoy giving pain to any living creature
ASR     : im quite sure youre too kind hearted to enjoy giving pain to any living creature
WER     : 26.67%
```
Two issues: (1) "you are" → "youre" — the speaker contracted it, ASR followed, WER correctly
penalises this as a real intelligibility difference from the prompt. (2) "kindhearted" (hyphen
stripped) vs "kind hearted" (two words from ASR) — a minor normalisation artefact from hyphen
removal. Hyphens in compound words could arguably be expanded to a space, but edge cases
(e.g. "kind-of") make this non-trivial.

**Example 101/449 — open-ended prompt, speaker went off-script**
```
Prompt Text : Tell us about one of your favorite bands or singers.
Transcript  : [Tell us about one of your favorite bands or singers.] One of my
              favorite singers was linda ronstat she sounded like a songbird...
GT norm     : one of my favorite singers was linda ronstat she sounded like a songbird...
ASR         : one of my favorite singers was linda ronstat she sounded like a songbird...
WER         : ~0%
```
This is a `Spontaneous Speech Prompts` utterance (`Category Description`). With the old
`--gt-source prompt-text` default, WER would exceed 100% because the speaker's free-form
response produces far more words than the cue. With the current default (`--gt-source transcript`),
the `[...]` annotation is stripped and WER is scored against what was actually said, giving a
meaningful result. The `category_description` field is preserved in the pairs CSV and Lhotse
manifests so these utterances can be analysed or filtered separately downstream.

**Example 201/449 — proper noun, dysarthric distortion**
```
Prompt  : Cortana, play music.
GT norm : cortana play music
ASR     : krtana play music
WER     : 33.33%
```
"Cortana" → "krtana" — the speaker dropped the initial vowel. The ASR picked up the distorted
form. This is a genuine intelligibility error correctly captured by WER.

**Example 301/449 — clean transcription**
```
Prompt  : Get help.
GT norm : get help
ASR     : get help
WER     : 0.00%
```
Short, clear command. No errors.

**Example 401/449 — medication name, partial distortion**
```
Prompt  : Set a reminder to take ibuprofen at 11:30 AM.
GT norm : set a reminder to take ibuprofen at eleven thirty am
ASR     : set a reminder to take ibuprofenle at eleven thirty am
WER     : 10.00%
```
"ibuprofen" → "ibuprofenle" — a subtle articulation error on a difficult word. The number
normalisation (`11:30` → `eleven thirty`) worked correctly here. Note: `11:30` contains a
colon, which is stripped as punctuation; the digits are expanded separately by `num2words`.

**On contractions (e.g. "when's" → "whens" vs "when is")**

The apostrophe in contractions is stripped by normalisation, producing e.g. "whens" (1 token).
The ASR may expand this to "when is" (2 tokens), yielding inflated WER. However, this is
intentionally left as-is: if the speaker said "whens" (dropping the contraction), that is a
real intelligibility signal. The ASR expanding it reflects the language model filling in
expected speech, not what was actually articulated.

---

## Remaining Sources of WER Inflation

Beyond transcript artifacts, genuine ASR failures on dysarthric speech include:

- **Phoneme substitutions**: reduced articulation causes the model to hear a different
  phoneme. E.g. `"puzzled"` → `"puzzle"` (final /d/ dropped).
- **Word merging**: running speech causes the model to merge adjacent words.
  E.g. `"hey siri"` → `"heysari"`.
- **Severe dysarthria**: highly distorted speech may produce near-random ASR output.
  E.g. `"acetaminophen"` → `"a seat of mdifin"`.

These are **not** artifacts — they reflect real intelligibility difficulty and should
be preserved in the WER score.

---

## Interpreting WER as an Intelligibility Proxy

- SAP rating scale: **1 = most intelligible, 7 = least intelligible** (inverted from intuition)
- Expected relationship: **higher WER ↔ higher SAP rating** (both indicate lower intelligibility)
- The output CSV (`pd_train_wer.csv`) includes both `Average_Rating` (null for unrated speakers)
  and `Average_WER` side by side for direct comparison.
- Before using WER for validation set stratification, it is worth plotting the correlation
  between WER and SAP rating for the rated speakers to validate the proxy.

---

## Validation Set Stratification Plan

Once WER scores are available for all PD TRAIN speakers:

1. For rated speakers: use `Average_Rating` as the intelligibility score.
2. For unrated speakers: use `Average_WER` as the intelligibility score
   (after verifying the correlation with ratings).
3. Bin speakers into intelligibility tiers (exact boundaries TBD based on the distribution).
4. Sample proportionally from each tier into a validation set.
5. Pass the selected speaker IDs to `select_validation_speakers.py` → `val_speakers_PD.csv`.
6. Pass to `sap.py --val-speakers val_speakers_PD.csv` to produce train/val/test manifests.
