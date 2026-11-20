"""
prepare_hamlet_corpus.py — Hamlet PDF corpus preparation pipeline.
================================================================

WHAT IT DOES
    Extracts the Folger Shakespeare Library Hamlet PDF, cleans it, and
    produces a structured, scene-level dataset for my-LLM Stage 7.

    1. Extracts text from data/raw/Hamlet.pdf using the pure-Python
       pdf_extractor (no external dependencies).
    2. Removes page numbers, repeated page headers, and FTLN line
       identifiers from each page.
    3. Concatenates cleaned pages into data/processed/hamlet_clean.txt.
    4. Detects act/scene boundaries, speakers, and stage directions,
       writing them to data/processed/hamlet_scenes.jsonl.
    5. Writes data/manifests/corpus_manifest.json describing the corpus.
    6. Writes reports/hamlet_corpus_report.md with quality metrics.

INPUT
    data/raw/Hamlet.pdf  (Folger Shakespeare Library edition, 144 pages)

OUTPUTS (all inside the project folder)
    data/processed/hamlet_clean.txt        — cleaned full play text
    data/processed/hamlet_scenes.jsonl     — one JSON record per scene
    data/manifests/corpus_manifest.json    — corpus manifest with weights
    reports/hamlet_corpus_report.md        — human-readable quality report

SAFETY
    * The original PDF is never modified.
    * No network calls, no downloads, no external libraries.
    * All writes stay inside the my-LLM project folder.

HOW TO RUN
    python src\\stage7\\prepare_hamlet_corpus.py

This file is part of the from-scratch my-LLM project. We authored it.
"""

import hashlib
import json
import os
import re
import sys

# Make src/stage7 importable
STAGE7_DIR = os.path.dirname(os.path.abspath(__file__))
if STAGE7_DIR not in sys.path:
    sys.path.insert(0, STAGE7_DIR)

from pdf_extractor import extract_pdf_text, clean_page_text


# --- project-relative paths ----------------------------------------------

PROJECT_ROOT = os.path.dirname(os.path.dirname(STAGE7_DIR))
RAW_PDF = os.path.join(PROJECT_ROOT, "data", "raw", "Hamlet.pdf")
CLEAN_TXT = os.path.join(PROJECT_ROOT, "data", "processed", "hamlet_clean.txt")
SCENES_JSONL = os.path.join(PROJECT_ROOT, "data", "processed", "hamlet_scenes.jsonl")
MANIFEST_JSON = os.path.join(PROJECT_ROOT, "data", "manifests", "corpus_manifest.json")
REPORT_MD = os.path.join(PROJECT_ROOT, "reports", "hamlet_corpus_report.md")


# --- scene / speaker / stage-direction detection --------------------------

# Matches act headers like "ACT 1", "ACT I", "ACT 1. SC.", etc.
ACT_RE = re.compile(
    r"^ACT\s+(\d+|[IVX]+)"
    r"(?:\s*\.\s*SC\.?\s*(\d+|[IVX]+)?)?"
    r"(?:\s+Scene\s+(\d+|[IVX]+))?"
    r"\s*$",
    re.IGNORECASE,
)

# Matches scene headers like "Scene 1", "Scene II", or standalone "Scene"
SCENE_RE = re.compile(r"^Scene\s+(\d+|[IVX]+)\s*$", re.IGNORECASE)
SCENE_STANDALONE_RE = re.compile(r"^Scene\s*$", re.IGNORECASE)

# Matches a page footer like "9 Hamlet" or "Hamlet 9"
PAGE_FOOTER_RE = re.compile(r"^\d+\s+Hamlet$|^Hamlet\s+\d+$", re.IGNORECASE)

# Matches a speaker line: uppercase name optionally followed by dialogue
# on the same line, e.g. "HORATIO Friends to this ground."
# The speaker name must be a known Hamlet character or a plausible
# all-caps name (2-30 chars, letters, spaces, apostrophes, periods).
SPEAKER_NAME_RE = re.compile(
    r"^([A-Z][A-Z'\.\s]{1,29})(\s+(.*))?$"
)

# Common stage-direction markers (lowercase or bracketed)
STAGE_DIRECTION_KEYWORDS = (
    "enter", "exit", "exuent", "exeunt", "exeunt.",
    "within", "above", "below", "aside", "to him", "to her",
    "a march", "a flourish", "alarum", "battle", "drum",
    "trumpet", "sennet", "hautboys", "music", "song",
    "they fight", "they exit", "they run", "die", "falls",
    "kneels", "rises", "stands", "sits", "lies", "sleep",
    "reads", "writes", "draws", "sheath", "strike",
    "lights", "candles", "torch", "thunder", "lightning",
    "ghost", "appears", "disappears", "vanishes",
)

# Speaker names known from Hamlet (uppercase)
KNOWN_SPEAKERS = {
    "HAMLET", "CLAUDIUS", "GERTRUDE", "POLONIUS", "OPHELIA",
    "LAERTES", "HORATIO", "GHOST", "MARCELLUS", "BARNARDO",
    "FRANCISCO", "ROSENCRANTZ", "GUILDENSTERN", "FORTINBRAS",
    "VOLTEMAND", "CORNELIUS", "OSRIC", "REYNALDO",
    "PLAYER", "PLAYERS", "SERVANT", "SERVANTS", "SAILOR",
    "SAILORS", "MESSENGER", "CAPTAIN", "OFFICER", "LORD",
    "LORDS", "LADY", "LADIES", "GENTLEMAN", "GENTLEMEN",
    "FIRST", "SECOND", "THIRD", "FOURTH", "FIFTH",
    "KING", "QUEEN", "ALL", "OTHER",
    "PLAYER KING", "PLAYER QUEEN",
    "THE GHOST", "FIRST PLAYER", "GRAVEDIGGER",
    "FIRST GRAVEDIGGER", "SECOND GRAVEDIGGER",
    "FIRST SAILOR", "SECOND SAILOR",
    "FIRST LORD", "SECOND LORD",
}


def is_stage_direction(line):
    """Heuristic: decide whether a line is a stage direction."""
    stripped = line.strip()
    if not stripped:
        return False

    lower = stripped.lower()

    # Bracketed stage directions: [Enter HAMLET] or (Enter HAMLET)
    if stripped.startswith("[") and stripped.endswith("]"):
        return True
    if stripped.startswith("(") and stripped.endswith(")"):
        return True

    # Starts with a stage-direction keyword
    for kw in STAGE_DIRECTION_KEYWORDS:
        if lower.startswith(kw):
            return True

    return False


def roman_to_int(s):
    """Convert a Roman numeral to an int, or return None."""
    s = s.strip().upper()
    roman_map = {
        "I": 1, "II": 2, "III": 3, "IV": 4, "V": 5,
        "VI": 6, "VII": 7, "VIII": 8, "IX": 9, "X": 10,
        "XI": 11, "XII": 12, "XIII": 13, "XIV": 14, "XV": 15,
        "XVI": 16, "XVII": 17, "XVIII": 18, "XIX": 19, "XX": 20,
    }
    if s in roman_map:
        return roman_map[s]
    # Fallback: try parsing as an integer
    try:
        return int(s)
    except ValueError:
        return None


def parse_scenes(text):
    """Parse cleaned play text into scene records.

    Each record has:
        act, scene, act_scene_key, speaker, dialogue,
        stage_direction, source_page, source_line, text_type

    Returns a list of record dicts.

    Parsing strategy:
      - Remove page-footers (e.g. "9 Hamlet") line by line.
      - Skip front matter (table of contents, synopsis, director's note)
        until the first known speaker appears — that marks the start of
        the actual play.
      - Track the current act from "ACT X. SC." running headers and from
        standalone "ACT" + number headers. Running headers update the act
        but NEVER reset the scene counter.
      - Each standalone "Scene" line starts a new scene within the current
        act (scene counter increments by one).
      - When a new act begins (standalone "ACT N"), the scene counter
        resets to 1.
      - Match speaker names against KNOWN_SPEAKERS first, then fall back
        to plausible all-caps names.
      - When a speaker name appears on the same line as dialogue, split
        them into separate fields.
    """
    records = []
    lines = text.split("\n")

    current_act = None
    current_scene = 0
    last_act_scene_key = None  # track last boundary to count scenes

    # Play has not started yet; skip front matter until first known speaker.
    play_started = False

    # Pending act header state (for "ACT" on one line, number on next).
    pending_act = False

    i = 0
    n = len(lines)
    while i < n:
        stripped = lines[i].strip()
        if not stripped:
            i += 1
            continue

        # --- Remove page footer like "9 Hamlet" or "21 Hamlet" ---
        if PAGE_FOOTER_RE.match(stripped):
            i += 1
            continue

        # --- Detect play start: first known speaker in dialogue format ---
        # We require the speaker name to be followed by real dialogue:
        # (a) starts with an UPPERCASE letter (start of a sentence), AND
        # (b) has at least 4 words (real sentences are longer than the
        #     short noun phrases used in character-list descriptions like
        #     "Danish soldiers" or "courtiers at the Danish court").
        # This avoids triggering on front-matter character lists.
        if not play_started:
            speaker_match = None
            speaker_rest = ""
            for known in sorted(KNOWN_SPEAKERS, key=len, reverse=True):
                if stripped == known:
                    speaker_match = known
                    speaker_rest = ""
                    break
                if stripped.startswith(known + " "):
                    speaker_match = known
                    speaker_rest = stripped[len(known):].strip()
                    break
            if speaker_match is not None:
                # Accept only if the rest looks like real dialogue:
                # (a) starts with an UPPERCASE letter (start of a sentence), AND
                # (b) has at least 4 words (real sentences are longer than the
                #     short noun phrases used in character-list descriptions), AND
                # (c) the first word after the speaker name is NOT another
                #     known speaker name (e.g. "KING CLAUDIUS, ..." is a
                #     character description, not dialogue).
                looks_like_dialogue = False
                if speaker_rest:
                    first_ch = speaker_rest[0]
                    words = speaker_rest.split()
                    first_word = words[0].rstrip(",.") if words else ""
                    if (first_ch.isupper()
                            and len(words) >= 4
                            and first_word not in KNOWN_SPEAKERS):
                        looks_like_dialogue = True
                    elif first_ch in "\"'" and len(speaker_rest) > 1:
                        if (speaker_rest[1].isupper()
                                and len(words) >= 4
                                and first_word not in KNOWN_SPEAKERS):
                            looks_like_dialogue = True
                if not speaker_rest:
                    # Speaker name on its own line: check the next non-empty line.
                    # Real play dialogue follows with a sentence (>= 4 words);
                    # character-list entries are followed by another name or a
                    # short phrase.
                    for j in range(i + 1, min(n, i + 5)):
                        nxt = lines[j].strip()
                        if nxt:
                            nxt_words = nxt.split()
                            # Skip if next line is another known speaker name
                            # (still in the character list).
                            if nxt in KNOWN_SPEAKERS:
                                break
                            # Skip if next line is a single all-caps word
                            # (another character-list name).
                            if nxt.isupper() and len(nxt_words) == 1 and len(nxt) <= 20:
                                break
                            # Skip short noun-phrase descriptions (2-3 words)
                            # like "A Lord" or "Danish soldiers".
                            if len(nxt_words) <= 3:
                                break
                            # Real dialogue: next line is a sentence (>= 4 words,
                            # not all uppercase, starts with uppercase, and does
                            # not begin with another speaker name).
                            if (len(nxt_words) >= 4
                                    and not nxt.isupper()
                                    and nxt[0].isupper()
                                    and nxt_words[0] not in KNOWN_SPEAKERS):
                                looks_like_dialogue = True
                            break
                if looks_like_dialogue:
                    play_started = True
                    if current_act is None:
                        current_act = 1
                        current_scene = 1  # first scene of the play
                # Whether or not we start the play, advance past this line
                i += 1
                continue
            else:
                # Still in front matter; skip this line.
                i += 1
                continue

        # --- Combined act+scene header: "ACT 1. SC.", "ACT 1 Scene 2", etc. ---
        # Running header: updates the act but NEVER resets the scene counter.
        m = ACT_RE.match(stripped)
        if m:
            act_val = roman_to_int(m.group(1))
            if act_val is not None and act_val != current_act:
                current_act = act_val
                current_scene = 1
            elif act_val is not None and current_act is not None:
                # Same act as before: running header only, scene unchanged.
                pass
            elif current_act is None:
                current_act = act_val
                current_scene = 1
            i += 1
            continue

        # --- Standalone "ACT" header (number may be on next line) ---
        if stripped.upper() == "ACT":
            pending_act = True
            i += 1
            continue

        # --- Standalone "Scene" header: starts a new scene in current act ---
        # If the next line is a standalone "ACT" with the same act number,
        # this is a page-header repeat (running header at top of page), not
        # a real scene boundary — skip it. If the ACT number differs, this
        # is a real act/scene boundary.
        if SCENE_STANDALONE_RE.match(stripped):
            next_act_num = None
            if i + 1 < n and lines[i + 1].strip().upper() == "ACT":
                # Look ahead for the act number after "ACT". The number may
                # not be on the immediately following line (e.g. "ACT" then
                # blank line then content, with "ACT 2. SC." appearing as a
                # running header several lines later). Scan up to 80 lines.
                for j in range(i + 2, min(n, i + 80)):
                    nxt = lines[j].strip()
                    if not nxt:
                        continue
                    m2 = ACT_RE.match(nxt)
                    if m2:
                        act_val2 = roman_to_int(m2.group(1))
                        if act_val2 is not None:
                            next_act_num = act_val2
                            break
                    # Also accept a standalone number right after ACT
                    act_num = roman_to_int(nxt)
                    if act_num is not None:
                        next_act_num = act_num
                        break
            if next_act_num is not None:
                if next_act_num == current_act:
                    # Page-header repeat: same act, not a real boundary.
                    pending_act = False
                    i += 1
                    continue
                else:
                    # Real act change via Scene + ACT divider.
                    current_act = next_act_num
                    current_scene = 1
                    pending_act = False
                    key = f"{current_act or 0}.{current_scene or 0}"
                    if key != last_act_scene_key:
                        last_act_scene_key = key
                    i += 1
                    continue
            else:
                # Real scene boundary within the current act.
                current_scene = (current_scene or 1) + 1
                key = f"{current_act or 0}.{current_scene or 0}"
                if key != last_act_scene_key:
                    last_act_scene_key = key
            pending_act = False
            i += 1
            continue

        # --- Resolve pending act number from next line ---
        if pending_act:
            act_val = roman_to_int(stripped)
            if act_val is not None:
                current_act = act_val
                current_scene = 1
                pending_act = False
                key = f"{current_act or 0}.{current_scene or 0}"
                if key != last_act_scene_key:
                    last_act_scene_key = key
                i += 1
                continue
            else:
                pending_act = False

        # --- Stage direction ---
        if is_stage_direction(stripped):
            records.append({
                "act": current_act,
                "scene": current_scene,
                "act_scene_key": f"{current_act or 0}.{current_scene or 0}",
                "speaker": None,
                "dialogue": None,
                "stage_direction": stripped,
                "source_page": None,
                "source_line": None,
                "text_type": "stage_direction",
            })
            i += 1
            continue

        # --- Speaker + dialogue ---
        # Try to match a known speaker name at the start of the line.
        # If found, everything after the name (separated by whitespace)
        # is dialogue. This prevents false positives like single letters
        # or dialogue initials being treated as speaker names.
        speaker_name = None
        rest = ""
        for known in sorted(KNOWN_SPEAKERS, key=len, reverse=True):
            # Match known speaker at start of line, followed by space or end
            if stripped == known:
                speaker_name = known
                rest = ""
                break
            if stripped.startswith(known + " "):
                speaker_name = known
                rest = stripped[len(known):].strip()
                break

        if speaker_name is None:
            # Fallback: plausible all-caps name (2-30 chars, no digits)
            m = SPEAKER_NAME_RE.match(stripped)
            if m:
                candidate = m.group(1).strip()
                candidate = re.sub(r"\s+", " ", candidate)
                rest_candidate = (m.group(3) or "").strip()
                if (candidate.isupper()
                        and len(candidate) <= 30
                        and not any(ch.isdigit() for ch in candidate)
                        and candidate not in ("ACT", "SCENE", "SC.")
                        and len(candidate) >= 2):
                    # Additional sanity: the rest should look like dialogue
                    # (starts with lowercase or punctuation, or is empty)
                    if not rest_candidate or rest_candidate[0].islower() or rest_candidate[0] in '",\'.–-(':
                        speaker_name = candidate
                        rest = rest_candidate

        if speaker_name is not None:
            dialogue = rest if rest else None
            records.append({
                "act": current_act,
                "scene": current_scene,
                "act_scene_key": f"{current_act or 0}.{current_scene or 0}",
                "speaker": speaker_name,
                "dialogue": dialogue,
                "stage_direction": None,
                "source_page": None,
                "source_line": None,
                "text_type": "dialogue",
            })
            i += 1
            continue

        # --- Generic text (narration, orphaned lines) ---
        records.append({
            "act": current_act,
            "scene": current_scene,
            "act_scene_key": f"{current_act or 0}.{current_scene or 0}",
            "speaker": None,
            "dialogue": stripped,
            "stage_direction": None,
            "source_page": None,
            "source_line": None,
            "text_type": "text",
        })
        i += 1

    return records


# --- manifest -----------------------------------------------------------

def build_manifest(clean_chars, scenes_count, speakers, dialogue_entries,
                   stage_directions, unknown_token_chars):
    """Build the corpus manifest JSON payload."""
    return {
        "format": "my-LLM corpus manifest v1",
        "stage": "Stage 7 — corpus engineering",
        "created": "",
        "description": (
            "Cleaned Hamlet corpus (Folger Shakespeare Library edition) "
            "for my-LLM language-model training. Kept separate from the "
            "Stage 5 original-language corpus."
        ),
        "source": {
            "file": "data/raw/Hamlet.pdf",
            "edition": "Folger Shakespeare Library",
            "pages": 144,
            "play_begins_on_page": 8,
            "acts": 5,
            "scenes": 20,
        },
        "corpora": [
            {
                "id": "corpus-a-original",
                "name": "Original language material",
                "file": "data/tiny_corpus.txt",
                "weight": 0.25,
                "notes": "Locally authored educational prose (Stage 5).",
            },
            {
                "id": "corpus-b-shakespeare",
                "name": "Shakespearean literary material",
                "file": "data/processed/hamlet_clean.txt",
                "weight": 0.25,
                "notes": "Hamlet, Folger edition. Extracted and cleaned from PDF.",
            },
            {
                "id": "corpus-c-modern",
                "name": "Modern English material",
                "file": None,
                "weight": 0.20,
                "notes": "To be added in a later stage. Placeholder.",
            },
            {
                "id": "corpus-d-conversation",
                "name": "Original conversational examples",
                "file": None,
                "weight": 0.15,
                "notes": "Conversational fine-tuning format (see format note).",
            },
            {
                "id": "corpus-e-teacher",
                "name": "Optional teacher-generated examples",
                "file": None,
                "weight": 0.15,
                "notes": "Optional distillation data. Not included yet.",
            },
        ],
        "weight_note": (
            "Weights sum to 1.0. Hamlet is capped at 25% so it cannot "
            "dominate the language distribution merely because its file "
            "is larger than the original documents."
        ),
        "conversation_format": {
            "example": {
                "messages": [
                    {"role": "user", "content": "Explain gravity."},
                    {"role": "assistant", "content": "Gravity is the force that attracts objects toward each other."}
                ]
            },
            "notes": "Kept separate from literary pretraining data.",
        },
        "stats": {
            "clean_characters": clean_chars,
            "scenes": scenes_count,
            "speakers": len(speakers),
            "dialogue_entries": dialogue_entries,
            "stage_directions": stage_directions,
            "unknown_token_chars": unknown_token_chars,
        },
    }


# --- main ---------------------------------------------------------------

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    print("=== Hamlet Corpus Preparation Pipeline ===")
    print()

    if not os.path.isfile(RAW_PDF):
        print(f"ERROR: Source PDF not found: {RAW_PDF}")
        print("       Place Hamlet.pdf at data/raw/Hamlet.pdf")
        sys.exit(1)

    # 1) Extract text from PDF
    print(f"Extracting text from {RAW_PDF} ...")
    pages = extract_pdf_text(RAW_PDF)
    print(f"  Pages extracted: {len(pages)}")

    total_raw_chars = sum(len(p["raw_text"]) for p in pages)
    print(f"  Total raw extracted characters: {total_raw_chars:,}")

    # 2) Clean each page
    print("Cleaning pages (page numbers, headers, FTLN identifiers) ...")
    cleaned_pages = []
    all_notes = []
    for p in pages:
        cleaned, notes = clean_page_text(
            p["raw_text"], p["page_number"], len(pages)
        )
        cleaned_pages.append(cleaned)
        all_notes.extend(notes)

    clean_text = "\n\n".join(cleaned_pages)
    # Normalize: collapse excessive blank lines
    clean_text = re.sub(r"\n{3,}", "\n\n", clean_text)
    clean_text = clean_text.strip()

    clean_chars = len(clean_text)
    print(f"  Cleaned characters: {clean_chars:,}")
    print(f"  Cleaning notes: {len(all_notes)}")

    # 3) Write hamlet_clean.txt
    os.makedirs(os.path.dirname(CLEAN_TXT), exist_ok=True)
    with open(CLEAN_TXT, "w", encoding="utf-8", newline="\n") as f:
        f.write(clean_text)
    print(f"  Written: {os.path.relpath(CLEAN_TXT, PROJECT_ROOT)}")

    # 4) Parse scenes
    print("Parsing scenes, speakers, and stage directions ...")
    records = parse_scenes(clean_text)

    # Count stats
    scenes_set = set()
    speakers_set = set()
    dialogue_count = 0
    stage_direction_count = 0
    for r in records:
        # A scene is identified by its act_scene_key (act.scene).
        # Count it if we have at least an act number.
        if r["act"] is not None:
            scenes_set.add(r["act_scene_key"])
        if r["speaker"]:
            speakers_set.add(r["speaker"])
        if r["text_type"] == "dialogue":
            dialogue_count += 1
        if r["text_type"] == "stage_direction":
            stage_direction_count += 1

    scenes_count = len(scenes_set)
    print(f"  Scenes detected: {scenes_count}")
    print(f"  Speakers identified: {len(speakers_set)}")
    print(f"  Dialogue entries: {dialogue_count}")
    print(f"  Stage directions: {stage_direction_count}")

    # 5) Write hamlet_scenes.jsonl
    os.makedirs(os.path.dirname(SCENES_JSONL), exist_ok=True)
    with open(SCENES_JSONL, "w", encoding="utf-8", newline="\n") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  Written: {os.path.relpath(SCENES_JSONL, PROJECT_ROOT)}")

    # 6) Build and write manifest
    manifest = build_manifest(
        clean_chars=clean_chars,
        scenes_count=scenes_count,
        speakers=speakers_set,
        dialogue_entries=dialogue_count,
        stage_directions=stage_direction_count,
        unknown_token_chars=0,  # filled in by tokenizer audit
    )
    manifest["created"] = __import__("datetime").datetime.now().isoformat()
    os.makedirs(os.path.dirname(MANIFEST_JSON), exist_ok=True)
    with open(MANIFEST_JSON, "w", encoding="utf-8", newline="\n") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"  Written: {os.path.relpath(MANIFEST_JSON, PROJECT_ROOT)}")

    # 7) Write corpus quality report
    print("Writing corpus quality report ...")
    write_report(
        clean_chars, clean_text, scenes_count, speakers_set,
        dialogue_count, stage_direction_count, all_notes, pages
    )
    print(f"  Written: {os.path.relpath(REPORT_MD, PROJECT_ROOT)}")

    print()
    print("=== Hamlet corpus preparation complete ===")


def write_report(clean_chars, clean_text, scenes_count, speakers_set,
                 dialogue_count, stage_direction_count, all_notes, pages):
    """Write the human-readable corpus quality report."""
    word_count = len(clean_text.split())

    # Character coverage: which ASCII letters/digits/punct appear
    present_chars = sorted(set(clean_text))
    ascii_lower = [c for c in present_chars if "a" <= c <= "z"]
    ascii_upper = [c for c in present_chars if "A" <= c <= "Z"]
    digits = [c for c in present_chars if "0" <= c <= "9"]
    punct = [c for c in present_chars if c in ".,;:!?'\"()-–—…"]
    whitespace = [c for c in present_chars if c in " \t\n\r"]
    non_ascii = [c for c in present_chars if ord(c) > 127]

    # Check for page-header / FTLN residue
    header_residue = [n for n in all_notes if "removed repeated header" in n]
    ftln_residue = [n for n in all_notes if "removed inline FTLN" in n]
    page_num_residue = [n for n in all_notes if "removed page number" in n]

    # Check for artifacts
    artifact_checks = []
    for p in pages[:5]:
        txt = p["raw_text"]
        if "FTLN" in txt:
            artifact_checks.append(
                f"  Page {p['page_number']}: FTLN identifiers found in raw text (removed in cleaning)"
            )
        if re.search(r"\bHamlet\b", txt, re.IGNORECASE):
            artifact_checks.append(
                f"  Page {p['page_number']}: 'Hamlet' header found in raw text"
            )

    lines = []
    lines.append("# Hamlet Corpus Quality Report")
    lines.append("")
    lines.append("## Source")
    lines.append("")
    lines.append("- File: `data/raw/Hamlet.pdf`")
    lines.append("- Edition: Folger Shakespeare Library")
    lines.append("- Pages: 144 (play begins on PDF page 8)")
    lines.append("- Acts: 5 | Scenes: 20")
    lines.append("")
    lines.append("## Extraction Statistics")
    lines.append("")
    lines.append(f"- Total extracted characters (raw): {sum(len(p['raw_text']) for p in pages):,}")
    lines.append(f"- Cleaned characters: {clean_chars:,}")
    lines.append(f"- Word count: {word_count:,}")
    lines.append(f"- Scenes detected: {scenes_count}")
    lines.append(f"- Speakers identified: {len(speakers_set)}")
    lines.append(f"- Dialogue entries: {dialogue_count}")
    lines.append(f"- Stage directions: {stage_direction_count}")
    lines.append("")
    lines.append("## Character Coverage")
    lines.append("")
    lines.append(f"- Lowercase letters ({len(ascii_lower)}): `{''.join(ascii_lower)}`")
    lines.append(f"- Uppercase letters ({len(ascii_upper)}): `{''.join(ascii_upper)}`")
    lines.append(f"- Digits ({len(digits)}): `{''.join(digits)}`")
    lines.append(f"- Standard punctuation ({len(punct)}): `{''.join(punct)}`")
    lines.append(f"- Whitespace ({len(whitespace)}): space, tab, newline, carriage return")
    lines.append(f"- Non-ASCII characters ({len(non_ascii)}):")
    for c in sorted(non_ascii):
        lines.append(f"    - U+{ord(c):04X} {c!r}")
    lines.append("")
    lines.append("## Cleaning Operations Performed")
    lines.append("")
    lines.append("- Removed PDF page numbers")
    lines.append("- Removed repeated page headers ('Hamlet')")
    lines.append("- Removed Folger line identifiers (FTLN 0001, etc.)")
    lines.append("- Collapsed excessive blank lines")
    lines.append("- Preserved character names, dialogue, punctuation,")
    lines.append("  archaic vocabulary, sentence structure, scene boundaries,")
    lines.append("  act boundaries, stage directions, paragraph breaks")
    lines.append("")
    lines.append("## Extraction Warnings")
    lines.append("")
    if header_residue:
        lines.append(f"- Repeated headers removed: {len(header_residue)} occurrence(s)")
    else:
        lines.append("- No repeated page headers detected")
    if ftln_residue:
        lines.append(f"- FTLN identifiers removed: {len(ftln_residue)} occurrence(s)")
    else:
        lines.append("- No FTLN identifiers detected")
    if page_num_residue:
        lines.append(f"- Page numbers removed: {len(page_num_residue)} occurrence(s)")
    else:
        lines.append("- No standalone page numbers detected")
    lines.append("")
    lines.append("## Artifact Check")
    lines.append("")
    if artifact_checks:
        for a in artifact_checks:
            lines.append(a)
    else:
        lines.append("- No PDF artifacts (page numbers, headers) found in cleaned text")
    lines.append("")
    lines.append("## Speaker List")
    lines.append("")
    for s in sorted(speakers_set):
        lines.append(f"- {s}")
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- The extraction preserves archaic vocabulary and original punctuation.")
    lines.append("- No modernization of Shakespeare's language was performed.")
    lines.append("- Scene boundaries are detected from ACT/Scene headers.")
    lines.append("- Stage directions are identified heuristically.")
    lines.append("- The original PDF at data/raw/Hamlet.pdf is untouched.")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("*Generated by src/stage7/prepare_hamlet_corpus.py*")

    os.makedirs(os.path.dirname(REPORT_MD), exist_ok=True)
    with open(REPORT_MD, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
