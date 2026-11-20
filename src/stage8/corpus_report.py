"""
corpus_report.py — Phase C: Modern English Corpus Report.
=========================================================

Researches and documents suitable modern English corpus options for
Corpus C (20% weight in the planned training mix).

This is a RESEARCH-ONLY report. No datasets are downloaded. No external
material is copied. The report documents:

  * Dataset name
  * Source
  * License / rights status
  * Approximate size
  * Language
  * Content type
  * Suitability for a small CPU-trained model

Candidates are limited to legally reusable, openly licensed material with
clear provenance. Copyrighted material is not recommended without
appropriate authorization.

Corpus D (conversational) and Corpus E (teacher-generated) remain
optional — no fake data is generated to fill these categories.

This file is part of the from-scratch my-LLM project. We authored it.
"""

import json
import os
import sys


# --- project-relative paths ----------------------------------------------

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

MANIFEST_PATH = os.path.join(PROJECT_ROOT, "data", "manifests", "corpus_manifest.json")
REPORT_PATH = os.path.join(PROJECT_ROOT, "reports", "stage8_dataset_report.md")

# Note: stage8_dataset_report.md is written by dataset_splitter.py.
# This script writes a separate Phase C section or standalone report.


def fail(message):
    print("\nERROR: " + message, file=sys.stderr)
    sys.exit(1)


def rel(path):
    return os.path.relpath(path, PROJECT_ROOT)


# ---------------------------------------------------------------------------
# Corpus C candidate research
# ---------------------------------------------------------------------------

CORPUS_C_CANDIDATES = [
    {
        "name": "Project Gutenberg — Modern English fiction (pre-1929)",
        "source": "https://www.gutenberg.org/",
        "license": "Public domain (US) — works published before 1929, or "
                   "author died more than 70 years ago",
        "size_estimate": "100 KB – 2 MB per book",
        "language": "English (modern, early 20th century)",
        "content_type": "Novels, short stories, essays",
        "suitability": "High — clean OCR'd text, well-edited, natural dialogue "
                       "and narrative. Choose a single novel or short story "
                       "collection under 1 MB for CPU training.",
        "example_titles": [
            "The Secret Garden (1911) — Frances Hodgson Burnett",
            "The Velveteen Rabbit (1922) — Margery Williams",
            "The Tale of Peter Rabbit (1902) — Beatrix Potter",
        ],
    },
    {
        "name": "Standard Ebooks — Curated public domain ebooks",
        "source": "https://standardebooks.org/",
        "license": "Public domain (CC0 for the ebook files themselves; "
                   "underlying works are public domain)",
        "size_estimate": "50 KB – 1 MB per book",
        "language": "English (modern, late 19th–early 20th century)",
        "content_type": "Carefully formatted classic literature",
        "suitability": "High — professionally typeset, clean UTF-8 text, "
                       "consistent formatting. Good for small-model training.",
        "example_titles": [
            "Alice's Adventures in Wonderland — Lewis Carroll",
            "The Jungle Book — Rudyard Kipling",
        ],
    },
    {
        "name": "US Government publications (public domain)",
        "source": "https://www.govinfo.gov/",
        "license": "Public domain — US government works are not subject to "
                   "copyright (17 U.S.C. § 105)",
        "size_estimate": "Variable — 10 KB – several MB",
        "language": "English (modern, formal)",
        "content_type": "Reports, speeches, hearings, regulations",
        "suitability": "Medium — good for formal/expository English, but "
                       "less natural dialogue. Better as a supplement than "
                       "a primary modern English source.",
        "example_titles": [
            "White House press briefings",
            "Congressional hearings",
            "Federal agency reports",
        ],
    },
    {
        "name": "Project Gutenberg — Sherlock Holmes stories (1887–1927)",
        "source": "https://www.gutenberg.org/",
        "license": "Public domain (US)",
        "size_estimate": "~500 KB for the complete collection",
        "language": "English (modern, Victorian–Edwardian)",
        "content_type": "Detective fiction, dialogue-heavy",
        "suitability": "High — excellent dialogue examples, clear prose, "
                       "well-known public domain status.",
        "example_titles": [
            "The Adventures of Sherlock Holmes (1892)",
            "The Memoirs of Sherlock Holmes (1894)",
        ],
    },
]

CORPUS_D_NOTE = """
Corpus D — Original conversational examples (placeholder in manifest).

No conversational data has been created. This is intentional — the project
rules prohibit fabricating training data to inflate dataset size. Corpus D
should be populated only with hand-written, original conversational examples
created specifically for this project, such as:

  {"messages": [
    {"role": "user", "content": "Explain gravity."},
    {"role": "assistant", "content": "Gravity is the force that attracts objects toward each other."}
  ]}

Recommended size: 50–200 high-quality examples. Not auto-generated.
"""

CORPUS_E_NOTE = """
Corpus E — Optional teacher-generated examples (placeholder in manifest).

No teacher-generated data exists. This is optional distillation data that
would require a larger "teacher" model to generate. Not part of Stage 8.
"""


def load_manifest():
    if not os.path.isfile(MANIFEST_PATH):
        fail(f"Corpus manifest not found: {MANIFEST_PATH}")
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def write_report(manifest):
    """Write the modern English corpus recommendation report."""
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)

    lines = []
    lines.append("# Stage 8 Corpus C — Modern English Recommendations")
    lines.append("")
    lines.append("## Purpose")
    lines.append("")
    lines.append("Corpus C is planned as 20% of the training mix (per the corpus "
                 "manifest). It should provide modern, natural English that "
                 "complements the Stage 5 original documents (everyday prose) "
                 "and the Hamlet corpus (Shakespearean English).")
    lines.append("")
    lines.append("## Candidate Datasets (Research Only — No Downloads)")
    lines.append("")
    lines.append("No datasets were downloaded. All candidates below are "
                 "documented for future manual selection and download, "
                 "subject to the project's rules: no network calls at runtime, "
                 "no copyrighted material without authorization.")
    lines.append("")

    for i, cand in enumerate(CORPUS_C_CANDIDATES, 1):
        lines.append(f"### Candidate {i}: {cand['name']}")
        lines.append("")
        lines.append(f"- **Source:** {cand['source']}")
        lines.append(f"- **License:** {cand['license']}")
        lines.append(f"- **Estimated size:** {cand['size_estimate']}")
        lines.append(f"- **Language:** {cand['language']}")
        lines.append(f"- **Content type:** {cand['content_type']}")
        lines.append(f"- **Suitability:** {cand['suitability']}")
        lines.append("")
        lines.append("  Example titles:")
        for title in cand["example_titles"]:
            lines.append(f"  - {title}")
        lines.append("")

    lines.append("## Recommended Selection")
    lines.append("")
    lines.append("For a small CPU-trained model (~822K parameters), a single "
                 "public-domain novel or short story collection of "
                 "100–500 KB is appropriate. Recommended priority order:")
    lines.append("")
    lines.append("1. **Project Gutenberg — Sherlock Holmes stories** "
                 "(~500 KB, dialogue-rich, public domain)")
    lines.append("2. **Standard Ebooks — Alice's Adventures in Wonderland** "
                 "(clean, curated, public domain)")
    lines.append("3. **Project Gutenberg — The Secret Garden** "
                 "(1911, natural narrative, public domain)")
    lines.append("")
    lines.append("Download manually, save as `data/stage8/modern_english.txt`, "
                 "and update `data/manifests/corpus_manifest.json` to point "
                 "at it. Do not download programmatically at runtime.")
    lines.append("")

    lines.append("## Corpus D — Conversational Examples")
    lines.append("")
    lines.append(CORPUS_D_NOTE.strip())
    lines.append("")

    lines.append("## Corpus E — Teacher-Generated Examples")
    lines.append("")
    lines.append(CORPUS_E_NOTE.strip())
    lines.append("")

    lines.append("## Constraints")
    lines.append("")
    lines.append("- No downloads at runtime (project rule).")
    lines.append("- No copyrighted material without authorization.")
    lines.append("- All files must remain inside the project folder.")
    lines.append("- Corpus D and E remain optional placeholders.")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("*Generated by src/stage8/corpus_report.py*")

    with open(REPORT_PATH, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Report written to: {rel(REPORT_PATH)}")


def main():
    print("=" * 60)
    print("Stage 8 Phase C — Modern English Corpus Report")
    print("=" * 60)
    print()
    print("This is research only. No datasets are downloaded.")
    print()

    manifest = load_manifest()
    print(f"Loaded manifest: {rel(MANIFEST_PATH)}")

    # Show current corpus C status
    for entry in manifest.get("corpora", []):
        if entry.get("id") == "corpus-c-modern":
            print(f"\nCorpus C status: {entry.get('notes')}")
            print(f"  File: {entry.get('file')}")
            print(f"  Weight: {entry.get('weight', 0):.0%}")

    write_report(manifest)
    print("\nPhase C complete.")


if __name__ == "__main__":
    main()
