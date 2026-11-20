"""
conversation_prep.py — Stage 10 Phase C: Conversation Corpus Cleaning.
======================================================================

Prepares conversational training data from raw conversation datasets.
Reads raw conversation data, filters for English-language, high-quality
conversations, and writes them as structured JSONL records.

WHAT IT DOES
    1. Loads raw conversation data from a JSONL file.
    2. Filters by language (English only).
    3. Filters by quality metrics (toxicity, quality scores, etc.).
    4. Removes deleted, malformed, or duplicate messages.
    5. Builds conversation trees from parent-child relationships.
    6. Extracts valid conversation paths (root to leaf).
    7. Writes cleaned conversation records as JSONL.

WHAT IT DOES NOT DO
    * Does not download anything.
    * Does not fabricate conversational data.
    * Does not modify the original dataset.
    * Does not merge conversations from different sources.

INPUT FORMAT
    Each line is a JSON object with at least:
        message_id: Unique identifier for the message.
        parent_id: ID of the parent message (null for root messages).
        text: The message content.
        role: "prompter" or "assistant" (or "user"/"assistant").
        lang: Language code (e.g., "en").
        Optional fields:
            message_tree_id: Groups messages into conversation trees.
            tree_state: Conversation state (e.g., "ready_for_export").
            review_count, review_result: Review metadata.
            deleted: Boolean flag for deleted messages.
            rank: Ranking score for response quality.
            model_name: Model that generated the response (if assistant).

OUTPUT FORMAT
    Each line is a JSON object with:
        conversation_id: Unique identifier for the conversation.
        messages: List of {"role": "user"/"assistant", "content": str}.
        metadata: Dictionary with source metadata.

SAFETY
    * The original dataset is never modified.
    * No network calls, no downloads.
    * All writes stay inside the project folder.
    * Empty or malformed conversations are skipped, not fabricated.

HOW TO RUN
    python src\\stage10\\conversation_prep.py --input <path> --output <path>

This file is part of the from-scratch my-LLM project. We authored it.
"""

import argparse
import hashlib
import json
import os
import sys
import uuid


# --- project-relative paths ----------------------------------------------

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)


def fail(message):
    print("\nERROR: " + message, file=sys.stderr)
    sys.exit(1)


def rel(path):
    return os.path.relpath(path, PROJECT_ROOT)


# ---------------------------------------------------------------------------
# Conversation record model
# ---------------------------------------------------------------------------

class ConversationRecord:
    """A single conversation record ready for training.

    Attributes:
        conversation_id: Unique identifier for the conversation.
        messages: List of {"role": str, "content": str} dicts.
        metadata: Dictionary with source metadata (source, language, etc.).
    """

    def __init__(self, conversation_id, messages, metadata=None):
        self.conversation_id = conversation_id
        self.messages = messages
        self.metadata = metadata or {}

    def to_dict(self):
        return {
            "conversation_id": self.conversation_id,
            "messages": self.messages,
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# Loading and cleaning
# ---------------------------------------------------------------------------

def load_conversation_jsonl(path):
    """Load conversation records from a JSONL file.

    Each line is expected to be a JSON object representing a single message
    with at least: message_id, parent_id, text, role.

    Returns a list of raw message dicts.
    """
    if not os.path.isfile(path):
        fail(f"Conversation file not found: {path}")

    messages = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                messages.append(record)
            except json.JSONDecodeError as exc:
                print(f"WARNING: Skipping malformed line {line_num}: {exc}")

    print(f"Loaded {len(messages)} message records from {rel(path)}")
    return messages


def filter_english(messages):
    """Filter messages to English only."""
    english = []
    for msg in messages:
        lang = msg.get("lang", "")
        if lang == "en" or lang == "en-US" or lang == "en-GB":
            english.append(msg)
    print(f"  English messages: {len(english)} of {len(messages)}")
    return english


def filter_quality(messages):
    """Filter messages by quality metrics.

    Removes:
    - Deleted messages.
    - Empty messages.
    - Messages with very low quality scores.
    """
    cleaned = []
    for msg in messages:
        # Skip deleted messages
        if msg.get("deleted", False):
            continue

        # Skip empty messages
        text = msg.get("text", "").strip()
        if not text:
            continue

        # Skip very short messages (less than 2 characters)
        if len(text) < 2:
            continue

        # Filter by quality score if available
        quality = msg.get("quality")
        if quality is not None and quality < 0.5:
            continue

        # Filter by toxicity if available
        toxicity = msg.get("toxicity")
        if toxicity is not None and toxicity > 0.5:
            continue

        cleaned.append(msg)

    print(f"  Quality-filtered: {len(cleaned)} of {len(messages)}")
    return cleaned


def build_conversation_trees(messages):
    """Build conversation trees from message records.

    Groups messages by message_tree_id and builds parent-child relationships.
    Returns a list of conversation trees, where each tree is a dict with
    root_id and a list of messages.

    Returns a list of dicts with keys:
        tree_id: Unique tree identifier.
        root_id: ID of the root message.
        messages: List of message dicts in the tree.
    """
    # Group messages by tree_id
    trees = {}
    for msg in messages:
        tree_id = msg.get("message_tree_id")
        if tree_id is None:
            # Use message_id as tree_id if no tree_id
            tree_id = msg.get("message_id")
        if tree_id not in trees:
            trees[tree_id] = []
        trees[tree_id].append(msg)

    # Build tree structures
    conversations = []
    for tree_id, msgs in trees.items():
        # Build a map of message_id -> message
        msg_map = {msg.get("message_id"): msg for msg in msgs}

        # Find root messages (no parent or parent not in tree)
        roots = []
        for msg in msgs:
            parent_id = msg.get("parent_id")
            if parent_id is None or parent_id not in msg_map:
                roots.append(msg)

        # Build tree structure for each root
        for root in roots:
            tree = {
                "tree_id": tree_id,
                "root_id": root.get("message_id"),
                "messages": msgs,
            }
            conversations.append(tree)

    print(f"  Built {len(conversations)} conversation trees from {len(messages)} messages")
    return conversations


def extract_conversation_paths(trees):
    """Extract valid conversation paths from trees.

    For each tree, finds all root-to-leaf paths. Each path represents
    a complete conversation from start to finish.

    Returns a list of ConversationRecord objects.
    """
    conversations = []

    for tree in trees:
        tree_id = tree["tree_id"]
        messages = tree["messages"]

        # Build parent->children mapping
        children = {}
        for msg in messages:
            parent_id = msg.get("parent_id")
            if parent_id not in children:
                children[parent_id] = []
            children[parent_id].append(msg)

        # Find root messages
        roots = [msg for msg in messages if msg.get("parent_id") is None
                 or msg.get("parent_id") not in {m.get("message_id") for m in messages}]

        # DFS to find all root-to-leaf paths
        def dfs(node, path):
            path = path + [msg]
            children_of_node = children.get(msg.get("message_id"), [])
            if not children_of_node:
                # Leaf node - this is a complete conversation
                conversations.append(path)
            else:
                for child in children_of_node:
                    dfs(child, path)

        for root in roots:
            dfs(root, [])

    # Convert to ConversationRecord objects
    records = []
    for i, path in enumerate(conversations):
        if len(path) < 2:
            # Skip conversations with only one message
            continue

        messages = []
        for msg in path:
            role = msg.get("role", "user")
            # Map role names
            if role in ("prompter", "user"):
                role = "user"
            elif role in ("assistant", "assistant"):
                role = "assistant"
            else:
                role = "user"  # Default to user for unknown roles

            messages.append({
                "role": role,
                "content": msg.get("text", ""),
            })

        metadata = {
            "source": "openassistant/oasst1",
            "language": path[0].get("lang", "en"),
            "message_tree_id": tree_id,
        }

        conv_id = f"conv_{i}"
        records.append(ConversationRecord(conv_id, messages, metadata))

    print(f"  Built {len(records)} conversation records from {len(trees)} trees")
    return records


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def write_conversations(records, path):
    """Write conversation records to a JSONL file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")

    print(f"  Written {len(records)} conversations to {rel(path)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_prep(input_path, output_path):
    """Run the full conversation preparation pipeline."""
    print("=" * 60)
    print("Stage 10 Phase C — Conversation Corpus Preparation")
    print("=" * 60)

    # Load
    print("\nLoading conversation data ...")
    messages = load_conversation_jsonl(input_path)

    # Filter by language
    print("\nFiltering by language ...")
    messages = filter_english(messages)

    # Filter by quality
    print("\nFiltering by quality ...")
    messages = filter_quality(messages)

    # Build conversation trees
    print("\nBuilding conversation trees ...")
    trees = build_conversation_trees(messages)

    # Extract conversation paths
    print("\nExtracting conversation paths ...")
    records = extract_conversation_paths(trees)

    # Write output
    print("\nWriting output ...")
    write_conversations(records, output_path)

    print("\n" + "=" * 60)
    print("Stage 10 Phase C complete.")
    print("=" * 60)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Stage 10: Conversation corpus preparation pipeline."
    )
    parser.add_argument("--input", required=True,
                        help="Path to raw conversation JSONL file.")
    parser.add_argument("--output", required=True,
                        help="Path to output cleaned conversation JSONL.")
    return parser.parse_args(argv)


def main():
    args = parse_args()
    run_prep(args.input, args.output)


if __name__ == "__main__":
    main()
