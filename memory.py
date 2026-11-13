"""Memory of failed ideas, stored with mem0.

After every gate run, one lesson is saved for each BLOCKED candidate. A lesson
holds the candidate name and family, every failed check exactly as the gate
printed it, the numbers behind it, and the round number.

Before a round proposes anything, the lessons are read back and used to skip
candidates (and whole families) that already failed the same way.

Print what is in memory at any time:

    python memory.py

Configuration (see the block below):
- Everything runs on this machine. No API key is needed and none is read.
- Settings come from environment variables, each with a safe default.
"""

import json
import os
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent

# ============================================================================
# SETTINGS - read from environment variables. No API keys are used.
# ============================================================================
# Where mem0 keeps its files (vector store + history database).
MEMORY_DIR = Path(os.environ.get("TT_MEMORY_DIR", PROJECT_DIR / "memory_store"))

# Local embedding model, run on the CPU by FastEmbed. Turns a lesson's text into
# numbers so similar lessons can be found. Downloaded once from Hugging Face.
EMBED_MODEL = os.environ.get("TT_EMBED_MODEL", "BAAI/bge-small-en-v1.5")
EMBED_DIMS = int(os.environ.get("TT_EMBED_DIMS", "384"))

# Where the downloaded embedding model is cached (about 65 MB, not committed).
EMBED_CACHE_DIR = Path(os.environ.get("TT_EMBED_CACHE", PROJECT_DIR / "models"))

# mem0 groups memories under an owner id. All lessons use this one.
MEMORY_OWNER = "trading-terminal"
COLLECTION_NAME = "gate_lessons"

# These must be set BEFORE mem0 is imported, so they are set here at the top.
# setdefault keeps any value you already set yourself.
os.environ.setdefault("MEM0_TELEMETRY", "False")               # no usage data sent to mem0
os.environ.setdefault("MEM0_DIR", str(MEMORY_DIR / "mem0_home"))  # keep mem0's own files in the project
os.environ.setdefault("FASTEMBED_CACHE_PATH", str(EMBED_CACHE_DIR))
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")               # plain HTTPS download of the model
# ============================================================================


def build_mem0_config() -> dict:
    """The mem0 configuration this project uses.

    - embedder: FastEmbed, a small local model. No key, no network after download.
    - vector_store: Qdrant in local file mode. No server to run.
    - llm: mem0 insists on creating an LLM client at start-up even though we
      never use it (we store lessons as written, with infer=False). "lmstudio"
      is chosen because it starts without any key. It is never called.
    """
    return {
        "embedder": {
            "provider": "fastembed",
            "config": {"model": EMBED_MODEL, "embedding_dims": EMBED_DIMS},
        },
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "path": str(MEMORY_DIR / "qdrant"),
                "collection_name": COLLECTION_NAME,
                "embedding_model_dims": EMBED_DIMS,
                "on_disk": True,
            },
        },
        "llm": {"provider": "lmstudio", "config": {}},
        "history_db_path": str(MEMORY_DIR / "history.db"),
    }


def open_memory():
    """Open the mem0 store (creating it on first use)."""
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    EMBED_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    from mem0 import Memory   # imported here so the settings above are already in place
    return Memory.from_config(build_mem0_config())


# ----------------------------------------------------------------------------
# Writing lessons
# ----------------------------------------------------------------------------
def build_lesson_text(verdict: dict, round_number: int) -> str:
    """Write one lesson as a sentence a person can read.

    It always contains the failed checks and the numbers, never just "failed".
    """
    numbers = verdict["numbers"]
    params = ", ".join(f"{key}={value}" for key, value in verdict["params"].items())
    return (
        f"Round {round_number}: BLOCK {verdict['name']} "
        f"(family {verdict['family']}; params {params}). "
        f"Failed checks: {'; '.join(verdict['failures'])}. "
        f"Numbers: train return {numbers['train_return_pct']:+.2f}%, "
        f"validation return {numbers['validation_return_pct']:+.2f}%, "
        f"validation max drawdown {numbers['validation_drawdown_pct']:.2f}%, "
        f"validation trades {numbers['validation_trades']}."
    )


def build_lesson_metadata(verdict: dict, round_number: int,
                          family_proposed: int, family_blocked: int) -> dict:
    """The same facts as separate fields, so code can read them without parsing text."""
    numbers = verdict["numbers"]
    return {
        "kind": "gate_lesson",
        "round": round_number,
        "candidate": verdict["name"],
        "family": verdict["family"],
        "params": json.dumps(verdict["params"], sort_keys=True),
        "failed_checks": " | ".join(verdict["failures"]),
        "failure_codes": ",".join(verdict["failure_codes"]),
        "train_return_pct": round(numbers["train_return_pct"], 4),
        "validation_return_pct": round(numbers["validation_return_pct"], 4),
        "validation_drawdown_pct": round(numbers["validation_drawdown_pct"], 4),
        "validation_trades": numbers["validation_trades"],
        # How the whole family did in this round, for the family-level rule.
        "family_proposed_in_round": family_proposed,
        "family_blocked_in_round": family_blocked,
    }


def save_lessons(memory, verdicts: list, round_number: int) -> list:
    """Save one lesson per BLOCKED candidate. Returns the lesson texts saved."""
    # Count, per family, how many candidates were proposed and how many blocked.
    proposed, blocked = {}, {}
    for verdict in verdicts:
        family = verdict["family"]
        proposed[family] = proposed.get(family, 0) + 1
        if verdict["verdict"] == "BLOCK":
            blocked[family] = blocked.get(family, 0) + 1

    # Do not save the same candidate twice for the same round (safe to rerun).
    already = {(lesson["candidate"], lesson["round"]) for lesson in read_lessons(memory)}

    saved = []
    for verdict in verdicts:
        if verdict["verdict"] != "BLOCK":
            continue
        if (verdict["name"], round_number) in already:
            continue
        text = build_lesson_text(verdict, round_number)
        metadata = build_lesson_metadata(
            verdict, round_number, proposed[verdict["family"]], blocked[verdict["family"]]
        )
        # infer=False: store the text exactly as written. No LLM rewrites it.
        memory.add(text, user_id=MEMORY_OWNER, metadata=metadata, infer=False)
        saved.append(text)
    return saved


# ----------------------------------------------------------------------------
# Reading lessons
# ----------------------------------------------------------------------------
def read_lessons(memory) -> list:
    """Read every lesson back from mem0, oldest round first."""
    found = memory.get_all(filters={"user_id": MEMORY_OWNER}, top_k=10_000)
    lessons = []
    for item in found.get("results", []):
        metadata = item.get("metadata") or {}
        if metadata.get("kind") != "gate_lesson":
            continue
        lessons.append({
            "text": item["memory"],
            "round": metadata["round"],
            "candidate": metadata["candidate"],
            "family": metadata["family"],
            "failed_checks": metadata["failed_checks"],
            "failure_codes": [code for code in metadata["failure_codes"].split(",") if code],
            "family_proposed_in_round": metadata["family_proposed_in_round"],
            "family_blocked_in_round": metadata["family_blocked_in_round"],
        })
    lessons.sort(key=lambda lesson: (lesson["round"], lesson["candidate"]))
    return lessons


def find_similar_lessons(memory, description: str, limit: int = 5) -> list:
    """Find lessons whose text is closest in meaning to `description`.

    This is mem0's semantic search. It is for free-text ideas (a later step);
    the exact-match rules below do not depend on it.
    """
    found = memory.search(description, filters={"user_id": MEMORY_OWNER}, top_k=limit)
    return [{"text": item["memory"], "score": item.get("score")} for item in found.get("results", [])]


def print_memory(lessons: list, title: str = "Memory contents") -> None:
    """Print every lesson in full."""
    print(f"{title}: {len(lessons)} lesson(s)")
    if not lessons:
        print("  (empty)")
    for number, lesson in enumerate(lessons, start=1):
        print(f"  {number:>2}. {lesson['text']}")


# ----------------------------------------------------------------------------
# Using lessons: what must not be proposed again
# ----------------------------------------------------------------------------
def failed_candidates(lessons: list) -> dict:
    """Map each blocked candidate name to its most recent lesson."""
    latest = {}
    for lesson in lessons:   # lessons are sorted oldest first, so later ones win
        latest[lesson["candidate"]] = lesson
    return latest


def failed_families(lessons: list) -> dict:
    """Find families that failed as a whole, the same way.

    A family counts as failed when, in one round, EVERY candidate proposed from
    it was blocked AND all of them failed at least one check in common.
    Returns {family: {"round": n, "blocked": k, "shared_codes": [...]}}.
    """
    # Group the lessons by (family, round).
    groups = {}
    for lesson in lessons:
        groups.setdefault((lesson["family"], lesson["round"]), []).append(lesson)

    failed = {}
    for (family, round_number), group in sorted(groups.items(), key=lambda pair: pair[0][1]):
        proposed = group[0]["family_proposed_in_round"]
        blocked = group[0]["family_blocked_in_round"]
        if blocked < proposed or len(group) < proposed:
            continue   # at least one candidate passed (or lessons are missing)

        # Checks that every single lesson in the group failed.
        shared = set(group[0]["failure_codes"])
        for lesson in group[1:]:
            shared &= set(lesson["failure_codes"])
        if shared:
            failed[family] = {"round": round_number, "blocked": blocked,
                              "shared_codes": sorted(shared)}
    return failed


def filter_proposals(candidates: list, lessons: list) -> tuple:
    """Split proposed candidates into (allowed, skipped) using memory.

    A candidate is skipped when:
    - its whole family already failed the same way, or
    - this exact parameter set was already blocked.
    Each skipped entry carries the reason, quoting the lesson's failed checks.
    """
    family_failures = failed_families(lessons)
    candidate_failures = failed_candidates(lessons)

    allowed, skipped = [], []
    for candidate in candidates:
        family = candidate["family"]
        name = candidate["name"]
        if family in family_failures:
            info = family_failures[family]
            skipped.append({
                "candidate": candidate,
                "reason": (f"family {family} failed as a whole in round {info['round']}: "
                           f"all {info['blocked']} candidates blocked, every one on "
                           f"{', '.join(info['shared_codes'])}"),
            })
        elif name in candidate_failures:
            lesson = candidate_failures[name]
            skipped.append({
                "candidate": candidate,
                "reason": (f"same parameters blocked in round {lesson['round']}: "
                           f"{lesson['failed_checks']}"),
            })
        else:
            allowed.append(candidate)
    return allowed, skipped


def main() -> int:
    memory = open_memory()
    print(f"Store: {MEMORY_DIR}")
    print(f"Embedder: FastEmbed {EMBED_MODEL} (local). LLM: not used. API keys: none.")
    print()
    print_memory(read_lessons(memory))
    return 0


if __name__ == "__main__":
    sys.exit(main())
