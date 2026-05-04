"""Load Markdown knowledge files used for community-specific answers."""

import logging
import os
from pathlib import Path

log = logging.getLogger(__name__)

KNOWLEDGE_DIR = Path(__file__).parent / "knowledge"
MAX_KNOWLEDGE_CHARS = int(os.getenv("MAX_KNOWLEDGE_CHARS", "4000"))


def load_knowledge_bank() -> str:
    """Load all Markdown files and return them as one combined string."""
    if not KNOWLEDGE_DIR.exists():
        log.warning("Knowledge directory not found: %s", KNOWLEDGE_DIR)
        return ""

    md_files = sorted(KNOWLEDGE_DIR.glob("*.md"))
    if not md_files:
        log.warning("No Markdown files found in %s", KNOWLEDGE_DIR)
        return ""

    sections = []
    for filepath in md_files:
        try:
            content = filepath.read_text(encoding="utf-8").strip()
            if content:
                sections.append(content)
                log.info("Loaded knowledge file: %s", filepath.name)
        except Exception as e:
            log.error("Failed to load %s: %s", filepath.name, e)

    if not sections:
        return ""

    combined = "\n\n---\n\n".join(sections)
    if len(combined) > MAX_KNOWLEDGE_CHARS:
        truncated = combined[:MAX_KNOWLEDGE_CHARS].rsplit("\n", 1)[0]
        log.warning(
            "Knowledge bank truncated: %s to %s chars (limit %s)",
            f"{len(combined):,}",
            f"{len(truncated):,}",
            f"{MAX_KNOWLEDGE_CHARS:,}",
        )
        combined = truncated + "\n\n[... truncated to fit model context limit ...]"

    log.info("Knowledge bank ready: %s file(s), %s characters", len(md_files), f"{len(combined):,}")
    return combined


def get_knowledge_stats() -> dict:
    """Return info about loaded knowledge files."""
    if not KNOWLEDGE_DIR.exists():
        return {"files": 0, "names": [], "total_chars": 0}

    md_files = sorted(KNOWLEDGE_DIR.glob("*.md"))
    total = 0
    names = []
    for filepath in md_files:
        try:
            total += len(filepath.read_text(encoding="utf-8"))
            names.append(filepath.name)
        except Exception:
            log.warning("Could not count knowledge file: %s", filepath.name)

    return {"files": len(md_files), "names": names, "total_chars": total}
