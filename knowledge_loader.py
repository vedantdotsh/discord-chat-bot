"""
knowledge_loader.py — Reads all .md files from the /knowledge folder
─────────────────────────────────────────────────────────────────────
HOW TO ADD NEW KNOWLEDGE:
  1. Create a new .md file inside the /knowledge/ folder
  2. Name it with a number prefix for ordering: 03_tokenomics.md
  3. Restart the bot — it loads automatically, no code changes needed

Files load alphabetically, so use number prefixes:
  01_orbitguild_project.md
  02_discord_server.md
  03_tokenomics.md
  04_faq.md  ...etc
"""

import logging
from pathlib import Path

log = logging.getLogger(__name__)

KNOWLEDGE_DIR = Path(__file__).parent / "knowledge"

# Max characters of knowledge to include in the system prompt.
# ~4 chars ≈ 1 token, so 4000 chars ≈ 1000 tokens — safe for small models.
import os as _os
MAX_KNOWLEDGE_CHARS = int(_os.getenv("MAX_KNOWLEDGE_CHARS", "4000"))


def load_knowledge_bank() -> str:
    """Load all .md files and return them as one combined string."""
    if not KNOWLEDGE_DIR.exists():
        log.warning(f"Knowledge directory not found: {KNOWLEDGE_DIR}")
        return ""

    md_files = sorted(KNOWLEDGE_DIR.glob("*.md"))
    if not md_files:
        log.warning("No .md files found in /knowledge/ folder.")
        return ""

    sections = []
    for filepath in md_files:
        try:
            content = filepath.read_text(encoding="utf-8").strip()
            if content:
                sections.append(content)
                log.info(f"  ✅ Loaded: {filepath.name}")
        except Exception as e:
            log.error(f"  ❌ Failed to load {filepath.name}: {e}")

    if not sections:
        return ""

    combined = "\n\n---\n\n".join(sections)

    # Truncate if over budget to avoid 413 Payload Too Large errors
    if len(combined) > MAX_KNOWLEDGE_CHARS:
        truncated = combined[:MAX_KNOWLEDGE_CHARS].rsplit("\n", 1)[0]
        log.warning(
            f"Knowledge bank truncated: {len(combined):,} → {len(truncated):,} chars "
            f"(limit {MAX_KNOWLEDGE_CHARS:,})"
        )
        combined = truncated + "\n\n[... truncated to fit model context limit ...]"

    log.info(f"Knowledge bank ready: {len(md_files)} file(s), {len(combined):,} characters")
    return combined


def get_knowledge_stats() -> dict:
    """Return info about loaded knowledge files."""
    if not KNOWLEDGE_DIR.exists():
        return {"files": 0, "names": [], "total_chars": 0}

    md_files = sorted(KNOWLEDGE_DIR.glob("*.md"))
    total    = 0
    names    = []
    for f in md_files:
        try:
            total += len(f.read_text(encoding="utf-8"))
            names.append(f.name)
        except Exception:
            pass

    return {"files": len(md_files), "names": names, "total_chars": total}

