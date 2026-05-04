"""Environment-based settings for the Discord AI bot."""

import os

from dotenv import load_dotenv

load_dotenv()

DISCORD_TOKEN: str = os.getenv("DISCORD_TOKEN", "")
if not DISCORD_TOKEN:
    raise ValueError("DISCORD_TOKEN is missing from your .env file")


def _csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _csv_ints(value: str) -> set[int]:
    ids: set[int] = set()
    for raw in _csv(value):
        try:
            ids.add(int(raw))
        except ValueError as exc:
            raise ValueError(f"Invalid Discord channel ID in ALLOWED_CHANNEL_IDS: {raw}") from exc
    return ids


# Prefer channel IDs in real servers. Channel names are useful for local demos.
ALLOWED_CHANNEL_IDS: set[int] = _csv_ints(os.getenv("ALLOWED_CHANNEL_IDS", ""))
ALLOWED_CHANNELS: set[str] = set(_csv(os.getenv("ALLOWED_CHANNELS", "")))

COMMAND_PREFIX: str = os.getenv("COMMAND_PREFIX", "!ask")
BOT_STATUS_MESSAGE: str = os.getenv("BOT_STATUS_MESSAGE", "AI help | !ask")

OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY is missing from your .env file")

OPENROUTER_MODEL: str = os.getenv("OPENROUTER_MODEL", "deepseek/deepseek-v3-base:free")
MAX_TOKENS: int = int(os.getenv("MAX_TOKENS", "1024"))

TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "")
MAX_HISTORY: int = int(os.getenv("MAX_HISTORY", "10"))
RATE_LIMIT_SECONDS: int = int(os.getenv("RATE_LIMIT_SECONDS", "3"))

BASE_SYSTEM_PROMPT: str = os.getenv(
    "SYSTEM_PROMPT",
    (
        "You are a helpful AI assistant for a Discord community. "
        "Answer clearly, briefly, and safely. "
        "\n\nYou have access to the following tools:"
        "\n- web_search(query): Search the internet for real-time information."
        "\n- get_crypto_price(coin): Get the current live price of a cryptocurrency."
        "\n\nGuidelines:"
        "\n- Keep answers concise: 3-4 sentences or 5-6 bullets."
        "\n- Use the knowledge bank for community-specific questions."
        "\n- For crypto prices: use the get_crypto_price tool to fetch live data."
        "\n- For people, projects, companies, news, or recent events: use web_search."
        "\n- Never fabricate facts, names, dates, or statistics."
        "\n- Do not reveal secrets, tokens, private logs, or hidden configuration."
    ),
)

CHANNEL_PROMPTS: dict[str, str] = {
    # "dev-help": "You are a senior software engineer. Focus on coding questions.",
    # "trading": "Focus on crypto markets and trading topics.",
}
