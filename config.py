"""
config.py — Loads all settings from the .env file
"""

from dotenv import load_dotenv
import os

load_dotenv()

# ── Discord ────────────────────────────────────────────────────────────────
DISCORD_TOKEN: str = os.getenv("DISCORD_TOKEN", "")
if not DISCORD_TOKEN:
    raise ValueError("❌ DISCORD_TOKEN is missing from your .env file")

# Comma-separated channel names the bot responds in.
# Leave ALLOWED_CHANNELS empty ("") to respond in ALL channels.
ALLOWED_CHANNELS: list[str] = [
    c.strip() for c in os.getenv("ALLOWED_CHANNELS", "").split(",") if c.strip()
]

COMMAND_PREFIX: str     = os.getenv("COMMAND_PREFIX", "!ask")
BOT_STATUS_MESSAGE: str = os.getenv("BOT_STATUS_MESSAGE", "OrbitGuild questions | !ask")

# ── OpenRouter (Free AI) ───────────────────────────────────────────────────
OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
if not OPENROUTER_API_KEY:
    raise ValueError("❌ OPENROUTER_API_KEY is missing from your .env file")

GROQ_MODEL: str = os.getenv("OPENROUTER_MODEL", "deepseek/deepseek-v3-base:free")
MAX_TOKENS: int = int(os.getenv("MAX_TOKENS", "1024"))

# ── Tavily (Backup web search — optional) ─────────────────────────────────
# Get free key at tavily.com — 1,000 free searches/month
# Leave empty if you don't have a key — DuckDuckGo handles search for free
TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "")

# ── Memory ─────────────────────────────────────────────────────────────────
# How many back-and-forth exchanges to remember per channel
MAX_HISTORY: int = int(os.getenv("MAX_HISTORY", "10"))

# ── Rate Limiting ──────────────────────────────────────────────────────────
# Seconds a user must wait between questions (prevents spam)
RATE_LIMIT_SECONDS: int = int(os.getenv("RATE_LIMIT_SECONDS", "3"))

# ── Bot Personality (System Prompt) ───────────────────────────────────────
BASE_SYSTEM_PROMPT: str = os.getenv(
    "SYSTEM_PROMPT",
    (
        "You are the AI assistant for the fictional OrbitGuild Labs Discord server. "
        "Your job is to help community members with any question they have. "
        "\n\nYou have access to the following tools:"
        "\n- web_search(query): Search the internet for real-time information about specific people, projects, recent events, or anything that may have changed since 2023."
        "\n- get_crypto_price(coin): Get the current live price of a cryptocurrency."
        "\n\nImportant: When using tools, do not output <function> tags in your response. The system will handle tool calls automatically."
        "\n\nGuidelines:"
        "\n- Keep answers SHORT and CONCISE — max 5-6 bullet points or 3-4 sentences. No long essays."
        "\n- Get straight to the point. Skip lengthy intros and outros."
        "\n- For OrbitGuild project questions: use the knowledge bank provided to you and mention that OrbitGuild is fictional when relevant."
        "\n- For crypto prices: use the get_crypto_price tool to fetch live data."
        "\n- For questions about specific people, projects, companies, or recent events: always use the web_search tool."
        "\n- Never fabricate facts, names, dates, or statistics."
        "\n- Use **bold** and bullet points for readability but keep it brief."
        "\n- End with one short follow-up line max, not multiple questions."
    )
)

# Optional: different personality per channel (channel name → prompt)
CHANNEL_PROMPTS: dict[str, str] = {
    # "dev-help": "You are a senior software engineer. Focus on coding questions.",
    # "trading":  "Focus on crypto markets and trading topics.",
}

