"""
OrbitGuild Discord AI Bot — bot.py
─────────────────────────────
Main entry point. Handles all Discord events and commands.

Commands available to users:
  !ask <question>   — Ask the AI anything
  @BotName <msg>    — Mention the bot to ask
  !ping             — Check bot status
  !knowledge        — List loaded knowledge files
  !clear            — Reset channel conversation history
  !help             — Show all commands
"""

import discord
import logging
import time
from config import (
    DISCORD_TOKEN, ALLOWED_CHANNELS,
    COMMAND_PREFIX, BOT_STATUS_MESSAGE, RATE_LIMIT_SECONDS
)
from groq_client import ask_groq, get_system_info
from memory import clear_history, stats as memory_stats

# ── Logging setup ──────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("bot.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

# ── Discord client setup ───────────────────────────────────────────────────
intents = discord.Intents.default()
intents.message_content = True   # Must be enabled in Discord Dev Portal
intents.members = True

bot = discord.Client(intents=intents)

# ── Per-user rate limiting ─────────────────────────────────────────────────
_cooldowns: dict[int, float] = {}

def is_on_cooldown(user_id: int) -> bool:
    now = time.monotonic()
    if now - _cooldowns.get(user_id, 0) < RATE_LIMIT_SECONDS:
        return True
    _cooldowns[user_id] = now
    return False


# ── Bot events ─────────────────────────────────────────────────────────────
@bot.event
async def on_ready():
    info = get_system_info()
    log.info(f"Logged in as {bot.user} (ID: {bot.user.id})")
    log.info(f"Model: {info['model']} via OpenRouter")
    log.info(f"Knowledge: {info['knowledge_files']} file(s) | {info['total_knowledge_chars']:,} chars")
    await bot.change_presence(
        status=discord.Status.online,
        activity=discord.Activity(
            type=discord.ActivityType.listening,
            name=BOT_STATUS_MESSAGE,
        ),
    )
    log.info("Bot is ready ✅")


@bot.event
async def on_message(message: discord.Message):
    # Ignore messages from bots
    if message.author.bot:
        return

    channel_name = getattr(message.channel, "name", "")
    is_allowed   = (not ALLOWED_CHANNELS) or (channel_name in ALLOWED_CHANNELS)
    is_mentioned = bot.user in message.mentions
    has_prefix   = message.content.startswith(COMMAND_PREFIX + " ")

    # Only respond in allowed channels or when mentioned
    if not is_allowed and not is_mentioned:
        return

    # ── Extract the question from the message ──────────────────────────
    content = message.content.strip()
    if has_prefix:
        question = content[len(COMMAND_PREFIX):].strip()
    elif is_mentioned:
        question = (
            content
            .replace(f"<@{bot.user.id}>", "")
            .replace(f"<@!{bot.user.id}>", "")
            .strip()
        )
    else:
        question = content

    # Empty message — show greeting
    if not question:
        await message.reply(
            f"👋 Hi! I'm the OrbitGuild AI assistant.\n"
            f"Ask me anything — OrbitGuild info, crypto prices, live news, or general questions!\n"
            f"Type `{COMMAND_PREFIX} <your question>` or just @mention me.\n"
            f"Type `!help` to see all commands."
        )
        return

    q_lower = question.lower().strip()

    # ── Built-in commands ──────────────────────────────────────────────
    if q_lower in ("!clear", "!reset"):
        clear_history(message.channel.id)
        await message.reply("✅ Conversation history cleared for this channel.")
        return

    if q_lower in ("!ping", "!status"):
        latency = round(bot.latency * 1000)
        mem     = memory_stats()
        info    = get_system_info()
        await message.reply(
            f"🏓 **Bot Status**\n"
            f"📡 Latency: `{latency}ms`\n"
            f"🤖 AI Model: `{info['model']}` via OpenRouter (Free)\n"
            f"📚 Knowledge files: `{info['knowledge_files']}` (`{info['total_knowledge_chars']:,}` chars)\n"
            f"🔍 Web Search: DuckDuckGo → Tavily fallback\n"
            f"💰 Crypto Prices: CoinGecko (live, free)\n"
            f"🧠 Active channel memories: `{mem['channels_with_history']}`\n"
            f"💬 Total messages remembered: `{mem['total_messages']}`"
        )
        return

    if q_lower == "!knowledge":
        info = get_system_info()
        if info["knowledge_files"] == 0:
            await message.reply("📚 No knowledge files loaded. Add .md files to the /knowledge/ folder and restart.")
        else:
            names = "\n".join(f"  • `{n}`" for n in info["knowledge_file_names"])
            await message.reply(
                f"📚 **Knowledge Bank** — `{info['knowledge_files']}` file(s) loaded:\n"
                f"{names}\n\n"
                f"Total context: `{info['total_knowledge_chars']:,}` characters"
            )
        return

    if q_lower == "!help":
        await message.reply(
            f"**🤖 OrbitGuild AI Bot — Commands**\n\n"
            f"`{COMMAND_PREFIX} <question>` — Ask the AI anything\n"
            f"`@{bot.user.name} <question>` — Mention me to ask\n"
            f"`!knowledge` — See which knowledge files are loaded\n"
            f"`!ping` — Check bot status and all systems\n"
            f"`!clear` — Reset conversation history in this channel\n"
            f"`!help` — Show this message\n\n"
            f"**What I can answer:**\n"
            f"📖 OrbitGuild project questions (from knowledge files)\n"
            f"💰 Live crypto prices — BTC, ETH, SOL, and more\n"
            f"🌐 Real-time web search — news, projects, people\n"
            f"🧠 General knowledge questions"
        )
        return

    # ── Rate limit check ───────────────────────────────────────────────
    if is_on_cooldown(message.author.id):
        await message.reply(
            f"⏳ Please wait `{RATE_LIMIT_SECONDS}s` between questions.",
            delete_after=5,
        )
        return

    # ── Send to Groq (with tool calling) ──────────────────────────────
    log.info(f"[#{channel_name}] {message.author}: {question[:80]}")

    async with message.channel.typing():
        try:
            reply  = await ask_groq(message.channel.id, question, channel_name)
            # Split into 2000-char chunks (Discord limit)
            chunks = [reply[i : i + 1990] for i in range(0, len(reply), 1990)]
            for i, chunk in enumerate(chunks):
                if i == 0:
                    await message.reply(chunk)
                else:
                    await message.channel.send(chunk)
        except Exception as e:
            log.error(f"Error handling message: {e}", exc_info=True)
            await message.reply("⚠️ Something went wrong. Please try again in a moment.")


@bot.event
async def on_error(event, *args, **kwargs):
    log.error(f"Unhandled Discord error in event: {event}", exc_info=True)


# ── Start the bot ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    log.info("Starting OrbitGuild Discord AI Bot...")
    bot.run(DISCORD_TOKEN, reconnect=True, log_handler=None)

