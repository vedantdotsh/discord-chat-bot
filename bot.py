"""Discord client entry point and command routing."""

import logging
import time

import discord

from ai_client import ask_ai, get_system_info
from config import (
    ALLOWED_CHANNEL_IDS,
    ALLOWED_CHANNELS,
    BOT_STATUS_MESSAGE,
    COMMAND_PREFIX,
    DISCORD_TOKEN,
    RATE_LIMIT_SECONDS,
)
from memory import clear_history, stats as memory_stats

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("bot.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

intents = discord.Intents.default()
intents.message_content = True

bot = discord.Client(intents=intents)

_cooldowns: dict[int, float] = {}
COMMANDS = {"!clear", "!reset", "!ping", "!status", "!knowledge", "!help"}


def is_on_cooldown(user_id: int) -> bool:
    now = time.monotonic()
    if now - _cooldowns.get(user_id, 0) < RATE_LIMIT_SECONDS:
        return True
    _cooldowns[user_id] = now
    return False


def is_channel_allowed(message: discord.Message) -> bool:
    if not ALLOWED_CHANNEL_IDS and not ALLOWED_CHANNELS:
        return True

    channel_id = getattr(message.channel, "id", None)
    channel_name = getattr(message.channel, "name", "")
    return channel_id in ALLOWED_CHANNEL_IDS or channel_name in ALLOWED_CHANNELS


def strip_bot_mention(content: str) -> str:
    if not bot.user:
        return content.strip()

    return (
        content.replace(f"<@{bot.user.id}>", "")
        .replace(f"<@!{bot.user.id}>", "")
        .strip()
    )


@bot.event
async def on_ready():
    info = get_system_info()
    log.info("Logged in as %s (ID: %s)", bot.user, bot.user.id)
    log.info("Model: %s via OpenRouter", info["model"])
    log.info(
        "Knowledge: %s file(s) | %s chars",
        info["knowledge_files"],
        f"{info['total_knowledge_chars']:,}",
    )
    await bot.change_presence(
        status=discord.Status.online,
        activity=discord.Activity(
            type=discord.ActivityType.listening,
            name=BOT_STATUS_MESSAGE,
        ),
    )
    log.info("Bot is ready")


@bot.event
async def on_message(message: discord.Message):
    if message.author.bot:
        return

    content = message.content.strip()
    q_lower = content.lower()
    channel_name = getattr(message.channel, "name", "")
    is_allowed = is_channel_allowed(message)
    is_mentioned = bool(bot.user and bot.user in message.mentions)
    has_prefix = content == COMMAND_PREFIX or content.startswith(COMMAND_PREFIX + " ")
    is_command = q_lower in COMMANDS

    if not is_allowed and not is_mentioned:
        return

    # Avoid interrupting normal chat. The bot only responds to explicit triggers.
    if not has_prefix and not is_mentioned and not is_command:
        return

    if has_prefix:
        question = content[len(COMMAND_PREFIX):].strip()
    elif is_mentioned:
        question = strip_bot_mention(content)
    else:
        question = content

    if not question:
        await message.reply(
            "Hi, I am the Discord AI assistant.\n"
            "I can answer community questions, search the web, and fetch live crypto prices.\n"
            f"Type `{COMMAND_PREFIX} <your question>` or just @mention me.\n"
            "Type `!help` to see all commands."
        )
        return

    q_lower = question.lower().strip()

    if q_lower in ("!clear", "!reset"):
        clear_history(message.channel.id)
        await message.reply("Conversation history cleared for this channel.")
        return

    if q_lower in ("!ping", "!status"):
        latency = round(bot.latency * 1000)
        mem = memory_stats()
        info = get_system_info()
        await message.reply(
            f"**Bot Status**\n"
            f"Latency: `{latency}ms`\n"
            f"AI model: `{info['model']}` via OpenRouter\n"
            f"Knowledge files: `{info['knowledge_files']}` (`{info['total_knowledge_chars']:,}` chars)\n"
            f"Web search: DuckDuckGo with Tavily fallback\n"
            f"Crypto prices: CoinGecko\n"
            f"Active channel memories: `{mem['channels_with_history']}`\n"
            f"Total messages remembered: `{mem['total_messages']}`"
        )
        return

    if q_lower == "!knowledge":
        info = get_system_info()
        if info["knowledge_files"] == 0:
            await message.reply("No knowledge files loaded. Add Markdown files to `knowledge/` and restart.")
        else:
            names = "\n".join(f"- `{name}`" for name in info["knowledge_file_names"])
            await message.reply(
                f"**Knowledge Bank** - `{info['knowledge_files']}` file(s) loaded:\n"
                f"{names}\n\n"
                f"Total context: `{info['total_knowledge_chars']:,}` characters"
            )
        return

    if q_lower == "!help":
        await message.reply(
            f"**Discord AI Bot - Commands**\n\n"
            f"`{COMMAND_PREFIX} <question>` - Ask the AI anything\n"
            f"`@{bot.user.name} <question>` - Mention me to ask\n"
            f"`!knowledge` - See which knowledge files are loaded\n"
            f"`!ping` - Check bot status\n"
            f"`!clear` - Reset conversation history in this channel\n"
            f"`!help` - Show this message\n\n"
            f"**What I can answer:**\n"
            f"- Community questions from the knowledge bank\n"
            f"- Live crypto prices such as BTC, ETH, and SOL\n"
            f"- Current web information for news, projects, and people\n"
            f"- General knowledge questions"
        )
        return

    if is_on_cooldown(message.author.id):
        await message.reply(
            f"Please wait `{RATE_LIMIT_SECONDS}s` between questions.",
            delete_after=5,
        )
        return

    log.info(
        "Prompt received in #%s from user_id=%s (%s chars)",
        channel_name,
        message.author.id,
        len(question),
    )

    async with message.channel.typing():
        try:
            reply = await ask_ai(message.channel.id, question, channel_name)
            chunks = [reply[i : i + 1990] for i in range(0, len(reply), 1990)]
            for i, chunk in enumerate(chunks):
                if i == 0:
                    await message.reply(chunk)
                else:
                    await message.channel.send(chunk)
        except Exception as e:
            log.error("Error handling message: %s", e, exc_info=True)
            await message.reply("Something went wrong. Please try again in a moment.")


@bot.event
async def on_error(event, *args, **kwargs):
    log.error("Unhandled Discord error in event: %s", event, exc_info=True)


if __name__ == "__main__":
    log.info("Starting Discord AI Bot...")
    bot.run(DISCORD_TOKEN, reconnect=True, log_handler=None)
