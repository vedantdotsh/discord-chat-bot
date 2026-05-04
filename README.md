# Discord AI Bot

A Python Discord assistant that answers explicit user questions, uses a Markdown knowledge bank for community-specific context, searches the web for current information, and fetches live crypto prices.

This is portfolio work focused on practical bot engineering: command routing, provider integration, tool calling, local knowledge loading, rate limiting, and deployment hygiene.

## Highlights

- Responds only to `!ask`, direct mentions, and built-in commands, so it does not interrupt normal chat.
- Uses OpenRouter through the OpenAI Python SDK.
- Loads local Markdown knowledge from `knowledge/*.md`.
- Provides DuckDuckGo search with optional Tavily fallback.
- Fetches live crypto prices from CoinGecko.
- Keeps short per-channel conversation memory.
- Supports channel allowlists by Discord channel ID or channel name.

## Commands

- `!ask <question>` - Ask the AI assistant.
- `@BotName <question>` - Ask by mentioning the bot.
- `!ping` - Show latency, model, tools, and memory stats.
- `!knowledge` - List loaded knowledge files.
- `!clear` - Reset memory for the current channel.
- `!help` - Show available commands.

## Setup

1. Create and activate a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Copy the example environment file:

```bash
copy .env.example .env
```

4. Fill in the required values in `.env`.

5. Start the bot:

```bash
python bot.py
```

## Configuration

Required:

- `DISCORD_TOKEN` - Discord bot token.
- `OPENROUTER_API_KEY` - OpenRouter API key.

Optional:

- `ALLOWED_CHANNEL_IDS` - Comma-separated Discord channel IDs. Recommended for real servers.
- `ALLOWED_CHANNELS` - Comma-separated channel names for local demos.
- `COMMAND_PREFIX` - Defaults to `!ask`.
- `OPENROUTER_MODEL` - Defaults to `deepseek/deepseek-v3-base:free`.
- `TAVILY_API_KEY` - Optional fallback search provider.
- `MAX_HISTORY`, `MAX_TOKENS`, `MAX_KNOWLEDGE_CHARS`, `RATE_LIMIT_SECONDS`.

## Knowledge Bank

Add or update Markdown files in `knowledge/`. Files load alphabetically, so numbered names like `01_project.md` and `02_server_rules.md` keep context predictable.

## Repository Layout

- `bot.py` - Discord client and command handling
- `config.py` - environment-based settings
- `ai_client.py` - OpenRouter client and tool-calling loop
- `search.py` - web search and crypto price tools
- `memory.py` - per-channel chat history
- `knowledge_loader.py` - Markdown knowledge loader
- `knowledge/` - bot knowledge files
- `docs/` - reference and setup documents
- `scripts/` - deployment and operations helpers

## Security

Never commit `.env`, logs, caches, or local virtual environments. The `.gitignore` file is set up to keep secrets and generated files out of git.

For production Discord servers, prefer `ALLOWED_CHANNEL_IDS` over channel names and rotate any token that has ever been shared outside your private environment.

## Portfolio Notes

Good next upgrades would be slash commands, persistent memory, tests around Discord event routing, and a deployment guide for a small VPS or container host.
