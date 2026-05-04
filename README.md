# Discord AI Bot

Discord AI assistant for a fictional OrbitGuild Labs community. The bot answers Discord questions, uses a local Markdown knowledge bank for OrbitGuild context, remembers short channel history, can search the web, and can fetch live crypto prices.

## Features

- Discord message handling with `!ask`, mentions, `!ping`, `!knowledge`, `!clear`, and `!help`
- OpenRouter chat completions through the OpenAI Python SDK
- Local knowledge bank loaded from `knowledge/*.md`
- DuckDuckGo search with Tavily fallback
- CoinGecko crypto price lookups
- Per-channel in-memory conversation history

## Setup

1. Create a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Copy the example environment file:

```bash
cp .env.example .env
```

4. Fill in the required values in `.env`.
5. Start the bot:

```bash
python bot.py
```

## Knowledge Bank

Add or update Markdown files in `knowledge/`. Files are loaded alphabetically, so numbered filenames like `01_orbitguild_project.md` and `02_discord_server.md` keep context in a predictable order.

## Repository Layout

- `bot.py` - Discord client and command handling
- `config.py` - environment-based settings
- `groq_client.py` - OpenRouter client and tool-calling loop
- `search.py` - web search and crypto price tools
- `memory.py` - per-channel chat history
- `knowledge_loader.py` - Markdown knowledge loader
- `knowledge/` - bot knowledge files
- `docs/` - reference and setup documents
- `scripts/` - deployment and operations helpers

## Security

Never commit `.env`, logs, caches, or local virtual environments. The `.gitignore` file is set up to keep secrets and generated files out of git.

