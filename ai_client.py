"""OpenRouter chat client with tool calling."""

import json
import logging
import re

from openai import (
    APIConnectionError,
    APIStatusError,
    AsyncOpenAI,
    AuthenticationError,
    BadRequestError,
    RateLimitError,
)

from config import (
    BASE_SYSTEM_PROMPT,
    CHANNEL_PROMPTS,
    MAX_TOKENS,
    OPENROUTER_API_KEY,
    OPENROUTER_MODEL,
)
from knowledge_loader import get_knowledge_stats, load_knowledge_bank
from memory import add_message, get_history
from search import get_crypto_price, web_search

log = logging.getLogger(__name__)

log.info("Loading knowledge bank...")
_knowledge_text = load_knowledge_bank()
_knowledge_stats = get_knowledge_stats()

log.info(
    "System prompt built | %s knowledge file(s) | %s total chars",
    _knowledge_stats["files"],
    f"{_knowledge_stats['total_chars']:,}",
)

_client = AsyncOpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1",
    default_headers={
        "HTTP-Referer": "https://discord.com",
        "X-Title": "Discord AI Bot",
    },
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search for real-time or current information. Use this for recent "
                "events, specific people, companies, projects, or anything likely "
                "to have changed."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "A focused search query, ideally 3-8 words.",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_crypto_price",
            "description": "Get current live price data for a cryptocurrency.",
            "parameters": {
                "type": "object",
                "properties": {
                    "coin": {
                        "type": "string",
                        "description": "Coin name or ticker, such as bitcoin, btc, eth, or sol.",
                    }
                },
                "required": ["coin"],
            },
        },
    },
]

_FUNC_TAG_RE = re.compile(
    r"<function=\w+[^>]*>.*?</function>"
    r"|<function>\w+</function>[^\n]*",
    re.DOTALL,
)

_KNOWLEDGE_KEYWORDS = {
    "community",
    "discord",
    "server",
    "rules",
    "channels",
    "bot",
    "project",
    "knowledge",
}


def _clean_reply(text: str) -> str:
    cleaned = _FUNC_TAG_RE.sub("", text).strip()
    return cleaned if cleaned else text


def _should_include_knowledge(question: str) -> bool:
    q = question.lower()
    return any(keyword in q for keyword in _KNOWLEDGE_KEYWORDS)


def _get_system_prompt(channel_name: str = "", question: str = "") -> str:
    base = CHANNEL_PROMPTS.get(channel_name, BASE_SYSTEM_PROMPT)
    if _knowledge_text and _should_include_knowledge(question):
        return (
            base
            + "\n\n"
            + "=" * 60
            + "\nKNOWLEDGE BANK - Use this for community-specific questions:\n"
            + "=" * 60
            + "\n\n"
            + _knowledge_text
        )
    return base


async def _execute_tool(tool_name: str, args: dict) -> str:
    log.info("Tool called: %s", tool_name)

    if tool_name == "web_search":
        query = args.get("query", "").strip()
        return await web_search(query) if query else "No search query provided."

    if tool_name == "get_crypto_price":
        coin = args.get("coin", "").strip()
        return await get_crypto_price(coin) if coin else "No coin name provided."

    return f"Unknown tool '{tool_name}'"


async def ask_ai(channel_id: int, user_message: str, channel_name: str = "") -> str:
    """Send a question to OpenRouter with optional tool calling."""
    add_message(channel_id, "user", user_message)
    messages = [
        {"role": "system", "content": _get_system_prompt(channel_name, user_message)},
        *get_history(channel_id),
    ]

    try:
        max_rounds = 3
        for round_num in range(max_rounds):
            is_last_round = round_num == max_rounds - 1
            request_kwargs = {
                "model": OPENROUTER_MODEL,
                "max_tokens": MAX_TOKENS,
                "messages": messages,
                "temperature": 0.7,
            }
            if not is_last_round:
                request_kwargs["tools"] = TOOLS
                request_kwargs["tool_choice"] = "auto"

            response = await _client.chat.completions.create(**request_kwargs)
            if not response.choices:
                log.warning("OpenRouter returned empty choices")
                break

            choice = response.choices[0]

            if choice.finish_reason == "stop":
                reply = _clean_reply(choice.message.content or "")
                add_message(channel_id, "assistant", reply)
                log.info("OpenRouter replied in round %s with %s chars", round_num + 1, len(reply))
                return reply

            if choice.finish_reason == "tool_calls":
                tool_calls = choice.message.tool_calls or []
                messages.append(
                    {
                        "role": "assistant",
                        "content": choice.message.content or "",
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments,
                                },
                            }
                            for tc in tool_calls
                        ],
                    }
                )

                for tc in tool_calls:
                    try:
                        args = json.loads(tc.function.arguments)
                    except json.JSONDecodeError:
                        args = {}

                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": await _execute_tool(tc.function.name, args),
                        }
                    )
                continue

            log.warning("Unexpected finish_reason: %s", choice.finish_reason)
            break

        reply = "I could not form a complete answer. Please try rephrasing your question."
        add_message(channel_id, "assistant", reply)
        return reply

    except BadRequestError as e:
        if "tool_use_failed" not in str(e):
            raise

        log.warning("Malformed tool call from model; retrying without tools")
        fallback = await _client.chat.completions.create(
            model=OPENROUTER_MODEL,
            max_tokens=MAX_TOKENS,
            messages=messages,
            temperature=0.7,
        )
        reply = _clean_reply(fallback.choices[0].message.content or "")
        add_message(channel_id, "assistant", reply)
        return reply
    except APIStatusError as e:
        if e.status_code == 413:
            return (
                "My knowledge base is too large for the current model. "
                "Please reduce MAX_KNOWLEDGE_CHARS or use a larger-context model."
            )
        raise
    except RateLimitError:
        return "The AI provider is rate limited right now. Please try again in a moment."
    except AuthenticationError:
        return "The AI provider API key is invalid. Please check the bot configuration."
    except APIConnectionError:
        return "I cannot reach the AI provider right now. Please try again shortly."


def get_system_info() -> dict:
    return {
        "model": OPENROUTER_MODEL,
        "knowledge_files": _knowledge_stats["files"],
        "knowledge_file_names": _knowledge_stats["names"],
        "total_knowledge_chars": _knowledge_stats["total_chars"],
    }
