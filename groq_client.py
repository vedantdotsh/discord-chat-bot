"""
groq_client.py — OpenRouter AI with Tool Calling
──────────────────────────────────────────────────
The AI decides on its own when to:
  • Call web_search()       → for any live/recent info, specific people/projects
  • Call get_crypto_price() → for live coin prices
"""

import json
import logging
import re
from openai import AsyncOpenAI, RateLimitError, AuthenticationError, APIConnectionError, BadRequestError, APIStatusError

from config import (
    OPENROUTER_API_KEY, GROQ_MODEL, MAX_TOKENS,
    BASE_SYSTEM_PROMPT, CHANNEL_PROMPTS
)
from memory import get_history, add_message
from knowledge_loader import load_knowledge_bank, get_knowledge_stats
from search import web_search, get_crypto_price

log = logging.getLogger(__name__)

# ── Load knowledge bank once at startup ───────────────────────────────────
log.info("Loading knowledge bank...")
_knowledge_text  = load_knowledge_bank()
_knowledge_stats = get_knowledge_stats()

log.info(
    f"System prompt built | {_knowledge_stats['files']} knowledge file(s) | "
    f"{_knowledge_stats['total_chars']:,} total chars"
)

# ── Groq async client ──────────────────────────────────────────────────────
_client = AsyncOpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1",
    default_headers={
        "HTTP-Referer": "https://discord.com",
        "X-Title": "OrbitGuild Discord Bot",
    },
)

# ── Tool definitions (given to Groq so it knows what tools exist) ─────────
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the internet for real-time, current information. "
                "ALWAYS use this tool when the question involves: "
                "(1) Any specific person, founder, team, or individual you are not 100% certain about. "
                "(2) Any blockchain project, protocol, game, or platform not covered in the knowledge bank. "
                "(3) Recent news, events, updates, or announcements. "
                "(4) Anything that may have changed since your training cutoff of December 2023. "
                "(5) Questions where being wrong would mislead the user. "
                "When in doubt — always search. It is much better to search and confirm than to guess."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "A specific, focused search query. "
                            "Good examples: 'indie game jam schedule', "
                            "'Discord moderation best practices', "
                            "'latest game developer conference'. "
                            "Use 3-8 words for best results."
                        )
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_crypto_price",
            "description": (
                "Get the current live price of any cryptocurrency. "
                "Use this whenever someone asks about the price, value, market cap, "
                "24h change, or trading data of any crypto token or coin. "
                "Supports BTC, ETH, SOL, APT, DOGE, and hundreds of others."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "coin": {
                        "type": "string",
                        "description": (
                            "The coin name or ticker symbol. "
                            "Examples: 'bitcoin', 'btc', 'ethereum', 'eth', "
                            "'solana', 'sol', 'dogecoin', 'doge'."
                        )
                    }
                },
                "required": ["coin"]
            }
        }
    }
]

# Regex to strip malformed tool-call markup the model sometimes leaks into text
_FUNC_TAG_RE = re.compile(
    r"<function=\w+[^>]*>.*?</function>"
    r"|<function>\w+</function>[^\n]*",
    re.DOTALL,
)


def _clean_reply(text: str) -> str:
    """Remove any raw <function>…</function> tags the model leaked into its reply."""
    cleaned = _FUNC_TAG_RE.sub("", text).strip()
    return cleaned if cleaned else text


_ORB_KEYWORDS = {
    "orbitguild", "guild", "questline", "signal drop", "orbit credits",
    "pulsescore", "event", "playtest", "moderation", "community",
    "indie studio", "creator", "discord server",
}

def _is_orbitguild_question(question: str) -> bool:
    q = question.lower()
    return any(kw in q for kw in _ORB_KEYWORDS)


def _get_system_prompt(channel_name: str = "", question: str = "") -> str:
    base = CHANNEL_PROMPTS.get(channel_name, BASE_SYSTEM_PROMPT)
    if _knowledge_text and _is_orbitguild_question(question):
        return (
            base
            + "\n\n" + "=" * 60
            + "\nKNOWLEDGE BANK — Use this to answer OrbitGuild-specific questions:\n"
            + "=" * 60
            + "\n\n" + _knowledge_text
        )
    return base


async def _execute_tool(tool_name: str, args: dict) -> str:
    """Run the tool the AI requested and return its result."""
    log.info(f"🔧 Tool called: {tool_name}({args})")

    if tool_name == "web_search":
        query = args.get("query", "").strip()
        return await web_search(query) if query else "No search query provided."

    elif tool_name == "get_crypto_price":
        coin = args.get("coin", "").strip()
        return await get_crypto_price(coin) if coin else "No coin name provided."

    return f"Unknown tool '{tool_name}'"


async def ask_groq(
    channel_id: int,
    user_message: str,
    channel_name: str = ""
) -> str:
    """
    Send a question to Groq with tool calling enabled.

    The AI will:
    - Answer directly if it's confident (general facts, knowledge bank content)
    - Call web_search for anything about specific people/projects/recent events
    - Call get_crypto_price when user asks for coin prices

    Supports up to 3 tool-call rounds per message (for complex questions
    that need multiple searches).
    """
    add_message(channel_id, "user", user_message)
    system_prompt = _get_system_prompt(channel_name, user_message)

    # Build full conversation to send to Groq
    messages = [
        {"role": "system", "content": system_prompt},
        *get_history(channel_id),
    ]

    try:
        max_rounds = 3
        for round_num in range(max_rounds):  # Allow up to 3 rounds of tool use

            # On the last round, don't offer tools so the model MUST give a final answer
            is_last_round = (round_num == max_rounds - 1)

            request_kwargs = dict(
                model=GROQ_MODEL,
                max_tokens=MAX_TOKENS,
                messages=messages,
                temperature=0.7,
            )
            if not is_last_round:
                request_kwargs["tools"] = TOOLS
                request_kwargs["tool_choice"] = "auto"

            response = await _client.chat.completions.create(**request_kwargs)

            if not response.choices:
                log.warning("OpenRouter returned empty choices — retrying")
                break

            choice        = response.choices[0]
            finish_reason = choice.finish_reason

            # ── Groq gave a final answer ───────────────────────────────
            if finish_reason == "stop":
                reply = _clean_reply(choice.message.content or "")
                add_message(channel_id, "assistant", reply)
                log.info(f"OpenRouter replied (round {round_num + 1}) — {len(reply)} chars")
                return reply

            # ── Groq wants to call a tool ──────────────────────────────
            elif finish_reason == "tool_calls":
                tool_calls = choice.message.tool_calls or []

                # Record Groq's tool call request in the conversation
                messages.append({
                    "role":    "assistant",
                    "content": choice.message.content or "",
                    "tool_calls": [
                        {
                            "id":       tc.id,
                            "type":     "function",
                            "function": {
                                "name":      tc.function.name,
                                "arguments": tc.function.arguments,
                            }
                        }
                        for tc in tool_calls
                    ]
                })

                # Execute each tool and add results back to conversation
                for tc in tool_calls:
                    try:
                        args = json.loads(tc.function.arguments)
                    except json.JSONDecodeError:
                        args = {}

                    tool_result = await _execute_tool(tc.function.name, args)

                    messages.append({
                        "role":         "tool",
                        "tool_call_id": tc.id,
                        "content":      tool_result,
                    })

                    log.info(
                        f"  ↳ {tc.function.name} returned "
                        f"{len(tool_result)} chars"
                    )
                # Loop — Groq now reads the tool results and forms a reply
                continue

            else:
                log.warning(f"Unexpected finish_reason: {finish_reason}")
                break

        # If we hit max rounds without a final answer
        log.warning("Tool calling loop reached max rounds")
        reply = (
            "I searched for information but couldn't form a complete answer. "
            "Please try rephrasing your question."
        )
        add_message(channel_id, "assistant", reply)
        return reply

    except BadRequestError as e:
        # Model generated a malformed tool call (e.g. <function=web_search(...)>)
        # Try to parse the intended tool call and execute it ourselves
        if "tool_use_failed" not in str(e):
            raise  # Re-raise unrelated BadRequestErrors

        error_body = getattr(e, "body", {}) or {}
        if isinstance(error_body, dict):
            failed_gen = error_body.get("error", {}).get("failed_generation", "")
        else:
            failed_gen = str(e)

        log.warning(f"Malformed tool call, attempting manual parse: {failed_gen}")

        # Try to extract tool name + query from the many malformed formats:
        #   <function=web_search(query="...")></function>
        #   <function=web_search.query="..."></function>
        #   <function=web_search.query>"..."</function>
        #   <function=get_crypto_price>{"coin": "solana"}
        tool_name = None
        tool_args = {}

        # Pattern 1: <function=TOOL(key="value")>
        m = re.search(r'<function=(\w+)\((\w+)=["\']?(.+?)["\']?\)', failed_gen)
        if not m:
            # Pattern 2: <function=TOOL.key="value">  or  <function=TOOL.key>"value"
            m = re.search(r'<function=(\w+)\.(\w+)=["\']?(.+?)["\']?\s*>', failed_gen)
        if not m:
            # Pattern 3: <function=TOOL.key>"value"</function>
            m = re.search(r'<function=(\w+)\.(\w+)>["\']?(.+?)["\']?</function>', failed_gen)

        # Pattern 4: <function=TOOL>{"key": "value"} — JSON body after tag
        if not m:
            m4 = re.search(r'<function=(\w+)>\s*(\{.+?\})', failed_gen, re.DOTALL)
            if m4:
                tool_name = m4.group(1)
                try:
                    tool_args = json.loads(m4.group(2))
                except json.JSONDecodeError:
                    tool_args = {}

        if m and not tool_name:
            tool_name = m.group(1)
            tool_args = {m.group(2): m.group(3).strip()}

        if tool_name and tool_name in ("web_search", "get_crypto_price"):
            log.info(f"🔧 Recovered malformed tool call: {tool_name}({tool_args})")
            try:
                tool_result = await _execute_tool(tool_name, tool_args)
                log.info(f"  ↳ {tool_name} returned {len(tool_result)} chars")

                # Add the tool result as context and ask the model to answer
                messages.append({
                    "role": "user",
                    "content": (
                        f"Here are the {tool_name} results:\n\n{tool_result}\n\n"
                        "Now please answer the original question using this information."
                    ),
                })
                followup = await _client.chat.completions.create(
                    model=GROQ_MODEL,
                    max_tokens=MAX_TOKENS,
                    messages=messages,
                    temperature=0.7,
                )
                reply = _clean_reply(followup.choices[0].message.content or "")
                add_message(channel_id, "assistant", reply)
                return reply
            except Exception as inner_err:
                log.error(f"Recovered tool call also failed: {inner_err}", exc_info=True)

        # Final fallback: retry without tools
        log.info("Could not recover tool call, retrying without tools")
        try:
            fallback = await _client.chat.completions.create(
                model=GROQ_MODEL,
                max_tokens=MAX_TOKENS,
                messages=messages,
                temperature=0.7,
            )
            reply = _clean_reply(fallback.choices[0].message.content or "")
            add_message(channel_id, "assistant", reply)
            return reply
        except Exception as fallback_err:
            log.error(f"Fallback also failed: {fallback_err}", exc_info=True)
            return (
                "⚠️ I had trouble looking that up. "
                "Please try rephrasing your question."
            )
    except APIStatusError as e:
        if e.status_code == 413:
            log.error(f"Payload too large for model: {e}")
            return (
                "⚠️ My knowledge base is too large for the current model. "
                "Please ask a server admin to increase `MAX_KNOWLEDGE_CHARS` "
                "or switch to a model with a larger context window."
            )
        raise  # Re-raise other API status errors
    except RateLimitError:
        log.warning("OpenRouter rate limit reached")
        return (
            "⏳ I've hit the rate limit on the free AI tier. "
            "Please try again in a moment!"
        )
    except AuthenticationError:
        log.critical("Invalid OpenRouter API key!")
        return "⚠️ Invalid API key — please ask a server admin to check the bot configuration."
    except APIConnectionError:
        log.error("Cannot connect to OpenRouter API")
        return "⚠️ Can't reach the AI service right now. Please try again in a moment."
    except Exception as e:
        log.error(f"Unexpected OpenRouter error: {e}", exc_info=True)
        raise


def get_system_info() -> dict:
    """Return bot configuration info (used in !status command)."""
    return {
        "model":                 GROQ_MODEL,
        "knowledge_files":       _knowledge_stats["files"],
        "knowledge_file_names":  _knowledge_stats["names"],
        "total_knowledge_chars": _knowledge_stats["total_chars"],
    }

