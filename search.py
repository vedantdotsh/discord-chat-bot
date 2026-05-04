"""
search.py — Real-time data tools
──────────────────────────────────
Two tools called by Groq when it needs live information:

  Tool 1: get_crypto_price(coin)  — CoinGecko live prices (free, no key)
  Tool 2: web_search(query)       — DuckDuckGo → Tavily fallback (free)

These are called automatically by the AI (Groq tool calling).
The AI decides WHEN to call them based on the question.
"""

import asyncio
import logging
import aiohttp
from config import TAVILY_API_KEY

log = logging.getLogger(__name__)

# ── Coin symbol/name → CoinGecko ID ───────────────────────────────────────
COIN_ID_MAP: dict[str, str] = {
    "btc": "bitcoin",       "bitcoin": "bitcoin",
    "eth": "ethereum",      "ethereum": "ethereum",
    "bnb": "binancecoin",   "binance": "binancecoin",
    "sol": "solana",        "solana": "solana",
    "apt": "aptos",         "aptos": "aptos",
    "usdt": "tether",       "tether": "tether",
    "usdc": "usd-coin",
    "xrp": "ripple",        "ripple": "ripple",
    "doge": "dogecoin",     "dogecoin": "dogecoin",
    "ada": "cardano",       "cardano": "cardano",
    "matic": "matic-network", "polygon": "matic-network",
    "dot": "polkadot",      "polkadot": "polkadot",
    "avax": "avalanche-2",  "avalanche": "avalanche-2",
    "link": "chainlink",    "chainlink": "chainlink",
    "ltc": "litecoin",      "litecoin": "litecoin",
    "sui": "sui",
    "arb": "arbitrum",      "arbitrum": "arbitrum",
    "op": "optimism",       "optimism": "optimism",
    "atom": "cosmos",       "cosmos": "cosmos",
    "near": "near",
    "ftm": "fantom",        "fantom": "fantom",
    "sand": "the-sandbox",  "sandbox": "the-sandbox",
    "mana": "decentraland", "decentraland": "decentraland",
    "axs": "axie-infinity",  "axie": "axie-infinity",
}

COIN_DISPLAY_NAMES: dict[str, str] = {
    "bitcoin": "Bitcoin (BTC)",         "ethereum": "Ethereum (ETH)",
    "binancecoin": "BNB (BNB)",         "solana": "Solana (SOL)",
    "aptos": "Aptos (APT)",
    "tether": "Tether (USDT)",          "ripple": "XRP (XRP)",
    "dogecoin": "Dogecoin (DOGE)",      "cardano": "Cardano (ADA)",
    "matic-network": "Polygon (MATIC)", "avalanche-2": "Avalanche (AVAX)",
    "chainlink": "Chainlink (LINK)",    "polkadot": "Polkadot (DOT)",
    "litecoin": "Litecoin (LTC)",       "usd-coin": "USD Coin (USDC)",
    "sui": "Sui (SUI)",                 "arbitrum": "Arbitrum (ARB)",
    "optimism": "Optimism (OP)",        "cosmos": "Cosmos (ATOM)",
    "near": "NEAR Protocol (NEAR)",     "fantom": "Fantom (FTM)",
    "the-sandbox": "The Sandbox (SAND)","decentraland": "Decentraland (MANA)",
    "axie-infinity": "Axie Infinity (AXS)",
}


def _format_number(n: float) -> str:
    """Format large numbers with T/B/M suffixes."""
    if n >= 1_000_000_000_000: return f"${n/1_000_000_000_000:.2f}T"
    if n >= 1_000_000_000:     return f"${n/1_000_000_000:.2f}B"
    if n >= 1_000_000:         return f"${n/1_000_000:.2f}M"
    return f"${n:,.0f}"


# ── Auto-resolve coin via CoinGecko Search API ────────────────────────────
async def _resolve_coin_id(query: str) -> tuple[str, str] | None:
    """
    Search CoinGecko for a coin by name or symbol.
    Returns (coingecko_id, display_name) or None.
    """
    url = f"https://api.coingecko.com/api/v3/search?query={query}"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=8)) as resp:
                if resp.status != 200:
                    return None
                data = await resp.json()

        coins = data.get("coins", [])
        if not coins:
            return None

        # Prefer exact symbol match, then exact name match, then first result
        q = query.lower()
        for c in coins:
            if c.get("symbol", "").lower() == q:
                cid = c["id"]
                name = f"{c['name']} ({c['symbol'].upper()})"
                # Cache for next time
                COIN_ID_MAP[q] = cid
                COIN_DISPLAY_NAMES[cid] = name
                log.info(f"Auto-resolved '{query}' → {cid}")
                return cid, name

        for c in coins:
            if c.get("name", "").lower() == q:
                cid = c["id"]
                name = f"{c['name']} ({c['symbol'].upper()})"
                COIN_ID_MAP[q] = cid
                COIN_DISPLAY_NAMES[cid] = name
                log.info(f"Auto-resolved '{query}' → {cid}")
                return cid, name

        # Fallback: use the top result
        c = coins[0]
        cid = c["id"]
        name = f"{c['name']} ({c['symbol'].upper()})"
        COIN_ID_MAP[q] = cid
        COIN_DISPLAY_NAMES[cid] = name
        log.info(f"Auto-resolved '{query}' → {cid} (top result)")
        return cid, name

    except Exception as e:
        log.warning(f"CoinGecko search failed for '{query}': {e}")
        return None


# ── Tool 1: CoinGecko Live Crypto Price ───────────────────────────────────
async def get_crypto_price(coin: str) -> str:
    """
    Fetch live crypto price from CoinGecko.
    Called by Groq when user asks about a coin's current price.
    coin: ticker symbol (btc, eth) or full name (bitcoin, ethereum)
    """
    coin_lower = coin.lower().strip().lstrip("$")  # Strip $ prefix (AI often sends "$HYPE")
    coin_id    = COIN_ID_MAP.get(coin_lower)

    if not coin_id:
        # Try partial match in local cache
        for key, val in COIN_ID_MAP.items():
            if key in coin_lower or coin_lower in key:
                coin_id = val
                break

    # Auto-resolve via CoinGecko Search API if not in local map
    if not coin_id:
        resolved = await _resolve_coin_id(coin_lower)
        if resolved:
            coin_id = resolved[0]

    if not coin_id:
        return (
            f"I couldn't find '{coin}' on CoinGecko. "
            f"Please check the spelling or try a different name/ticker."
        )

    url = (
        f"https://api.coingecko.com/api/v3/simple/price"
        f"?ids={coin_id}&vs_currencies=usd"
        f"&include_24hr_change=true&include_market_cap=true&include_24hr_vol=true"
    )

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=8)) as resp:
                if resp.status != 200:
                    return f"CoinGecko returned an error (HTTP {resp.status}). Please try again shortly."
                data = await resp.json()

        if coin_id not in data:
            return f"No price data found for '{coin}'. It may not be listed on CoinGecko yet."

        d          = data[coin_id]
        price      = d.get("usd", 0)
        change_24h = d.get("usd_24h_change", 0)
        market_cap = d.get("usd_market_cap", 0)
        volume_24h = d.get("usd_24h_vol", 0)

        # Format price based on magnitude
        if price < 0.001:    price_str = f"${price:.8f}"
        elif price < 1:      price_str = f"${price:.6f}"
        elif price < 100:    price_str = f"${price:.4f}"
        else:                price_str = f"${price:,.2f}"

        change_str = f"+{change_24h:.2f}%" if change_24h >= 0 else f"{change_24h:.2f}%"
        name       = COIN_DISPLAY_NAMES.get(coin_id, coin_id.upper())

        result = (
            f"**{name}** — Live Price via CoinGecko\n"
            f"💰 Price: **{price_str}**\n"
            f"📊 24h Change: **{change_str}**\n"
            f"🏦 Market Cap: **{_format_number(market_cap)}**\n"
            f"📦 24h Volume: **{_format_number(volume_24h)}**"
        )
        log.info(f"CoinGecko: {name} = {price_str} ({change_str})")
        return result

    except asyncio.TimeoutError:
        return "CoinGecko request timed out. Please try again in a moment."
    except Exception as e:
        log.error(f"CoinGecko error: {e}")
        return f"Error fetching price for '{coin}': {str(e)}"


# ── Tool 2a: DuckDuckGo Web Search (Free, no API key) ─────────────────────
async def _search_duckduckgo(query: str, max_results: int = 5) -> str | None:
    """Search using DuckDuckGo — completely free, no key needed."""
    try:
        from duckduckgo_search import DDGS

        def _run():
            results = []
            with DDGS() as ddgs:
                for r in ddgs.text(query, max_results=max_results):
                    results.append({
                        "title": r.get("title", ""),
                        "body":  r.get("body", ""),
                        "href":  r.get("href", ""),
                    })
            return results

        results = await asyncio.to_thread(_run)
        if not results:
            return None

        formatted = []
        for r in results:
            if r["body"]:
                snippet = r["body"][:300]
                formatted.append(
                    f"[{r['title']}]\n{snippet}\nSource: {r['href']}"
                )

        if not formatted:
            return None

        log.info(f"DuckDuckGo: {len(results)} results for '{query[:50]}'")
        return "\n\n".join(formatted)

    except Exception as e:
        log.warning(f"DuckDuckGo failed: {type(e).__name__}: {e}")
        return None


# ── Tool 2b: Tavily Search (Reliable fallback) ─────────────────────────────
async def _search_tavily(query: str) -> str | None:
    """
    Tavily backup search — only used when DuckDuckGo fails.
    Requires TAVILY_API_KEY. Uses 1 of 1,000 free monthly credits.
    """
    if not TAVILY_API_KEY:
        log.debug("No TAVILY_API_KEY set — skipping Tavily fallback")
        return None

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://api.tavily.com/search",
                json={
                    "api_key":      TAVILY_API_KEY,
                    "query":        query,
                    "search_depth": "basic",
                    "max_results":  5,
                    "include_answer": True,
                },
                timeout=aiohttp.ClientTimeout(total=10)
            ) as resp:
                if resp.status != 200:
                    log.warning(f"Tavily returned HTTP {resp.status}")
                    return None
                data = await resp.json()

        # Use Tavily's pre-generated answer if available
        if data.get("answer"):
            log.info(f"Tavily answered: '{query[:50]}'")
            return data["answer"]

        results = data.get("results", [])
        if not results:
            return None

        formatted = []
        for r in results[:4]:
            snippet = r.get("content", "")[:300]
            title   = r.get("title", "")
            url     = r.get("url", "")
            if snippet:
                formatted.append(f"[{title}]\n{snippet}\nSource: {url}")

        log.info(f"Tavily: {len(results)} results for '{query[:50]}'")
        return "\n\n".join(formatted) if formatted else None

    except Exception as e:
        log.error(f"Tavily error: {e}")
        return None


# ── Main web search function (DuckDuckGo → Tavily fallback) ───────────────
async def web_search(query: str) -> str:
    """
    Search the web — tries DuckDuckGo first, falls back to Tavily.
    Called by Groq tool calling when the AI decides it needs live info.
    Always returns a string.
    """
    log.info(f"Web search: '{query[:70]}'")

    # Try DuckDuckGo first (completely free, no key needed)
    result = await _search_duckduckgo(query)
    if result:
        return result

    # Fall back to Tavily
    log.info("DuckDuckGo failed — trying Tavily backup...")
    result = await _search_tavily(query)
    if result:
        return result

    log.warning(f"Both search engines failed for: '{query[:60]}'")
    return (
        "I was unable to retrieve search results for this query. "
        "Please suggest the user check official sources or try rephrasing."
    )

