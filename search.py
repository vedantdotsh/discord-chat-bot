"""Real-time search and crypto price tools used by the AI client."""

import asyncio
import logging

import aiohttp

from config import TAVILY_API_KEY

log = logging.getLogger(__name__)

COIN_ID_MAP: dict[str, str] = {
    "btc": "bitcoin",
    "bitcoin": "bitcoin",
    "eth": "ethereum",
    "ethereum": "ethereum",
    "bnb": "binancecoin",
    "binance": "binancecoin",
    "sol": "solana",
    "solana": "solana",
    "apt": "aptos",
    "aptos": "aptos",
    "usdt": "tether",
    "tether": "tether",
    "usdc": "usd-coin",
    "xrp": "ripple",
    "ripple": "ripple",
    "doge": "dogecoin",
    "dogecoin": "dogecoin",
    "ada": "cardano",
    "cardano": "cardano",
    "matic": "matic-network",
    "polygon": "matic-network",
    "dot": "polkadot",
    "polkadot": "polkadot",
    "avax": "avalanche-2",
    "avalanche": "avalanche-2",
    "link": "chainlink",
    "chainlink": "chainlink",
    "ltc": "litecoin",
    "litecoin": "litecoin",
    "sui": "sui",
    "arb": "arbitrum",
    "arbitrum": "arbitrum",
    "op": "optimism",
    "optimism": "optimism",
    "atom": "cosmos",
    "cosmos": "cosmos",
    "near": "near",
}

COIN_DISPLAY_NAMES: dict[str, str] = {
    "bitcoin": "Bitcoin (BTC)",
    "ethereum": "Ethereum (ETH)",
    "binancecoin": "BNB (BNB)",
    "solana": "Solana (SOL)",
    "aptos": "Aptos (APT)",
    "tether": "Tether (USDT)",
    "ripple": "XRP (XRP)",
    "dogecoin": "Dogecoin (DOGE)",
    "cardano": "Cardano (ADA)",
    "matic-network": "Polygon (MATIC)",
    "avalanche-2": "Avalanche (AVAX)",
    "chainlink": "Chainlink (LINK)",
    "polkadot": "Polkadot (DOT)",
    "litecoin": "Litecoin (LTC)",
    "usd-coin": "USD Coin (USDC)",
    "sui": "Sui (SUI)",
    "arbitrum": "Arbitrum (ARB)",
    "optimism": "Optimism (OP)",
    "cosmos": "Cosmos (ATOM)",
    "near": "NEAR Protocol (NEAR)",
}


def _format_number(n: float) -> str:
    if n >= 1_000_000_000_000:
        return f"${n / 1_000_000_000_000:.2f}T"
    if n >= 1_000_000_000:
        return f"${n / 1_000_000_000:.2f}B"
    if n >= 1_000_000:
        return f"${n / 1_000_000:.2f}M"
    return f"${n:,.0f}"


async def _resolve_coin_id(query: str) -> tuple[str, str] | None:
    """Search CoinGecko for a coin by name or symbol."""
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

        q = query.lower()
        for coin in coins:
            if coin.get("symbol", "").lower() == q:
                return _cache_coin(q, coin)

        for coin in coins:
            if coin.get("name", "").lower() == q:
                return _cache_coin(q, coin)

        return _cache_coin(q, coins[0])
    except Exception as e:
        log.warning("CoinGecko search failed for '%s': %s", query, e)
        return None


def _cache_coin(query: str, coin: dict) -> tuple[str, str]:
    coin_id = coin["id"]
    name = f"{coin['name']} ({coin['symbol'].upper()})"
    COIN_ID_MAP[query] = coin_id
    COIN_DISPLAY_NAMES[coin_id] = name
    log.info("Auto-resolved '%s' to %s", query, coin_id)
    return coin_id, name


async def get_crypto_price(coin: str) -> str:
    """Fetch live crypto price from CoinGecko."""
    coin_lower = coin.lower().strip().lstrip("$")
    coin_id = COIN_ID_MAP.get(coin_lower)

    if not coin_id:
        for key, val in COIN_ID_MAP.items():
            if key in coin_lower or coin_lower in key:
                coin_id = val
                break

    if not coin_id:
        resolved = await _resolve_coin_id(coin_lower)
        if resolved:
            coin_id = resolved[0]

    if not coin_id:
        return f"I could not find '{coin}' on CoinGecko. Check the spelling or try another ticker."

    url = (
        "https://api.coingecko.com/api/v3/simple/price"
        f"?ids={coin_id}&vs_currencies=usd"
        "&include_24hr_change=true&include_market_cap=true&include_24hr_vol=true"
    )

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=8)) as resp:
                if resp.status != 200:
                    return f"CoinGecko returned HTTP {resp.status}. Please try again shortly."
                data = await resp.json()

        if coin_id not in data:
            return f"No price data found for '{coin}'. It may not be listed on CoinGecko yet."

        details = data[coin_id]
        price = details.get("usd", 0)
        change_24h = details.get("usd_24h_change", 0)
        market_cap = details.get("usd_market_cap", 0)
        volume_24h = details.get("usd_24h_vol", 0)

        if price < 0.001:
            price_str = f"${price:.8f}"
        elif price < 1:
            price_str = f"${price:.6f}"
        elif price < 100:
            price_str = f"${price:.4f}"
        else:
            price_str = f"${price:,.2f}"

        change_str = f"+{change_24h:.2f}%" if change_24h >= 0 else f"{change_24h:.2f}%"
        name = COIN_DISPLAY_NAMES.get(coin_id, coin_id.upper())

        log.info("CoinGecko: %s = %s (%s)", name, price_str, change_str)
        return (
            f"**{name}** - Live Price via CoinGecko\n"
            f"Price: **{price_str}**\n"
            f"24h Change: **{change_str}**\n"
            f"Market Cap: **{_format_number(market_cap)}**\n"
            f"24h Volume: **{_format_number(volume_24h)}**"
        )
    except asyncio.TimeoutError:
        return "CoinGecko request timed out. Please try again in a moment."
    except Exception as e:
        log.error("CoinGecko error: %s", e)
        return f"Error fetching price for '{coin}': {e}"


async def _search_duckduckgo(query: str, max_results: int = 5) -> str | None:
    """Search using DuckDuckGo."""
    try:
        from duckduckgo_search import DDGS

        def _run():
            results = []
            with DDGS() as ddgs:
                for result in ddgs.text(query, max_results=max_results):
                    results.append(
                        {
                            "title": result.get("title", ""),
                            "body": result.get("body", ""),
                            "href": result.get("href", ""),
                        }
                    )
            return results

        results = await asyncio.to_thread(_run)
        formatted = [
            f"[{result['title']}]\n{result['body'][:300]}\nSource: {result['href']}"
            for result in results
            if result["body"]
        ]
        if not formatted:
            return None

        log.info("DuckDuckGo: %s results for '%s'", len(results), query[:50])
        return "\n\n".join(formatted)
    except Exception as e:
        log.warning("DuckDuckGo failed: %s: %s", type(e).__name__, e)
        return None


async def _search_tavily(query: str) -> str | None:
    """Search with Tavily when an API key is available."""
    if not TAVILY_API_KEY:
        log.debug("No TAVILY_API_KEY set; skipping Tavily fallback")
        return None

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": TAVILY_API_KEY,
                    "query": query,
                    "search_depth": "basic",
                    "max_results": 5,
                    "include_answer": True,
                },
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                if resp.status != 200:
                    log.warning("Tavily returned HTTP %s", resp.status)
                    return None
                data = await resp.json()

        if data.get("answer"):
            log.info("Tavily answered '%s'", query[:50])
            return data["answer"]

        formatted = []
        for result in data.get("results", [])[:4]:
            snippet = result.get("content", "")[:300]
            if snippet:
                formatted.append(f"[{result.get('title', '')}]\n{snippet}\nSource: {result.get('url', '')}")

        return "\n\n".join(formatted) if formatted else None
    except Exception as e:
        log.error("Tavily error: %s", e)
        return None


async def web_search(query: str) -> str:
    """Search the web with DuckDuckGo first, then Tavily fallback."""
    log.info("Web search: '%s'", query[:70])

    result = await _search_duckduckgo(query)
    if result:
        return result

    log.info("DuckDuckGo failed; trying Tavily backup")
    result = await _search_tavily(query)
    if result:
        return result

    log.warning("Both search engines failed for '%s'", query[:60])
    return "I could not retrieve search results. Please check official sources or try rephrasing."
