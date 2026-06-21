import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import httpx

import config

# Shared sync client with SSL bypass (corporate proxy)
_http = httpx.Client(verify=False, timeout=20.0)


def web_search(query: str) -> str:
    """Search the web via Tavily and return the top 3 results."""
    try:
        resp = _http.post(
            "https://api.tavily.com/search",
            json={"api_key": config.TAVILY_API_KEY, "query": query, "max_results": 3},
        )
        resp.raise_for_status()
        results = resp.json().get("results", [])
        if not results:
            return "=== WEB SEARCH RESULTS ===\n\nNo results found for this query.\n\n=== END SEARCH RESULTS ==="

        lines = ["=== WEB SEARCH RESULTS ===\n"]
        for i, r in enumerate(results, 1):
            lines.append(f"[Result {i}]")
            lines.append(f"Title: {r.get('title', '')}")
            lines.append(f"URL: {r.get('url', '')}")
            lines.append(f"Content: {r.get('content', '')}")
            lines.append("")
        lines.append("=== END SEARCH RESULTS ===")
        return "\n".join(lines)
    except Exception as exc:
        return f"=== WEB SEARCH RESULTS ===\n\nError: {exc}\n\n=== END SEARCH RESULTS ==="
