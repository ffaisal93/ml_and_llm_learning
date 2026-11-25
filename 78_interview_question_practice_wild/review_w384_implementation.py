from collections import deque
from urllib.parse import urlsplit, urlunsplit, urljoin, urldefrag


def crawl(start_url, fetch_page, parse_urls, max_attempts=3):
    if max_attempts < 1:
        raise ValueError("positive attempts required")

    def canonical(url):
        parts = urlsplit(urldefrag(url)[0])
        if (
            parts.scheme.lower() not in ("http", "https")
            or not parts.hostname
            or parts.username is not None
        ):
            return None
        host = parts.hostname.lower()
        netloc = ("[" + host + "]") if ":" in host else host
        if parts.port is not None:
            netloc += ":" + str(parts.port)
        return urlunsplit(
            (parts.scheme.lower(), netloc, parts.path or "/", parts.query, "")
        )

    start = canonical(start_url)
    if start is None:
        raise ValueError("valid HTTP(S) start required")
    hostname = urlsplit(start).hostname
    seen = {start}
    queue = deque([start])
    errors = {}
    while queue:
        url = queue.popleft()
        content = None
        last_error = None
        for _ in range(max_attempts):
            try:
                content = fetch_page(url)
                break
            except Exception as exc:
                last_error = exc
        if content is None:
            errors[url] = f"fetch_failed: {type(last_error).__name__}"
            continue
        links = None
        for _ in range(max_attempts):
            try:
                links = list(parse_urls(content))
                break
            except Exception as exc:
                last_error = exc
        if links is None:
            errors[url] = f"parse_failed: {type(last_error).__name__}"
            continue
        for link in links:
            if not isinstance(link, str):
                continue
            try:
                target = canonical(urljoin(url, link))
            except (ValueError, TypeError):
                continue
            if (
                target is not None
                and urlsplit(target).hostname == hostname
                and target not in seen
            ):
                seen.add(target)
                queue.append(target)
    return {"urls": sorted(seen), "errors": errors}


if __name__ == "__main__":
    graph = {
        "https://site.test/": [
            "/a",
            "https://evilsite.test/a",
            "/a#x",
            "/bad",
            "/parse",
        ],
        "https://site.test/a": ["/", "https://site.test/b"],
        "https://site.test/b": [],
    }
    calls = {}

    def fetch(url):
        calls[url] = calls.get(url, 0) + 1
        if url.endswith("/bad"):
            raise TimeoutError()
        if url.endswith("/parse"):
            return "unparseable"
        if url.endswith("/a") and calls[url] == 1:
            raise TimeoutError()
        return url

    def parse(content):
        if content == "unparseable":
            raise ValueError("bad document")
        return graph[content]

    result = crawl("https://SITE.test", fetch, parse)
    assert result["urls"] == [
        "https://site.test/",
        "https://site.test/a",
        "https://site.test/b",
        "https://site.test/bad",
        "https://site.test/parse",
    ]
    assert (
        len(result["errors"]) == 2
        and calls["https://site.test/a"] == 2
        and calls["https://site.test/"] == 1
    )
    assert not any("evil" in url for url in calls)
    print(
        "W384: same-host BFS, cycles/fragments, exact logical dedup and fetch/parse failures passed"
    )
