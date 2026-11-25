import React, { useEffect, useRef, useState } from "react";

// fetchPage({query,sort,direction,page,pageSize,signal}) -> {items,total,version}
export function LargeList({ fetchPage, dataVersion = 0 }) {
  const [text, setText] = useState("");
  const [query, setQuery] = useState("");
  const [sort, setSort] = useState("name");
  const [direction, setDirection] = useState("asc");
  const [page, setPage] = useState(0);
  const [result, setResult] = useState({ items: [], total: 0 });
  const [status, setStatus] = useState("loading");
  const [scrollTop, setScrollTop] = useState(0);
  const generation = useRef(0);
  const viewport = useRef(null);
  const pageSize = 1000,
    rowHeight = 40,
    height = 400,
    overscan = 4;
  useEffect(() => {
    const timer = setTimeout(() => {
      setQuery(text.trim());
      setPage(0);
    }, 250);
    return () => clearTimeout(timer);
  }, [text]);
  useEffect(() => {
    const current = ++generation.current;
    const abort = new AbortController();
    setStatus("loading");
    setScrollTop(0);
    if (viewport.current) viewport.current.scrollTop = 0;
    fetchPage({ query, sort, direction, page, pageSize, signal: abort.signal })
      .then((data) => {
        if (current === generation.current && !abort.signal.aborted) {
          setResult(data);
          setStatus("ready");
        }
      })
      .catch((error) => {
        if (current === generation.current && !abort.signal.aborted) setStatus("error");
      });
    return () => abort.abort();
  }, [query, sort, direction, page, fetchPage, dataVersion]);
  const start = Math.max(0, Math.floor(scrollTop / rowHeight) - overscan);
  const end = Math.min(
    result.items.length,
    Math.ceil((scrollTop + height) / rowHeight) + overscan,
  );
  return (
    <section>
      <label>
        Search <input value={text} onChange={(e) => setText(e.target.value)} />
      </label>
      <label>
        Sort{" "}
        <select
          value={sort}
          onChange={(e) => {
            setSort(e.target.value);
            setPage(0);
          }}
        >
          <option value="name">Name</option>
          <option value="updatedAt">Updated</option>
        </select>
      </label>
      <button
        onClick={() => {
          setDirection((d) => (d === "asc" ? "desc" : "asc"));
          setPage(0);
        }}
      >
        {direction === "asc" ? "Ascending" : "Descending"}
      </button>
      <button
        disabled={page === 0 || status === "loading"}
        onClick={() => setPage((p) => p - 1)}
      >
        Previous
      </button>
      <button
        disabled={(page + 1) * pageSize >= result.total || status === "loading"}
        onClick={() => setPage((p) => p + 1)}
      >
        Next
      </button>
      <p aria-live="polite">
        {status === "ready" ? `${result.total} results, page ${page + 1}` : status}
      </p>
      {status === "ready" && (
        <div
          ref={viewport}
          style={{ height, overflowY: "auto" }}
          onScroll={(e) => setScrollTop(e.currentTarget.scrollTop)}
        >
          <ul
            aria-label="Results"
            style={{
              height: result.items.length * rowHeight,
              position: "relative",
              margin: 0,
              padding: 0,
              listStyle: "none",
            }}
          >
            {result.items.slice(start, end).map((item, offset) => {
              const index = start + offset;
              return (
                <li
                  key={item.id}
                  aria-setsize={result.total}
                  aria-posinset={page * pageSize + index + 1}
                  style={{
                    position: "absolute",
                    top: index * rowHeight,
                    height: rowHeight,
                    left: 0,
                    right: 0,
                  }}
                >
                  {item.name}
                </li>
              );
            })}
          </ul>
        </div>
      )}
    </section>
  );
}
