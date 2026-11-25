import React, { useMemo, useState } from "react";
import { matches } from "./filter_508.mjs";
export function FilterableList({ items }) {
  const [query, setQuery] = useState("");
  const [status, setStatus] = useState("all");
  const [type, setType] = useState("all");
  const visible = useMemo(
    () => items.filter((item) => matches(item, { query, status, type })),
    [items, query, status, type],
  );
  function reset() {
    setQuery("");
    setStatus("all");
    setType("all");
  }
  return (
    <section aria-label="Device list">
      <label>
        Search <input value={query} onChange={(e) => setQuery(e.target.value)} />
      </label>
      <label>
        Status{" "}
        <select value={status} onChange={(e) => setStatus(e.target.value)}>
          <option value="all">All</option>
          <option value="active">Active</option>
          <option value="inactive">Inactive</option>
        </select>
      </label>
      <label>
        Type{" "}
        <select value={type} onChange={(e) => setType(e.target.value)}>
          <option value="all">All</option>
          <option value="camera">Camera</option>
          <option value="sensor">Sensor</option>
        </select>
      </label>
      <button type="button" onClick={reset}>
        Reset
      </button>
      <p aria-live="polite">{visible.length} results</p>
      {visible.length ? (
        <ul>
          {visible.map((item) => (
            <li key={item.id}>
              {item.name} — {item.status} — {item.type}
            </li>
          ))}
        </ul>
      ) : (
        <p>No results</p>
      )}
    </section>
  );
}
