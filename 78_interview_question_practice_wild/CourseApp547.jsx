import React, { useEffect, useRef, useState } from "react";
export function CourseApp() {
  const csrf = useRef("");
  const detailRequest = useRef(0);
  const [user, setUser] = useState(null),
    [message, setMessage] = useState("");
  const [username, setUsername] = useState(""),
    [password, setPassword] = useState("");
  const [query, setQuery] = useState(""),
    [category, setCategory] = useState("");
  const [view, setView] = useState("courses"),
    [items, setItems] = useState([]);
  const [detail, setDetail] = useState(null),
    [revision, setRevision] = useState(0);
  async function api(path, method = "GET", body, signal) {
    const response = await fetch("/api" + path, {
      method,
      credentials: "same-origin",
      signal,
      headers: {
        "Content-Type": "application/json",
        ...(method === "GET" ? {} : { "X-CSRF-Token": csrf.current }),
      },
      ...(body === undefined ? {} : { body: JSON.stringify(body) }),
    });
    const data = response.status === 204 ? null : await response.json();
    if (!response.ok) throw new Error(data?.error || `HTTP ${response.status}`);
    return data;
  }
  useEffect(() => {
    let live = true;
    api("/csrf")
      .then((data) => {
        if (live) csrf.current = data.csrf;
      })
      .catch((e) => setMessage(e.message));
    return () => {
      live = false;
    };
  }, []);
  useEffect(() => {
    const abort = new AbortController();
    const timer = setTimeout(() => {
      const params = new URLSearchParams({ q: query, category });
      api(
        view === "favorites" ? "/favorites" : `/courses?${params}`,
        "GET",
        undefined,
        abort.signal,
      )
        .then((data) => {
          if (!abort.signal.aborted) {
            setItems(data.items);
            setMessage("");
          }
        })
        .catch((e) => {
          if (!abort.signal.aborted) setMessage(e.message);
        });
    }, 200);
    return () => {
      clearTimeout(timer);
      abort.abort();
    };
  }, [query, category, view, revision]);
  async function authenticate(register) {
    try {
      if (!csrf.current) csrf.current = (await api("/csrf")).csrf;
      if (register) await api("/register", "POST", { username, password });
      const data = await api("/login", "POST", { username, password });
      csrf.current = data.csrf;
      setUser(data.id);
      setPassword("");
      setMessage("Logged in");
      setRevision((r) => r + 1);
    } catch (error) {
      setMessage(error.message);
    }
  }
  async function showCourse(id) {
    const request = ++detailRequest.current;
    try {
      const data = await api(`/courses/${id}`);
      if (request === detailRequest.current) setDetail(data);
    } catch (error) {
      if (request === detailRequest.current) setMessage(error.message);
    }
  }
  async function favorite(id, remove = false) {
    try {
      await api(`/favorites/${id}`, remove ? "DELETE" : "PUT");
      setMessage(remove ? "Favorite removed" : "Favorited");
      setRevision((r) => r + 1);
    } catch (error) {
      setMessage(error.message);
    }
  }
  return (
    <main>
      <h1>Courses</h1>
      {!user ? (
        <form
          onSubmit={(e) => {
            e.preventDefault();
            authenticate(false);
          }}
        >
          <label>
            Username{" "}
            <input
              autoComplete="username"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
            />
          </label>
          <label>
            Password{" "}
            <input
              type="password"
              autoComplete="current-password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
            />
          </label>
          <button type="submit">Log in</button>
          <button type="button" onClick={() => authenticate(true)}>
            Register and log in
          </button>
        </form>
      ) : (
        <button
          onClick={async () => {
            try {
              await api("/logout", "POST");
              csrf.current = (await api("/csrf")).csrf;
              setUser(null);
              setView("courses");
            } catch (e) {
              setMessage(e.message);
            }
          }}
        >
          Log out
        </button>
      )}
      <p role="status">{message}</p>
      <nav>
        <button onClick={() => setView("courses")}>Browse</button>
        <button disabled={!user} onClick={() => setView("favorites")}>
          My favorites
        </button>
      </nav>
      {view === "courses" && (
        <section>
          <label>
            Search <input value={query} onChange={(e) => setQuery(e.target.value)} />
          </label>
          <label>
            Category{" "}
            <input
              value={category}
              onChange={(e) => setCategory(e.target.value)}
              placeholder="e.g. ML"
            />
          </label>
        </section>
      )}
      <ul>
        {items.map((item) => (
          <li key={item.id}>
            <button onClick={() => showCourse(item.id)}>{item.title}</button>{" "}
            {item.category}
            {user && (
              <button onClick={() => favorite(item.id, view === "favorites")}>
                {view === "favorites" ? "Unfavorite" : "Favorite"}
              </button>
            )}
          </li>
        ))}
      </ul>
      {detail && (
        <article>
          <h2>{detail.title}</h2>
          <p>{detail.introduction}</p>
          {detail.chapters.map((chapter) => (
            <details key={chapter.id}>
              <summary>{chapter.title}</summary>
              <p>{chapter.body}</p>
            </details>
          ))}
        </article>
      )}
    </main>
  );
}
