import argparse, json, re, socket, urllib.request, urllib.error
from pathlib import Path


def load_samples(path):
    samples = []
    seen = set()
    for number, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if (
            not isinstance(row, dict)
            or not isinstance(row.get("id"), str)
            or not isinstance(row.get("prompt"), str)
        ):
            raise ValueError(f"line {number}: string id/prompt required")
        if row["id"] in seen:
            raise ValueError("duplicate sample id")
        if row.get("reference") is not None and not isinstance(row["reference"], str):
            raise ValueError("string reference required")
        seen.add(row["id"])
        samples.append(row)
    if not samples:
        raise ValueError("no evaluation samples")
    return samples


def ollama_generate(prompt, model, url="http://127.0.0.1:11434", timeout=30):
    payload = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0},
        }
    ).encode()
    request = urllib.request.Request(
        url.rstrip("/") + "/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = json.load(response)
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"Ollama HTTP {e.code}") from e
    except (urllib.error.URLError, TimeoutError, socket.timeout) as e:
        raise RuntimeError(f"Ollama unavailable/timeout: {type(e).__name__}") from e
    except (ValueError, UnicodeError) as e:
        raise RuntimeError("Ollama invalid JSON") from e
    if not isinstance(data, dict) or not isinstance(data.get("response"), str):
        raise RuntimeError("Ollama response field missing or nonstring")
    return data["response"]


def metrics(output, reference):
    normalize = lambda s: " ".join(s.casefold().split())
    exact = float(normalize(output) == normalize(reference))
    a = set(re.findall(r"\w+", normalize(output)))
    b = set(re.findall(r"\w+", normalize(reference)))
    overlap = len(a & b) / len(a | b) if a | b else 1.0
    return {"exact_match": exact, "jaccard": overlap}


def evaluate(samples, caller, model):
    rows = []
    for sample in samples:
        eligible = sample.get("reference") is not None
        row = {
            "id": sample["id"],
            "output": None,
            "metrics": None,
            "score": None,
            "pass": None,
            "error": None,
            "scorable": eligible,
        }
        try:
            row["output"] = caller(sample["prompt"])
            if not isinstance(row["output"], str):
                raise ValueError("caller did not return text")
            if eligible:
                row["metrics"] = metrics(row["output"], sample["reference"])
                row["score"] = sum(row["metrics"].values()) / 2
                row["pass"] = row["metrics"]["exact_match"] == 1
        except Exception as e:
            row["error"] = f"{type(e).__name__}: {e}"
            if eligible:
                row["metrics"] = {"exact_match": 0.0, "jaccard": 0.0}
                row["score"] = 0.0
                row["pass"] = False
        rows.append(row)
    scored = [r for r in rows if r["scorable"]]
    return {
        "model": model,
        "settings": {"temperature": 0, "stream": False},
        "sample_count": len(rows),
        "scored_count": len(scored),
        "unscored_count": len(rows) - len(scored),
        "error_count": sum(r["error"] is not None for r in rows),
        "average_score": (
            sum(r["score"] for r in scored) / len(scored) if scored else None
        ),
        "pass_rate": sum(r["pass"] for r in scored) / len(scored) if scored else None,
        "failed_ids": [
            r["id"] for r in rows if r["pass"] is False or r["error"] is not None
        ],
        "samples": rows,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("samples")
    p.add_argument("--model", required=True)
    p.add_argument("--url", default="http://127.0.0.1:11434")
    p.add_argument("--timeout", type=float, default=30)
    p.add_argument("--out")
    a = p.parse_args()
    report = evaluate(
        load_samples(a.samples),
        lambda text: ollama_generate(text, a.model, a.url, a.timeout),
        a.model,
    )
    encoded = json.dumps(report, indent=2, ensure_ascii=False)
    if a.out:
        Path(a.out).write_text(encoded)
    else:
        print(encoded)
