"""W052: provider-independent Messages-style tool loop, using plain dictionaries."""

import json

SCHEMAS = [
    {
        "name": "get_ticker",
        "description": "Resolve a company name to its stock ticker.",
        "input_schema": {
            "type": "object",
            "properties": {"company_name": {"type": "string"}},
            "required": ["company_name"],
            "additionalProperties": False,
        },
    },
    {
        "name": "get_stock_price",
        "description": "Return historical price and currency on a trading date.",
        "input_schema": {
            "type": "object",
            "properties": {"ticker": {"type": "string"}, "date": {"type": "string"}},
            "required": ["ticker", "date"],
            "additionalProperties": False,
        },
    },
]


def answer(question, call_model, tools, max_turns=8):
    """call_model(messages, schemas) returns a list of content-block dictionaries.
    Convert SDK response blocks to dictionaries inside that supplied wrapper.
    """
    messages = [{"role": "user", "content": question}]
    seen_ids = set()
    for _ in range(max_turns):
        blocks = call_model(messages, SCHEMAS)
        messages.append({"role": "assistant", "content": blocks})
        requests = [b for b in blocks if b["type"] == "tool_use"]
        if not requests:
            text = "\n".join(b["text"] for b in blocks if b["type"] == "text")
            if not text:
                raise ValueError("Model returned neither tools nor answer")
            return text, messages
        results = []
        for request in requests:
            request_id = request["id"]
            if request_id in seen_ids:
                raise ValueError("Duplicate tool-use ID")
            seen_ids.add(request_id)
            try:
                name = request["name"]
                args = request["input"]
                required = {
                    "get_ticker": {"company_name"},
                    "get_stock_price": {"ticker", "date"},
                }
                if name not in required or set(args) != required[name]:
                    raise ValueError("Unknown tool or invalid arguments")
                if not all(isinstance(v, str) and v.strip() for v in args.values()):
                    raise ValueError("Arguments must be nonempty strings")
                value = tools[name](**args)
                result = {
                    "type": "tool_result",
                    "tool_use_id": request_id,
                    "content": json.dumps(value),
                }
            except (
                ValueError,
                KeyError,
                TypeError,
                TimeoutError,
                ConnectionError,
            ) as exc:
                result = {
                    "type": "tool_result",
                    "tool_use_id": request_id,
                    "content": str(exc),
                    "is_error": True,
                }
            results.append(result)
        # One user message returns a result for EVERY tool-use block.
        messages.append({"role": "user", "content": results})
    raise RuntimeError("Model-turn budget exhausted")


if __name__ == "__main__":

    def mock(messages, schemas):
        if len(messages) == 1:
            return [
                dict(
                    type="tool_use",
                    id="a",
                    name="get_ticker",
                    input={"company_name": "Apple"},
                ),
                dict(
                    type="tool_use",
                    id="b",
                    name="get_ticker",
                    input={"company_name": "Nvidia"},
                ),
            ]
        results = messages[-1]["content"]
        assert [r["tool_use_id"] for r in results] == ["a", "b"]
        return [dict(type="text", text="Resolved both companies.")]

    text, transcript = answer(
        "Resolve Apple and Nvidia",
        mock,
        {
            "get_ticker": lambda company_name: {"Apple": "AAPL", "Nvidia": "NVDA"}[
                company_name
            ]
        },
    )
    assert text == "Resolved both companies." and len(transcript) == 4
    print("W052 multiple-tool correlation and transcript checks passed")
