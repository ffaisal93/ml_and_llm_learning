from copy import deepcopy


def simulate(initial_state, events, transition):
    """Input events are ordered dicts with unique string id; rules are injected.
    transition(state,event) -> (new_state,output). A duplicate id/payload reuses
    the original output without applying the transition twice.
    """
    state = deepcopy(initial_state)
    seen = {}
    outputs = []
    for index, event in enumerate(events):
        if not isinstance(event, dict) or not isinstance(event.get("id"), str):
            raise ValueError(f"event {index}: string id required")
        identity = event["id"]
        if identity in seen:
            previous, result = seen[identity]
            if event != previous:
                raise ValueError("conflicting duplicate event id")
            outputs.append(
                {"id": identity, "duplicate": True, "result": deepcopy(result)}
            )
            continue
        # Isolated candidate prevents a failing rule from partially mutating state.
        candidate, result = transition(deepcopy(state), deepcopy(event))
        state = candidate
        seen[identity] = (deepcopy(event), deepcopy(result))
        outputs.append({"id": identity, "duplicate": False, "result": deepcopy(result)})
    return {"state": state, "outputs": outputs}


def inventory_rule(state, event):
    # Chosen demonstration only; original domain rules are not supplied.
    item = event["item"]
    quantity = event["quantity"]
    if type(quantity) is not int or quantity <= 0:
        raise ValueError("positive integer quantity required")
    if event["type"] == "receive":
        state[item] = state.get(item, 0) + quantity
    elif event["type"] == "ship":
        if state.get(item, 0) < quantity:
            return state, {"accepted": False, "reason": "insufficient"}
        state[item] -= quantity
    else:
        raise ValueError("unknown event type")
    return state, {"accepted": True, "remaining": state[item]}


if __name__ == "__main__":
    start = {"book": 2}
    events = [
        {"id": "a", "type": "receive", "item": "book", "quantity": 3},
        {"id": "b", "type": "ship", "item": "book", "quantity": 4},
        {"id": "c", "type": "ship", "item": "book", "quantity": 2},
    ]
    result = simulate(start, events + [events[1]], inventory_rule)
    assert (
        result["state"] == {"book": 1}
        and result["outputs"][2]["result"]["accepted"] is False
    )
    assert result["outputs"][3]["duplicate"] and start == {"book": 2}
    assert simulate(start, [], inventory_rule) == {"state": start, "outputs": []}
    try:
        simulate(start, [events[0], dict(events[0], quantity=8)], inventory_rule)
    except ValueError:
        pass
    else:
        raise AssertionError("conflicting id reuse accepted")
    print(
        "W369: final state book=1; rejected shipment, duplicate replay, empty and conflict checks passed"
    )
