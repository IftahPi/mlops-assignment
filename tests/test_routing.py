from agent.graph import AgentState, MAX_ITERATIONS, route_after_verify


def _state(verify_ok: bool, iteration: int) -> AgentState:
    return AgentState(question="q", db_id="db", verify_ok=verify_ok, iteration=iteration)


def test_routes_to_end_when_verify_ok():
    assert route_after_verify(_state(verify_ok=True, iteration=1)) == "end"


def test_routes_to_revise_when_not_ok_and_under_cap():
    assert route_after_verify(_state(verify_ok=False, iteration=1)) == "revise"


def test_routes_to_end_when_cap_reached():
    assert route_after_verify(_state(verify_ok=False, iteration=MAX_ITERATIONS)) == "end"
