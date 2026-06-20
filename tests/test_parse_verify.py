from agent.parsing import parse_verify_result


def test_parses_clean_json():
    r = parse_verify_result('{"ok": true, "issue": ""}')
    assert r.ok is True
    assert r.issue == ""


def test_parses_fenced_json():
    r = parse_verify_result('```json\n{"ok": false, "issue": "no rows"}\n```')
    assert r.ok is False
    assert r.issue == "no rows"


def test_parses_json_embedded_in_prose():
    text = 'The result looks wrong. {"ok": false, "issue": "wrong columns"} Done.'
    r = parse_verify_result(text)
    assert r.ok is False
    assert r.issue == "wrong columns"


def test_strips_think_block_and_uses_final_json():
    text = '<think>{"ok": true, "issue": ""} hmm</think>\n{"ok": false, "issue": "empty"}'
    r = parse_verify_result(text)
    assert r.ok is False
    assert r.issue == "empty"


def test_defaults_to_not_ok_on_garbage():
    r = parse_verify_result("totally not json")
    assert r.ok is False
