from agent import prompts


def test_system_prompts_are_nonempty():
    assert prompts.GENERATE_SQL_SYSTEM.strip()
    assert prompts.VERIFY_SYSTEM.strip()
    assert prompts.REVISE_SYSTEM.strip()


def test_generate_sql_user_substitutes_schema_and_question():
    out = prompts.GENERATE_SQL_USER.format(schema="SCHEMA_X", question="QUESTION_Y")
    assert "SCHEMA_X" in out
    assert "QUESTION_Y" in out


def test_verify_user_substitutes_its_inputs():
    out = prompts.VERIFY_USER.format(question="Q1", sql="SELECT 1", result="OK: 1 rows.")
    assert "Q1" in out
    assert "SELECT 1" in out
    assert "OK: 1 rows." in out


def test_revise_user_substitutes_its_inputs():
    out = prompts.REVISE_USER.format(
        schema="SCH", question="Q2", sql="SELECT 2", result="ERROR: boom", issue="bad cols"
    )
    for token in ("SCH", "Q2", "SELECT 2", "ERROR: boom", "bad cols"):
        assert token in out
