from evals.run_eval import summarize


def _result(correct_per_iter):
    return {
        "correct_per_iter": correct_per_iter,
        "final_correct": correct_per_iter[-1] if correct_per_iter else False,
    }


def test_per_iteration_pass_rate_with_carry_forward():
    # Q1 fixed on the revise; Q2 correct immediately (carries forward);
    # Q3 never fixed. Demonstrates the loop earning its keep: iter0 1/3 -> iter2 2/3.
    results = [
        _result([False, True]),
        _result([True]),
        _result([False, False, False]),
    ]
    s = summarize(results)
    assert s["overall_pass_rate"] == 2 / 3
    assert s["pass_rate_per_iteration"][0] == 1 / 3
    assert s["pass_rate_per_iteration"][1] == 2 / 3
    assert s["pass_rate_per_iteration"][2] == 2 / 3


def test_summarize_handles_empty_correct_per_iter():
    # agent errored on a question -> contributes 0 at every iteration, no crash
    s = summarize([_result([]), _result([True])])
    assert s["overall_pass_rate"] == 1 / 2
    assert s["pass_rate_per_iteration"][0] == 1 / 2


def test_summarize_reports_only_observed_iterations():
    # No question ran a 3rd iteration, so only iterations 0 and 1 are reported.
    # (Decoupled from the agent's MAX_ITERATIONS -- the count comes from the data.)
    s = summarize([_result([False, True]), _result([True])])
    assert len(s["pass_rate_per_iteration"]) == 2
    assert s["pass_rate_per_iteration"] == [1 / 2, 1.0]
