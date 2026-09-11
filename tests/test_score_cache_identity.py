"""Regression tests for score-cache identity: an entry may only be reused for
the content it was produced from.

The scorer is stubbed, so these exercise `score_directed_pairs`' cache
bookkeeping (which entry is reused, which is re-scored) without any API."""

import json

import pytest

from llm_verifier import fine_grained_reward as fgr

CRITERION = {"id": "correctness", "name": "Correctness",
             "description": "Is it right?"}
NOTE = "the note"


def _tasks(problem, traces):
    return {"task": [{"problem": problem, "trace": t, "reward": 0,
                      "images": None} for t in traces]}


@pytest.fixture
def scorer(monkeypatch):
    """Stub verifier: records every comparison it is asked to score and
    returns a score derived from the traces, so a replayed entry is
    distinguishable from a freshly scored one."""
    calls = []

    def fake_score_pair_criterion(client, problem, trace_a, trace_b, crit,
                                  note, model, images=None):
        calls.append({"problem": problem, "trace_a": trace_a,
                      "trace_b": trace_b, "crit": crit["id"], "model": model})
        return float(len(trace_a)), float(len(trace_b))

    monkeypatch.setattr(fgr, "score_pair_criterion", fake_score_pair_criterion)
    return calls


def _score(tmp_path, tasks, scorer_model="model-x", note=NOTE,
           criterion=CRITERION, cache_name="cache.json"):
    lazy = fgr.LazyClient()
    lazy._client = object()  # the stubbed scorer never uses it
    return fgr.score_directed_pairs(
        lazy, tasks, {"task": [(0, 1)]}, [criterion], note,
        n_reps=1, max_workers=1, cache_file=str(tmp_path / cache_name),
        model=scorer_model, progress=False)


def test_a_different_problem_is_not_served_from_the_cache(tmp_path, scorer):
    """The reported bug: two `select()` calls share the synthetic task name
    `"task"`, so one cache file replayed the first call's scores."""
    first = _score(tmp_path, _tasks("reverse a string", ["aa", "bbb"]))
    calls_after_first = len(scorer)
    assert calls_after_first == 1

    second = _score(tmp_path, _tasks("solve a quadratic", ["cccc", "ddddd"]))

    assert len(scorer) == calls_after_first + 1, (
        "different content must be scored, not replayed")
    assert scorer[-1]["problem"] == "solve a quadratic"
    key = fgr.cache_key("correctness", "task", 0, 1, 0)
    assert first[key] != second[key]
    assert second[key] == {"score_A": 4.0, "score_B": 5.0}


def test_identical_content_still_hits_the_cache(tmp_path, scorer):
    tasks = _tasks("reverse a string", ["aa", "bbb"])
    first = _score(tmp_path, tasks)
    assert len(scorer) == 1

    second = _score(tmp_path, _tasks("reverse a string", ["aa", "bbb"]))

    assert len(scorer) == 1, "identical content must not be re-scored"
    assert second == first


@pytest.mark.parametrize("changed", ["trace", "criterion_text", "note",
                                     "model"])
def test_every_fingerprinted_input_invalidates_a_stale_entry(tmp_path, scorer,
                                                             changed):
    tasks = _tasks("reverse a string", ["aa", "bbb"])
    _score(tmp_path, tasks)
    assert len(scorer) == 1

    kwargs = {}
    if changed == "trace":
        tasks = _tasks("reverse a string", ["aa", "bbbb"])
    elif changed == "criterion_text":
        kwargs["criterion"] = dict(CRITERION, description="Is it correct?")
    elif changed == "note":
        kwargs["note"] = "a different note"
    elif changed == "model":
        kwargs["scorer_model"] = "model-y"

    _score(tmp_path, tasks, **kwargs)

    assert len(scorer) == 2, f"changing the {changed} must re-score"


def test_stale_entries_are_kept_on_disk_beside_the_new_ones(tmp_path, scorer):
    cache = tmp_path / "cache.json"
    _score(tmp_path, _tasks("reverse a string", ["aa", "bbb"]))
    after_first = json.loads(cache.read_text())
    assert len(after_first) == 1

    _score(tmp_path, _tasks("solve a quadratic", ["cccc", "ddddd"]))
    after_second = json.loads(cache.read_text())

    assert len(after_second) == 2, "the earlier entry must not be overwritten"
    assert set(after_first) < set(after_second)
    for key in after_second:
        assert key.startswith("correctness|task|0,1|0|")


def test_legacy_entries_without_a_fingerprint_are_ignored(tmp_path, scorer):
    """A cache written before content binding cannot be shown to match, so it
    is re-scored rather than trusted."""
    cache = tmp_path / "cache.json"
    legacy_key = fgr.cache_key("correctness", "task", 0, 1, 0)
    cache.write_text(json.dumps({legacy_key: {"score_A": 9.0,
                                              "score_B": 9.0}}))

    results = _score(tmp_path, _tasks("reverse a string", ["aa", "bbb"]))

    assert len(scorer) == 1, "a legacy entry must not be replayed"
    assert results[legacy_key] == {"score_A": 2.0, "score_B": 3.0}
