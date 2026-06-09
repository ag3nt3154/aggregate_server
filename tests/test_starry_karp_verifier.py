from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.starry_karp import RequestResult, verify_phase1, verify_phase2, verify_phase3

# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_results(n: int, status: int = 200) -> list[RequestResult]:
    return [RequestResult(status_code=status, body={}) for _ in range(n)]


def make_stats(
    backend_id: str,
    hit_count: int,
    models: list[str],
) -> dict[str, object]:
    return {
        "backend_id": backend_id,
        "hit_count": hit_count,
        "requests": [{"model": m} for m in models],
    }


# ── Phase 1 ───────────────────────────────────────────────────────────────────

def test_phase1_all_pass() -> None:
    results = make_results(20, 200)
    stats = {
        "backend_1": make_stats("backend_1", 10, ["test-model"] * 10),
        "backend_2": make_stats("backend_2", 5, ["test-model"] * 5),
        "backend_3": make_stats("backend_3", 3, ["test-model"] * 3),
        "backend_4": make_stats("backend_4", 2, ["test-model"] * 2),
    }
    checks = verify_phase1(results, stats, elapsed=8.0)
    failures = [c for c in checks if not c.passed]
    assert failures == [], [c.message for c in failures]


def test_phase1_fails_when_not_200() -> None:
    results = make_results(19, 200) + [RequestResult(status_code=404, body={})]
    stats = {
        "backend_1": make_stats("backend_1", 10, ["test-model"] * 10),
        "backend_2": make_stats("backend_2", 5, ["test-model"] * 5),
        "backend_3": make_stats("backend_3", 3, ["test-model"] * 3),
        "backend_4": make_stats("backend_4", 2, ["test-model"] * 2),
    }
    checks = verify_phase1(results, stats, elapsed=8.0)
    assert any(not c.passed and "200" in c.message for c in checks)


def test_phase1_fails_when_backend_not_hit() -> None:
    results = make_results(20, 200)
    stats = {
        "backend_1": make_stats("backend_1", 15, ["test-model"] * 15),
        "backend_2": make_stats("backend_2", 5, ["test-model"] * 5),
        "backend_3": make_stats("backend_3", 0, []),
        "backend_4": make_stats("backend_4", 0, []),
    }
    checks = verify_phase1(results, stats, elapsed=8.0)
    assert any(not c.passed and "all" in c.message.lower() for c in checks)


def test_phase1_fails_when_b1_not_faster_than_b4() -> None:
    results = make_results(20, 200)
    stats = {
        "backend_1": make_stats("backend_1", 2, ["test-model"] * 2),
        "backend_2": make_stats("backend_2", 6, ["test-model"] * 6),
        "backend_3": make_stats("backend_3", 6, ["test-model"] * 6),
        "backend_4": make_stats("backend_4", 6, ["test-model"] * 6),
    }
    checks = verify_phase1(results, stats, elapsed=8.0)
    assert any(not c.passed and "backend_1" in c.message for c in checks)


# ── Phase 2 ───────────────────────────────────────────────────────────────────

def test_phase2_all_pass() -> None:
    results = make_results(20, 200)
    stats = {
        "backend_1": make_stats("backend_1", 6, ["model-a"] * 6),
        "backend_2": make_stats("backend_2", 4, ["model-a"] * 4),
        "backend_3": make_stats("backend_3", 7, ["model-b"] * 7),
        "backend_4": make_stats("backend_4", 3, ["model-b"] * 3),
    }
    checks = verify_phase2(results, stats)
    failures = [c for c in checks if not c.passed]
    assert failures == [], [c.message for c in failures]


def test_phase2_fails_when_wrong_model_on_backend() -> None:
    results = make_results(20, 200)
    stats = {
        "backend_1": make_stats("backend_1", 5, ["model-a"] * 5),
        "backend_2": make_stats("backend_2", 5, ["model-a"] * 5),
        "backend_3": make_stats("backend_3", 5, ["model-a"] * 3 + ["model-b"] * 2),
        "backend_4": make_stats("backend_4", 5, ["model-b"] * 5),
    }
    checks = verify_phase2(results, stats)
    assert any(not c.passed for c in checks)


def test_phase2_fails_when_hit_counts_wrong() -> None:
    results = make_results(20, 200)
    stats = {
        "backend_1": make_stats("backend_1", 8, ["model-a"] * 8),
        "backend_2": make_stats("backend_2", 8, ["model-a"] * 8),
        "backend_3": make_stats("backend_3", 2, ["model-b"] * 2),
        "backend_4": make_stats("backend_4", 2, ["model-b"] * 2),
    }
    checks = verify_phase2(results, stats)
    assert any(not c.passed for c in checks)


# ── Phase 3 ───────────────────────────────────────────────────────────────────

def test_phase3_all_pass() -> None:
    results = [RequestResult(status_code=200, body={})]
    stats = {
        "backend_5": make_stats("backend_5", 2, ["retry-model", "retry-model"]),
        "backend_1": make_stats("backend_1", 1, ["retry-model"]),
    }
    checks = verify_phase3(results, stats, elapsed=10.5)
    failures = [c for c in checks if not c.passed]
    assert failures == [], [c.message for c in failures]


def test_phase3_fails_when_no_200() -> None:
    # Also triggers the healthy-hits check (backend_1=0), but we only assert on 200.
    results = [RequestResult(status_code=502, body={})]
    stats = {
        "backend_5": make_stats("backend_5", 2, ["retry-model"] * 2),
        "backend_1": make_stats("backend_1", 0, []),
    }
    checks = verify_phase3(results, stats, elapsed=10.5)
    assert any(not c.passed and "200" in c.message for c in checks)


def test_phase3_fails_when_flaky_not_hit() -> None:
    results = [RequestResult(status_code=200, body={})]
    stats = {
        "backend_5": make_stats("backend_5", 0, []),
        "backend_1": make_stats("backend_1", 1, ["retry-model"]),
    }
    checks = verify_phase3(results, stats, elapsed=10.5)
    assert any(not c.passed and "backend_5" in c.message for c in checks)


def test_phase3_fails_when_elapsed_too_short() -> None:
    results = [RequestResult(status_code=200, body={})]
    stats = {
        "backend_5": make_stats("backend_5", 2, ["retry-model"] * 2),
        "backend_1": make_stats("backend_1", 1, ["retry-model"]),
    }
    checks = verify_phase3(results, stats, elapsed=3.0)
    assert any(not c.passed and "error_latency" in c.message for c in checks)
