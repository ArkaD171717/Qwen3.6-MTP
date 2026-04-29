from qwen_mtp.bugs import (
    BUG_REPORTS,
    check_prefix_cache_degradation,
    check_turboquant_conflict,
    get_critical_bugs,
    get_open_bugs,
)
from qwen_mtp.types import BugSeverity, BugStatus


def test_bug_reports_non_empty():
    assert len(BUG_REPORTS) >= 4


def test_l457_is_open():
    l457 = next(b for b in BUG_REPORTS if b.id == "L457")
    assert l457.status == BugStatus.OPEN
    assert l457.severity == BugSeverity.HIGH
    assert "38182" in l457.upstream_issue


def test_tq40831_is_closed():
    tq = next(b for b in BUG_REPORTS if b.id == "TQ-40831")
    assert tq.status == BugStatus.CLOSED
    assert tq.severity == BugSeverity.CRITICAL


def test_tq40880_is_closed():
    tq = next(b for b in BUG_REPORTS if b.id == "TQ-40880")
    assert tq.status == BugStatus.CLOSED


def test_ngram_bug_exists():
    ngram = next(b for b in BUG_REPORTS if b.id == "NGRAM-40875")
    assert "prompt_lookup_min" in ngram.workaround


def test_get_open_bugs():
    open_bugs = get_open_bugs()
    assert all(b.status == BugStatus.OPEN for b in open_bugs)
    assert any(b.id == "L457" for b in open_bugs)


def test_get_critical_bugs():
    critical = get_critical_bugs()
    assert all(b.severity == BugSeverity.CRITICAL for b in critical)


def test_turboquant_conflict_detected():
    bug = check_turboquant_conflict(enable_turboquant=True, num_spec_tokens=2)
    assert bug is not None
    assert bug.id == "TQ-40831"


def test_turboquant_no_conflict_when_disabled():
    assert check_turboquant_conflict(False, 2) is None


def test_turboquant_no_conflict_when_no_spec():
    assert check_turboquant_conflict(True, 0) is None


def test_prefix_cache_degradation_detected():
    bug = check_prefix_cache_degradation(enable_prefix_cache=True, num_spec_tokens=1)
    assert bug is not None
    assert bug.id == "L457"


def test_prefix_cache_no_issue_when_off():
    assert check_prefix_cache_degradation(False, 2) is None


def test_prefix_cache_no_issue_when_no_spec():
    assert check_prefix_cache_degradation(True, 0) is None
