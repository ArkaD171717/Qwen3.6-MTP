from qwen3_6_mtp.bench import generate_benchmark_data
from qwen3_6_mtp.crossover import (
    find_crossover_points,
    quick_crossover,
    summarize_crossovers,
)


def test_find_crossover_points():
    data = generate_benchmark_data()
    points = find_crossover_points(data)
    assert len(points) > 0


def test_crossover_points_exclude_baseline():
    data = generate_benchmark_data()
    points = find_crossover_points(data)
    assert all(p.spec_tokens > 0 for p in points)


def test_crossover_delta_range():
    data = generate_benchmark_data()
    points = find_crossover_points(data)
    for p in points:
        assert -50 < p.delta_pct < 150


def test_small_batch_generally_positive():
    data = generate_benchmark_data()
    points = find_crossover_points(data)
    bs1_mtp1 = [p for p in points if p.batch_size == 1 and p.spec_tokens == 1]
    assert len(bs1_mtp1) > 0
    assert any(p.is_net_positive for p in bs1_mtp1)


def test_summarize_crossovers():
    data = generate_benchmark_data()
    points = find_crossover_points(data)
    summaries = summarize_crossovers(points)
    assert len(summaries) == 5
    assert summaries[0].spec_tokens == 1
    assert summaries[4].spec_tokens == 5


def test_summary_max_delta_positive():
    data = generate_benchmark_data()
    points = find_crossover_points(data)
    summaries = summarize_crossovers(points)
    for s in summaries:
        assert s.max_positive_delta_pct >= 0


def test_mtp1_no_crossover_without_prefix_cache():
    data = generate_benchmark_data()
    points = find_crossover_points(data)
    summaries = summarize_crossovers(points, prefix_cache=False)
    mtp1 = summaries[0]
    assert mtp1.crossover_batch_size is None


def test_higher_spec_tokens_crossover_earlier_with_prefix_cache():
    data = generate_benchmark_data()
    points = find_crossover_points(data)
    summaries = summarize_crossovers(points, prefix_cache=True)
    with_crossover = [s for s in summaries if s.crossover_batch_size is not None]
    assert len(with_crossover) >= 2
    sorted_by_tokens = sorted(with_crossover, key=lambda s: s.spec_tokens)
    assert (
        sorted_by_tokens[-1].crossover_batch_size
        <= sorted_by_tokens[0].crossover_batch_size
    )


def test_quick_crossover():
    summaries = quick_crossover()
    assert len(summaries) == 5
    assert all(s.spec_tokens >= 1 for s in summaries)


def test_quick_crossover_with_gpu():
    summaries = quick_crossover(gpu_id="h100-sxm")
    assert len(summaries) == 5
