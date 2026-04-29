from qwen3_6_mtp.bench import ACCEPTANCE_RATES, generate_benchmark_data
from qwen3_6_mtp.types import BenchmarkPoint


def test_generates_data():
    data = generate_benchmark_data()
    # 7 batch sizes * 6 spec token counts * 2 prefix cache settings = 84
    assert len(data) == 84
    for point in data:
        assert isinstance(point, BenchmarkPoint)
        assert point.batch_size > 0
        assert point.num_spec_tokens >= 0
        assert isinstance(point.prefix_cache, bool)
        assert point.latency_ms > 0
        assert point.throughput_tps > 0
        assert 0.0 <= point.acceptance_rate <= 1.0
        assert 0 <= point.kv_cache_util_pct <= 100


def test_default_sweep_dimensions():
    data = generate_benchmark_data()
    batch_sizes = {d.batch_size for d in data}
    spec_tokens = {d.num_spec_tokens for d in data}
    assert batch_sizes == {1, 2, 4, 8, 16, 32, 64}
    assert spec_tokens == {0, 1, 2, 3, 4, 5}


def test_both_prefix_cache_settings():
    data = generate_benchmark_data()
    prefix_values = {d.prefix_cache for d in data}
    assert prefix_values == {True, False}


def test_custom_batch_sizes():
    data = generate_benchmark_data(batch_sizes=[1, 4], spec_token_counts=[0, 2])
    batch_sizes = {d.batch_size for d in data}
    assert batch_sizes == {1, 4}


def test_baseline_throughput_positive():
    data = generate_benchmark_data()
    baselines = [d for d in data if d.num_spec_tokens == 0]
    assert all(d.throughput_tps > 0 for d in baselines)


def test_latency_positive():
    data = generate_benchmark_data()
    assert all(d.latency_ms > 0 for d in data)


def test_kv_util_bounded():
    data = generate_benchmark_data()
    assert all(0 <= d.kv_cache_util_pct <= 100 for d in data)


def test_acceptance_rate_baseline():
    data = generate_benchmark_data()
    baselines = [d for d in data if d.num_spec_tokens == 0]
    assert all(d.acceptance_rate == 1.0 for d in baselines)


def test_acceptance_rate_decreases_with_spec_tokens():
    assert ACCEPTANCE_RATES[1] > ACCEPTANCE_RATES[3]
    assert ACCEPTANCE_RATES[3] > ACCEPTANCE_RATES[5]


def test_datacenter_gpu_faster():
    consumer = generate_benchmark_data(gpu_id="rtx-3090")
    dc = generate_benchmark_data(gpu_id="h100-sxm")
    c_bs1 = next(
        d
        for d in consumer
        if d.batch_size == 1 and d.num_spec_tokens == 0 and not d.prefix_cache
    )
    d_bs1 = next(
        d
        for d in dc
        if d.batch_size == 1 and d.num_spec_tokens == 0 and not d.prefix_cache
    )
    assert d_bs1.throughput_tps > c_bs1.throughput_tps


def test_mtp_reduces_latency_at_batch1():
    data = generate_benchmark_data()
    baseline = next(
        d
        for d in data
        if d.batch_size == 1 and d.num_spec_tokens == 0 and not d.prefix_cache
    )
    mtp2 = next(
        d
        for d in data
        if d.batch_size == 1 and d.num_spec_tokens == 2 and not d.prefix_cache
    )
    assert mtp2.latency_ms < baseline.latency_ms
