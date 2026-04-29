from qwen3_6_mtp.backends.sglang import sglang_mtp_command
from qwen3_6_mtp.backends.vllm import vllm_mtp_command
from qwen3_6_mtp.types import Backend


class TestVllmBackend:
    def test_default_command(self):
        cfg = vllm_mtp_command()
        assert "mtp" in cfg.command
        assert '"method": "mtp"' in cfg.command or '"method":"mtp"' in cfg.command
        assert Backend.VLLM == cfg.backend

    def test_uses_mtp_not_qwen3_next_mtp(self):
        cfg = vllm_mtp_command()
        assert "qwen3_next_mtp" not in cfg.command
        assert "mtp" in cfg.command

    def test_speculative_tokens_in_command(self):
        cfg = vllm_mtp_command(num_speculative_tokens=3)
        assert (
            '"num_speculative_tokens": 3' in cfg.command
            or '"num_speculative_tokens":3' in cfg.command
        )

    def test_no_prefix_caching_by_default(self):
        cfg = vllm_mtp_command()
        assert "--no-enable-prefix-caching" in cfg.command

    def test_prefix_caching_enabled(self):
        cfg = vllm_mtp_command(enable_prefix_caching=True)
        assert "--enable-prefix-caching" in cfg.command
        assert any("L457" in c for c in cfg.caveats)

    def test_prefix_caching_no_caveat_without_spec(self):
        cfg = vllm_mtp_command(num_speculative_tokens=0, enable_prefix_caching=True)
        assert not any("L457" in c for c in cfg.caveats)

    def test_tensor_parallel(self):
        cfg = vllm_mtp_command(tensor_parallel=4)
        assert "--tensor-parallel-size 4" in cfg.command

    def test_reasoning_parser(self):
        cfg = vllm_mtp_command()
        assert "--reasoning-parser qwen3" in cfg.command

    def test_extra_flags(self):
        cfg = vllm_mtp_command(extra_flags=["--enforce-eager"])
        assert "--enforce-eager" in cfg.command

    def test_zero_spec_tokens(self):
        cfg = vllm_mtp_command(num_speculative_tokens=0)
        assert "--speculative-config" not in cfg.command


class TestSglangBackend:
    def test_default_command(self):
        cfg = sglang_mtp_command()
        assert "NEXTN" in cfg.command
        assert Backend.SGLANG == cfg.backend

    def test_step_count_is_tokens_plus_one(self):
        cfg = sglang_mtp_command(num_speculative_tokens=2)
        assert "--speculative-num-steps 3" in cfg.command

    def test_draft_tokens(self):
        cfg = sglang_mtp_command(num_speculative_tokens=2)
        assert "--speculative-num-draft-tokens 4" in cfg.command

    def test_eagle_topk_one(self):
        cfg = sglang_mtp_command()
        assert "--speculative-eagle-topk 1" in cfg.command

    def test_reasoning_parser(self):
        cfg = sglang_mtp_command()
        assert "--reasoning-parser qwen3" in cfg.command

    def test_zero_spec_tokens(self):
        cfg = sglang_mtp_command(num_speculative_tokens=0)
        assert "--speculative-algo" not in cfg.command

    def test_tensor_parallel(self):
        cfg = sglang_mtp_command(tensor_parallel=8)
        assert "--tp-size 8" in cfg.command

    def test_apple_silicon_caveat(self):
        cfg = sglang_mtp_command()
        assert any("Apple Silicon" in c for c in cfg.caveats)
